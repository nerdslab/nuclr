from typing import Dict, Iterator, List, Optional, Tuple
import math
import numpy as np
import torch
import torch.distributed as dist
from temporaldata import Interval
from torch_brain.data.dataset import DatasetIndex


class RandomTwoViewSampler(torch.utils.data.Sampler):
    """Samples pairs of temporal windows from recordings for contrastive learning.

    Note: The number of samples may vary between epochs due to random sampling.

    Args:
        sampling_intervals: Dictionary mapping recording IDs to Interval objects
            specifying valid time ranges for sampling.
        window_length: Length of each sampled window in seconds.
        max_time_diff: Maximum allowed time difference between paired windows.
            If None, no limit is imposed.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        max_time_diff: Optional[float],  # None => no limit
        seed: int,
    ):

        self.sampling_intervals = sampling_intervals
        self._sanitize_intervals(2 * window_length)

        self.window_length = window_length
        self.init_seed = seed
        self.np_rng = np.random.default_rng(seed)
        self.max_time_diff = max_time_diff

        self._indices_cache = None

    def _sanitize_intervals(self, min_length: float):
        """Remove interval periods that are shorter than min_length"""
        for rec_id, interval in self.sampling_intervals.items():
            valid_idx = (interval.end - interval.start) >= min_length

            if not valid_idx.any():
                raise ValueError(
                    f"None of the periods in sampling interval for {rec_id} "
                    "are long enough."
                )

            self.sampling_intervals[rec_id] = Interval(
                interval.start[valid_idx], interval.end[valid_idx]
            )

    def _get_one_recording_sample_indices(
        self,
        rec_id: str,
    ) -> List[Tuple[DatasetIndex, DatasetIndex]]:
        """Returns sampling indices for a single recording"""

        interval = self.sampling_intervals[rec_id]
        sample_indices = []
        for start, end in zip(interval.start, interval.end):
            offset = self.np_rng.uniform(0, self.window_length)

            for _start in np.arange(
                start + offset, end - self.window_length, self.window_length
            ):
                # First view
                view1 = DatasetIndex(rec_id, _start, _start + self.window_length)

                # Second view
                if self.max_time_diff == 0.0:
                    sample_indices.append((view1, view1))
                else:
                    # 1. Construct Interval from which to sample
                    view2_samp_interval = interval
                    if self.max_time_diff is not None:
                        view2_samp_interval &= Interval(
                            view1.start - self.max_time_diff,
                            view1.end + self.max_time_diff,
                        )
                    # 2. Sample and check
                    view2_interval = self._sample_window_from_interval(
                        view2_samp_interval
                    )
                    if view2_interval is not None:
                        view2 = DatasetIndex(
                            rec_id,
                            view2_interval.start[0],
                            view2_interval.end[0],
                        )
                        sample_indices.append((view1, view2))

        assert len(sample_indices) > 0

        return sample_indices

    def _sample_window_from_interval(self, interval: Interval) -> Optional[Interval]:
        """
        Algo:
            An interval is made of smaller periods:
            [(intrvl.start[0], intrvl.end[0]), (intrvl.start[1], intrvl.end[1]), ...]
            So first we decide which period to sample from, and then sample from it.
        """

        # 1. Measure duration from which "start" time can be sampled from each period
        start_sample_durs = interval.end - interval.start - self.window_length
        start_sample_durs = start_sample_durs.clip(0)
        if not (start_sample_durs > 0).any():
            return None

        # 2. Choose which period to sample from
        probs = start_sample_durs / sum(start_sample_durs)
        period_idx = self.np_rng.choice(len(interval), p=probs)
        period_start, period_end = interval.start[period_idx], interval.end[period_idx]

        # 3. Sample from the chosen period
        start = self.np_rng.uniform(period_start, period_end - self.window_length)
        return Interval(start, start + self.window_length)

    def _prepare_indices_cache(self):
        if self._indices_cache is not None:
            return

        self._indices_cache = []
        for rec_id in self.sampling_intervals.keys():
            self._indices_cache.extend(self._get_one_recording_sample_indices(rec_id))

        self.np_rng.shuffle(self._indices_cache)

    def set_epoch(self, epoch: int):
        self.np_rng = np.random.default_rng(self.init_seed + epoch)

    def __len__(self) -> int:
        self._prepare_indices_cache()
        return len(self._indices_cache)

    def __iter__(self) -> Iterator[Tuple[DatasetIndex, DatasetIndex]]:
        self._prepare_indices_cache()
        for index in self._indices_cache:
            yield index
        self._indices_cache = None


class DistributedSamplerWrapper(torch.utils.data.Sampler):
    """Wrapper for distributing any sampler across multiple processes.

    This wrapper takes an existing sampler and distributes its indices across
    multiple replicas (processes) in a distributed training setup. It ensures
    each replica gets a unique subset of the data by splitting the indices
    evenly across replicas.

    Note: This wrapper supports samplers that may return a different number of
    samples each epoch, as it recomputes the indices on each iteration.

    Note: If the length of the sampler is not a multiple of num_replicas, some
    samples will be dropped to ensure equal distribution across replicas.

    Args:
        sampler: The base sampler to wrap
        num_replicas: Number of processes in distributed training
        rank: Rank of the current process
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        self.sampler = sampler

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank
        self._indices_cache = None

    def _prepare_indices_cache(self):
        if self._indices_cache is not None:
            return

        indices = list(self.sampler)

        # Force len(indices) to be a multiple of num_replicas
        num_samples_per_rank = math.floor(len(indices) / self.num_replicas)
        total_size = num_samples_per_rank * self.num_replicas
        indices = indices[:total_size]

        self._indices_cache = indices[self.rank : total_size : self.num_replicas]

    def __len__(self):
        self._prepare_indices_cache()
        return len(self._indices_cache)

    def __iter__(self):
        self._prepare_indices_cache()
        for idx in self._indices_cache:
            yield idx
        self._indices_cache = None

    def set_epoch(self, epoch: int):
        """Pass the epoch to the underlying sampler if it supports it."""
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class RandomFixedWindowSampler(torch.utils.data.Sampler):
    """Samples pairs of temporal windows from recordings for contrastive learning.

    Note: The number of samples may vary between epochs due to random sampling.

    Args:
        sampling_intervals: Dictionary mapping recording IDs to Interval objects
            specifying valid time ranges for sampling.
        window_length: Length of each sampled window in seconds.
        max_time_diff: Maximum allowed time difference between paired windows.
            If None, no limit is imposed.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        sampling_intervals: Dict[str, Interval],
        window_length: float,
        seed: int,
    ):

        self.sampling_intervals = sampling_intervals
        self.window_length = window_length
        self.init_seed = seed
        self.np_rng = np.random.default_rng(seed)

        self._indices_cache = None

    def set_epoch(self, epoch: int):
        self.np_rng = np.random.default_rng(self.init_seed + epoch)

    def _get_one_recording_sample_indices(
        self,
        rec_id: str,
    ) -> List[Tuple[DatasetIndex, DatasetIndex]]:
        """Returns sampling indices for a single recording"""

        interval = self.sampling_intervals[rec_id]
        sample_indices = []
        for start, end in zip(interval.start, interval.end):
            offset = self.np_rng.uniform(0, self.window_length)

            for _start in np.arange(
                start + offset, end - self.window_length, self.window_length
            ):
                # First view
                sample_indices.append(
                    DatasetIndex(rec_id, _start, _start + self.window_length)
                )

        assert len(sample_indices) > 0
        return sample_indices

    def _prepare_indices_cache(self):
        if self._indices_cache is not None:
            return

        self._indices_cache = []
        for rec_id in self.sampling_intervals.keys():
            self._indices_cache.extend(self._get_one_recording_sample_indices(rec_id))

        self.np_rng.shuffle(self._indices_cache)

    def __len__(self) -> int:
        self._prepare_indices_cache()
        return len(self._indices_cache)

    def __iter__(self) -> Iterator[Tuple[DatasetIndex, DatasetIndex]]:
        self._prepare_indices_cache()
        for index in self._indices_cache:
            yield index
        self._indices_cache = None

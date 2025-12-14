from pathlib import Path
from tqdm import tqdm
import pickle
import torch
import numpy as np


def load_emb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    r"""Load embeddings from a single checkpoint"""
    data = torch.load(path, weights_only=False, map_location="cpu")
    return data["embs"].float().numpy(), data["ids"]


def load_embs(embs_dir: Path) -> tuple[list[np.ndarray], np.ndarray]:
    r"""Load embeddings from a directory full of checkpoints"""

    if not embs_dir.is_dir():
        embs, ids = load_emb(embs_dir)
        return [embs], ids

    paths = sorted(embs_dir.iterdir(), key=lambda x: int(x.stem.split("_")[-1]))
    print(f"Found {len(paths)} checkpoints")

    embs = []
    ids = None
    for path in tqdm(paths):
        _embs, _ids = load_emb(path)
        embs.append(_embs)

        if ids is None:
            ids = _ids
        else:
            assert (ids == _ids).all()

    assert ids is not None
    return embs, ids


def load_splits(splits_filepath: Path):
    splits_file = Path(splits_filepath)
    with open(splits_file, "rb") as f:
        splits_data = pickle.load(f)

    label_map = splits_data["label_map"]
    splits = splits_data["splits"]

    return label_map, splits


def presence_mask(check_ids, base_ids):
    """
    Returns a bool array indicating if elements of `check_ids` are present
    in `base_ids`.
    Output shape: (len(check_ids),)
    """
    base_ids_set = set(base_ids)
    return np.array([x in base_ids_set for x in check_ids])

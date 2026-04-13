# Allen VBN 2022 VIS Pipeline Domain Review

This note reviews how `preprocess/allen_vbn_2022_vis/pipeline.py` and
`preprocess/allen_vbn_2022_vis/session_extractor.py` currently define the
top-level HDF5 `domain`, how that differs from the other pipelines in this
repo, and what downstream training/evaluation code will do with that domain.

## Short Answer

The current VBN pipeline is probably not using the `domain` in the same way as
the established pipelines.

Right now, the VBN top-level `Data.domain` is a single continuous interval from:

1. The first spike among the filtered VIS units, to
2. The last optotagging pulse stop time.

That means downstream samplers can draw training/evaluation windows from
optotagging time and from any long gaps between visual behavior/mapping/replay
and optotagging. The other ephys pipelines generally treat `Data.domain` as the
valid neural sampling interval, often trimming off optotagging/control tail time
and sometimes excluding invalid/dropout periods.

The key thing to keep in mind: in this repo, `Data.domain` is not just passive
metadata. It is what the training and forward-pass samplers use as the interval
from which to generate fixed windows.

## Where The Current VBN Domain Is Built

Look first at:

- `preprocess/allen_vbn_2022_vis/pipeline.py`
- `preprocess/allen_vbn_2022_vis/session_extractor.py`

Important sections:

- `preprocess/allen_vbn_2022_vis/pipeline.py:142`
  calls `extract_session_data(...)`.
- `preprocess/allen_vbn_2022_vis/pipeline.py:163`
  creates the top-level `Data(...)`.
- `preprocess/allen_vbn_2022_vis/pipeline.py:170`
  writes `domain=domain` into that top-level `Data(...)`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:35`
  defines `extract_spikes(...)`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:56`
  creates `spikes = IrregularTimeSeries(..., domain="auto")`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:72`
  defines `interval_sorter(...)`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:80`
  reads `session.optotagging_table`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:81`
  creates `opto_intervals = Interval(...)`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:89`
  creates `intervals = Data(optotagging=opto_intervals, domain="auto")`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:96`
  defines `extract_session_data(...)`.
- `preprocess/allen_vbn_2022_vis/session_extractor.py:108`
  creates the current top-level domain:

```python
domain = Interval(start = spikes.domain.start[0], # Beginning of image change task
                  end = intervals.optotagging.end[-1]) # End of optotagging task
```

That comment is misleading. `spikes.domain.start[0]` is the first spike among the
filtered VIS units, not necessarily the beginning of the image-change task.

Because `intervals.optotagging.end[-1]` is the end of the last optotagging
interval, the top-level HDF5 domain includes optotagging by construction.

## What `domain="auto"` Means Here

The VBN extractor uses `domain="auto"` when building `spikes`:

- `preprocess/allen_vbn_2022_vis/session_extractor.py:56`

In `temporaldata`, `IrregularTimeSeries(domain="auto")` makes the time series
domain span from the first timestamp to the last timestamp. See local package
code:

- `.venv/Lib/site-packages/temporaldata/irregular_ts.py:100`

So, for spikes, `spikes.domain.start[0]` is just the minimum selected spike time.
It is not task-aware and does not know about visual behavior, passive replay,
mapping, invalid times, or optotagging.

The intervals object also uses `domain="auto"`:

- `preprocess/allen_vbn_2022_vis/session_extractor.py:89`

Since that `Data` object currently contains only `optotagging`, its automatic
domain is the optotagging interval union. It does not describe the visual task.

## Why This Matters Downstream

The downstream code uses the top-level `Data.domain` as the sampling interval
unless a split-specific domain exists.

Key files:

- `.venv/Lib/site-packages/torch_brain/data/dataset.py`
- `src/trainers/contrastive.py`
- `src/trainers/forward.py`
- `src/samplers.py`
- `.venv/Lib/site-packages/torch_brain/data/sampler.py`

Important sections:

- `.venv/Lib/site-packages/torch_brain/data/dataset.py:347`
  defines `Dataset.get_sampling_intervals()`.
- `.venv/Lib/site-packages/torch_brain/data/dataset.py:356`
  chooses `domain` when no split is provided.
- `.venv/Lib/site-packages/torch_brain/data/dataset.py:360`
  reads that interval from the loaded HDF5 `Data` object.
- `.venv/Lib/site-packages/torch_brain/data/dataset.py:293`
  defines `Dataset.get(...)`.
- `.venv/Lib/site-packages/torch_brain/data/dataset.py:305`
  slices the recording with `sample = data.slice(start, end)`.
- `src/trainers/contrastive.py:247`
  creates the training sampler from `ds.sanitized_sampling_intervals`.
- `src/trainers/contrastive.py:303`
  creates the validation sampler from `ds.get_sampling_intervals()`.
- `src/trainers/forward.py:99`
  reads `sampling_intervals = ds.get_sampling_intervals()`.
- `src/trainers/forward.py:117`
  uses `SequentialFixedWindowSampler(...)` over those intervals.

The practical consequence is:

If the top-level VBN HDF5 domain is `[first_VIS_spike, last_optotagging_stop]`,
then contrastive training, validation, and forward-pass embedding extraction can
sample fixed windows anywhere in that entire interval.

That includes:

- The visual behavior task.
- Passive replay or mapping blocks, if they fall inside the interval.
- Gaps between blocks.
- Optotagging.
- Any late tail period before or during optotagging.

The model tokenizers will then operate on whatever spikes exist in the sampled
window:

- `src/models/nuclr_v2ga_spikes.py:197`
  defines spike-token model tokenization.
- `src/models/nuclr_v2ga_spikes.py:200`
  computes `active_units` from units that fired in the sampled window.
- `src/models/nuclr_v2ga_spikes.py:212`
  sets `unit_seqlen` to `len(active_units)`.
- `src/models/nuclr_v2ga_bin.py:143`
  defines binned-spike model tokenization.
- `src/models/nuclr_v2ga_bin.py:146`
  computes `active_units`.
- `src/models/nuclr_v2ga_bin.py:167`
  sets `unit_seqlen` to `len(active_units)`.

Empty or very low-spike windows are partly filtered later by `ViewData` /
`TwoViewData`, but they still waste sampler capacity and can bias which windows
actually contribute. For two-view training, windows with too few active units can
become `None` in `TwoViewData.from_two_views(...)`.

## Comparison With Existing Pipelines

### Allen Visual Coding 2019

File:

- `preprocess/allen_vc_2019_vis/pipeline.py`

Important sections:

- `preprocess/allen_vc_2019_vis/pipeline.py:100`
  extracts spikes.
- `preprocess/allen_vc_2019_vis/pipeline.py:102`
  extracts invalid intervals.
- `preprocess/allen_vc_2019_vis/pipeline.py:105`
  extracts stimulus epochs.
- `preprocess/allen_vc_2019_vis/pipeline.py:109`
  tries to extract optotagging info.
- `preprocess/allen_vc_2019_vis/pipeline.py:110`
  finds the first optotagging start.
- `preprocess/allen_vc_2019_vis/pipeline.py:111`
  slices spikes before optotagging with a 5 minute gap:

```python
spikes = spikes.slice(0, optotag_start - 300.0)  # 5 minute gap
```

- `preprocess/allen_vc_2019_vis/pipeline.py:123`
  writes:

```python
domain=spikes.domain.difference(invalid_interval)
```

This is the opposite of the current VBN behavior. Allen VC removes the
optotagging tail from the spikes and then defines the sampling domain as the
remaining spike domain minus invalid intervals.

### IBL Brainwide Map

File:

- `preprocess/ibl_bwm/pipeline.py`

Important sections:

- `preprocess/ibl_bwm/pipeline.py:120`
  extracts spikes and units.
- `preprocess/ibl_bwm/pipeline.py:122`
  defines:

```python
domain = Interval(spikes.domain.start[0], spikes.domain.end[0] - 300.0)
```

- `preprocess/ibl_bwm/pipeline.py:134`
  writes that top-level `domain`.

This is also a sampling-domain convention: the pipeline trims the last 5 minutes
because the last few minutes can be unreliable or empty.

### Steinmetz 2019

Files:

- `preprocess/steinmetz_2019/prepare_data.py`
- `preprocess/steinmetz_2019/steinmetz_split_probes.py`
- `utils/split_probes.py`

Important sections:

- `preprocess/steinmetz_2019/prepare_data.py:57`
  creates the spike time series.
- `preprocess/steinmetz_2019/prepare_data.py:89`
  writes `domain=spikes.domain`.
- `preprocess/steinmetz_2019/steinmetz_split_probes.py:48`
  starts a per-probe invalid-domain calculation.
- `preprocess/steinmetz_2019/steinmetz_split_probes.py:55`
  creates a per-probe domain from that probe's first and last spike.
- `preprocess/steinmetz_2019/steinmetz_split_probes.py:56`
  subtracts invalid/dropout intervals.
- `preprocess/steinmetz_2019/steinmetz_split_probes.py:63`
  writes the per-probe spike domain.
- `preprocess/steinmetz_2019/steinmetz_split_probes.py:65`
  also sets the top-level `_domain` to the same per-probe valid domain.

The generic splitter does similar work:

- `utils/split_probes.py:20`
  reads `data.units.probe_id`.
- `utils/split_probes.py:26`
  changes `brainset.id` to `<old>_probes`.
- `utils/split_probes.py:27`
  changes session IDs to `<session>_p<i>`.
- `utils/split_probes.py:54`
  builds a per-probe domain from that probe's first and last spike.
- `utils/split_probes.py:55`
  subtracts inferred zero-spike/dropout intervals.
- `utils/split_probes.py:62`
  sets the per-probe spike domain.
- `utils/split_probes.py:64`
  sets the top-level `_domain` to the per-probe valid domain.

This matters for VBN because the repo's ephys training configs are generally
named around probe-split datasets. If VBN is eventually split with
`utils/split_probes.py`, the splitter will recompute the domain from each probe's
spikes. If the unsplit VBN file still contains optotagging spikes, the splitter
can reintroduce optotagging into the probe-level domains unless the spikes are
trimmed before splitting or the splitter intersects with the parent `data.domain`.

### Bugeon 2022

File:

- `preprocess/bugeon/prepare_data.py`

Important sections:

- `preprocess/bugeon/prepare_data.py:177`
  builds multiple experiment traces.
- `preprocess/bugeon/prepare_data.py:183`
  starts with the first experiment domain.
- `preprocess/bugeon/prepare_data.py:185`
  unions each later experiment domain:

```python
domain = domain | calcium_traces_list[i].domain
```

- `preprocess/bugeon/prepare_data.py:201`
  writes `domain=calcium_traces.domain`.

This is a useful pattern for VBN if the valid visual epochs are naturally
discontinuous. You do not need to force everything into one continuous interval.
You can represent multiple valid blocks as an `Interval` with multiple
start/end pairs.

## Current VBN Risks

### 1. Optotagging Is In The Sampling Domain

The current end point is `intervals.optotagging.end[-1]`.

That means optotagging windows are eligible for standard NuCLR training and
forward-pass embeddings. The Allen VC pipeline explicitly avoids this by slicing
before optotagging with a 5 minute gap.

If optotagging is only meant to label or characterize units, it should probably
be stored as an auxiliary interval but excluded from the default top-level
sampling domain.

### 2. The Start Time Is Not Actually The Image-Change Task Start

The current start point is `spikes.domain.start[0]`.

That is the first spike among filtered VIS units. It can be close to the
recording start, but it is not guaranteed to be the start of the behavioral task
or the first visual stimulus. If you intend the domain to mean "the active visual
behavior epoch", this is the wrong source.

Use `session.trials` or `session.stimulus_presentations` to define task epochs.

Relevant AllenSDK properties in the installed local package:

- `.venv/Lib/site-packages/allensdk/brain_observatory/behavior/behavior_session.py:1067`
  `stimulus_presentations`
- `.venv/Lib/site-packages/allensdk/brain_observatory/behavior/behavior_session.py:1074`
  notes that presentations are divided by `stimulus_block` and
  `stimulus_block_name`.
- `.venv/Lib/site-packages/allensdk/brain_observatory/behavior/behavior_session.py:1142`
  documents the `active` column.
- `.venv/Lib/site-packages/allensdk/brain_observatory/behavior/behavior_session.py:1271`
  `trials`
- `.venv/Lib/site-packages/allensdk/brain_observatory/ecephys/behavior_ecephys_session.py:338`
  `optotagging_table`
- `.venv/Lib/site-packages/allensdk/brain_observatory/ecephys/behavior_ecephys_session.py:372`
  `spike_times`
- `.venv/Lib/site-packages/allensdk/brain_observatory/ecephys/behavior_ecephys_session.py:394`
  `get_channels(...)`
- `.venv/Lib/site-packages/allensdk/brain_observatory/ecephys/behavior_ecephys_session.py:413`
  `get_units(...)`

### 3. Visual Behavior Intervals Are Not Stored Yet

`interval_sorter(...)` currently stores only:

```python
intervals = Data(
    optotagging = opto_intervals,
    domain = "auto"
)
```

So the output HDF5 has optotagging intervals, but not the actual visual behavior
stimulus presentations, trial intervals, or stimulus-block intervals.

This makes it hard to later answer:

- Which windows came from active behavior?
- Which windows came from passive replay?
- Which windows came from mapping?
- Which image or omitted-flash events were inside a sampled window?
- Which time periods should be eligible for training?

### 4. Individual Stimulus Presentations Are Not A Good Default Domain By Themselves

For VBN, `session.stimulus_presentations` is event-like: individual image
presentations or omitted flashes. Those intervals are often much shorter than a
NuCLR context window.

If you make the top-level domain the raw union of every individual flash
interval, most fixed windows may be too long to fit inside any single interval.
The samplers will drop short intervals.

Better pattern:

- Store individual presentations as `intervals.stimulus_presentations`.
- Store trials as `intervals.trials`.
- Store larger block-level intervals as something like
  `intervals.stimulus_blocks` or `intervals.visual_behavior`.
- Use block-level or trial-level intervals for the default top-level
  `Data.domain`.

### 5. Split-Probe Processing Can Undo A Parent-Domain-Only Fix

If you only change the unsplit VBN file's top-level domain while keeping all
spikes, then `utils/split_probes.py` may undo that choice later.

Why:

- `utils/split_probes.py:39`
  selects spikes for one probe.
- `utils/split_probes.py:54`
  computes `valid_domain = Interval(new_timestamps[0], new_timestamps[-1])`.
- `utils/split_probes.py:64`
  sets the new top-level domain to this per-probe spike-derived interval.

So if `new_timestamps` includes optotagging spikes, the split probe file can
again include optotagging in its domain.

If VBN will use probe-split files, either:

1. Trim or select the spikes themselves before writing the unsplit HDF5, or
2. Change `utils/split_probes.py` or a VBN-specific splitter to intersect the
   per-probe valid domain with the original parent `data.domain`:

```python
valid_domain = valid_domain.difference(invalid_domain)
valid_domain = valid_domain & data.domain
```

That preserves the semantic sampling domain from the parent file while still
removing probe dropouts.

### 6. Config Naming Is A Little Misleading Right Now

The VBN data configs are named with `probes`, but they currently point at the
unsplit brainset ID:

- `configs/data/train_dataset/allen_vb_vis_probes_all.yaml:2`
- `configs/data/train_dataset/allen_vb_vis_probes_medium.yaml:2`
- `configs/data/train_dataset/allen_vb_vis_probes_tiny.yaml:2`
- `configs/data/train_dataset/allen_vb_vis_probes_transductive.yaml:2`
- `configs/data/val_dataset/allen_vb_vis_probes_test.yaml:2`
- `configs/data/val_dataset/allen_vb_vis_test.yaml:2`

Each uses:

```yaml
brainset: "allen_vbn_2022"
```

Meanwhile the generic splitter would produce `allen_vbn_2022_probes` because it
sets:

```python
new_brainset_id = old_brainset_id + "_probes"
```

See:

- `utils/split_probes.py:15`
- `utils/split_probes.py:26`

This is not strictly a domain bug, but it is part of the same "how the HDF5 files
are organized downstream" question.

## Recommended Direction

I would separate two ideas:

1. Auxiliary event intervals: what happened when.
2. Sampling domain: where NuCLR is allowed to sample windows by default.

For VBN, that probably means:

- Keep optotagging as an auxiliary interval.
- Add visual behavior/task intervals.
- Set top-level `Data.domain` to a visual/neural sampling domain that excludes
  optotagging unless you intentionally want optotagging windows in training.

### Recommended Stored Intervals

In `preprocess/allen_vbn_2022_vis/session_extractor.py`, extend
`interval_sorter(session)` to return something closer to:

```python
intervals = Data(
    stimulus_presentations=stimulus_presentations,
    trials=trials,
    stimulus_blocks=stimulus_blocks,
    optotagging=opto_intervals,
    domain="auto",
)
```

Candidate sources:

- `session.stimulus_presentations`
  for per-presentation intervals and metadata.
- `session.trials`
  for active task trial start/stop intervals.
- `session.optotagging_table`
  for optotagging intervals.

Metadata worth preserving from `stimulus_presentations`, depending on available
columns:

- `stimulus_block`
- `stimulus_block_name`
- `active`
- `image_name`
- `omitted`
- `is_change`
- `flashes_since_change`
- `duration`

Metadata worth preserving from `trials`, depending on available columns:

- `go`
- `catch`
- `hit`
- `miss`
- `false_alarm`
- `correct_reject`
- `aborted`
- `auto_rewarded`
- `change_time`
- `response_time`
- `initial_image_name`
- `change_image_name`

### Recommended Default Domain Choices

There are three reasonable choices. Pick based on what you want NuCLR to learn
from by default.

#### Option A: All Non-Optotag Neural Recording

Closest to Allen VC and IBL.

Use the spike domain, but end before optotagging:

```python
opto_start = intervals.optotagging.start.min()
domain = spikes.domain & Interval(spikes.domain.start[0], opto_start - 300.0)
```

This keeps all pre-opto neural windows: active behavior, passive replay, mapping,
and any gaps between them. It avoids optotagging but does not make the domain
strictly task-specific.

If you also keep optotagging spikes in `spikes`, remember to update
`utils/split_probes.py` or use a VBN-specific splitter so probe-level files
intersect with the parent domain.

#### Option B: Active Visual Behavior Only

Most semantically clean if the target dataset is "visual behavior".

Use `session.trials` or `stimulus_presentations.active` to build a block/trial
domain. Prefer trial or coalesced block intervals over individual stimulus
presentation intervals.

Sketch:

```python
trial_table = session.trials
active_domain = Interval(
    start=trial_table["start_time"].to_numpy(),
    end=trial_table["stop_time"].to_numpy(),
).coalesce(eps=1.0)

domain = active_domain & spikes.domain
```

The `eps` argument controls how close intervals must be to merge. Use a value
that matches the gaps you want treated as continuous task time.

#### Option C: All Visual Stimulus Blocks Except Optotagging

Good if you want active behavior plus passive replay/mapping, but not
optotagging.

Use `session.stimulus_presentations`, group or coalesce by `stimulus_block` /
`stimulus_block_name`, and exclude optotagging.

Sketch:

```python
presentations = session.stimulus_presentations

visual_blocks = []
for _, block in presentations.groupby("stimulus_block"):
    visual_blocks.append(
        Interval(
            start=block["start_time"].min(),
            end=block["stop_time"].max(),
        )
    )

domain = visual_blocks[0]
for block_domain in visual_blocks[1:]:
    domain = domain | block_domain

domain = domain & spikes.domain
```

This avoids the "individual flashes are too short" problem while preserving
discontinuous valid blocks.

## Implementation Notes

### Do Not Accidentally Shift Time Origins While Trimming

`temporaldata.Data.slice(...)` and `IrregularTimeSeries.slice(...)` default to
`reset_origin=True`.

Relevant package lines:

- `.venv/Lib/site-packages/temporaldata/data.py:208`
- `.venv/Lib/site-packages/temporaldata/data.py:239`
- `.venv/Lib/site-packages/temporaldata/irregular_ts.py:186`
- `.venv/Lib/site-packages/temporaldata/irregular_ts.py:217`

That is correct during downstream window extraction because each sampled window
should become time-relative to its own start. But during preprocessing, if you
slice spikes without slicing all interval metadata the same way, you can misalign
spikes and intervals.

Safer preprocessing patterns:

- Build a top-level `domain` without mutating timestamps.
- Or slice the entire `Data` object consistently.
- Or call `slice(..., reset_origin=False)` if you only want to remove samples
  while keeping global session timestamps.

Allen VC gets away with `spikes.slice(0, optotag_start - 300.0)` because the
slice starts at 0, so no time origin shift occurs.

### Validate The Domain Before Writing HDF5

Add explicit checks before `data.to_hdf5(...)`:

```python
assert domain.is_sorted()
assert domain.is_disjoint()
assert (domain.end > domain.start).all()
assert ((domain & intervals.optotagging).end - (domain & intervals.optotagging).start).sum() == 0
```

Also log:

```python
domain_duration = float((domain.end - domain.start).sum())
spike_duration = float((spikes.domain.end - spikes.domain.start).sum())
logging.info(f"domain duration: {domain_duration:.1f}s")
logging.info(f"spike domain duration: {spike_duration:.1f}s")
logging.info(f"domain intervals: {len(domain)}")
```

The optotagging-overlap assertion above may need a small helper if empty
intersections behave differently than expected, but the principle is important:
the default sampling domain should intentionally include or exclude optotagging,
not include it by accident.

### Watch Empty Sessions Or Empty Unit Filters

`extract_units(...)` filters to VIS units and by QC thresholds:

- `preprocess/allen_vbn_2022_vis/session_extractor.py:13`

`extract_spikes(...)` then loops over selected units:

- `preprocess/allen_vbn_2022_vis/session_extractor.py:35`

If a session has no units after filtering, `spike_times` will be empty and
`IrregularTimeSeries(domain="auto")` may fail or create an unusable domain. A
multi-session pipeline should guard this explicitly.

Suggested check:

```python
if len(selected_units) == 0:
    raise ValueError("No VIS units passed filtering")
```

or skip the session with a logged status, depending on how `brainsets prepare`
expects failures to be handled.

## Proposed Minimal Fix

If you want the smallest change that aligns VBN more closely with Allen VC:

1. Keep extracting optotagging intervals.
2. Add a pre-opto sampling domain.
3. Exclude optotagging from top-level `Data.domain`.
4. Make sure probe splitting does not reintroduce optotagging.

Conceptually:

```python
opto_start = intervals.optotagging.start.min()
domain = spikes.domain & Interval(spikes.domain.start[0], opto_start - 300.0)
```

Then either trim spikes before writing:

```python
spikes = spikes.slice(
    spikes.domain.start[0],
    opto_start - 300.0,
    reset_origin=False,
)
domain = spikes.domain
```

or update splitting to preserve the parent domain:

```python
valid_domain = valid_domain.difference(invalid_domain)
valid_domain = valid_domain & data.domain
```

The second approach lets the unsplit HDF5 retain optotagging spikes as auxiliary
data while keeping the default sampling domain pre-opto. The first approach
matches the current Allen VC behavior more closely.

## Proposed Better Fix

For a more complete VBN pipeline:

1. Extend `interval_sorter(session)` to include:
   - `stimulus_presentations`
   - `trials`
   - `stimulus_blocks`
   - `optotagging`
2. Decide the default sampling domain:
   - active trials only,
   - all visual blocks except optotagging,
   - or all pre-opto neural recording.
3. Store that domain as the top-level `Data.domain`.
4. Preserve optotagging as `intervals.optotagging`, but keep it out of
   `Data.domain` unless intentionally training on optotagging.
5. Update probe splitting so probe-level domains are:

```python
per_probe_valid_domain = spike_coverage_domain.difference(dropout_domain)
per_probe_valid_domain = per_probe_valid_domain & parent_data.domain
```

6. Add validation/logging for:
   - domain total duration,
   - number of domain intervals,
   - optotagging overlap,
   - number of windows available for the configured NuCLR context duration,
   - number of selected VIS units,
   - number of spikes after filtering/trimming.

## Exact Files To Look At

### VBN Pipeline

- `preprocess/allen_vbn_2022_vis/pipeline.py`
  - `Pipeline.get_manifest(...)`
  - `Pipeline.download(...)`
  - `Pipeline.process(...)`
  - especially lines `142`, `163`, and `170`
- `preprocess/allen_vbn_2022_vis/session_extractor.py`
  - `extract_units(...)`, line `13`
  - `extract_spikes(...)`, line `35`
  - `interval_sorter(...)`, line `72`
  - `extract_session_data(...)`, line `96`
  - current domain construction, line `108`

### Comparison Pipelines

- `preprocess/allen_vc_2019_vis/pipeline.py`
  - `Pipeline.process(...)`
  - optotagging trim, lines `109` to `111`
  - invalid interval extraction, line `102`
  - stimulus epoch extraction, line `105`
  - final domain, line `123`
  - `extract_invalid_interval(...)`, line `202`
  - `extract_stimulus_epochs(...)`, line `209`
- `preprocess/ibl_bwm/pipeline.py`
  - domain trim, line `122`
  - final `Data(... domain=domain)`, line `134`
- `preprocess/steinmetz_2019/prepare_data.py`
  - final `Data(... domain=spikes.domain)`, line `89`
- `preprocess/steinmetz_2019/steinmetz_split_probes.py`
  - per-probe invalid/dropout domain, lines `48` to `56`
  - top-level `_domain` reset, line `65`
- `preprocess/bugeon/prepare_data.py`
  - disjoint domain union, lines `183` to `185`
  - final `Data(... domain=calcium_traces.domain)`, line `201`
- `utils/split_probes.py`
  - probe ID split, lines `20` and `30`
  - per-probe spike-derived domain, lines `54` to `55`
  - top-level `_domain` reset, line `64`

### Downstream Sampling And Models

- `.venv/Lib/site-packages/torch_brain/data/dataset.py`
  - `Dataset.get(...)`, lines `293` to `305`
  - `Dataset.get_sampling_intervals(...)`, lines `347` to `360`
- `.venv/Lib/site-packages/torch_brain/data/sampler.py`
  - `RandomFixedWindowSampler`, line `13`
  - `SequentialFixedWindowSampler`, line `141`
- `src/trainers/contrastive.py`
  - train sampler, lines `247` to `248`
  - validation sampler, lines `302` to `303`
- `src/trainers/forward.py`
  - sampling intervals, line `99`
  - forward-pass sampler, line `117`
- `src/models/nuclr_v2ga_spikes.py`
  - `tokenize(...)`, line `197`
  - `active_units`, line `200`
- `src/models/nuclr_v2ga_bin.py`
  - `tokenize(...)`, line `143`
  - `active_units`, line `146`

### AllenSDK Sources For VBN Session Tables

- `.venv/Lib/site-packages/allensdk/brain_observatory/behavior/behavior_session.py`
  - `stimulus_presentations`, line `1067`
  - `stimulus_block_name` docs, line `1074`
  - `active` docs, line `1142`
  - `trials`, line `1271`
- `.venv/Lib/site-packages/allensdk/brain_observatory/ecephys/behavior_ecephys_session.py`
  - `optotagging_table`, line `338`
  - `spike_times`, line `372`
  - `get_channels(...)`, line `394`
  - `get_units(...)`, line `413`

## My Recommendation

For the next code change, I would not keep the current VBN domain.

The lowest-risk alignment with the rest of the repo is:

1. Build and store visual intervals from `session.stimulus_presentations` and
   `session.trials`.
2. Set the default top-level `Data.domain` to pre-optotagging visual/neural time,
   not through the end of optotagging.
3. If using probe-split VBN files, update the splitter to intersect probe
   domains with the parent domain.

That gives you HDF5 files that downstream NuCLR sampling will treat the same way
it treats the other ephys datasets: `Data.domain` means "valid time to sample
model windows", while task/control/event information lives in auxiliary
intervals.

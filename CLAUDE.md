# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Official code for the NeurIPS 2025 paper "Know Thyself by Knowing Others: Learning Neuron Identity from Population Context" (NuCLR). A contrastive objective trains an encoder that gives every neuron (called a "unit" in the code) an embedding. The embeddings are then evaluated by predicting cell type and brain region with logistic regression.

The code is built on [`torch_brain`](https://github.com/neuro-galaxy/torch_brain) 0.2.1, which provides the data objects (`torch_brain.data`), the dataset base class, samplers, collation, and the `brainsets` command. The tag `v0.1.0` is the code as it was released with the paper, before it was migrated to this `torch_brain` version.

## Ground rules for changes

- **Behaviour must be preserved.** This is the code of a published paper. A change must not alter what the models compute or what the data pipelines produce, unless that is the stated purpose of the change.
- **There is no test suite and no CI.** A change is verified by hand: run the code before and after the change on real data with the same seed, and compare the results (processed files, sampled windows, collated batches, model outputs). The pull request should say what was compared and what was not covered.
- With a fixed seed on one GPU, training the binned model (`nuclr_v2ga_bin`) has reproduced bit for bit from run to run. Training the spikes model and the calcium model has not: two identical runs give outputs that differ on the order of 1e-2 with `bf16`. Compare those two against that run-to-run noise instead of expecting exact equality.
- **Never push to `main`.** Work on a branch and open a pull request, one topic per pull request. Descriptions are written in plain, full sentences and contain no machine-specific paths.
- Code is formatted with `black` (pinned in `requirements.txt`). Check with `black --check .`
- `img/arch-diagram.png` shows up as modified on machines that have git-lfs installed, because `.gitattributes` marks `*.png` as LFS while the image is committed as a regular file. Leave it out of commits: stage files by name instead of using `git add -A`.
- Known follow-ups are tracked as GitHub issues (`gh issue list`). The YAML dataset selection files under `configs/data/` are planned to be replaced (issue #13), so avoid building new features on that format.

## Commands

Everything is run from the repository root. The evaluators, the evaluation scripts and several utility scripts use paths relative to it.

```bash
# Environment (Python 3.10, managed with uv)
uv venv venv -p 3.10
source venv/bin/activate
uv pip install -r requirements.txt

# Train on electrophysiology data (IBL, Allen, Steinmetz) or on calcium imaging data (Bugeon)
python train.py --config-name train_ephys data=<data-config> num_epochs=<n>
python train.py --config-name train_ca data=<data-config> num_epochs=<n>

# Compute embeddings for every checkpoint of a run, or for one checkpoint
bash utils/forward_all_epochs.sh <run_id> <data-config> [batch_size] [epoch_stride]
python train.py --config-name forward ckpt.load_from=<path/to/epoch_N.pt> data=<data-config>

# Quick check that the training loop still runs
CUDA_VISIBLE_DEVICES=0 python train.py --config-name train_ephys \
    data=ibl_bwm_probes_dev data/train_dataset=ibl_bwm_probes_tiny debug=train
```

Things to know about these commands:

- `--config-name` is always required (`train_ephys`, `train_ca` or `forward`). `train.py` has no working default. For `forward`, `data=` is also required, because the default in `configs/forward.yaml` does not exist.
- `<data-config>` is the name of a file in `configs/data/`, for example `ibl_bwm_probes_dev`.
- `debug=train` turns off W&B, checkpointing, validation and the evaluator. A run with it only needs the processed data. There is no option that limits the number of steps, so stop the run by hand once it is stepping.
- Without `debug`, a training run needs a W&B login (or `wandb.mode=offline` / `wandb.mode=disabled`) and the metadata files in `./neuron_metadata` (download link in `README.md`).
- The default locations are outside the repository: `data_root=../data/processed`, `ckpt.dir=../ckpt` (the environment variable `CKPT_DIR` overrides it) and `embs_root=../embs`. Pass `data_root=<dir>` when the data is somewhere else.
- A CUDA GPU is required. `train.py` starts one process for every visible GPU and uses DDP when there is more than one, so limit the GPUs with `CUDA_VISIBLE_DEVICES`. `batch_size` is the total over all GPUs.
- Preprocessing commands are in `preprocess/README.md`. Offline evaluation commands are in `eval_scripts/README.md`.

## Architecture

### Configuration

Configs are Hydra files in `configs/`, with the groups `data`, `model`, `loss`, `optim`, `trainer` and `train_transform`. Entries with a `_target_` key are built by `src.utils.instantiate`. This is a small local helper and not `hydra.utils.instantiate`: it is not recursive, and the caller passes extra arguments (for example `precision` for a model, and `model` for a loss).

A `data` config has three parts: `train_dataset` and `val_dataset`, which are selections of recordings, and the `evaluator` class for that dataset. A selection names a brainset and optionally `sessions` or `exclude_sessions`. `resolve_recording_ids` in `src/dataset.py` turns it into recording ids of the form `<brainset>/<session>`, which map to the files `<data_root>/<brainset>/<session>.h5`.

In the names of the data configs, "inductive" means that the validation sessions are held out from training, and "transductive" means that training uses all sessions. A "dev" config trains on a subset of the sessions and validates on a separate set, for development runs. Use a transductive config with `forward` to get embeddings for all neurons.

### Trainers

`src/trainers/trainer.py` has the base class `Trainer`: seeding, DDP setup, the W&B logger, loading and saving checkpoints, and the epoch loop. It has two subclasses, chosen by the `trainer` config group:

- `ContrastiveTrainer` (`src/trainers/contrastive.py`) does the training.
- `ForwardPass` (`src/trainers/forward.py`) reuses the same machinery to compute embeddings. It makes one pass over `data.train_dataset` (not `val_dataset`) and saves `{"embs", "ids"}` to `<embs_root>/<run_id>/embs_epoch_N.pt`. It builds the model from the config that is stored inside the checkpoint, not from the command line.

Checkpoints go to `<ckpt.dir>/<run_id>/epoch_N.pt`, where the run id is the W&B run id. The directory also gets `commit_id.txt` and `diff.patch`.

### How one training batch is built

This path goes through several files and is the main thing to understand before changing the data or model code.

1. `RandomFixedWindowSampler` (`src/samplers.py`) yields one window of `views.duration` seconds at a time, as a `DatasetIndex` (recording, start, end). It is reseeded every epoch.
2. `TwoViewDataset.__getitem__` (`src/dataset.py`) draws the second window itself: same recording, at most `views.max_distance` seconds away from the first one. Each window goes through `transform`, which is the `train_transform` (unit dropout) followed by `model.tokenize`. The result is a `ViewData`.
3. `common_transform` then runs `TwoViewData.from_two_views`, which computes `match_idx`: the units that are present in both windows. These are the positive pairs. The item is `None` when no second window can be found or when a view has 10 units or fewer. `TwoViewData.collate` drops those items.
4. The trainer calls `model(**batch.view1.data["enc_input"])`. The keys that a model's `tokenize` puts into `enc_input` therefore have to match the arguments of its `forward`. The output has one embedding per unit. The units of all windows in the batch are concatenated without padding, and xformers block-diagonal attention masks, built from the sequence lengths, keep the windows apart.
5. `SampleWiseContrastiveLoss` (`src/losses/sw_contrastive.py`) contrasts within each pair of windows: the positive for a unit is the same unit in the other view, and the negatives are the other units in the same pair of windows. The loss owns a projector, so it has parameters of its own. It is optimised, gradient-clipped and checkpointed together with the model.

There are three encoders in `src/models/`, all subclasses of `BaseNeuronEncoder`: `nuclr_v2ga_bin` (binned spikes, the default for electrophysiology), `nuclr_v2ga_spikes` (individual spike times) and `nuclr_v2ga_ca2` (calcium traces). The trainer sets `model.ctx_duration` from `views.duration`.

### Evaluation

During training, an `EmbeddingCollector` (`src/evaluators/collector.py`) averages the embedding of each unit over all windows in which it appeared. After a validation epoch, the evaluator of the dataset (`src/evaluators/<dataset>.py`) fits logistic regression probes with `ray`, using labels from `./neuron_metadata/*.csv`, and logs the metrics to W&B.

The unit id is the key that joins everything: the collectors, the metadata files and the split files. Unit ids are assumed to be unique across all recordings, and `NuclrDataset` does not add a prefix to them.

The scripts in `eval_scripts/` are separate from the training code. They read saved embeddings and the split files in `./splits`. `eval_scripts/README.md` says which script and which split file belong to each dataset.

Three different things are called `utils`: `src/utils.py` (helpers for the training code), the directory `utils/` (standalone scripts) and `eval_scripts/utils.py` (imported as `utils` by the evaluation scripts, which is why they are run as `python eval_scripts/<script>.py`).

### Preprocessing

Every recording becomes one HDF5 file, `<processed>/<brainset>/<session>.h5`, holding a `torch_brain.data.Data` object with `spikes` or `calcium_traces`, `units`, `session`, `subject` and `brainset`. There are two kinds of preprocessing code in `preprocess/`:

- **Allen and IBL** are `brainsets` pipelines (`pipeline.py`), run with `brainsets prepare --local <dir>`. This command builds a separate, temporary environment from the `# /// brainset-pipeline` header at the top of the file. The dependencies of a pipeline are therefore the ones in that header, and not the ones in `requirements.txt`. For example, the `ibllib` version in `requirements.txt` is not the one the IBL pipeline uses. Pin every package that matters in the header: both pipelines have stopped working before because a package that was not pinned moved on. Add `--single <id>` to process one session as a test.
- **Steinmetz and Bugeon** are plain scripts that run in the main environment, started with GNU `parallel`.

Electrophysiology sessions are then split into one file per probe, by `utils/split_probes.py` (Allen and IBL) or `preprocess/steinmetz_2019/steinmetz_split_probes.py`. The probe files are written to `<brainset>_probes/<session>_p<i>.h5`. The training configs use these brainsets: `allen_vc_2019_vis_probes`, `ibl_brainwide_map_qc_probes` and `steinmetz_2019_probes`. The calcium brainset `bugeon_transcriptomic_2022` is not split.

Two things to watch when testing a pipeline:

- The IBL pipeline uses the ONE API, which writes settings and a login token to `~/.one` and atlas files to `~/Downloads/ONE`. Set `HOME` to a scratch directory for test runs on a machine whose ONE setup should stay untouched.
- When `allensdk` cannot read a raw file, it downloads the file again and overwrites it in place. Do not give a test run access to existing raw data through symlinks. Use a copy.

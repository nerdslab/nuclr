# Evaluation Scripts

Run these scripts from the project root, and keep the splits directory in the project root.

## Data Splits

You can download the splits used in our paper from
[link](https://ik.imagekit.io/7tkfmw7hc/nuclr/splits.zip).
These should be placed (unzipped) in `<project_root>/splits`

## General Usage

The scripts are generally run like:

```bash
python eval_scripts/<the-script.py> <embs_path> --splits-path=<splits_path>
```

Here, `<embs_path>` can be either:

- A single embedding `.pt` file path, or
- A path to a directory containing embeddings from multiple epochs during training

If a directory path is given, the eval script will perform cross-validation to find the best epoch and report test results for that epoch.

Each dataset presents a unique situation and is evaluated slightly differently. See below for dataset-specific steps.

## Allen VC 2019

All evaluation settings are covered by a single script:

```bash
python eval_scripts/multi_split_eval.py <embs_path> --splits-path=splits/allen_vc_2019_l1out.pkl
```

## IBL Brainwide Map

For zero-shot evaluations:

```bash
python eval_scripts/single_split_eval.py <embs_path> --splits-path=splits/ibl_bwm/zeroshot.pkl
```

For transductive (non-zero-shot) evaluations:

```bash
python eval_scripts/single_split_eval.py <embs_path> --splits-path=splits/ibl_bwm/neuronwise.pkl
```

> [!NOTE]
> To reproduce results from the paper exactly, use `single_split_eval_original.py` instead of `single_split_eval.py`.
> The `single_split_eval.py` provides a slightly improved (and simplified) evaluation setup and gives better results than the paper.

## Bugeon

The following splits, with self-explanatory filenames, are included:

- `ei_transductive.pkl`, `subclass_transductive.pkl`, `11class_transductive.pkl`
- `ei_transductive_zs.pkl`, `subclass_transductive_zs.pkl`, `11class_transductive_zs.pkl`
- `ei_inductive_zs.pkl`, `subclass_inductive_zs.pkl`, `11class_inductive_zs.pkl`

For zero-shot evaluations:

```bash
python eval_scripts/multi_split_eval.py <embs_path> \
    --splits-path=splits/bugeon/<choose>_zs.pkl \
    --class-balance=resample
```

For transductive (non-zero-shot) evaluations:

```bash
python eval_scripts/single_split_eval.py <embs_path> \
    --splits-path=splits/bugeon/<choose>_transductive.pkl \
    --class-balance=resample
```

## Steinmetz

The following splits are included:

- `transductive.pkl`
- `transductive_zs.pkl`
- `inductive_zs_subject1.pkl` (this corresponds to results in paper, but we also provide splits for subject2, 3, and 4)

For zero-shot evaluations:

```bash
python eval_scripts/multi_split_eval.py <embs_path> \
    --splits-path=splits/steinmetz/<choose>.pkl \
    --class-balance=resample
```

For transductive (non-zero-shot) evaluations:

```bash
python eval_scripts/single_split_eval.py <embs_path> \
    --splits-path=splits/steinmetz/transductive.pkl \
    --class-balance=resample
```

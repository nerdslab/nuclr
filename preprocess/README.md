# Data Preprocessing

Given below are the steps needed to preprocess all datasets into formats that
our training and evaluation code can work with.

## Allen VC 2019

```bash
RAW_DIR=../data/raw
PROCESSED_DIR=../data/processed
brainsets prepare --local preprocess/allen_vc_2019_vis --raw-dir $RAW_DIR --processed-dir $PROCESSED_DIR -c 16
bash utils/split_probes_all.sh $PROCESSED_DIR/allen_vc_2019_vis
```

## IBL Brainwide Map

```bash
RAW_DIR=../data/raw
PROCESSED_DIR=../data/processed
brainsets prepare --local preprocess/ibl_bwm --raw-dir $RAW_DIR --processed-dir $PROCESSED_DIR -c 16
bash utils/split_probes_all.sh $PROCESSED_DIR/ibl_brainwide_map_qc
```

## Steinmetz et. al. 2019

First download the dataset into `../data/raw/steinmetz_2019` ([link](https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406)), then

```bash
parallel -j 16 python preprocess/steinmetz_2019/prepare_data.py \
    --raw_dir ../data/raw/steinmetz_2019 \
    --processed_dir ../data/processed/steinmetz_2019 \
    --session {} :::: preprocess/steinmetz_2019/session_ids.txt

# Split the data insertion-wise using
ls ../data/processed/steinmetz_2019 | parallel -j 16 python preprocess/steinmetz_2019/steinmetz_split_probes.py
```

## Bugeon et. al. 2022

First, download the dataset into `../data/raw/bugeon_transcriptomic_2022` ([link](https://figshare.com/articles/dataset/A_transcriptomic_axis_predicts_state_modulation_of_cortical_interneurons/19448531)), then

```bash
parallel -j 16 python preprocess/bugeon/prepare_data.py \
    --raw_dir ../data/raw/bugeon_transcriptomic_2022 \
    --processed_dir ../data/processed/bugeon_transcriptomic_2022 \
    --session {} :::: preprocess/bugeon/experiments.txt
```

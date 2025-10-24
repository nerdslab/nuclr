# Data Preprocessing
We use `brainsets` to manage our preprocessing pipelines.
Run the following commands from the root directory of this project.

**Allen Visual Coding Neurpixels 2019**
```bash
brainsets prepare preprocess/allen_vc_2019_vis --local \
    --raw-dir ../data/raw \
    --processed-dir ../data/processed \
    --cores 16
# This would store data in ../data/processed/allen_vc_2019_vis

# Split data insertion-wise
cat preprocess/allen_vc_2019_vis/session_ids.txt | parallel -j 16 python preprocess/allen_vc_2019_vis/allen_split_probes.py
# This would store data in ../data/processed/allen_vc_2019_vis_probes
```


**IBL Brainwide map**
```bash
# This one errors out from time to time, so you'll have to keep
# rerunning until all sessions have been processed
brainsets prepare preprocess/ibl_bwm --local \
    --raw-dir ../data/raw \
    --processed-dir ../data/processed \
    --cores 16  [<leader>aa: ask, <leader>ae: edit]
# This would store data in ../data/processed/ibl_brainwide_map_qc

# Split the data insertion-wise using
parallel -j 16 python preprocess/ibl_bwm/ibl_split_probes.py {} :::: preprocess/ibl_bwm/eids.txt
# This would store data in ../data/processed/ibl_brainwide_map_qc_probes
```


**Steinmetz et. al. 2019**
First download the dataset into `../data/raw/steinmetz_2019` ([link](https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406)), then
```bash
parallel -j 16 python preprocess/steinmetz_2019/prepare_data.py \
    --raw_dir ../data/raw/steinmetz_2019 \
    --processed_dir ../data/processed/steinmetz_2019 \
    --session {} :::: preprocess/steinmetz_2019/session_ids.txt

# Split the data insertion-wise using
ls ../data/processed/steinmetz_2019 | parallel -j 16 python preprocess/steinmetz_2019/steinmetz_split_probes.py
```


**Bugeon et. al. 2022**
First, download the dataset into `../data/raw/bugeon_transcriptomic_2022` ([link](https://figshare.com/articles/dataset/A_transcriptomic_axis_predicts_state_modulation_of_cortical_interneurons/19448531)), then
```bash
parallel -j 16 python preprocess/bugeon/prepare_data.py \
    --raw_dir ../data/raw/bugeon_transcriptomic_2022 \
    --processed_dir ../data/processed/bugeon_transcriptomic_2022 \
    --session {} :::: preprocess/bugeon/experiments.txt
```

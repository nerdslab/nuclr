# NuCLR
---

### 1. Setup virtual environemnt
We use `venv` to manage the Python environment. This code-base was developed using Python3.10, and should be tested on the same version.
```bash
source venv_setup.sh
```

### 2. Preprocessing datasets

#### Allen Visual Coding Neurpixels 2019
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


#### IBL Brainwide map
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


#### Steinmetz et. al. 2019
First download the dataset into `../data/raw/steinmetz_2019` ([link](https://figshare.com/articles/dataset/Dataset_from_Steinmetz_et_al_2019/9598406)), then
```bash
parallel -j 16 python preprocess/steinmetz_2019/prepare_data.py \
    --raw_dir ../data/raw/steinmetz_2019 \
    --processed_dir ../data/processed/steinmetz_2019 \
    --session {} :::: preprocess/steinmetz_2019/session_ids.txt

# Split the data insertion-wise using
ls ../data/processed/steinmetz_2019 | parallel -j 16 python preprocess/steinmetz_2019/steinmetz_split_probes.py
```


#### Bugeon et. al. 2022
First, download the dataset into `../data/raw/bugeon_transcriptomic_2022` ([link](https://figshare.com/articles/dataset/A_transcriptomic_axis_predicts_state_modulation_of_cortical_interneurons/19448531)), then
```bash
parallel -j 16 python preprocess/bugeon/prepare_data.py \
    --raw_dir ../data/raw/bugeon_transcriptomic_2022 \
    --processed_dir ../data/processed/bugeon_transcriptomic_2022 \
    --session {} :::: preprocess/bugeon/experiments.txt
```

### 3. Downloading neuron metadata
Download metadata (csv files) about neurons in all four datasets from this
[link](https://ik.imagekit.io/7tkfmw7hc/nuclr/neuron_metadata.zip?updatedAt=1747970653540)
and unzip into `./neuron_metadata`


### 4. Training
To train on ephys. datasets (IBL, Allen, Steinmetz et. al.):
```bash
python train.py --config-name train_ephys \
	data=<data-config> \
	batch_size=128 \
	num_epochs=<num_epochs>
```
- Options for `<data-config>` can be found in `configs/data/*.yaml`. E.g. `data=ibl_bwm_probes_dev`
- Set `num_epochs` such that the total number of training steps is roughly 50,000.
- The checkpoints would be stored in `../ckpt` by default.
- Other available configurations can be found in `configs/train_ephys.yaml`

To train on calcium imaging data (Bugeon et. al.):
```bash
python train.py --config-name train_ca \
	data=<data-config> batch_size=128 num_epochs=<num_epochs>
```
- Options for `<data-config>` can be found in `configs/data/*.yaml`. E.g. `data=bugeon_dev`
- Set `num_epochs` such that the total number of training steps is roughly 50,000.
- The checkpoints would be stored in `../ckpt` by default.
- Other available configurations can be found in `configs/train_ca.yaml`


### 5. Forward pass for final embeddings
A final forward pass over the entire data is needed to get the embeddings from a particular checkpoint.
The training script would print a "run_id" for the corresponding run. Use this to run the follwing command:

```bash
bash utils/forward_all_epochs.sh <run_id> <data-config-name> [batch_size] [epoch_stride]
```

This would store the embeddings in `../embs/<run_id>/embs_epoch_*.pt` depending on
the `run_id` and epoch number of the checkpoints used.
In most cases, you would want to use the "transductive" versions of each dataset, since we
want to compute embeddings for all neurons here.


### 6. Run evaluation on the produced embeddings
Evaluation notebooks are present and documented in the `eval_notebooks/` directory.

# NuCLR

Official codebase for [Know Thyself by Knowing Others: Learning Neuron Identity from Population Context
](https://neurips.cc/virtual/2025/poster/115008).

> [!NOTE]  
> We will be updating and cleaning this repository regularly until Nov 1, 2025.

## Usage:

This project use `venv` to manage the Python environment, and has only been tested on Python3.10.
```bash
source venv_setup.sh
```

**1. Preprocessing datasets**
Please follow the steps in `preprocess/README.md`


**2. Downloading neuron metadata**
Download metadata (csv files) about neurons in all four datasets from this
[link](https://ik.imagekit.io/7tkfmw7hc/nuclr/neuron_metadata.zip?updatedAt=1747970653540)
and unzip into `./neuron_metadata`


**3. Training**
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


**4. Forward pass for final embeddings**
A final forward pass over the entire data is needed to get the embeddings from a particular checkpoint.
The training script would print a "run_id" for the corresponding run. Use this to run the follwing command:

```bash
bash utils/forward_all_epochs.sh <run_id> <data-config-name> [batch_size] [epoch_stride]
```

This would store the embeddings in `../embs/<run_id>/embs_epoch_*.pt` depending on
the `run_id` and epoch number of the checkpoints used.
In most cases, you would want to use the "transductive" versions of each dataset, since we
want to compute embeddings for all neurons here.


**5. Run evaluation on the produced embeddings**
Evaluation notebooks are present and documented in the `eval_notebooks/` directory.


## Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```
@inproceedings{
    azabou2023unified,
    title={Know Thyself by Knowing Others: Learning Neuron Identity from Population Context},
    author={Vinam Arora and Divyansha Lachi and Ian J Knight and Mehdi Azabou and Blake Richards and Cole Hurwitz and Joshua H Siegle and Eva L Dyer},
    booktitle={Thirty-ninth Conference on Neural Information Processing Systems},
    year={2025},
    url={https://neurips.cc/virtual/2025/poster/115008}
}
```

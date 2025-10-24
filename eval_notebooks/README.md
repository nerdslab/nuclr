## Data splits
You can download the splits used in our paper from
[link](https://ik.imagekit.io/7tkfmw7hc/nuclr/splits.zip?updatedAt=1747973145010)
These should be placed (unzipped) in `<project_root>/splits`

Alternatively, you can also create the splits locally by following the instructions below:

### Creating splits locally
For Allen, Bugeon, and Steinmetz, we have a notebook for each dataset:
- `create_allen_splits.ipynb`
- `create_bugeon_splits.ipynb`
- `create_steinmetz_splits.ipynb`

These should be run from within the `eval_notebooks` (this) directory.
Make sure to create a directory called `splits` in the project root before running the notebook cells.

For IBL, run the following from the project root:
```bash
python preprocess/ibl_bwm/create_inductive_splits.py --splits_dir ./splits
python preprocess/ibl_bwm/create_transductive_splits.py --splits_dir ./splits
```

The notebooks and the scripts will create splits as pickle files
that will be read by the evaluation notebooks below.

## Evaluation
For evaluation, we have a notebook per dataset that cover all evaluation cases in the paper:
- `allen_eval.ipynb`
- `ibl_eval.ipynb`
- `steinmetz_eval.ipynb`
- `bugeon_eval.ipynb`

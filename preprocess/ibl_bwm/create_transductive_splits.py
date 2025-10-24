# pyright: reportAttributeAccessIssue=false


from pathlib import Path
from collections import defaultdict
import argparse
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import pandas as pd
import h5py
from iblatlas.regions import BrainRegions

from temporaldata import Data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="../data/processed/ibl_brainwide_map_qc_probes", type=Path
    )
    parser.add_argument("--splits_dir", default="./splits", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    # Gather neuron-level information
    info: dict[str, list | np.ndarray] = defaultdict(lambda: [])
    for path in args.data_dir.iterdir():
        path: Path
        if not path.name.endswith(".h5"):
            continue

        with h5py.File(path, "r") as f:
            data = Data.from_hdf5(f)  # type: ignore

            info["uuids"].append(data.units.id[:].astype(str))
            info["acronyms"].append(data.units.acronym[:].astype(str))
            info["pids"].append(
                np.array([data.session.id for _ in range(len(data.units))])
            )
    for k, v in info.items():
        info[k] = np.concatenate(v)

    # Add cosmos labels and remove units from "root" or "void"
    br = BrainRegions()
    cosmos_regions = br.acronym2acronym(info["acronyms"], mapping="Cosmos")
    info["labels"] = cosmos_regions

    mask = ~np.isin(info["labels"], ["root", "void"])
    for k, v in info.items():
        assert isinstance(v, np.ndarray)
        info[k] = v[mask]

    # Create splits
    np.random.seed(args.seed)
    ids = info["uuids"]
    labels = info["labels"]

    # 1. create main train-test split
    trainval_idx, test_idx = train_test_split(
        np.arange(len(ids)),
        test_size=0.2,
        stratify=labels,
    )
    # 2. create smaller validation set from this train set (1/8 size of training)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=1.0 / 8,
        stratify=labels[trainval_idx],
    )
    splits = {"train": ids[train_idx], "val": ids[val_idx], "test": ids[test_idx]}  # type: ignore

    # Also save label map for convenience
    label_map = pd.DataFrame(info["labels"], info["uuids"], columns=["label"])
    save_data = {
        "splits": splits,
        "label_map": label_map,
    }

    args.splits_dir.mkdir(exist_ok=True, parents=True)
    out_file = args.splits_dir / "ibl_transductive.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Saved output to {out_file}")


if __name__ == "__main__":
    main()

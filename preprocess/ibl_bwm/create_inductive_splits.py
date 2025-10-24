from pathlib import Path
import numpy as np
import yaml, pickle
import pandas as pd
import h5py
from iblatlas.regions import BrainRegions
from temporaldata import Data


def main():
    test_cfg_file = Path("./configs/data/val_dataset/ibl_bwm_probes_test.yaml")
    val_cfg_file = Path("./configs/data/val_dataset/ibl_bwm_probes_dev.yaml")
    train_cfg_file = Path("./configs/data/train_dataset/ibl_bwm_probes_all.yaml")
    data_dir = Path("../data/processed/ibl_brainwide_map_qc_probes")
    splits_dir = Path("./splits")

    test_sessions = load_session_list(test_cfg_file)
    test_ids, test_labels = load_unit_info(test_sessions, data_dir)

    trainval_sessions = load_session_list(train_cfg_file)
    trainval_ids, trainval_labels = load_unit_info(trainval_sessions, data_dir)
    # train_cfg contains train and val sessions, hence the name trainval_sessions

    val_sessions = load_session_list(val_cfg_file)
    val_ids, val_labels = load_unit_info(val_sessions, data_dir)

    # remove val sessions from trainval
    mask_val_in_trainval = np.isin(trainval_ids, val_ids)
    train_ids = trainval_ids[~mask_val_in_trainval]
    train_labels = trainval_labels[~mask_val_in_trainval]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    for k, v in splits.items():
        print(f"{k}: {len(v)}")

    # create a labelmap
    all_ids = np.concatenate((train_ids, val_ids, test_ids))
    all_labels = np.concatenate((train_labels, val_labels, test_labels))
    label_map = pd.DataFrame(all_labels, all_ids, columns=["label"])

    save_data = {
        "splits": splits,
        "label_map": label_map,
    }

    splits_dir.mkdir(exist_ok=True, parents=True)
    out_file = splits_dir / "ibl_inductive.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(save_data, f)
    print(f"Saved output to {out_file}")


def load_session_list(yaml_filepath: Path):
    with open(yaml_filepath, "r") as f:
        cfg = yaml.safe_load(f)

    assert len(cfg) == 1, "Dont know how to handle the other case yet"
    selection = cfg[0]["selection"]
    assert len(selection) == 1, "Dont know how to handle the other case yet"
    sessions = selection[0]["sessions"]
    return sessions


def clean_ids_and_labels(
    ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove "root" and "void" units"""
    mask = ~np.isin(labels, ["root", "void"])
    return ids[mask], labels[mask]


def load_unit_info(
    session_list: list[str],
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    uuids = []
    labels = []
    br = BrainRegions()
    for session_id in session_list:
        with h5py.File(data_dir / f"{session_id}.h5", "r") as f:
            data = Data.from_hdf5(f)

            uuids.append(data.units.id[:].astype(str))
            acronyms = data.units.acronym[:].astype(str)
            labels.append(br.acronym2acronym(acronyms, mapping="Cosmos"))

    uuids = np.concatenate(uuids)
    labels = np.concatenate(labels)
    return clean_ids_and_labels(uuids, labels)


if __name__ == "__main__":
    main()

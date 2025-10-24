from typing import List, Dict, Any
import pickle
import numpy as np
import pandas as pd
import torch
import ray
import matplotlib.pyplot as plt

from .rankme import rankme
from .classifier import (
    KFoldClassifier,
    ZeroshotClassifier,
    LeaveOneSessionOutClassifier,
)
from ..logger import get_cli_logger

logger = get_cli_logger()


class SteinmetzEvaluator:
    md_filepath = "./neuron_metadata/steinmetz.csv"

    def __init__(self, world_size: int):
        self.md = self.prep_md_df()
        ray.init(address="local", num_cpus=world_size * 7)

    @classmethod
    def prep_md_df(cls):
        df = pd.read_csv(cls.md_filepath)
        df["unit_id"] = [str(x) for x in df.id]
        df["brain_region"] = [region_map(x) for x in df.brain_area]
        return df.set_index("unit_id")

    def __call__(
        self,
        X_train: torch.Tensor,
        train_unit_ids: List[str] | np.ndarray,
        X_val: torch.Tensor,
        val_unit_ids: List[str] | np.ndarray,
    ) -> Dict[str, Any]:

        X_train_np = X_train.detach().cpu().numpy()
        X_val_np = X_val.detach().cpu().numpy()
        train_unit_ids = np.array(train_unit_ids)
        val_unit_ids = np.array(val_unit_ids)
        train_md = self.md.loc[train_unit_ids]
        val_md = self.md.loc[val_unit_ids]

        self.futures = []

        # Train embeddings brain region
        train_br = train_md.brain_region.values
        train_br_mask = train_br != "Remove"
        train_br_evaluator = KFoldClassifier("train/br", True)
        train_br_evaluator.launch(X_train_np[train_br_mask], train_br[train_br_mask])
        self.futures.append(train_br_evaluator)

        # Sessionwise Brain Region on train
        train_br_l1out_evaluator = LeaveOneSessionOutClassifier(
            "train/subj_l1out_br", True
        )
        train_br_l1out_evaluator.launch(
            X=X_train_np[train_br_mask],
            y=train_br[train_br_mask],
            s=train_md.subject_id[train_br_mask],
        )
        self.futures.append(train_br_l1out_evaluator)

        # Val embeddings brain region
        val_br = val_md.brain_region.values
        val_br_mask = val_br != "Remove"
        val_br_evaluator = KFoldClassifier("val/br", True)
        val_br_evaluator.launch(X_val_np[val_br_mask], val_br[val_br_mask])
        self.futures.append(val_br_evaluator)

        # Sessionwise Brain Region on validation
        if len(np.unique(val_md.subject_id[val_br_mask])) > 2:
            val_br_l1out_evaluator = LeaveOneSessionOutClassifier("val/l1out_br", True)
            val_br_l1out_evaluator.launch(
                X=X_val_np[val_br_mask],
                y=val_br[val_br_mask],
                s=val_md.subject_id[val_br_mask],
            )
            self.futures.append(val_br_l1out_evaluator)

        # Zeroshot brain region
        val_zs_br_evaluator = ZeroshotClassifier("val/zs_br", True)
        val_zs_br_evaluator.launch(
            X_train=X_train_np[train_br_mask],
            y_train=train_br[train_br_mask],
            X_test=X_val_np[val_br_mask],
            y_test=val_br[val_br_mask],
        )
        self.futures.append(val_zs_br_evaluator)

        # Rank me (not parallel, because on GPU)
        rankme_train = rankme(X_train)
        rankme_val = rankme(X_val)

        metrics = {
            "train/rankme": rankme_train,
            "val/rankme": rankme_val,
        }

        # Finalize all futures
        for future in self.futures:
            _metrics = future.finalize()
            metrics.update(_metrics)

        plt.close()
        return metrics


# Taken from:
# https://github.com/nsteinme/steinmetz-et-al-2019/blob/master/utils/brainRegionGroups.m
region_groups = {
    "MB": ["MRN", "SCm", "SCs", "APN", "PAG"],
    "VIS": ["VISp", "VISrl", "VISam", "VISpm", "VISl", "VISa"],
    "TH": ["LP", "LD", "RT", "MD", "MG", "LGd", "VPM", "VPL", "PO", "POL"],
    "HPF": ["POST", "SUB", "DG", "CA1", "CA3"],
}


def region_map(x: str) -> str:
    for k, v in region_groups.items():
        if x in v:
            return k
    return "Remove"


# TGT_REGIONS = ["VIS", "TH", "HPF", "MB"]
#
# with open("./neuron_metadata/allen_region_map.pkl", "rb") as f:
#    REGION_HEIRARCHY_DICT = pickle.load(f)
#
#    # Change all keys to upper-case to avoid confusion later
#    orig_keys = list(REGION_HEIRARCHY_DICT.keys())
#    for k in orig_keys:
#        v = REGION_HEIRARCHY_DICT.pop(k)
#        if k.lower() == k:  # we dont care about neurons that have all-small names
#            continue
#        k_new = k.upper()
#        assert k_new not in REGION_HEIRARCHY_DICT
#        REGION_HEIRARCHY_DICT[k_new] = v
#
#
# def region_map(x: str) -> str:
#    if x[0].lower() == x[0]:
#        # if all lower names, this does not belong with us
#        return "Remove"
#
#    x = x.upper()
#
#    if x == "CUL4 5":
#        x = "CUL4, 5"
#
#    if x not in REGION_HEIRARCHY_DICT:
#        raise ValueError(f"{x} not found in known heirarchy")
#
#    region_list = REGION_HEIRARCHY_DICT[x]
#    if "grey" not in region_list:
#        # we're ignoring neurons that are not in grey
#        return "Remove"
#
#    for region in region_list:
#        if region in TGT_REGIONS:
#            return region
#
#    return "Remove"

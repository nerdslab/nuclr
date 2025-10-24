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


class IBLBWMEvaluator:
    md_filepath = "./neuron_metadata/ibl_bwm.csv"

    def __init__(self, world_size: int):
        self.md = self.prep_md_df()
        ray.init(address="local", num_cpus=world_size * 7)

    @classmethod
    def prep_md_df(cls):
        df = pd.read_csv(cls.md_filepath)
        df["unit_id"] = [str(x) for x in df.id]
        # df["brain_region"] = [acryonym_to_region(x) for x in df.acronym]
        # df["brain_region"] = [area_map[x] for x in df.acronym]
        df["brain_region"] = [region_map(x) for x in df.acronym]
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

        def to_sw_cv(session_ids: np.ndarray):
            sw_cv_count = 5
            return np.array([hash(x) % sw_cv_count for x in session_ids], dtype=int)

        self.futures = []

        # Train embeddings brain region
        train_br = train_md.brain_region.values
        train_br_mask = train_br != "Remove"
        # train_br_evaluator = KFoldClassifier("train/br", True)
        # train_br_evaluator.launch(X_train_np[train_br_mask], train_br[train_br_mask])
        # self.futures.append(train_br_evaluator)

        # Sessionwise Brain Region on train
        train_br_l1out_evaluator = LeaveOneSessionOutClassifier("train/l1out_br", True)
        train_br_l1out_evaluator.launch(
            X=X_train_np[train_br_mask],
            y=train_br[train_br_mask],
            s=to_sw_cv(train_md.session_id[train_br_mask]),
        )
        self.futures.append(train_br_l1out_evaluator)

        # Val embeddings brain region
        val_br = val_md.brain_region.values
        val_br_mask = val_br != "Remove"
        # val_br_evaluator = KFoldClassifier("val/br", True)
        # val_br_evaluator.launch(X_val_np[val_br_mask], val_br[val_br_mask])
        # self.futures.append(val_br_evaluator)

        # Sessionwise Brain Region on validation
        val_br_l1out_evaluator = LeaveOneSessionOutClassifier("val/l1out_br", True)
        val_br_l1out_evaluator.launch(
            X=X_val_np[val_br_mask],
            y=val_br[val_br_mask],
            s=to_sw_cv(val_md.session_id[val_br_mask]),
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


TGT_REGIONS = ["CB", "CNU", "CTXsp", "HB", "HPF", "HY", "Isocortex", "MB", "OLF", "TH"]

with open("./neuron_metadata/allen_region_map.pkl", "rb") as f:
    REGION_HEIRARCHY_DICT = pickle.load(f)

    # Change all keys to upper-case to avoid confusion later
    orig_keys = list(REGION_HEIRARCHY_DICT.keys())
    for k in orig_keys:
        v = REGION_HEIRARCHY_DICT.pop(k)
        if k.lower() == k:  # we dont care about neurons that have all-small names
            continue
        k_new = k.upper()
        assert k_new not in REGION_HEIRARCHY_DICT
        REGION_HEIRARCHY_DICT[k_new] = v


def region_map(x: str) -> str:
    if x[0].lower() == x[0]:
        # if all lower names, this does not belong with us
        return "Remove"

    x = x.upper()

    if x == "CUL4 5":
        x = "CUL4, 5"

    if x not in REGION_HEIRARCHY_DICT:
        raise ValueError(f"{x} not found in known heirarchy")

    region_list = REGION_HEIRARCHY_DICT[x]
    if "grey" not in region_list:
        # we're ignoring neurons that
        return "Remove"

    for region in region_list:
        if region in TGT_REGIONS:
            return region

    raise ValueError(f"{x}: No region map found for. Possible options: {region_list}")


# Target regions: CB, CNU, CTXsp, HB, HPF, HY, Isocortex, MB, OLF, TH
# Tried to create by hand
area_map = {
    "ACAd6a": "Isocortex",
    "ACAv5": "Isocortex",
    "ACAv6a": "Isocortex",
    "ANcr2": "CB",
    "APN": "MB",
    "AUDd5": "Isocortex",
    "AUDp5": "Isocortex",
    "AUDv5": "Isocortex",
    "AUDv6a": "Isocortex",
    "BLAp": "CTXsp",
    "BMAa": "CTXsp",
    "BST": "CNU",
    "CA1": "HPF",
    "CA2": "HPF",
    "CA3": "HPF",
    "CB": "CB",
    "CEAm": "CNU",
    "CENT2": "CB",
    "CENT3": "CB",
    "CL": "TH",
    "CM": "TH",
    "COApl": "OLF",
    "COApm": "OLF",
    "COPY": "CB",
    "CP": "CNU",
    "CUL4 5": "CB",
    "DCO": "HB",
    "DG-mo": "HPF",
    "DG-po": "HPF",
    "DG-sg": "HPF",
    "DMH": "HY",
    "ECT2/3": "Isocortex",
    "ECT5": "Isocortex",
    "ENTl2": "HPF",
    "ENTl3": "HPF",
    "ENTl5": "HPF",
    "ENTl6a": "HPF",
    "ENTm1": "HPF",
    "ENTm2": "HPF",
    "ENTm3": "HPF",
    "ENTm5": "HPF",
    "EPd": "CTXsp",
    "Eth": "TH",
    "FF": "HY",
    "FS": "CNU",
    "GPi": "CNU",
    "GRN": "HB",
    "HATA": "HPF",
    "HY": "HY",
    "ICB": "HB",
    "IP": "CB",
    "IRN": "HB",
    "IntG": "TH",
    "LAV": "HB",
    "LD": "TH",
    "LGd-co": "TH",
    "LGd-ip": "TH",
    "LGd-sh": "TH",
    "LHA": "HY",
    "LM": "HY",
    "LP": "TH",
    "LSr": "CNU",
    "LSv": "CNU",
    "MA": "CNU",
    "MARN": "HB",
    "MB": "MB",
    "MD": "TH",
    "MEA": "CNU",
    "MGm": "TH",
    "MGv": "TH",
    "MH": "TH",
    "MOp6a": "Isocortex",
    "MOs5": "Isocortex",
    "MRN": "MB",
    "MV": "HB",
    "MY": "HB",
    "NDB": "CNU",
    "NOT": "MB",
    "NTS": "HB",
    "OLF": "OLF",
    "OT": "CNU",
    "P": "Remove",
    "PA": "CTXsp",
    "PAA": "OLF",
    "PAL": "CNU",
    "PARN": "HB",
    "PCN": "TH",
    "PF": "TH",
    "PGRNd": "HB",
    "PH": "HY",
    "PIL": "TH",
    "PIR": "OLF",
    "PL5": "Isocortex",
    "PL6a": "Isocortex",
    "PO": "TH",
    "POL": "TH",
    "POST": "Remove",
    "PP": "TH",
    "PPN": "MB",
    "PPT": "MB",
    "PRC": "MB",
    "PRE": "HPF",
    "PRM": "CB",
    "PRNr": "HB",
    "PRP": "HB",
    "PVT": "TH",
    "Pa5": "HB",
    "PeF": "HY",
    "PoT": "TH",
    "ProS": "HPF",
    "RE": "TH",
    "RH": "TH",
    "RN": "MB",
    "RR": "MB",
    "RSPd6a": "Isocortex",
    "RSPv1": "Isocortex",
    "RSPv2/3": "Isocortex",
    "RSPv5": "Isocortex",
    "SCdg": "MB",
    "SCig": "MB",
    "SCiw": "MB",
    "SCop": "MB",
    "SCsg": "MB",
    "SF": "CNU",
    "SGN": "TH",
    "SI": "CNU",
    "SIM": "CB",
    "SMT": "TH",
    "SNc": "MB",
    "SNr": "MB",
    "SOCl": "HB",
    "SPFp": "TH",
    "SPIV": "HB",
    "SPVC": "HB",
    "SPVI": "HB",
    "SSp-bfd1": "Isocortex",
    "SSp-bfd2/3": "Isocortex",
    "SSp-m5": "Isocortex",
    "SSp-m6a": "Isocortex",
    "SSp-m6b": "Isocortex",
    "SSp-tr2/3": "Isocortex",
    "SSp-tr4": "Isocortex",
    "SSp-tr5": "Isocortex",
    "SSp-tr6a": "Isocortex",
    "SSp-tr6b": "Isocortex",
    "SSp-ul4": "Isocortex",
    "SSp-ul5": "Isocortex",
    "SSs4": "Isocortex",
    "SSs5": "Isocortex",
    "STR": "CNU",
    "SUB": "HPF",
    "SUV": "HB",
    "TEa5": "Isocortex",
    "TH": "TH",
    "TR": "OLF",
    "TRS": "CNU",
    "TTd": "OLF",
    "V3": "Remove",
    "V4": "Remove",
    "VISa1": "Isocortex",
    "VISa2/3": "Isocortex",
    "VISa4": "Isocortex",
    "VISa5": "Isocortex",
    "VISa6a": "Isocortex",
    "VISa6b": "Isocortex",
    "VISam1": "Isocortex",
    "VISam2/3": "Isocortex",
    "VISam4": "Isocortex",
    "VISam5": "Isocortex",
    "VISam6a": "Isocortex",
    "VISam6b": "Isocortex",
    "VISli4": "Isocortex",
    "VISli5": "Isocortex",
    "VISp2/3": "Isocortex",
    "VISp4": "Isocortex",
    "VISp5": "Isocortex",
    "VISp6a": "Isocortex",
    "VISp6b": "Isocortex",
    "VISpm1": "Isocortex",
    "VISpm2/3": "Isocortex",
    "VISpm4": "Isocortex",
    "VISpm5": "Isocortex",
    "VISpm6a": "Isocortex",
    "VISpm6b": "Isocortex",
    "VISpor5": "Isocortex",
    "VISpor6a": "Isocortex",
    "VISrl6a": "Isocortex",
    "VISrl6b": "Isocortex",
    "VM": "TH",
    "VMH": "HY",
    "VPL": "TH",
    "VPLpc": "TH",
    "VPM": "TH",
    "VPMpc": "TH",
    "VeCB": "CB",
    "Xi": "TH",
    "ZI": "HY",
    "alv": "Remove",
    "arb": "Remove",
    "bsc": "Remove",
    "ccb": "Remove",
    "ccg": "Remove",
    "ccs": "Remove",
    "cing": "Remove",
    "cpd": "Remove",
    "das": "Remove",
    "dhc": "Remove",
    "fa": "Remove",
    "fiber tracts": "Remove",
    "fp": "Remove",
    "hbc": "Remove",
    "icp": "Remove",
    "int": "Remove",
    "ll": "Remove",
    "ml": "Remove",
    "mlf": "Remove",
    "opt": "Remove",
    "or": "Remove",
    "root": "Remove",
    "scp": "Remove",
    "scwm": "Remove",
    "sm": "Remove",
    "sptV": "Remove",
    "tspc": "Remove",
    "uf": "Remove",
    "void": "Remove",
    "y": "Remove",
}


# This is what was used in NEMO paper
def acryonym_to_region(acronym: str) -> str:
    if acronym.startswith("PO"):
        return "PO"
    elif acronym.startswith("LP"):
        return "LP"
    elif acronym.startswith("DG"):
        return "DG"
    elif acronym.startswith("CA1"):
        return "CA1"
    elif acronym.startswith("VISa"):
        return "VISa"
    else:
        return "Remove"

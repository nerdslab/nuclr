from typing import List, Any
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

class AllenVB2022VisEvaluator():
    def __init__(self, world_size: int):
        # Initialize dataset metadata 
        self.md = self.prep_md_df()
        # Ray cluster
        ray.init(address="local", num_cpus=world_size * 7)

    @staticmethod
    def prep_md_df():
        df = pd.read_csv("./neuron_metadata/allen_vbn_2022.csv")
        df["unit_id"] = df["id"].astype(str)
        df["brain_region"] = df["structure_acronym"]
        # TODO: Allen VC uses df.lolcat_celltype, but what should VB use? 
        # df["cell_type"] = df.lolcat_celltype
        return df.set_index("unit_id")
    
    def __call__(
        self,
        X_train: torch.Tensor,
        train_unit_ids: List[str] | np.ndarray,
        X_val: torch.Tensor,
        val_unit_ids: List[str] | np.ndarray,
        ) -> dict[str, Any]:
        
        # Tensor -> Detach -> CPU -> Numpy Array
        X_train_np = X_train.detach().cpu().numpy()
        X_val_np = X_val.detach().cpu().numpy()

        # Tensor -> Numpy Array
        train_unit_ids = np.array(train_unit_ids)
        val_unit_ids = np.array(val_unit_ids)

        # Find the metadata for the train & val units
        train_md = self.md.loc[train_unit_ids]
        val_md = self.md.loc[val_unit_ids]

        # Banned brain regions left empty for now
        banned_brs = []
        self.futures = []

        # Find brain regions for the train and val embeddings
        train_br = train_md.brain_region.values
        val_br = val_md.brain_region.values

        # Mask for train and val embeddings if in banned regions
        train_br_mask = ~np.isin(train_br, banned_brs)
        val_br_mask = ~np.isin(val_br, banned_brs)
        
        # train_br_evaluator = KFoldClassifier("train/br", True)
        # train_br_evaluator.launch(X_train_np[train_br_mask], train_br[train_br_mask])
        # self.futures.append(train_br_evaluator)
        # val_br_evaluator = KFoldClassifier("val/br", True)
        # val_br_evaluator.launch(X_val_np[val_br_mask], val_br[val_br_mask])
        # self.futures.append(val_br_evaluator)

        # BRAIN REGION | INDUCTIVE
        # Session-wise
        val_l1o_br_evaluator = LeaveOneSessionOutClassifier(
            name = "val/l1out_br", 
            balanced = True)
        val_l1o_br_evaluator.launch(
            X=X_val_np[val_br_mask],
            y=val_br[val_br_mask],
            s=val_md.session_id[val_br_mask],
        )
        self.futures.append(val_l1o_br_evaluator)

        # BRAIN REGION | ZERO SHOT
        # val_zs_br_evaluator = ZeroshotClassifier("val/zs_br", True)
        # val_zs_br_evaluator.launch(
        #     X_train=X_train_np,
        #     y_train=train_br,
        #     X_test=X_val_np,
        #     y_test=val_br,
        # )
        # self.futures.append(val_zs_br_evaluator)

        # CELLTYPE
        # Session-wise
        val_ctype = val_md.cell_type.values
        val_ctype_mask = val_ctype != "wt"
        val_ctype_evaluator = LeaveOneSessionOutClassifier("val/l1out_ctype", True)
        val_ctype_evaluator.launch(
            X=X_val_np[val_ctype_mask],
            y=val_ctype[val_ctype_mask],
            s=val_md.session_id[val_ctype_mask],
        )
        self.futures.append(val_ctype_evaluator)

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

from typing import List, Dict, Any
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


class BugeonEvaluator:
    def __init__(self, world_size: int):
        self.md = self.prep_md_df()
        ray.init(address="local", num_cpus=world_size * 7)

    @staticmethod
    def prep_md_df():
        df = pd.read_csv("./neuron_metadata/bugeon.csv")
        return df.set_index("id")

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

        banned_subclasses = ["unknown", "Serpinf1"]

        self.futures = []

        # Train embeddings brain region
        train_ei_evaluator = KFoldClassifier("train/ei", True)
        train_ei_evaluator.launch(X_train_np, train_md.ei_class.values)
        self.futures.append(train_ei_evaluator)

        train_l1o_ei_eval = LeaveOneSessionOutClassifier("train/subj_l1out_ei", True)
        train_l1o_ei_eval.launch(
            X=X_train_np,
            y=train_md.ei_class.values,
            s=train_md.subject_id,
        )
        self.futures.append(train_l1o_ei_eval)

        # Train embeddings subclass
        train_subclass_mask = ~np.isin(train_md.subclass, banned_subclasses)
        train_subclass_evaluator = KFoldClassifier("train/subclass", True)
        train_subclass_evaluator.launch(
            X=X_train_np[train_subclass_mask],
            y=train_md.subclass.values[train_subclass_mask],
        )
        self.futures.append(train_subclass_evaluator)

        train_l1o_subclass_eval = LeaveOneSessionOutClassifier(
            "train/subj_l1out_subclass", True
        )
        train_l1o_subclass_eval.launch(
            X=X_train_np[train_subclass_mask],
            y=train_md.subclass.values[train_subclass_mask],
            s=train_md.subject_id[train_subclass_mask],
        )
        self.futures.append(train_l1o_subclass_eval)

        # Validation
        # EI zeroshot
        val_zs_ei_evaluator = ZeroshotClassifier("val/zs_ei", True)
        val_zs_ei_evaluator.launch(
            X_train=X_train_np,
            y_train=train_md.ei_class.values,
            X_test=X_val_np,
            y_test=val_md.ei_class.values,
        )
        self.futures.append(val_zs_ei_evaluator)

        # Subclass zeroshot
        val_subclass_mask = ~np.isin(val_md.subclass, banned_subclasses)
        val_zs_subclass_evaluator = ZeroshotClassifier("val/zs_subclass", True)
        val_zs_subclass_evaluator.launch(
            X_train=X_train_np[train_subclass_mask],
            y_train=train_md.subclass.values[train_subclass_mask],
            X_test=X_val_np[val_subclass_mask],
            y_test=val_md.subclass.values[val_subclass_mask],
        )
        self.futures.append(val_zs_subclass_evaluator)

        #
        if len(np.unique(val_md.subject_id)) > 2:
            val_l1o_ei_eval = LeaveOneSessionOutClassifier("val/subj_l1out_ei", True)
            val_l1o_ei_eval.launch(
                X=X_val_np,
                y=val_md.ei_class.values,
                s=val_md.subject_id,
            )
            self.futures.append(val_l1o_ei_eval)

            val_l1o_subclass_eval = LeaveOneSessionOutClassifier(
                "val/subj_l1out_subclass", True
            )
            val_l1o_subclass_eval.launch(
                X=X_val_np[val_subclass_mask],
                y=val_md.subclass.values[val_subclass_mask],
                s=val_md.subject_id[val_subclass_mask],
            )
            self.futures.append(val_l1o_subclass_eval)

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

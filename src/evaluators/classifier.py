from abc import ABC, abstractmethod
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import ray
import wandb


@ray.remote(num_cpus=1)
def logistic_classification(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray | None,
    y_test: np.ndarray | None,
    balanced: bool,
):
    # Train the model
    class_weight = "balanced" if balanced else None
    model = LogisticRegression(
        max_iter=1000,
        solver="newton-cg",
        class_weight=class_weight,
    )
    model.fit(X, y)

    if X_test is None:
        return {"model": model}
    else:
        assert y_test is not None
        preds = model.predict(X_test)
        return {
            "model": model,
            "test_preds": preds,
            "test_targets": y_test,
        }


class ParallelClassifier(ABC):
    _labels: np.ndarray
    _futures: List[ray.ObjectRef]

    def __init__(self, name: str, balanced: bool):
        self.name = name
        self.balanced = balanced

    @abstractmethod
    def launch(self, *args, **kwargs):
        """Launch the classification processes."""
        pass

    def finalize(self):
        """
        Gather results and log
        """
        results = ray.get(self._futures)

        all_preds = np.concatenate([r["test_preds"] for r in results])
        all_targets = np.concatenate([r["test_targets"] for r in results])
        bacc = balanced_accuracy_score(all_targets, all_preds)
        acc = accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(
            y_true=all_targets,
            y_pred=all_preds,
            labels=self._labels,
            normalize="true",
        )
        f1_macro = f1_score(y_true=all_targets, y_pred=all_preds, average="macro")
        f1_micro = f1_score(y_true=all_targets, y_pred=all_preds, average="micro")

        size = 0.35 * len(self._labels) + 1.5
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(111)
        disp = ConfusionMatrixDisplay(cm, display_labels=self._labels)
        disp.plot(
            cmap="Blues",
            ax=ax,
            xticks_rotation=45,  # type: ignore
            colorbar=False,
            im_kw={"vmin": 0, "vmax": 1},
            text_kw={"fontsize": 8},
            values_format=".2f",
        )
        disp.figure_.colorbar(disp.im_, ax=disp.ax_, fraction=0.046, pad=0.04)
        disp.figure_.tight_layout()

        metrics = {
            f"{self.name}/acc": acc,
            f"{self.name}/bacc": bacc,
            f"{self.name}/f1_macro": f1_macro,
            f"{self.name}/f1_micro": f1_micro,
            f"{self.name}/cm": wandb.Image(disp.figure_),
        }
        return metrics


class KFoldClassifier(ParallelClassifier):
    def launch(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5):
        self._labels = np.unique(y)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Launch parallel training jobs
        self._futures = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self._futures.append(
                logistic_classification.remote(
                    X_train, y_train, X_val, y_val, self.balanced
                )
            )


class LeaveOneSessionOutClassifier(ParallelClassifier):
    def launch(self, X: np.ndarray, y: np.ndarray, s: np.ndarray):
        """s: Session id"""

        unique_s = np.unique(s)
        self._labels = np.unique(y)
        self._futures = []

        # For each unique session as test set
        for test_session in unique_s:
            # Get train/test indices
            train_idx = s != test_session
            test_idx = s == test_session

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self._futures.append(
                logistic_classification.remote(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    self.balanced,
                )
            )


class ZeroshotClassifier(ParallelClassifier):
    def launch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ):

        self._labels = np.unique(y_train)
        self._futures = []

        self._futures.append(
            logistic_classification.remote(
                X_train,
                y_train,
                X_test,
                y_test,
                self.balanced,
            )
        )

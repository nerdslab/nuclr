from typing import Literal
from pathlib import Path
from tqdm import tqdm
import argparse
import ray
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report

from utils import load_embs, load_splits, presence_mask


def eval_multisplit_with_cv(
    embs_list,
    ids,
    labels,
    splits,
    class_balance_type: Literal["resample", "weight"],
):

    BALANCED_RESAMPLING = class_balance_type == "resample"

    def balanced_resampling(X, y):
        resample = RandomOverSampler(random_state=42)
        resample_idx, _ = resample.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
        resample_idx = resample_idx.ravel()
        return X[resample_idx], y[resample_idx]

    def logreg_scores(X, y, train_mask, val_mask):
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        if BALANCED_RESAMPLING:
            X_train, y_train = balanced_resampling(X_train, y_train)

        scaler = StandardScaler()
        clf = LogisticRegression(
            max_iter=1000,
            tol=1e-4,
            C=1.0,
            solver="newton-cg",
            verbose=0,
            class_weight="balanced" if not BALANCED_RESAMPLING else None,
        )
        clf.fit(scaler.fit_transform(X_train), y_train)
        pred_val = clf.predict(scaler.transform(X_val))

        return {
            "pred": pred_val,
            "true": y_val,
        }

    @ray.remote
    def ray_logreg_scores(X, y, train_mask, val_mask):
        return logreg_scores(X, y, train_mask, val_mask)

    ray.init(address="local", num_cpus=32, ignore_reinit_error=True)

    num_ckpt = len(embs_list)
    num_splits = len(splits)

    # Launch all possible jobs
    cv_futures = [[[] for _ in range(num_ckpt)] for _ in range(num_splits)]
    test_futures = [[None for _ in range(num_ckpt)] for _ in range(num_splits)]
    for split_idx, split in enumerate(splits):
        test_mask = presence_mask(ids, split["test"])
        if "train" in split:
            train_mask = presence_mask(ids, split["train"])
        else:
            train_mask = ~test_mask

        # Do CV to find best epoch
        for ckpt_idx in range(num_ckpt):
            X = embs_list[ckpt_idx]
            y = labels

            if "train_cv" in split:
                # There are multiple CV splits
                for cv_idx, cv_split in enumerate(split["train_cv"]):
                    cv_train_mask = presence_mask(ids, cv_split["train"])
                    cv_val_mask = presence_mask(ids, cv_split["val"])
                    _future = ray_logreg_scores.remote(X, y, cv_train_mask, cv_val_mask)
                    cv_futures[split_idx][ckpt_idx].append(_future)

                _future = ray_logreg_scores.remote(X, y, train_mask, test_mask)
                test_futures[split_idx][ckpt_idx] = _future  # pyright: ignore
            elif "val" in split:
                # There is only one Train/Val split
                train_mask = presence_mask(ids, split["train"])
                val_mask = presence_mask(ids, split["val"])
                _future = ray_logreg_scores.remote(X, y, train_mask, val_mask)
                cv_futures[split_idx][ckpt_idx] = _future

                _future = ray_logreg_scores.remote(
                    X, y, train_mask | val_mask, test_mask
                )
                test_futures[split_idx][ckpt_idx] = _future  # pyright: ignore
            else:
                raise ValueError

    test_results_list = []
    best_ckpt_idx_list = []
    for split_idx, split in tqdm(enumerate(splits), total=len(splits)):
        scores_list = []
        for ckpt_idx in range(num_ckpt):
            results_list = ray.get(cv_futures[split_idx][ckpt_idx])
            if isinstance(results_list, list):
                pred = np.concatenate([result["pred"] for result in results_list])
                true = np.concatenate([result["true"] for result in results_list])
            else:
                pred = results_list["pred"]
                true = results_list["true"]

            scores = {
                "bacc": balanced_accuracy_score(true, pred),
                "f1": f1_score(true, pred, average="macro"),
                "index": ckpt_idx,
            }
            scores_list.append(scores)

        # Find results on best epoch
        best_scores = sorted(scores_list, key=lambda x: x["f1"], reverse=True)[0]
        best_ckpt_idx = best_scores["index"]
        best_ckpt_idx_list.append(best_ckpt_idx)

        test_results = ray.get(test_futures[split_idx][best_ckpt_idx])
        test_results["index"] = best_ckpt_idx  # pyright: ignore
        test_results_list.append(test_results)

    pred = np.concatenate([result["pred"] for result in test_results_list])
    true = np.concatenate([result["true"] for result in test_results_list])
    bacc = balanced_accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")
    print(classification_report(true, pred, digits=4))
    print(f"bacc: {bacc:.4f}")
    print(f"f1: {f1:.4f}")

    print(f"{best_ckpt_idx_list} (total ckpts = {num_ckpt})")

    ray.shutdown()


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "emb_path",
        type=Path,
        help="Directory containing embeddings for multiple epochs or a path to a single embedding file",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
    )
    parser.add_argument(
        "--class-balance",
        type=str,
        choices=["weight", "resample"],
        default="weight",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_cli()

    embs, ids = load_embs(args.emb_path)
    label_map, splits = load_splits(args.splits_path)

    if not isinstance(splits, list) and isinstance(splits, dict):
        splits = [splits]

    # Subset units based on what labels are available
    valid_id_mask = presence_mask(ids, label_map.index.values)
    print(f"Valid UUID stats: {valid_id_mask.mean()=}, {(~valid_id_mask).sum()=}")

    valid_ids = ids[valid_id_mask]
    valid_embs = [emb[valid_id_mask] for emb in embs]
    valid_labels = label_map.loc[valid_ids].to_numpy().ravel()

    eval_multisplit_with_cv(
        valid_embs,
        valid_ids,
        valid_labels,
        splits,
        class_balance_type=args.class_balance,
    )


if __name__ == "__main__":
    main()

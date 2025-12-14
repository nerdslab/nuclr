from pathlib import Path
import argparse
from tqdm import tqdm
import ray
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report

from utils import load_embs, load_splits, presence_mask


def eval_with_cv(valid_embs, valid_ids, valid_labels, split):

    def logreg_scores(X, y, train_mask, val_mask, index):
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        scaler = StandardScaler()
        clf = LogisticRegression(
            max_iter=1000,
            tol=1e-4,
            class_weight="balanced",
            C=1.0,
            solver="newton-cg",
            verbose=0,
        )
        clf.fit(scaler.fit_transform(X_train), y_train)
        pred_val = clf.predict(scaler.transform(X_val))

        return {
            "pred": pred_val,
            "true": y_val,
            "f1": f1_score(y_val, pred_val, average="macro"),
            "bacc": balanced_accuracy_score(y_val, pred_val),
            "index": index,
        }

    @ray.remote
    def ray_logreg_scores(X, y, train_mask, val_mask, index):
        return logreg_scores(X, y, train_mask, val_mask, index)

    ray.init(address="local", num_cpus=32, num_gpus=0, ignore_reinit_error=True)

    train_ids = split["train"]
    train_mask = presence_mask(valid_ids, train_ids)
    val_mask = presence_mask(valid_ids, split["val"])
    test_mask = presence_mask(valid_ids, split["test"])

    y = valid_labels
    val_futures = []
    test_futures = []
    for i, X in tqdm(enumerate(valid_embs), total=len(valid_embs)):
        _future = ray_logreg_scores.remote(X, y, train_mask, val_mask, i)
        val_futures.append(_future)

        _future = ray_logreg_scores.remote(X, y, train_mask | val_mask, test_mask, i)
        test_futures.append(_future)

    # scores_list = []
    val_result_list = []
    test_result_list = []
    for i, (val_future, test_future) in enumerate(zip(val_futures, test_futures)):
        val_result = ray.get(val_future)
        test_result = ray.get(test_future)
        val_result_list.append(val_result)
        test_result_list.append(test_result)
        print(
            f"Index {i} | "
            f"Validation - bacc: {val_result['bacc']:.4f}, F1: {val_result['f1']:.4f} | "
            f"Test - bacc: {test_result['bacc']:.4f}, F1: {test_result['f1']:.4f}"
        )

    # best_scores = sorted(scores_list, key=lambda x: x["val_f1"], reverse=True)[0]
    best_val_result = sorted(val_result_list, key=lambda x: x["f1"], reverse=True)[0]
    best_ckpt_idx = best_val_result["index"]
    print(f"Best index: {best_ckpt_idx}")
    print(f"Validation bal. acc.: {best_val_result['bacc']:.4f}")
    print(f"Validation F1: {best_val_result['f1']:.4f}")

    best_test_result = test_result_list[best_ckpt_idx]
    print(f"Test bal. acc.: {best_test_result['bacc']:.4f}")
    print(f"Test F1: {best_test_result['f1']:.4f}")
    # print(f"Best checkpoint: {ckpt_paths[best_scores['index']].name}")

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
    args = parser.parse_args()
    return args


def main():
    args = parse_cli()

    embs, ids = load_embs(args.emb_path)
    label_map, splits = load_splits(args.splits_path)

    # Subset units based on what labels are available
    valid_id_mask = presence_mask(ids, label_map.index)
    print(f"Valid UUID stats: {valid_id_mask.mean()=}, {(~valid_id_mask).sum()=}")

    valid_ids = ids[valid_id_mask]
    valid_embs = [emb[valid_id_mask] for emb in embs]
    valid_labels = label_map.loc[valid_ids].to_numpy().ravel()

    eval_with_cv(valid_embs, valid_ids, valid_labels, splits)


if __name__ == "__main__":
    main()

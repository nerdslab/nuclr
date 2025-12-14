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

    def balanced_resampling(X, y):
        resample = RandomOverSampler(random_state=42)
        resample_idx, _ = resample.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
        resample_idx = resample_idx.ravel()
        return X[resample_idx], y[resample_idx]

    def logreg_scores(X, y, train_mask, val_mask, test_mask):
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        X_train, y_train = balanced_resampling(X_train, y_train)

        scaler = StandardScaler()
        clf = LogisticRegression(
            max_iter=1000,
            tol=1e-5,
            C=1.0,
            verbose=0,
        )
        clf.fit(scaler.fit_transform(X_train), y_train)
        pred_val = clf.predict(scaler.transform(X_val))
        pred_test = clf.predict(scaler.transform(X_test))

        return {
            "val_f1": f1_score(y_val, pred_val, average="macro"),
            "val_bacc": balanced_accuracy_score(y_val, pred_val),
            "val_report": classification_report(y_val, pred_val),
            "test_f1": f1_score(y_test, pred_test, average="macro"),
            "test_bacc": balanced_accuracy_score(y_test, pred_test),
            "test_report": classification_report(y_test, pred_test),
            "test_pred": pred_test,
            "test_true": y_test,
        }

    @ray.remote(num_cpus=1)
    def ray_logreg_scores(X, y, train_mask, val_mask, test_mask):
        return logreg_scores(X, y, train_mask, val_mask, test_mask)

    ray.init(address="local", num_cpus=32, num_gpus=0, ignore_reinit_error=True)

    train_ids = split["train"]
    train_mask = presence_mask(valid_ids, train_ids)
    val_mask = presence_mask(valid_ids, split["val"])
    test_mask = presence_mask(valid_ids, split["test"])

    y = valid_labels
    futures = []
    for i, X in tqdm(enumerate(valid_embs), total=len(valid_embs)):
        _future = ray_logreg_scores.remote(X, y, train_mask, val_mask, test_mask)
        futures.append(_future)

    scores_list = []
    for i, future in enumerate(futures):
        scores = ray.get(future)
        scores["index"] = i
        scores_list.append(scores)
        print(
            f"Index {i} | "
            f"Validation - bacc: {scores['val_bacc']:.4f}, F1: {scores['val_f1']:.4f} | "
            f"Test - bacc: {scores['test_bacc']:.4f}, F1: {scores['test_f1']:.4f}"
        )

    best_scores = sorted(scores_list, key=lambda x: x["val_f1"], reverse=True)[0]
    print(f"Best index: {best_scores['index']}")
    print(f"Validation bal. acc.: {best_scores['val_bacc']:.4f}")
    print(f"Validation F1: {best_scores['val_f1']:.4f}")
    print(f"Test bal. acc.: {best_scores['test_bacc']:.4f}")
    print(f"Test F1: {best_scores['test_f1']:.4f}")
    # print(f"Best checkpoint: {ckpt_paths[best_scores['index']].name}")


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

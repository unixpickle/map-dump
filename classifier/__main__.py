"""
Evaluate embeddings by training a linear probe to predict store categories
from the embedding vectors.
"""

import argparse
import hashlib
import json
from typing import Dict, List, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings", type=str, required=True, help="path to embeddings JSON file"
    )
    parser.add_argument(
        "--categories", type=str, required=True, help="path to categories JSON file"
    )
    parser.add_argument("--seed", type=int, default=0, help="train/valid split seed")
    parser.add_argument(
        "--full-categories",
        action="store_true",
        help="use full categories, not just general names",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="linear",
        help="classifier type: dummy, linear, svm",
    )
    args = parser.parse_args()

    print("loading embeddings...")
    embs = load_embeddings(args.embeddings)

    print("loading dataset...")
    full_ds, num_labels = load_categories(args.categories, args.full_categories)
    print(f"number of labels: {num_labels}")
    filtered_ds = {k: v for k, v in full_ds.items() if k in embs}
    print(
        f"omitting {len(full_ds)-len(filtered_ds)}/{len(full_ds)} stores not present in embeddings..."
    )
    train_data, test_data = split_dataset(filtered_ds, args.seed)

    if args.classifier == "svm":
        clf = LinearSVC(verbose=True)
        scaler = Nystroem(n_components=1024)
    else:
        clf = {
            "dummy": DummyClassifier(),
            "linear": LogisticRegression(max_iter=10000, multi_class="multinomial"),
        }[args.classifier]
        scaler = StandardScaler()

    print("training classifier...")
    train_xs, train_ys, train_ws = embed_dataset(embs, train_data)
    scaler.fit(train_xs)
    print(f"num examples: {len(train_xs)}")
    clf.fit(scaler.transform(train_xs), train_ys, train_ws)
    print("evaluating...")
    test_xs, test_ys, test_ws = embed_dataset(embs, test_data)
    test_acc = clf.score(scaler.transform(test_xs), test_ys, test_ws)
    train_acc = clf.score(scaler.transform(train_xs), train_ys, train_ws)
    print(f"mean train accuracy: {train_acc}")
    print(f"mean test accuracy: {test_acc}")


def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        obj = json.load(f)
    return dict(zip(obj["store_names"], np.array(obj["vecs"], dtype=np.float32)))


def load_categories(
    path: str, full_categories: bool
) -> Tuple[Dict[str, List[int]], int]:
    store_to_cat_names = {}
    with open(path, "rb") as f:
        obj = json.load(f)
    for k, vs in obj.items():
        cat_names = [v["path"] if full_categories else v["name"] for v in vs]
        store_to_cat_names[k] = cat_names
    cat_names = set(x for y in store_to_cat_names.values() for x in y)
    cat_name_to_idx = {x: i for i, x in enumerate(sorted(cat_names))}
    return {
        k: [cat_name_to_idx[cat_name] for cat_name in v]
        for k, v in store_to_cat_names.items()
    }, len(cat_name_to_idx)


def split_dataset(
    name_to_cat: Dict[str, List[int]], seed: int
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Create a (train, test) split by splitting up the stores by name.
    """
    test_keys = ["a", "b"]  # Only 2/16 hex characters are mapped to test
    in_test = {
        k: hashlib.md5(bytes(f"{seed:010}{k}", "utf-8")).hexdigest()[0] in test_keys
        for k in name_to_cat.keys()
    }
    return (
        {k: v for k, v in name_to_cat.items() if not in_test[k]},
        {k: v for k, v in name_to_cat.items() if in_test[k]},
    )


def embed_dataset(
    embs: Dict[str, np.ndarray], name_to_cat: Dict[str, List[int]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the dataset into (vectors, labels, weights).
    """
    vectors = []
    labels = []
    weights = []
    for name, cats in name_to_cat.items():
        vec = embs[name]
        for cat in cats:
            vectors.append(vec)
            labels.append(cat)
            weights.append(1.0 / len(cats))
    return np.stack(vectors, axis=0), np.array(labels), np.array(weights)


if __name__ == "__main__":
    main()

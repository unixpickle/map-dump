"""
Compute GloVe-style embeddings given a co-occurrence matrix exported by the
`cooccurrence` sub-command of the main Rust program.
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.lib.npyio import NpzFile

from .sparse import SparseMatmul, SparseMatrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_max", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=3 / 4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--adam", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("cooc_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading co-occurrence info...")
    cooc = Cooc.load(args.cooc_path, device)
    print(f"total of {len(cooc.store_names)} stores")

    print("pre-computing weights and targets...")
    targets = cooc.log_cooc()
    weights = cooc.loss_weights(cutoff=args.x_max, power=args.alpha)
    num_nonzero = weights.values.shape[0]
    if args.dense:
        targets = targets.to_dense()
        weights = weights.to_dense()
    else:
        sparse_mm = SparseMatmul(targets)
    print(f" - density fraction: {num_nonzero / (cooc.cooccurrences.shape[0] ** 2)}")

    print("creating parameters and optimizer...")
    n_vocab = len(cooc.store_names)
    vecs = nn.Parameter(torch.randn(n_vocab, args.dim, device=device))
    vecs_bias = nn.Parameter(torch.zeros(n_vocab, device=device))
    contexts = nn.Parameter(torch.randn(n_vocab, args.dim, device=device))
    contexts_bias = nn.Parameter(torch.zeros(n_vocab, device=device))
    bias_lr_boost = math.sqrt(args.dim)
    params = [vecs, vecs_bias, contexts, contexts_bias]
    opt = (optim.Adam if args.adam else optim.Adagrad)(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    print("optimizing...")
    for i in range(args.iters):
        if args.dense:
            biases = bias_lr_boost * (vecs_bias[:, None] + contexts_bias)
            pred = (vecs @ contexts.T) + biases
        else:
            biases = bias_lr_boost * sparse_bias_sum(
                cooc.cooccurrences, vecs_bias, contexts_bias
            )
            pred = sparse_mm.mm(vecs, contexts.T) + biases
        losses = weights * ((pred - targets) ** 2)
        loss = losses.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"step {i+1}/{args.iters}: loss={loss.item()}")

    print("saving results...")
    with open(args.output_path, "w") as f:
        json.dump(
            dict(
                vecs=vecs.detach().cpu().numpy().tolist(),
                vecs_bias=vecs_bias.detach().cpu().numpy().tolist(),
                contexts=contexts.detach().cpu().numpy().tolist(),
                contexts_bias=contexts_bias.detach().cpu().numpy().tolist(),
                store_names=cooc.store_names,
                store_counts=cooc.store_counts,
            ),
            f,
        )


def sparse_bias_sum(
    target_mat: SparseMatrix, row_bias: torch.Tensor, col_bias: torch.Tensor
) -> SparseMatrix:
    indices = target_mat.indices
    return SparseMatrix(
        shape=target_mat.shape,
        indices=indices,
        values=row_bias[indices[0]] + col_bias[indices[1]],
    )


@dataclass
class Cooc:
    store_names: List[str]
    store_counts: List[int]
    cooccurrences: SparseMatrix

    @classmethod
    def load(cls, path: str, device: torch.device) -> "Cooc":
        if path.endswith(".json"):
            with open(path, "rb") as f:
                all_data = json.load(f)
            return cls(
                store_names=all_data["names"],
                store_counts=all_data["store_counts"],
                cooccurrences=_remove_diagonal(
                    _matrix_from_json(
                        len(all_data["names"]), all_data["binary_counts"], device
                    ).compact_index_dtype()
                ),
            )
        else:
            np_obj = np.load(path)
            names = np_obj["names"].tolist()
            return cls(
                store_names=names,
                store_counts=np_obj["store_counts"].tolist(),
                cooccurrences=_remove_diagonal(
                    _matrix_from_npz(
                        len(names), "binary_counts", np_obj, device
                    ).compact_index_dtype()
                ),
            )

    def log_cooc(self) -> SparseMatrix:
        return SparseMatrix(
            shape=self.cooccurrences.shape,
            indices=self.cooccurrences.indices,
            values=self.cooccurrences.values.log(),
        )

    def loss_weights(self, cutoff: float, power: float) -> SparseMatrix:
        values = (self.cooccurrences.values.clamp(max=cutoff) / cutoff) ** power
        return SparseMatrix(
            shape=self.cooccurrences.shape,
            indices=self.cooccurrences.indices,
            values=values,
        )


def _matrix_from_json(
    size: int, obj: Union[Dict[str, Any], List[List[float]]], device: torch.device
) -> SparseMatrix:
    if isinstance(obj, list):
        return SparseMatrix.from_dense(
            torch.tensor(obj, dtype=torch.float32, device=device)
        )
    else:
        return SparseMatrix(
            shape=(size, size),
            indices=torch.tensor(obj["indices"], dtype=torch.long, device=device),
            values=torch.tensor(obj["values"], dtype=torch.float32, device=device),
        )


def _matrix_from_npz(
    size: int, key: str, np_obj: NpzFile, device: torch.device
) -> SparseMatrix:
    if key in list(np_obj.keys()):
        dense = np_obj[key]
        return SparseMatrix.from_dense(
            torch.from_numpy(dense).to(dtype=torch.float32, device=device)
        )
    else:
        rows = np_obj[f"{key}_rows"]
        cols = np_obj[f"{key}_cols"]
        indices = np.stack([rows, cols], axis=0).astype(np.int32)
        values = np_obj[f"{key}_values"]
        return SparseMatrix(
            shape=(size, size),
            indices=torch.from_numpy(indices).to(dtype=torch.long, device=device),
            values=torch.from_numpy(values).to(dtype=torch.float32, device=device),
        )


def _remove_diagonal(matrix: SparseMatrix) -> SparseMatrix:
    values = matrix.values
    indices = matrix.indices
    non_diag = indices[0] != indices[1]
    return SparseMatrix(
        shape=matrix.shape,
        indices=indices[:, non_diag],
        values=values[non_diag],
    )


if __name__ == "__main__":
    main()

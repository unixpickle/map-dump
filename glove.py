"""
Compute GloVe-style embeddings given a co-occurrence matrix exported by the
`cooccurrence` sub-command of the main Rust program.
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_max", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=3 / 4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--dense", action="store_true")
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
    if args.dense:
        targets = targets.to_dense()
        weights = weights.to_dense()

    print("creating parameters and optimizer...")
    n_vocab = len(cooc.store_names)
    vecs = nn.Parameter(torch.randn(n_vocab, args.dim, device=device))
    vecs_bias = nn.Parameter(torch.zeros(n_vocab, device=device))
    contexts = nn.Parameter(torch.randn(n_vocab, args.dim, device=device))
    contexts_bias = nn.Parameter(torch.zeros(n_vocab, device=device))
    bias_lr_boost = math.sqrt(args.dim)
    opt = optim.Adagrad([vecs, vecs_bias, contexts, contexts_bias], lr=args.lr)

    print("optimizing...")
    for i in range(args.iters):
        if args.dense:
            biases = bias_lr_boost * (vecs_bias[:, None] + contexts_bias)
            pred = (vecs @ contexts.T) + biases
        else:
            biases = bias_lr_boost * sparse_bias_sum(
                cooc.cooccurrences, vecs_bias, contexts_bias
            )
            # Work-around for the fact that there is no gradient for to_sparse_csr().
            pred = (
                torch.sparse.sampled_addmm(
                    (biases * 0.0).detach().to_sparse_csr(), vecs, contexts.T
                ).to_sparse_coo()
                + biases
            )
        losses = weights * ((pred - targets) ** 2)
        if args.dense:
            loss = losses.sum()
        else:
            loss = losses.values().sum()
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
    target_mat: torch.Tensor, row_bias: torch.Tensor, col_bias: torch.Tensor
) -> torch.Tensor:
    indices = target_mat.indices()
    return torch.sparse_coo_tensor(
        indices=indices,
        values=row_bias[indices[0]] + col_bias[indices[1]],
        size=target_mat.shape,
    ).coalesce()


@dataclass
class Cooc:
    store_names: List[str]
    store_counts: List[int]
    cooccurrences: torch.Tensor  # must be coalesced sparse COO tensor

    @classmethod
    def load(cls, path: str, device: torch.device) -> "Cooc":
        with open(path, "rb") as f:
            all_data = json.load(f)
        return cls(
            store_names=all_data["names"],
            store_counts=all_data["store_counts"],
            cooccurrences=_remove_diagonal(
                _matrix_from_json(
                    len(all_data["names"]), all_data["binary_counts"], device
                )
            ),
        )

    def log_cooc(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            indices=self.cooccurrences.indices(),
            values=self.cooccurrences.values().log(),
            size=self.cooccurrences.shape,
        ).coalesce()

    def loss_weights(self, cutoff: float, power: float) -> torch.Tensor:
        values = (self.cooccurrences.values().clamp(max=cutoff) / cutoff) ** power
        indices = self.cooccurrences.indices()

        # Remove the diagonal of the matrix.
        mask = indices[0] == indices[1]
        values[mask] = 0.0
        return torch.sparse_coo_tensor(
            indices=indices, values=values, size=self.cooccurrences.shape
        ).coalesce()


def _matrix_from_json(
    size: int, obj: Union[Dict[str, Any], List[List[float]]], device=torch.device
) -> torch.Tensor:
    if isinstance(obj, list):
        return (
            torch.tensor(obj, dtype=torch.float32, device=device).to_sparse().coalesce()
        )
    else:
        return torch.sparse_coo_tensor(
            indices=torch.tensor(obj["indices"], dtype=torch.long, device=device),
            values=torch.tensor(obj["values"], dtype=torch.float32, device=device),
            size=(size, size),
        ).coalesce()


def _remove_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    values = matrix.values()
    indices = matrix.indices()
    non_diag = indices[0] != indices[1]
    return torch.sparse_coo_tensor(
        indices=indices[:, non_diag],
        values=values[non_diag],
        size=matrix.shape,
    ).coalesce()


if __name__ == "__main__":
    main()

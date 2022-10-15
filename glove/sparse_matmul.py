from typing import Tuple

import torch


class SparseMatmul:
    """
    Multiply two dense matrices into a sparse output matrix.
    """

    def __init__(self, output_pattern: torch.Tensor, block_size: int = 512):
        assert output_pattern.shape[0] == output_pattern.shape[1]
        self._block_size = block_size
        self._out_size = output_pattern.shape[0]

        all_indices = output_pattern.indices()

        self._block_take_indices = []
        new_out_indices = []
        for row, col in self._iterate_blocks():
            out_flags = (
                (all_indices[0] >= row)
                & (all_indices[0] < row + block_size)
                & (all_indices[1] >= col)
                & (all_indices[1] < col + block_size)
            )
            required_indices = all_indices[:, out_flags]

            block_cols = min(block_size, self._out_size - col)
            local_indices = required_indices - torch.tensor([row, col])[:, None].to(
                required_indices
            )
            block_flat_indices = local_indices[0] * block_cols + local_indices[1]
            self._block_take_indices.append(block_flat_indices)
            new_out_indices.append(required_indices)
        output_order = torch.cat(new_out_indices, dim=1)

        # We want the sparse matmul to have the same output
        # order as the output pattern.
        self._output_perm = _output_permutation(
            self._out_size, all_indices, output_order
        )
        self._output_indices = all_indices

    def mm(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        assert m1.shape[0] == self._out_size
        assert m2.shape[1] == self._out_size
        assert m1.shape[1] == m2.shape[0]

        outs = []
        for (row, col), take_indices in zip(
            self._iterate_blocks(), self._block_take_indices
        ):
            sub_rows = m1[row : row + self._block_size]
            sub_cols = m2[:, col : col + self._block_size]
            outs.append(IndexedMatmul.apply(sub_rows, sub_cols, take_indices))

        return torch.sparse_coo_tensor(
            indices=self._output_indices,
            values=torch.cat(outs, dim=0)[self._output_perm],
            size=(self._out_size,) * 2,
        ).coalesce()

    def _iterate_blocks(self) -> Tuple[int, int]:
        for row in range(0, self._out_size, self._block_size):
            for col in range(0, self._out_size, self._block_size):
                yield row, col


class IndexedMatmul(torch.autograd.Function):
    """
    Checkpoint a matmul + sparse indexing operation.

    Checkpointing this allows us to save a lot of memory, since we don't need
    to cache the whole output matrix in memory.
    """

    @staticmethod
    def forward(ctx, m1, m2, indices):
        ctx.save_for_backward(m1, m2, indices)
        with torch.no_grad():
            return (m1 @ m2).reshape(-1)[indices]

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2, indices = ctx.saved_tensors
        with torch.enable_grad():
            m1_req = m1.detach().requires_grad_(True)
            m2_req = m2.detach().requires_grad_(True)
            out = (m1_req @ m2_req).reshape(-1)[indices]
        m1_grad, m2_grad = torch.autograd.grad(out, (m1_req, m2_req), grad_output)
        return m1_grad, m2_grad, None


def _output_permutation(
    size: int, dst_indices: torch.Tensor, src_indices: torch.Tensor
):
    raw_dst = dst_indices[0] * size + dst_indices[1]
    raw_src = src_indices[0] * size + src_indices[1]

    dst_perm = torch.argsort(raw_dst)
    src_perm = torch.argsort(raw_src)
    out_perm = torch.zeros_like(dst_perm)
    out_perm[dst_perm] = src_perm

    return out_perm

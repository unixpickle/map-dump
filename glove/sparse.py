from dataclasses import dataclass
from typing import Tuple, Union

import torch

from .loss import SquaredError


@dataclass
class SparseMatrix:
    """
    Similar to PyTorch's sparse COO Tensors, but with stronger assumptions and
    seemingly more stable CUDA support.
    """

    shape: Tuple[int, int]
    indices: torch.Tensor  # [2 x N]
    values: torch.Tensor  # [N]

    def sum(self) -> torch.Tensor:
        return self.values.sum()

    def to_dense(self) -> torch.Tensor:
        res = torch.zeros(
            self.shape, dtype=self.values.dtype, device=self.values.device
        )
        res[self.indices[0].long(), self.indices[1].long()] = self.values
        return res

    @classmethod
    def from_dense(cls, obj: torch.Tensor) -> "SparseMatrix":
        sparse_tensor = obj.to_sparse().coalesce()
        return SparseMatrix(
            shape=sparse_tensor.shape,
            indices=sparse_tensor.indices().clone(),
            values=sparse_tensor.values().clone(),
        )

    def compact_index_dtype(self) -> "SparseMatrix":
        if self.indices.shape[1] < (1 << 31):
            return SparseMatrix(
                shape=self.shape,
                indices=self.indices.to(torch.int32),
                values=self.values,
            )
        else:
            return self

    def add_bias_vecs(
        self, row_bias: torch.Tensor, col_bias: torch.Tensor
    ) -> "SparseMatrix":
        return SparseMatrix(
            shape=self.shape,
            indices=self.indices,
            values=AddBiasVecs.apply(self.values, self.indices, row_bias, col_bias),
        )

    def squared_error(
        self, targets: "SparseMatrix", weights: "SparseMatrix"
    ) -> torch.Tensor:
        _check_compatible(self, targets)
        _check_compatible(self, weights)
        return SquaredError.apply(self.values, targets.values, weights.values)

    def __mul__(self, other: Union[float, "SparseMatrix"]) -> "SparseMatrix":
        return self._run_op(other, lambda x, y: x * y)

    def __rmul__(self, other: Union[float, "SparseMatrix"]) -> "SparseMatrix":
        return self.__mul__(other)

    def __add__(self, other: "SparseMatrix") -> "SparseMatrix":
        assert isinstance(other, SparseMatrix)
        return self._run_op(other, lambda x, y: x + y)

    def __sub__(self, other: "SparseMatrix") -> "SparseMatrix":
        assert isinstance(other, SparseMatrix)
        return self._run_op(other, lambda x, y: x - y)

    def __pow__(self, other: Union[float, "SparseMatrix"]) -> "SparseMatrix":
        return self._run_op(other, lambda x, y: x**y)

    def _run_op(self, other, op_fn):
        if isinstance(other, SparseMatrix):
            _check_compatible(self, other)
            return SparseMatrix(
                shape=self.shape,
                indices=self.indices,
                values=op_fn(self.values, other.values),
            )
        else:
            return SparseMatrix(
                shape=self.shape,
                indices=self.indices,
                values=op_fn(self.values, other),
            )


class AddBiasVecs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        base: torch.Tensor,
        indices: torch.Tensor,
        row_bias: torch.Tensor,
        col_bias: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(indices)
        ctx.bias_shapes = (len(row_bias), len(col_bias))
        with torch.no_grad():
            out = base.clone()
            out.add_(row_bias[indices[0].long()])
            out.add_(col_bias[indices[1].long()])
            return out

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        num_rows, num_cols = ctx.bias_shapes
        row_grad = torch.zeros(
            num_rows, dtype=grad_output.dtype, device=grad_output.device
        )
        row_grad.scatter_add_(0, indices[0].long(), grad_output)
        col_grad = torch.zeros(
            num_cols, dtype=grad_output.dtype, device=grad_output.device
        )
        col_grad.scatter_add_(0, indices[1].long(), grad_output)
        return grad_output, None, row_grad, col_grad


def _check_compatible(m1: SparseMatrix, m2: SparseMatrix):
    assert m1.shape == m2.shape
    assert m1.indices.shape == m2.indices.shape
    assert m1.indices is m2.indices or (m1.indices == m2.indices).all().item()


class SparseMatmul:
    """
    Multiply two dense matrices into a sparse output matrix.
    """

    def __init__(self, output_pattern: SparseMatrix, block_size: int = 2048):
        assert output_pattern.shape[0] == output_pattern.shape[1]
        self._block_size = block_size
        self._out_size = output_pattern.shape[0]

        all_indices = output_pattern.indices

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

    def mm(self, m1: torch.Tensor, m2: torch.Tensor) -> SparseMatrix:
        assert m1.shape[0] == self._out_size
        assert m2.shape[1] == self._out_size
        assert m1.shape[1] == m2.shape[0]

        class SparseMatmulFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
                ctx.save_for_backward(m1, m2)
                with torch.no_grad():
                    out_indices = self._output_perm
                    out = torch.zeros(
                        len(out_indices), device=m1.device, dtype=m1.dtype
                    )
                    for (row, col), take_indices in zip(
                        self._iterate_blocks(), self._block_take_indices
                    ):
                        sub_rows = m1[row : row + self._block_size]
                        sub_cols = m2[:, col : col + self._block_size]
                        product = (sub_rows @ sub_cols).view(-1)[take_indices.long()]
                        n = len(product)
                        out[out_indices[:n].long()] = product
                        out_indices = out_indices[n:]
                    return out

            @staticmethod
            def backward(ctx, grad_output):
                m1, m2 = ctx.saved_tensors
                m1_grad = torch.zeros_like(m1)
                m2_grad = torch.zeros_like(m2)
                in_indices = self._output_perm
                for (row, col), take_indices in zip(
                    self._iterate_blocks(), self._block_take_indices
                ):
                    sub_rows = m1[row : row + self._block_size]
                    sub_cols = m2[:, col : col + self._block_size]
                    n = len(take_indices)
                    sub_grad = grad_output[in_indices[:n].long()]
                    in_indices = in_indices[n:]

                    dense_sub_grad = torch.zeros(
                        sub_rows.shape[0],
                        sub_cols.shape[1],
                        dtype=sub_rows.dtype,
                        device=sub_rows.device,
                    )
                    dense_sub_grad.view(-1)[take_indices.long()] = sub_grad

                    sub_rows_grad = dense_sub_grad @ sub_cols.t()
                    sub_cols_grad = sub_rows.t() @ dense_sub_grad
                    m1_grad[row : row + self._block_size] += sub_rows_grad
                    m2_grad[:, col : col + self._block_size] += sub_cols_grad
                return m1_grad, m2_grad

        return SparseMatrix(
            shape=(self._out_size,) * 2,
            indices=self._output_indices,
            values=SparseMatmulFunc.apply(m1, m2),
        )

    def _iterate_blocks(self) -> Tuple[int, int]:
        for row in range(0, self._out_size, self._block_size):
            for col in range(0, self._out_size, self._block_size):
                yield row, col


def _output_permutation(
    size: int, dst_indices: torch.Tensor, src_indices: torch.Tensor
):
    # It might seem like there's a lot of memory gymnastics here.
    # This is needed to save memory for dense, large matrices.
    raw_dst = (dst_indices[0].long() * size + dst_indices[1].long()).cpu()
    dst_perm = torch.argsort(raw_dst)
    del raw_dst
    dst_perm = dst_perm.to(dst_indices)

    raw_src = (src_indices[0].long() * size + src_indices[1].long()).cpu()
    src_perm = torch.argsort(raw_src)
    del raw_src
    src_perm = src_perm.to(src_indices)

    out_perm = torch.zeros_like(dst_perm)
    out_perm[src_perm.long()] = dst_perm

    return out_perm

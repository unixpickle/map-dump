import torch
import torch.nn as nn

from .sparse import SparseMatmul, SparseMatrix


def test_sparse_ops():
    mat = random_sparsify(torch.randn(125, 74))
    dense = mat.to_dense()
    expected = ((dense * 2.0) - dense) ** 2
    actual = (((mat * 2.0) - mat) ** 2).to_dense()
    assert (actual - expected).abs().max().item() < 1e-5


def test_sparse_squared_error():
    size = 125
    block_size = 17

    mat1 = nn.Parameter(torch.randn(size, 8).double())
    mat2 = nn.Parameter(torch.randn(8, size).double())
    row_bias = nn.Parameter(torch.randn(size).double())
    col_bias = nn.Parameter(torch.randn(size).double())
    expected_out = random_sparsify(mat1 @ mat2 + row_bias[:, None] + col_bias)

    # Create some arbitrary targets and loss weights.
    targets = (expected_out.detach() * (1.0 / 3.0)) ** 3.0
    weights = expected_out.detach() ** 2

    expected_loss = (weights * (expected_out - targets) ** 2).values.sum()
    (
        expected_m1_grad,
        expected_m2_grad,
        expected_row_bias_grad,
        expected_col_bias_grad,
    ) = torch.autograd.grad(expected_loss, (mat1, mat2, row_bias, col_bias))

    actual_loss = SparseMatmul(
        expected_out, block_size=block_size
    ).weighted_squared_error(mat1, mat2, row_bias, col_bias, targets, weights)
    (
        actual_m1_grad,
        actual_m2_grad,
        actual_row_bias_grad,
        actual_col_bias_grad,
    ) = torch.autograd.grad(actual_loss, (mat1, mat2, row_bias, col_bias))

    assert (expected_loss - actual_loss).abs().max().item() < 1e-5
    assert (expected_m1_grad - actual_m1_grad).abs().max().item() < 1e-5
    assert (expected_m2_grad - actual_m2_grad).abs().max().item() < 1e-5
    assert (expected_row_bias_grad - actual_row_bias_grad).abs().max().item() < 1e-5
    assert (expected_col_bias_grad - actual_col_bias_grad).abs().max().item() < 1e-5


def random_sparsify(mat: torch.Tensor) -> SparseMatrix:
    all_indices = torch.tensor(
        [[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1])]
    ).T.contiguous()
    out_indices = all_indices[:, torch.rand(all_indices.shape[1]) < 0.25]
    return SparseMatrix(
        shape=mat.shape,
        indices=out_indices,
        values=mat.reshape(-1)[out_indices[0] * mat.shape[1] + out_indices[1]],
    ).compact_index_dtype()

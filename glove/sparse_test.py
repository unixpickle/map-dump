import torch

from .sparse import SparseMatmul, SparseMatrix


def test_sparse_ops():
    mat = random_sparsify(torch.randn(125, 74))
    dense = mat.to_dense()
    expected = ((dense * 2.0) - dense) ** 2
    actual = (((mat * 2.0) - mat) ** 2).to_dense()
    assert (actual - expected).abs().max().item() < 1e-5


def test_sparse_matmul():
    size = 125
    block_size = 17

    mat1 = torch.randn(size, 8)
    mat2 = torch.randn(8, size)
    expected_out = random_sparsify(mat1 @ mat2)
    actual_out = SparseMatmul(expected_out, block_size=block_size).mm(mat1, mat2)

    assert (expected_out.to_dense() - actual_out.to_dense()).abs().max().item() < 1e-5


def random_sparsify(mat: torch.Tensor) -> SparseMatrix:
    all_indices = torch.tensor(
        [[i, j] for i in range(mat.shape[0]) for j in range(mat.shape[1])]
    ).T.contiguous()
    out_indices = all_indices[:, torch.rand(all_indices.shape[1]) < 0.25]
    return SparseMatrix(
        shape=mat.shape,
        indices=out_indices,
        values=mat.reshape(-1)[out_indices[0] * mat.shape[1] + out_indices[1]],
    )

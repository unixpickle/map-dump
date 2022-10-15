import torch

from .sparse_matmul import SparseMatmul


def test_sparse_matmul():
    size = 125
    block_size = 17

    out_indices = torch.tensor([[i, j] for i in range(size) for j in range(size)]).T.contiguous()
    out_indices = out_indices[:, torch.rand(out_indices.shape[1]) < 0.25]

    mat1 = torch.randn(size, 8)
    mat2 = torch.randn(8, size)
    expected_out = torch.sparse_coo_tensor(
        indices=out_indices,
        values=(mat1 @ mat2).view(-1)[out_indices[0] * size + out_indices[1]],
        size=(size, size),
    ).coalesce()

    actual_out = SparseMatmul(expected_out, block_size=block_size).mm(mat1, mat2)

    assert (expected_out.to_dense() - actual_out.to_dense()).abs().max().item() < 1e-5

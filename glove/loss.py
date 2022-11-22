import torch


class SquaredError(torch.autograd.Function):
    """
    Compute the weighted total squared error sum(w*(x-y)**2).

    Gradients are only computed for x, not y or w.
    """

    @staticmethod
    def forward(ctx, x, y, w):
        ctx.save_for_backward(x, y, w)
        with torch.no_grad():
            x = x.reshape(-1)
            y = y.reshape(-1)
            w = w.reshape(-1)

            # Do the large accumulation in the highest possible precision.
            total_loss = torch.zeros((), dtype=torch.float64, device=x.device)

            # Chunk intermediate values to avoid extra allocation of size x.
            chunk_size = 2**20
            for i in range(0, len(x), chunk_size):
                diff = (
                    w[i : i + chunk_size].to(x)
                    * (x[i : i + chunk_size] - y[i : i + chunk_size].to(x)) ** 2
                )
                total_loss += diff.to(total_loss).sum()
            return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        x, y, w = ctx.saved_tensors
        res = x - y
        res.mul_(w)
        res.mul_(grad_output)
        res.mul_(2.0)
        return res, None, None

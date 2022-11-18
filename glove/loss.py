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
            diff = x - y
            diff.pow_(2.0)
            diff.mul_(w)
            return diff.sum()

    @staticmethod
    def backward(ctx, grad_output):
        x, y, w = ctx.saved_tensors
        res = x - y
        res.mul_(w)
        res.mul_(2.0 * grad_output)
        return res, None, None

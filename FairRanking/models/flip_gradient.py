import torch
from torch.autograd import Function

class FlipGradientFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

def flipGradient(x, lambda_=1.0):
    return FlipGradientFunction.apply(x, lambda_)
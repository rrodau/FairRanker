from torch import nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


def gradient_reversal_layer(x):
    return GradientReversalLayer.apply(x)
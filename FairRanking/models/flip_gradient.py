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

import torch

class gradient_reversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg_()


class CustomLayer(torch.nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()
        # Initialize your layer parameters if any

    def forward(self, input):
        # Apply the custom autograd function
        return gradient_reversal.apply(input)
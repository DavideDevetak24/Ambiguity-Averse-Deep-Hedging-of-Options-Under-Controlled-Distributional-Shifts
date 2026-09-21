"""
Scaled sigmoid activation function for the output neuron of the hedging network.

"""

# Import libraries
import torch
import torch.nn as nn

# Sigmoid function scaled and centered
class CustomSharpSigmoid(nn.Module):
    def __init__(self, center=0.5, scale=10.0):
        super().__init__()
        self.center = center
        self.scale = scale

    def forward(self, x):
        return torch.sigmoid((x - self.center) * self.scale)
    

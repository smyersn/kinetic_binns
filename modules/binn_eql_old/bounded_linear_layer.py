import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundedLinear(nn.Module):
    def __init__(self, in_features, out_features, param_bounds):
        super(BoundedLinear, self).__init__()
        self.param_min = -param_bounds
        self.param_max = param_bounds
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-4, 4))
        self.register_parameter('bias', None)

    def forward(self, x):
        # Sigmoid-transformed weights to lie within [weight_min, weight_max]
        weight = torch.sigmoid(self.raw_weight) * (self.param_max - self.param_min) + self.param_min
        return F.linear(x, weight, self.bias)

    @property
    def weight(self):
        # This transforms raw_weight to [weight_min, weight_max]
        return torch.sigmoid(self.raw_weight) * (self.param_max - self.param_min) + self.param_min

"""Surface fitter: a smooth neural interpolant u(x, t) of the data."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils

from modules.binn_eql.model.build_mlp import build_mlp


class FourierFeatureEncoding(nn.Module):
    """Fourier features [sin(2*pi*xB), cos(2*pi*xB)] (Tancik et al., 2020), with trainable B."""

    def __init__(self, in_features, mapping_size, scale=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale)

    def forward(self, x):
        projection = (2.0 * np.pi * x) @ self.B
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class SurfaceFitter(nn.Module):
    """
    Fourier-feature MLP mapping normalized (x, t) to concentrations.
    Softplus output keeps concentrations positive; weight normalization on
    every linear layer.
    """

    def __init__(self, input_features, species, mapping_size=256, scale=1.0, layers=None):
        super().__init__()
        layers = layers or [256, 256, 256, species]
        assert layers[-1] == species, "The final layer width must equal `species`."

        self.encoder = FourierFeatureEncoding(input_features, mapping_size, scale)
        self.mlp = build_mlp(
            input_features=2 * mapping_size,
            layers=layers,
            activation=nn.GELU(),
            linear_output=False,
            output_activation=nn.Softplus())

        for module in self.mlp.MLP:
            if isinstance(module, nn.Linear):
                utils.parametrizations.weight_norm(module)

    def forward(self, inputs):
        return self.mlp(self.encoder(inputs))
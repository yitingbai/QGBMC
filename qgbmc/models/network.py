import torch
from torch import nn

def mlp(dims):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class QGBMC(nn.Module):
    def __init__(self, input_dims, hidden=(256, 128), latent=64):
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError('at least two aligned views required')
        self.encoders = nn.ModuleList([mlp([d, *hidden, latent]) for d in input_dims])
        self.decoders = nn.ModuleList([mlp([latent, *reversed(hidden), d]) for d in input_dims])
        self.fusion = nn.Parameter(torch.eye(latent).repeat(len(input_dims), 1, 1) / len(input_dims))

    def forward(self, views):
        z = [net(x) for net, x in zip(self.encoders, views)]
        fused = sum(v @ self.fusion[i] for i, v in enumerate(z))
        # Final objective Eq.35 takes precedence over preliminary Eq.5.
        recon = [net(fused) for net in self.decoders]
        return z, fused, recon

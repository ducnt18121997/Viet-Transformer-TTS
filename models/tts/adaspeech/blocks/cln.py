import torch
import torch.nn as nn


class CondionalLayerNorm(nn.Module):

    def __init__(self, indims: int, outdims: int, epsilon=1e-5):
        super(CondionalLayerNorm, self).__init__()
        self.indims  = indims
        self.outdims = outdims

        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.indims, self.outdims)
        self.W_bias  = nn.Linear(self.indims, self.outdims)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.W_scale.weight, 0.0)
        nn.init.constant_(self.W_scale.bias, 1.0)
        nn.init.constant_(self.W_bias.weight, 0.0)
        nn.init.constant_(self.W_bias.bias, 0.0)
    
    def forward(self, x: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std 

        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y = (y * scale.unsqueeze(1)) + bias.unsqueeze(1)

        return y

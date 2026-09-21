"""Unintegrated H1 building blocks; no changes to Drama's training defaults.

Only allowed linear blocks own parameters. Dense materialization is for a
correctness prototype, not a claim of sparse-kernel speed or reduced FLOPs.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


ROLES = frozenset(('shared', 'latent', 'reward'))
READS = {'shared': {'shared'}, 'latent': {'shared', 'latent'},
         'reward': {'shared', 'reward'}}


def validate_sections(sections):
    sections = tuple(sections)
    if not sections or any(role not in ROLES or not isinstance(size, int)
                           or isinstance(size, bool) or size <= 0
                           for role, size in sections):
        raise ValueError('Expected nonempty positive-width named sections')
    return sections


class RoutedLinear(nn.Module):
    """Block-triangular projection with no trainable forbidden connections.

    Output roles may repeat to describe packed Mamba projection segments.
    Exposing weight allows later integration with fused-kernel call sites,
    but fused execution and CUDA Graph compatibility are not tested here.
    """
    def __init__(self, inputs, outputs, bias=False, *, device=None, dtype=None):
        super().__init__()
        self.inputs = validate_sections(inputs)
        self.outputs = validate_sections(outputs)
        self.in_features = sum(n for _, n in self.inputs)
        self.out_features = sum(n for _, n in self.outputs)
        self.blocks = nn.ParameterDict()
        for r, (target, rows) in enumerate(self.outputs):
            fan_in = sum(n for source, n in self.inputs if source in READS[target])
            if fan_in == 0:
                raise ValueError('Every output requires an allowed input')
            for c, (source, cols) in enumerate(self.inputs):
                if source in READS[target]:
                    p = nn.Parameter(torch.empty(rows, cols, device=device, dtype=dtype))
                    nn.init.uniform_(p, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
                    self.blocks[f'{r}_{c}'] = p
        self.bias = nn.Parameter(torch.zeros(self.out_features, device=device, dtype=dtype)) if bias else None

    @property
    def weight(self):
        reference = next(iter(self.blocks.values()))
        rows = []
        for r, (_, nr) in enumerate(self.outputs):
            blocks = []
            for c, (_, nc) in enumerate(self.inputs):
                key = f'{r}_{c}'
                blocks.append(self.blocks[key] if key in self.blocks
                              else reference.new_zeros((nr, nc)))
            rows.append(torch.cat(blocks, dim=1))
        return torch.cat(rows, dim=0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class PartitionRMSNorm(nn.Module):
    """Independent normalization per section; no cross-section denominator."""
    def __init__(self, widths, eps=1e-5, *, device=None, dtype=None):
        super().__init__()
        self.widths = tuple(widths)
        if not self.widths or any(not isinstance(n, int) or isinstance(n, bool)
                                  or n <= 0 for n in self.widths) or eps <= 0:
            raise ValueError('Positive widths and epsilon required')
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(sum(self.widths), device=device, dtype=dtype))

    def forward(self, x):
        if x.shape[-1] != sum(self.widths):
            raise ValueError('Feature width does not match partitions')
        # Preserve fp64 for numerical tests, accumulate half precision in fp32.
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        ys = [part * torch.rsqrt(part.square().mean(-1, keepdim=True) + self.eps)
              for part in work.split(self.widths, dim=-1)]
        return torch.cat(ys, dim=-1).to(x.dtype) * self.weight.to(x.dtype)

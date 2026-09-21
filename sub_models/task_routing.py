"""Opt-in H1 routing. Baseline construction never enters these helpers.

Allowed blocks own parameters; dense materialization preserves compatibility
with Mamba's weight-based code paths but is not sparse compute acceleration.
"""
import torch
from torch import nn
from torch.nn import functional as F

READS = {'shared': {'shared'}, 'latent': {'shared', 'latent'},
         'reward': {'shared', 'reward'}}


def sections(width):
    if width <= 0 or width % 4:
        raise ValueError('H1 requires a positive width divisible by four')
    return (('shared', width // 2), ('latent', width // 4), ('reward', width // 4))


class RoutedLinear(nn.Module):
    """Retain allowed weights from an existing Linear without drawing RNG."""
    def __init__(self, original, inputs, outputs):
        super().__init__()
        self.inputs, self.outputs = tuple(inputs), tuple(outputs)
        for role, size in self.inputs + self.outputs:
            if role not in READS or not isinstance(size, int) or size <= 0:
                raise ValueError('Invalid routed section')
        self.in_features = sum(n for _, n in self.inputs)
        self.out_features = sum(n for _, n in self.outputs)
        if (self.out_features, self.in_features) != original.weight.shape:
            raise ValueError('Routed layout differs from source projection')
        self.blocks = nn.ParameterDict()
        r0 = 0
        for r, (target, nr) in enumerate(self.outputs):
            c0 = 0
            for c, (source, nc) in enumerate(self.inputs):
                if source in READS[target]:
                    self.blocks[f'{r}_{c}'] = nn.Parameter(original.weight[r0:r0+nr, c0:c0+nc].detach().clone())
                c0 += nc
            if not any(f'{r}_{c}' in self.blocks for c in range(len(self.inputs))):
                raise ValueError('Output has no allowed inputs')
            r0 += nr
        self.bias = nn.Parameter(original.bias.detach().clone()) if original.bias is not None else None

    @property
    def weight(self):
        ref = next(iter(self.blocks.values()))
        return torch.cat([torch.cat([
            self.blocks[f'{r}_{c}'] if f'{r}_{c}' in self.blocks else ref.new_zeros((nr, nc))
            for c, (_, nc) in enumerate(self.inputs)], dim=1)
            for r, (_, nr) in enumerate(self.outputs)], dim=0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class PartitionRMSNorm(nn.Module):
    """Section-local RMS denominator, optionally with Mamba's SiLU gate."""
    def __init__(self, original, widths, gated=False):
        super().__init__()
        self.widths = tuple(widths)
        if not self.widths or any(n <= 0 for n in self.widths) or sum(self.widths) != original.weight.numel():
            raise ValueError('Invalid norm partition')
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.register_parameter('bias', None)
        self.eps = original.eps
        self.gated = gated
        self.norm_before_gate = getattr(original, 'norm_before_gate', False)

    def forward(self, x, z=None):
        if self.gated != (z is not None):
            raise ValueError('Gate argument does not match norm contract')
        dtype = x.dtype
        work = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        gate = F.silu(z.to(work.dtype)) if z is not None else None
        if gate is not None and not self.norm_before_gate:
            work = work * gate
        result = torch.cat([p * torch.rsqrt(p.square().mean(-1, keepdim=True) + self.eps)
                            for p in work.split(self.widths, -1)], -1)
        result = result * self.weight.to(result.dtype)
        if gate is not None and self.norm_before_gate:
            result = result * gate
        return result.to(dtype)


class RestrictedReadout(nn.Module):
    """Select shared plus one private section, storing only active weights."""
    def __init__(self, original, role):
        super().__init__()
        layout = sections(original.in_features)
        if role not in ('latent', 'reward'):
            raise ValueError('Readout requires a private task role')
        selected = []; offset = 0
        for source, n in layout:
            if source in READS[role]:selected.extend(range(offset, offset+n))
            offset += n
        self.register_buffer('indices', torch.tensor(selected, device=original.weight.device, dtype=torch.long), persistent=False)
        self.weight = nn.Parameter(original.weight[:, selected].detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone()) if original.bias is not None else None
        self.in_features, self.out_features = original.in_features, original.out_features

    def forward(self, x):
        return F.linear(x.index_select(-1, self.indices), self.weight, self.bias)


def install_task_routing(world_model, mode):
    """Called once during construction, before optimizer/cache creation."""
    if mode not in ('readout', 'routed'):
        raise ValueError('Unknown task routing mode: ' + str(mode))
    if world_model.model != 'Mamba2':
        raise ValueError('Task routing currently supports Mamba2 only')
    backbone = world_model.sequence_model.backbone
    layout = sections(world_model.hidden_state_dim)
    if mode == 'routed':
        for block in backbone.layers:
            m = block.mixer
            if (block.mlp is not None or m.ngroups != 1 or m.d_ssm != m.d_inner
                    or m.world_size != 1 or not m.rmsnorm or m.nheads % 4):
                raise ValueError('Unsupported Mamba configuration for H1-R1')
            inner = sections(m.d_inner)
            packed = inner + inner + (('shared', m.d_state), ('shared', m.d_state)) + sections(m.nheads)
            m.in_proj = RoutedLinear(m.in_proj, layout, packed)
            m.out_proj = RoutedLinear(m.out_proj, inner, layout)
            m.norm = PartitionRMSNorm(m.norm, [n for _, n in inner], gated=True)
            # The combined kernel hardcodes its norm grouping. The ordinary
            # scan and selective update kernels still run on GPU unchanged.
            m.use_mem_eff_path = False
            block.norm = PartitionRMSNorm(block.norm, [n for _, n in layout])
            block.fused_add_norm = False
            block.dropout = nn.Dropout(block.dropout_p)
        backbone.norm_f = PartitionRMSNorm(backbone.norm_f, [n for _, n in layout])
        backbone.fused_add_norm = False
        backbone.dropout = nn.Dropout(backbone.dropout_p)
    # Replacing the modules, rather than individual call sites, applies these
    # readouts equally in updates, imagination and real-decision prior calls.
    world_model.dist_head.prior_head = RestrictedReadout(world_model.dist_head.prior_head, 'latent')
    world_model.reward_decoder.backbone[0] = RestrictedReadout(world_model.reward_decoder.backbone[0], 'reward')
    world_model.termination_decoder.backbone[0] = RestrictedReadout(world_model.termination_decoder.backbone[0], 'reward')
    return {'mode': mode, 'widths': [n for _, n in layout], 'bc_groups': 1,
            'private_bc_dependency': False if mode == 'routed' else None,
            'dense_weight_materialization': mode == 'routed',
            'combined_mamba_kernel': mode != 'routed', 'claim': 'experimental; not performance validated'}

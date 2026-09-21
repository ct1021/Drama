"""B optimization control: rectified loss weighting adapted from HarmonyDream.

Reference: thuml/HarmonyDream, wmlib-torch/wmlib/agents/dreamerv2.py.
Three zero-initialized scalars; inference architecture and loss targets unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RectifiedLossHarmonizer(nn.Module):
    names = ('image', 'reward', 'kl')

    def __init__(self, device=None):
        super().__init__()
        # No random draws: initialization of the model and agent is preserved.
        self.log_scales = nn.Parameter(torch.zeros(3, device=device, dtype=torch.float32))

    def forward(self, reconstruction, reward, dynamics, representation, termination):
        losses = torch.stack((reconstruction.float(), reward.float(),
                              dynamics.float() + 0.1 * representation.float()))
        coefficients = torch.exp(-self.log_scales)
        regularizers = F.softplus(self.log_scales)
        total = (coefficients * losses + regularizers).sum() + termination.float()
        metrics = {}
        for index, name in enumerate(self.names):
            for key, value in (('log_scale', self.log_scales[index]),
                               ('coefficient', coefficients[index]),
                               ('regularizer', regularizers[index]),
                               ('weighted_loss', coefficients[index] * losses[index])):
                metrics[f'WorldModel/Harmony/{name}_{key}'] = value.detach().clone()
        metrics['WorldModel/Harmony/regularizer_total'] = regularizers.sum().detach()
        return total, metrics

    def optimizer_groups(self, model):
        scalar_ids = {id(p) for p in self.parameters()}
        return [{'params': [p for p in model.parameters() if id(p) not in scalar_ids]},
                {'params': list(self.parameters()), 'weight_decay': 0.0}]

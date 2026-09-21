"""CPU tests for the opt-in production routing components."""
import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sub_models.task_routing import RoutedLinear, RestrictedReadout, PartitionRMSNorm, sections


class RoutingTests(unittest.TestCase):
    def test_conversion_preserves_allowed_weights_and_rng(self):
        dense = nn.Linear(8, 8, dtype=torch.float64)
        before = torch.random.get_rng_state().clone()
        layer = RoutedLinear(dense, sections(8), sections(8))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 48)
        torch.testing.assert_close(layer.weight[:4, :4], dense.weight[:4, :4])
        self.assertEqual(layer.weight[:4, 4:].abs().sum().item(), 0)
        x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        grad = torch.autograd.grad(layer(x)[:, 4:6].sum(), x)[0]
        self.assertEqual(grad[:, 6:].abs().sum().item(), 0)
        self.assertGreater(grad[:, :6].abs().sum().item(), 0)

    def test_packed_bc_cannot_read_private_sections(self):
        # z, x, B, C, dt: role ordering matches Mamba's packed projection.
        packed = sections(16) * 2 + (('shared', 2), ('shared', 2)) + sections(4)
        layer = RoutedLinear(nn.Linear(8, 40), sections(8), packed)
        x = torch.randn(2, 8);changed=x.clone();changed[:, 4:] += 9
        torch.testing.assert_close(layer(x)[:, 32:36], layer(changed)[:, 32:36], atol=0, rtol=0)

    def test_readouts_have_identical_contract_for_all_leading_shapes(self):
        original = nn.Linear(8, 5)
        for role, hidden in [('latent', slice(6, 8)), ('reward', slice(4, 6))]:
            read = RestrictedReadout(original, role)
            for shape in [(2, 8), (2, 7, 8), (2, 1, 8)]:
                x = torch.randn(*shape); changed=x.clone();changed[..., hidden] += 20
                torch.testing.assert_close(read(x), read(changed), atol=0, rtol=0)
            self.assertEqual(read.weight.numel(), 30)
            copy=RestrictedReadout(original,role);copy.load_state_dict(read.state_dict())
            torch.testing.assert_close(copy(x),read(x))

    def test_gated_norm_preserves_partition_in_both_gate_orders(self):
        for before in (False, True):
            original=nn.LayerNorm(8,eps=1e-5);original.norm_before_gate=before
            norm=PartitionRMSNorm(original,(4,2,2),gated=True)
            x=torch.randn(2,8);z=torch.randn_like(x)
            xp=x.clone();zp=z.clone();xp[:,6:]+=10;zp[:,6:]+=8
            torch.testing.assert_close(norm(x,z)[:,:6],norm(xp,zp)[:,:6],atol=0,rtol=0)
            with self.assertRaises(ValueError):norm(x)

    def test_multilayer_residual_has_no_private_to_shared_leak(self):
        layers=[RoutedLinear(nn.Linear(8,8),sections(8),sections(8)) for _ in range(3)]
        norms=[PartitionRMSNorm(nn.LayerNorm(8),(4,2,2)) for _ in layers]
        def f(x):
            for layer,norm in zip(layers,norms):x=norm(x+torch.tanh(layer(x)))
            return x
        x=torch.randn(2,8);changed=x.clone();changed[:,6:]+=2
        torch.testing.assert_close(f(x)[:,:6],f(changed)[:,:6],atol=0,rtol=0)

    def test_layout_fails_early(self):
        with self.assertRaises(ValueError):sections(7)
        with self.assertRaises(ValueError):RoutedLinear(nn.Linear(8,8),sections(4),sections(8))
        with self.assertRaises(ValueError):RestrictedReadout(nn.Linear(8,8),'invalid')


if __name__ == '__main__':unittest.main(verbosity=2)

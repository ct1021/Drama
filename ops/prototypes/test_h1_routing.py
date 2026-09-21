"""CPU dependency tests only, not RL evidence or full Mamba integration."""
import unittest

import torch

from h1_routing import PartitionRMSNorm, RoutedLinear


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3710)
        self.sections = (('shared', 4), ('latent', 2), ('reward', 2))

    def test_private_perturbation_stays_private_through_layers_and_residuals(self):
        layers = [RoutedLinear(self.sections, self.sections, dtype=torch.float64)
                  for _ in range(3)]
        norms = [PartitionRMSNorm((4, 2, 2), dtype=torch.float64) for _ in layers]
        def run(x):
            for layer, norm in zip(layers, norms):
                x = norm(x + torch.tanh(layer(x)))
            return x
        x = torch.randn(2, 5, 8, dtype=torch.float64)
        y = run(x)
        for private, visible in [(slice(6, 8), slice(0, 6)),
                                 (slice(4, 6), [0, 1, 2, 3, 6, 7])]:
            changed = x.clone(); changed[..., private] += 3
            torch.testing.assert_close(run(changed)[..., visible], y[..., visible], rtol=0, atol=0)
            self.assertGreater((run(changed)[..., private] - y[..., private]).abs().max().item(), 0)

    def test_forbidden_gradient_is_zero_and_allowed_gradient_exists(self):
        layer = RoutedLinear(self.sections, self.sections, dtype=torch.float64)
        x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
        grad = torch.autograd.grad(layer(x)[..., 4:6].sum(), x)[0]
        self.assertEqual(grad[..., 6:].abs().max().item(), 0)
        self.assertGreater(grad[..., :6].abs().max().item(), 0)
        layer(x).square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in layer.parameters()))

    def test_global_norm_would_break_dependency_contract(self):
        x = torch.ones(1, 8, dtype=torch.float64)
        changed = x.clone(); changed[..., 6:] *= 10
        norm = PartitionRMSNorm((4, 2, 2), dtype=torch.float64)
        torch.testing.assert_close(norm(x)[..., :6], norm(changed)[..., :6])
        def global_norm(v):
            return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-5)
        self.assertFalse(torch.allclose(global_norm(x)[..., :6], global_norm(changed)[..., :6]))

    def test_effective_parameters_and_packed_segments(self):
        layer = RoutedLinear(self.sections, self.sections)
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 40)
        self.assertEqual(layer.weight.numel(), 64)
        self.assertEqual(torch.count_nonzero(layer.weight[:4, 4:]).item(), 0)
        packed = RoutedLinear(self.sections, (('latent', 2), ('shared', 3),
                                              ('reward', 2), ('shared', 1)), bias=True)
        copy = RoutedLinear(self.sections, packed.outputs, bias=True)
        copy.load_state_dict(packed.state_dict())
        x = torch.randn(3, 8)
        torch.testing.assert_close(packed(x), copy(x))
        self.assertEqual(packed(x).shape, (3, 8))

    def test_invalid_layout_is_rejected(self):
        with self.assertRaises(ValueError):RoutedLinear((('unknown', 2),), self.sections)
        with self.assertRaises(ValueError):RoutedLinear((('reward', 2),), (('shared', 2),))
        with self.assertRaises(ValueError):PartitionRMSNorm((4, 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)

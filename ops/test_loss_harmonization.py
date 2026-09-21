import io
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sub_models.loss_harmonization import RectifiedLossHarmonizer


class HarmonizationTests(unittest.TestCase):
    def test_initial_model_gradient_matches_baseline(self):
        x = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
        losses = x.square().unbind()
        h = RectifiedLossHarmonizer()
        result, metrics = h(*losses)
        baseline = losses[0] + losses[1] + losses[2] + 0.1 * losses[3] + losses[4]
        expected = torch.autograd.grad(baseline, x, retain_graph=True)[0]
        result.backward()
        torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
        torch.testing.assert_close(h.log_scales.grad, torch.tensor([-.5, -3.5, -10.1]))
        self.assertEqual(len(metrics), 13)
        torch.testing.assert_close(result.detach()-baseline.detach(), 3*torch.log(torch.tensor(2.)))

    def test_scalar_update_and_checkpoint(self):
        h = RectifiedLossHarmonizer();opt=torch.optim.AdamW(h.parameters(),lr=.01,weight_decay=0)
        losses = [torch.tensor(v) for v in (2., .2, 3., 4., .1)]
        loss, before = h(*losses);loss.backward();opt.step()
        self.assertTrue(torch.isfinite(h.log_scales).all())
        self.assertTrue((h.log_scales != 0).all())
        self.assertEqual(before['WorldModel/Harmony/image_log_scale'].item(), 0.)
        buf=io.BytesIO();torch.save(h.state_dict(),buf);buf.seek(0)
        restored=RectifiedLossHarmonizer();restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(restored(*losses)[0],h(*losses)[0],rtol=0,atol=0)

    def test_groups_cover_model_once_without_scalar_decay(self):
        model=nn.Module();model.net=nn.Linear(2,2);model.h=RectifiedLossHarmonizer()
        opt=torch.optim.AdamW(model.h.optimizer_groups(model),lr=.003,weight_decay=.01)
        flat=[p for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(flat),len({id(p) for p in flat}))
        self.assertEqual({id(p) for p in flat},{id(p) for p in model.parameters()})
        self.assertEqual(opt.param_groups[0]['weight_decay'],.01)
        self.assertEqual(opt.param_groups[1]['weight_decay'],0.)
        self.assertTrue(all(group['lr']==.003 for group in opt.param_groups))

    def test_no_rng_draw_or_extra_inference_parameters(self):
        before=torch.random.get_rng_state().clone();h=RectifiedLossHarmonizer()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in h.parameters()),3)

    def test_rectification_stable_for_large_positive_scale(self):
        h=RectifiedLossHarmonizer()
        with torch.no_grad():h.log_scales.fill_(100.)
        value,_=h(*[torch.tensor(1.) for _ in range(5)])
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(h.log_scales.grad).all())


if __name__=='__main__':unittest.main()

"""Temporary tensor timing only: no optimizer step, training run, or checkpoint write."""
import contextlib
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import yaml

ROOT = Path('/root/autodl-tmp/drama-research-20260919/code')
sys.path.insert(0, str(ROOT))
from train import DotDict, build_world_model, build_agent

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def timed(fn, warmups, repeats):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return dict(median_seconds=statistics.median(samples), min_seconds=min(samples), max_seconds=max(samples), samples=samples)


results = []
for label, mults in [('public_21m', [1, 2, 4, 8]), ('lightweight_five_stage', [1, 2, 3, 4, 4])]:
    torch.manual_seed(3710)
    raw = yaml.safe_load((ROOT/'config_files/configure.yaml').read_text())
    raw['Models']['WorldModel']['dtype'] = torch.float32
    raw['Models']['Agent']['dtype'] = torch.float32
    raw['Models']['WorldModel']['Encoder']['Mults'] = mults
    raw['Models']['WorldModel']['Decoder']['Mults'] = mults
    raw['BasicSettings']['Compile'] = False
    config = DotDict(raw)
    with contextlib.redirect_stdout(sys.stderr):
        wm = build_world_model(config, 18, 'cuda:0')
        agent = build_agent(config, 18, 'cuda:0')
    initial_versions = [p._version for p in wm.parameters()] + [p._version for p in agent.parameters()]
    sizes = {name: sum(p.numel() for p in child.parameters()) for name, child in wm.named_children() if sum(p.numel() for p in child.parameters())}
    cache = wm.sequence_model.allocate_inference_cache(1, 24, dtype=torch.float32)
    cache_bytes = sum(t.numel()*t.element_size() for values in cache.values() for t in values)
    del cache
    obs = torch.rand(16, 128, 3, 64, 64, device='cuda')
    actions = torch.randint(18, (16, 128), device='cuda')
    rewards = torch.zeros(16, 128, device='cuda')
    terminals = torch.zeros(16, 128, device='cuda')
    wm.train()

    def prediction_backward():
        wm.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            embedding = wm.encoder(obs)
            post = wm.dist_head.forward_post(embedding)
            sample = wm.stright_throught_gradient(post, sample_mode='random_sample')
            latent = wm.flatten_sample(sample)
            reconstruction = wm.image_decoder(latent)
            feature = wm.sequence_model(latent, actions)
            prior = wm.dist_head.forward_prior(feature)
            reward = wm.reward_decoder(feature)
            termination = wm.termination_decoder(feature)
            dyn, _ = wm.categorical_kl_div_loss(post[:, 1:].detach(), prior[:, :-1])
            rep, _ = wm.categorical_kl_div_loss(post[:, 1:], prior[:, :-1].detach())
            loss = (wm.mse_loss_func(reconstruction, obs) + wm.symlog_twohot_loss_func(reward, rewards)
                    + wm.bce_with_logits_loss_func(termination, terminals) + dyn + 0.1*rep)
        loss.backward()
        assert torch.isfinite(loss).item()

    torch.cuda.reset_peak_memory_stats()
    backward_time = timed(prediction_backward, 3, 10)
    wm.zero_grad(set_to_none=True)
    del obs, actions, rewards, terminals
    wm.eval()
    agent.eval()
    context = torch.rand(1024, 8, 3, 64, 64, device='cuda')
    context_actions = torch.randint(18, (1024, 8), device='cuda')

    @torch.no_grad()
    def imagination():
        values = wm.imagine_data2(agent, context, context_actions, 1024, 16, False, None, 0)
        assert all(torch.isfinite(v).all().item() for v in values if isinstance(v, torch.Tensor))

    with contextlib.redirect_stdout(sys.stderr):
        imagination_time = timed(imagination, 2, 6)
    final_versions = [p._version for p in wm.parameters()] + [p._version for p in agent.parameters()]
    assert initial_versions == final_versions, 'Parameter changed during measurement'
    result = dict(profile=label, encoder_mults=mults, encoder_output_shape=list(wm.encoder.output_dim),
        world_model_parameters=sum(p.numel() for p in wm.parameters()), agent_parameters=sum(p.numel() for p in agent.parameters()),
        parameter_breakdown=sizes, recursive_cache_bytes_per_batch_row_float32=cache_bytes,
        prediction_forward_backward=backward_time, imagination_1024x16=imagination_time,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        optimizer_steps=0, parameter_versions_unchanged=True)
    results.append(result)
    print(json.dumps(result), flush=True)
    del wm, agent, context, context_actions
    gc.collect()
    torch.cuda.empty_cache()
print(json.dumps(dict(scope='synthetic tensor timing; excludes optimizer, actor backward, replay sampling, actual environment and evaluation; not RL performance evidence', results=results)), flush=True)

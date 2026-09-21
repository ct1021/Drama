"""Bounded GPU routing checks and component timing; no optimizer steps."""
import contextlib,copy,gc,json,statistics,sys,time,traceback
from pathlib import Path
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train import DotDict,build_world_model,build_agent
from mamba_ssm import InferenceParams
from experiment_profiles import apply_profile

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True


def timed(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(5):
        start=time.perf_counter();fn();torch.cuda.synchronize();times.append(time.perf_counter()-start)
    return {'median_seconds':statistics.median(times),'samples':times}


def run(profile):
    torch.manual_seed(3710)
    raw=apply_profile(yaml.safe_load((ROOT/'config_files/configure.yaml').read_text()),profile)
    raw['Models']['WorldModel']['dtype']=torch.float32
    raw['Models']['Agent']['dtype']=torch.float32
    conf=DotDict(raw)
    wm=build_world_model(conf,18,'cuda:0').eval();agent=build_agent(conf,18,'cuda:0').eval()
    initial=[p._version for p in wm.parameters()]
    opt_ids={id(p) for group in wm.optimizer.param_groups for p in group['params']}
    assert opt_ids=={id(p) for p in wm.parameters()},'Optimizer misses routed parameters'
    seq=wm.sequence_model
    latent=torch.randn(2,16,wm.stoch_flattened_dim,device='cuda')
    actions=torch.randint(18,(2,16),device='cuda')
    errors=[]
    with torch.no_grad():
        full=seq(latent,actions)
        for prefill in (1,8):
            cache=InferenceParams(max_seqlen=16,max_batch_size=2)
            pieces=[seq(latent[:,:prefill].contiguous(),actions[:,:prefill].contiguous(),inference_params=cache)]
            for t in range(prefill,16):
                cache.seqlen_offset=t
                pieces.append(seq(latent[:,t:t+1].contiguous(),actions[:,t:t+1].contiguous(),inference_params=cache))
            incremental=torch.cat(pieces,1)
            torch.testing.assert_close(full,incremental,rtol=2e-3,atol=2e-3)
            errors.append(float((full-incremental).abs().max()))
            # reset() resets the offset; next prefill must overwrite both caches.
            for values in cache.key_value_memory_dict.values():
                for value in values:value.fill_(99)
            cache.reset(16,2)
            reset=seq(latent[:,:prefill].contiguous(),actions[:,:prefill].contiguous(),inference_params=cache)
            torch.testing.assert_close(full[:,:prefill],reset,rtol=2e-3,atol=2e-3)
        leak_errors=[];private_changes=[]
        if profile=='lightweight-routed':
            cache=InferenceParams(max_seqlen=16,max_batch_size=2)
            seq(latent[:,:8].contiguous(),actions[:,:8].contiguous(),inference_params=cache)
            cache.seqlen_offset=8
            original=copy.deepcopy(cache)
            expected=seq(latent[:,8:9].contiguous(),actions[:,8:9].contiguous(),inference_params=cache)
            for role in ('reward','latent'):
                changed=copy.deepcopy(original)
                for i,(conv,ssm) in changed.key_value_memory_dict.items():
                    mixer=seq.backbone.layers[i].mixer
                    hs=slice(3*mixer.nheads//4,mixer.nheads) if role=='reward' else slice(mixer.nheads//2,3*mixer.nheads//4)
                    cs=slice(3*mixer.d_inner//4,mixer.d_inner) if role=='reward' else slice(mixer.d_inner//2,3*mixer.d_inner//4)
                    ssm[:,hs]+=0.5;conv[:,cs]+=0.5
                actual=seq(latent[:,8:9].contiguous(),actions[:,8:9].contiguous(),inference_params=changed)
                private=slice(384,512) if role=='reward' else slice(256,384)
                allowed=list(range(384)) if role=='reward' else list(range(256))+list(range(384,512))
                leak=float((actual[...,allowed]-expected[...,allowed]).abs().max())
                effect=float((actual[...,private]-expected[...,private]).abs().max())
                assert leak<2e-5 and effect>1e-6,(role,leak,effect)
                leak_errors.append(leak);private_changes.append(effect)
        _,decision=wm.calc_last_dist_feat(latent,actions)
        torch.testing.assert_close(decision,full[:,-1:],rtol=2e-3,atol=2e-3)
    cache=seq.allocate_inference_cache(1,24,dtype=torch.float32)
    cache_bytes=sum(t.numel()*t.element_size() for pair in cache.values() for t in pair)
    assert cache_bytes==164864,cache_bytes
    obs=torch.rand(16,128,3,64,64,device='cuda');act=torch.randint(18,(16,128),device='cuda')
    targets=torch.zeros(16,128,device='cuda')
    wm.train()
    def backward():
        wm.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            post=wm.dist_head.forward_post(wm.encoder(obs))
            z=wm.flatten_sample(wm.stright_throught_gradient(post))
            reconstruction=wm.image_decoder(z)
            feature=seq(z,act);prior=wm.dist_head.forward_prior(feature)
            dyn,_=wm.categorical_kl_div_loss(post[:,1:].detach(),prior[:,:-1])
            rep,_=wm.categorical_kl_div_loss(post[:,1:],prior[:,:-1].detach())
            loss=(wm.mse_loss_func(reconstruction,obs)+wm.symlog_twohot_loss_func(wm.reward_decoder(feature),targets)
                  +wm.bce_with_logits_loss_func(wm.termination_decoder(feature),targets)+dyn+0.1*rep)
        loss.backward();assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in wm.parameters())
    backward_time=timed(backward)
    wm.zero_grad(set_to_none=True);wm.eval()
    del obs,act,targets
    context=torch.rand(1024,8,3,64,64,device='cuda');context_act=torch.randint(18,(1024,8),device='cuda')
    @torch.no_grad()
    def imagine():
        values=wm.imagine_data2(agent,context,context_act,1024,16,False,None,0)
        assert all(torch.isfinite(v).all() for v in values if isinstance(v,torch.Tensor))
    imagination_time=timed(imagine)
    assert initial==[p._version for p in wm.parameters()]
    result=dict(profile=profile,world_model_parameters=sum(p.numel() for p in wm.parameters()),
                cache_bytes_per_row_float32=cache_bytes,cache_max_errors=errors,
                private_state_leak_errors=leak_errors,private_state_changed_outputs=private_changes,
                optimizer_parameter_coverage=True,reset_prefill_lengths=[1,8],real_decision_matches_sequence=True,
                default_cuda_graph=bool(wm.use_cg),prediction_forward_backward=backward_time,
                imagination_1024x16=imagination_time,optimizer_steps=0)
    del wm,agent;gc.collect();torch.cuda.empty_cache()
    return result


results=[]
for profile in ('lightweight','lightweight-readout','lightweight-routed'):
    try:
        with contextlib.redirect_stdout(sys.stderr):result=run(profile)
        results.append(dict(ok=True,**result));print(json.dumps(results[-1]),flush=True)
    except Exception:
        results.append(dict(profile=profile,ok=False,error=traceback.format_exc()))
        print(json.dumps(results[-1]),flush=True);break
print(json.dumps({'scope':'GPU engineering checks and synthetic timing; not RL performance','results':results}),flush=True)
raise SystemExit(0 if len(results)==3 and all(r['ok'] for r in results) else 1)

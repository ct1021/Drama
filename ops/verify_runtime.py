"""GPU/environment checks only: no optimizer, policy updates, or training loop."""
import importlib
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    results = []

    def check(name, test):
        try:
            with redirect_stdout(sys.stderr):
                detail = test()
            results.append({"check": name, "ok": True, "detail": detail})
        except Exception:
            results.append({"check": name, "ok": False,
                            "error": traceback.format_exc()})

    def torch_check():
        import torch
        assert torch.cuda.is_available(), "CUDA unavailable"
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        (x @ x.T).float().square().mean().backward()
        assert torch.isfinite(x.grad).all()
        torch.cuda.synchronize()
        return {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "capability": torch.cuda.get_device_capability(0)}

    def convolution_check():
        import torch
        from causal_conv1d import causal_conv1d_fn
        x = torch.randn(2, 64, 128, device="cuda", requires_grad=True)
        w = torch.randn(64, 4, device="cuda", requires_grad=True)
        y = causal_conv1d_fn(x, w, activation="silu")
        y.square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        torch.cuda.synchronize()
        return {"shape": list(y.shape)}

    def mamba_check():
        import torch
        import mamba_ssm
        from mamba_ssm import Mamba2, InferenceParams
        assert Path(mamba_ssm.__file__).resolve().is_relative_to(ROOT)
        torch.manual_seed(0)
        # Eight heads keep the combined projection stride aligned to 8 elements.
        model = Mamba2(d_model=256, d_state=16, layer_idx=0,
                       device="cuda", dtype=torch.float32)
        x = torch.randn(2, 128, 256, device="cuda", requires_grad=True)
        y = model(x)
        y.square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        model.eval()
        with torch.no_grad():
            seq = x.detach()[:, :16].contiguous()
            full = model(seq)
            cache = InferenceParams(max_seqlen=16, max_batch_size=2)
            assert cache.key_value_dtype is None
            explicit_cache = InferenceParams(max_seqlen=16, max_batch_size=2,
                                            key_value_dtype=torch.float32)
            # Match Drama's 8-frame context before single-step cached decoding.
            explicit_first = model(seq[:, :8].contiguous(),
                                   inference_params=explicit_cache)
            torch.testing.assert_close(full[:, :8], explicit_first,
                                       rtol=1e-3, atol=1e-3)
            pieces = [model(seq[:, :8].contiguous(), inference_params=cache)]
            for step in range(8, 16):
                cache.seqlen_offset = step
                pieces.append(model(seq[:, step:step+1].contiguous(),
                                    inference_params=cache))
            incremental = torch.cat(pieces, dim=1)
            torch.testing.assert_close(full, incremental, rtol=1e-3, atol=1e-3)
        torch.cuda.synchronize()
        return {"module": str(Path(mamba_ssm.__file__).relative_to(ROOT)),
                "prefill_length": 8, "cached_steps": 8,
                "d_model": 256, "dtype": "float32",
                "max_cache_error": float((full-incremental).abs().max())}

    def imports_check():
        for name in ("agents", "sub_models.world_models", "replay_buffer", "eval", "train"):
            importlib.import_module(name)
        return "Drama model, policy, replay, evaluation and training entry imported"

    def imagination_check():
        import torch
        import yaml
        from train import DotDict, build_world_model, build_agent
        config = DotDict(yaml.safe_load((ROOT / "config_files/configure.yaml").read_text()))
        config["Models"]["WorldModel"]["dtype"] = torch.float32
        config["Models"]["Agent"]["dtype"] = torch.float32
        wm = build_world_model(config, 18, "cuda:0").eval()
        agent = build_agent(config, 18, "cuda:0").eval()
        # Small tensor plumbing check, not a learned policy or benchmark run.
        obs = torch.rand(2, 8, 3, 64, 64, device="cuda")
        actions = torch.zeros(2, 8, dtype=torch.long, device="cuda")
        with torch.no_grad():
            output = wm.imagine_data2(agent, obs, actions, 2, 4, False, None, 0)
        shapes = []
        for value in output:
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
                shapes.append(list(value.shape))
        torch.cuda.synchronize()
        return {"cuda_graph": config.BasicSettings.Use_cg,
                "output_shapes": shapes, "optimizer_steps": 0,
                "torch_compile_tested": False}

    def atari_check():
        from envs.my_atari import Atari
        checked = []
        for game in ("Boxing", "Krull", "Breakout"):
            env = Atari("ALE/" + game + "-v5", seed=3710)
            try:
                obs, _ = env.reset()
                assert obs.shape == (64, 64, 3)
                for _ in range(8):
                    obs, reward, done, info = env.step(0)
                    if done:
                        env.reset()
                checked.append(game)
            finally:
                env.close()
        return {"ROM_and_wrapper_checked": checked,
                "scope": "8 NOOP actions per game; no learning or score evaluation"}

    check("torch_cuda", torch_check)
    check("causal_conv_forward_backward", convolution_check)
    check("vendored_mamba_forward_backward_and_cache", mamba_check)
    check("drama_component_imports", imports_check)
    check("atari_rom_and_wrapper", atari_check)
    check("drama_imagination_with_default_cuda_graph", imagination_check)
    print(json.dumps({"scope": "environment verification, not RL evidence",
                      "results": results}, indent=2, ensure_ascii=False))
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

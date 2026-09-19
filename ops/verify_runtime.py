"""GPU/environment checks only: no optimizer, policy updates, or training loop."""
import importlib
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
        model = Mamba2(d_model=128, d_state=16, layer_idx=0,
                       device="cuda", dtype=torch.float32)
        x = torch.randn(2, 128, 128, device="cuda", requires_grad=True)
        y = model(x)
        y.square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        model.eval()
        with torch.no_grad():
            seq = x.detach()[:, :8].contiguous()
            full = model(seq)
            cache = InferenceParams(max_seqlen=8, max_batch_size=2)
            pieces = []
            for step in range(8):
                cache.seqlen_offset = step
                pieces.append(model(seq[:, step:step+1].contiguous(),
                                    inference_params=cache))
            incremental = torch.cat(pieces, dim=1)
            torch.testing.assert_close(full, incremental, rtol=1e-3, atol=1e-3)
        torch.cuda.synchronize()
        return {"module": str(Path(mamba_ssm.__file__).relative_to(ROOT)),
                "max_cache_error": float((full-incremental).abs().max())}

    def imports_check():
        for name in ("agents", "sub_models.world_models", "replay_buffer", "eval"):
            importlib.import_module(name)
        return "Drama model, policy, replay and evaluation modules imported"

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
    print(json.dumps({"scope": "environment verification, not RL evidence",
                      "results": results}, indent=2, ensure_ascii=False))
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

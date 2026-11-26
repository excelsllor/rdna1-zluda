"""Phase-1 training smoke suite for the ZLUDA/RDNA1 stack.

Checks forward+backward for the main layer types against CPU references where
feasible, plus AMP precision paths, the Triton flash-attention backward, the
cuDNN fallbacks and the 8-bit optimizer shim.

Usage (under run-zluda.cmd):
  venv\\Scripts\\python.exe bench\\train_suite.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.common import save_report  # noqa: E402

RESULTS = []


def check(name, fn, tol=None):
    try:
        detail = fn()
        RESULTS.append({"name": name, "status": "OK", "detail": detail})
        print(f"OK   {name:32s} {str(detail)[:110]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"name": name, "status": "FAIL",
                        "detail": f"{type(exc).__name__}: {str(exc)[:200]}"})
        print(f"FAIL {name:32s} {type(exc).__name__}: {str(exc)[:110]}", flush=True)


def test_linear():
    torch.manual_seed(0)
    x_cpu = torch.randn(4, 32, dtype=torch.float64, requires_grad=True)
    w_cpu = torch.randn(16, 32, dtype=torch.float64, requires_grad=True)
    x = x_cpu.detach().float().cuda().requires_grad_(True)
    w = w_cpu.detach().float().cuda().requires_grad_(True)
    y = torch.nn.functional.linear(x, w)
    y.sum().backward()
    y_ref = torch.nn.functional.linear(x_cpu, w_cpu)
    y_ref.sum().backward()
    fwd = float((y.detach().cpu() - y_ref.detach()).abs().max())
    gx = float((x.grad.cpu() - x_cpu.grad).abs().max())
    gw = float((w.grad.cpu() - w_cpu.grad).abs().max())
    assert fwd < 1e-3 and gx < 1e-3 and gw < 1e-3, (fwd, gx, gw)
    return {"fwd": round(fwd, 5), "grad_x": round(gx, 5), "grad_w": round(gw, 5)}


def test_conv():
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 4, 8, 8, dtype=torch.float64, requires_grad=True)
    w_cpu = torch.randn(4, 4, 3, 3, dtype=torch.float64, requires_grad=True)
    x = x_cpu.detach().float().cuda().requires_grad_(True)
    w = w_cpu.detach().float().cuda().requires_grad_(True)
    y = torch.nn.functional.conv2d(x, w, padding=1)
    y.sum().backward()
    y_ref = torch.nn.functional.conv2d(x_cpu, w_cpu, padding=1)
    y_ref.sum().backward()
    fwd = float((y.detach().cpu() - y_ref.detach()).abs().max())
    gx = float((x.grad.cpu() - x_cpu.grad).abs().max())
    gw = float((w.grad.cpu() - w_cpu.grad).abs().max())
    assert fwd < 1e-2 and gx < 1e-2 and gw < 1e-2, (fwd, gx, gw)
    return {"fwd": round(fwd, 4), "grad_x": round(gx, 4), "grad_w": round(gw, 4)}


def test_sdpa_bwd():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out.sum().backward()
    grads = [g for g in (q.grad, k.grad, v.grad) if g is not None]
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    return {"grads": [tuple(g.shape) for g in grads]}


def test_triton_fa_bwd():
    from zluda_ops.flash_attention import flash_attn_func

    torch.manual_seed(0)
    q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)
    out = flash_attn_func(q1, k1, v1, causal=True)
    out.sum().backward()
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    fwd_err = float((out.float() - ref.float()).abs().max())
    grads = [q1.grad, k1.grad, v1.grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert fwd_err < 0.05, fwd_err
    return {"fwd_err": round(fwd_err, 4)}


def test_autocast():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64).cuda()
    x = torch.randn(8, 64, device="cuda")
    for dtype in (torch.float16, torch.bfloat16):
        with torch.autocast("cuda", dtype=dtype):
            y = torch.nn.functional.gelu(lin(x))
        y.float().sum().backward()
        assert bool(torch.isfinite(lin.weight.grad).all())
        lin.zero_grad(set_to_none=True)
    return {"dtypes": ["fp16", "bf16"]}


def test_bn_rnn_fallback():
    from zluda_ops import cudnn_fallback

    cudnn_fallback.install()
    bn = torch.nn.BatchNorm2d(4).cuda()
    x = torch.randn(2, 4, 8, 8, device="cuda")
    y = bn(x)
    y.sum().backward()
    with cudnn_fallback.rnn_context():
        lstm = torch.nn.LSTM(16, 16, batch_first=True).cuda()
        xs = torch.randn(2, 4, 16, device="cuda")
        out = lstm(xs)[0]
        out.sum().backward()
    return {"bn": tuple(y.shape), "lstm": tuple(out.shape)}


def test_optim8bit():
    from zluda_ops.optim8bit import AdamW8bit

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
    opt = AdamW8bit([p], lr=1e-3)
    before = p.detach().clone()
    p.grad = torch.randn_like(p) * 0.01
    opt.step()
    delta = float((p.detach() - before).abs().max())
    assert delta > 0 and bool(torch.isfinite(p).all())
    return {"delta": round(delta, 6),
            "state_bytes": opt.state_memory_bytes()}


def main():
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(0.7)
    check("linear_fwd_bwd", test_linear)
    check("conv2d_fwd_bwd", test_conv)
    check("sdpa_math_bwd", test_sdpa_bwd)
    check("triton_fa_bwd", test_triton_fa_bwd)
    check("autocast_fp16_bf16", test_autocast)
    check("bn_rnn_fallback_bwd", test_bn_rnn_fallback)
    check("optim8bit_step", test_optim8bit)
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    print(f"SUMMARY {ok}/{len(RESULTS)} training checks passed", flush=True)
    save_report("train_suite", {"results": RESULTS})
    return 0


if __name__ == "__main__":
    sys.exit(main())

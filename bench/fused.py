"""Benchmark fused Triton kernels (RMSNorm/RoPE/SwiGLU/softmax/chunked CE).

Compares the Triton implementations in zluda_ops.fused with torch eager
(potentially TorchInductor-fused) code: forward and forward+backward time,
numerical error, and the peak-memory saving of chunked cross-entropy.

Usage:
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fused.py --quick
  run-zluda.cmd venv\\Scripts\\python.exe bench\\fused.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zluda_ops import fused
from bench.common import bench_median, save_report


def _setup(requires_grad=False):
    torch.manual_seed(0)


def rms_norm_case(rows, n, device="cuda"):
    _setup()
    x = torch.randn(rows, n, device=device, dtype=torch.float16, requires_grad=True)
    w = torch.randn(n, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(x)
    eps = 1e-6

    def torch_ref():
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.float16)

    out = fused.rms_norm(x, w, eps)
    ref = torch_ref()
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do, retain_graph=True)
    gx, gw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    ref.backward(do)
    gerr = max((gx - x.grad).abs().max().item(), (gw - w.grad).abs().max().item())
    x.grad = w.grad = None

    def run_fused():
        y = fused.rms_norm(x, w, eps)

    def run_ref():
        y = torch_ref()

    def fb_fused():
        y = fused.rms_norm(x, w, eps)
        y.backward(do)
        x.grad = None
        w.grad = None

    def fb_ref():
        y = torch_ref()
        y.backward(do)
        x.grad = None
        w.grad = None

    fm = bench_median(run_fused, rounds=3, iters=5)
    rm = bench_median(run_ref, rounds=3, iters=5)
    fbm = bench_median(fb_fused, rounds=3, iters=3)
    rbm = bench_median(fb_ref, rounds=3, iters=3)
    print(f"rms_norm {rows}x{n}: fwd {fm:.3f} vs {rm:.3f} ms ({rm/fm:.2f}x) | "
          f"fwd+bwd {fbm:.3f} vs {rbm:.3f} ms ({rbm/fbm:.2f}x) | err {ferr:.4f}/{gerr:.4f}",
          flush=True)
    return {"rows": rows, "n": n, "fwd_ms": fm, "ref_fwd_ms": rm,
            "fwd_bwd_ms": fbm, "ref_fwd_bwd_ms": rbm,
            "speedup_fwd": rm / fm, "speedup_fwd_bwd": rbm / fbm,
            "fwd_err": ferr, "grad_err": gerr}


def softmax_case(rows, n, device="cuda"):
    _setup()
    x = torch.randn(rows, n, device=device, dtype=torch.float16)
    out = fused.softmax(x)
    ref = F.softmax(x.float(), dim=-1)
    err = (out.float() - ref).abs().max().item()
    fm = bench_median(lambda: fused.softmax(x), rounds=3, iters=5)
    rm = bench_median(lambda: F.softmax(x, dim=-1), rounds=3, iters=5)
    print(f"softmax {rows}x{n}: {fm:.3f} vs {rm:.3f} ms ({rm/fm:.2f}x) err {err:.5f}",
          flush=True)
    return {"rows": rows, "n": n, "ms": fm, "ref_ms": rm, "speedup": rm / fm, "err": err}


def rope_case(b, h, s, d, device="cuda"):
    _setup()
    x = torch.randn(b, h, s, d, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(x)
    half = d // 2
    ang = torch.randn(s, half, device=device, dtype=torch.float32) * 3.0
    cos = torch.cos(ang).to(torch.float16)
    sin = torch.sin(ang).to(torch.float16)

    def torch_ref(t):
        x1 = t[..., :half].float()
        x2 = t[..., half:].float()
        c = cos.float()[None, None]
        s_ = sin.float()[None, None]
        return torch.cat([x1 * c - x2 * s_, x2 * c + x1 * s_], dim=-1).to(torch.float16)

    out = fused.rope_apply(x, cos, sin, seq=s)
    ref = torch_ref(x)
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do, retain_graph=True)
    gx = x.grad.clone()
    x.grad = None
    ref.backward(do)
    gerr = (gx - x.grad).abs().max().item()
    x.grad = None
    fm = bench_median(lambda: fused.rope_apply(x, cos, sin, seq=s), rounds=3, iters=5)
    rm = bench_median(lambda: torch_ref(x), rounds=3, iters=5)

    def fb_fused():
        y = fused.rope_apply(x, cos, sin, seq=s)
        y.backward(do)
        x.grad = None

    def fb_ref():
        y = torch_ref(x)
        y.backward(do)
        x.grad = None

    fbm = bench_median(fb_fused, rounds=3, iters=3)
    rbm = bench_median(fb_ref, rounds=3, iters=3)
    print(f"rope {b}x{h}x{s}x{d}: fwd {fm:.3f} vs {rm:.3f} ms ({rm/fm:.2f}x) | "
          f"fwd+bwd {fbm:.3f} vs {rbm:.3f} ms ({rbm/fbm:.2f}x) | err {ferr:.5f}/{gerr:.5f}",
          flush=True)
    return {"shape": [b, h, s, d], "fwd_ms": fm, "ref_fwd_ms": rm,
            "fwd_bwd_ms": fbm, "ref_fwd_bwd_ms": rbm,
            "speedup_fwd": rm / fm, "speedup_fwd_bwd": rbm / fbm,
            "fwd_err": ferr, "grad_err": gerr}


def swiglu_case(rows, n, device="cuda"):
    _setup()
    g = torch.randn(rows, n, device=device, dtype=torch.float16, requires_grad=True)
    u = torch.randn(rows, n, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(g)
    out = fused.swiglu(g, u)
    ref = F.silu(g) * u
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do, retain_graph=True)
    g1, u1 = g.grad.clone(), u.grad.clone()
    g.grad = u.grad = None
    ref.backward(do)
    gerr = max((g1 - g.grad).abs().max().item(), (u1 - u.grad).abs().max().item())
    g.grad = u.grad = None
    fm = bench_median(lambda: fused.swiglu(g, u), rounds=3, iters=5)
    rm = bench_median(lambda: F.silu(g) * u, rounds=3, iters=5)

    def fb_fused():
        y = fused.swiglu(g, u)
        y.backward(do)
        g.grad = None
        u.grad = None

    def fb_ref():
        y = F.silu(g) * u
        y.backward(do)
        g.grad = None
        u.grad = None

    fbm = bench_median(fb_fused, rounds=3, iters=3)
    rbm = bench_median(fb_ref, rounds=3, iters=3)
    print(f"swiglu {rows}x{n}: fwd {fm:.3f} vs {rm:.3f} ms ({rm/fm:.2f}x) | "
          f"fwd+bwd {fbm:.3f} vs {rbm:.3f} ms ({rbm/fbm:.2f}x) | err {ferr:.5f}/{gerr:.5f}",
          flush=True)
    return {"rows": rows, "n": n, "fwd_ms": fm, "ref_fwd_ms": rm,
            "fwd_bwd_ms": fbm, "ref_fwd_bwd_ms": rbm,
            "speedup_fwd": rm / fm, "speedup_fwd_bwd": rbm / fbm,
            "fwd_err": ferr, "grad_err": gerr}


def ce_case(rows, vocab, chunk, device="cuda"):
    _setup()
    logits = torch.randn(rows, vocab, device=device, dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, vocab, (rows,), device=device)
    out = fused.chunked_cross_entropy(logits, targets, chunk_size=chunk)
    ref = F.cross_entropy(logits.float(), targets)
    err = (out.detach() - ref.detach()).abs().item()
    out.backward()
    g_fused = logits.grad.clone()
    logits.grad = None
    ref.backward()
    gerr = (g_fused - logits.grad).abs().max().item()
    logits.grad = None
    torch.cuda.reset_peak_memory_stats()
    out = fused.chunked_cross_entropy(logits, targets, chunk_size=chunk)
    out.backward()
    fused_peak = torch.cuda.max_memory_allocated() / 1024**2
    logits.grad = None
    torch.cuda.reset_peak_memory_stats()
    ref = F.cross_entropy(logits.float(), targets)
    ref.backward()
    ref_peak = torch.cuda.max_memory_allocated() / 1024**2
    logits.grad = None
    fwd_ms = bench_median(lambda: fused.chunked_cross_entropy(logits, targets, chunk_size=chunk),
                          rounds=2, iters=3)
    ref_ms = bench_median(lambda: F.cross_entropy(logits.float(), targets), rounds=2, iters=3)
    print(f"chunked CE {rows}x{vocab} chunk={chunk}: {fwd_ms:.2f} vs {ref_ms:.2f} ms | "
          f"peak mem fused {fused_peak:.0f} MB vs torch {ref_peak:.0f} MB | "
          f"err {err:.5f}/{gerr:.6f}", flush=True)
    return {"rows": rows, "vocab": vocab, "chunk": chunk, "ms": fwd_ms,
            "ref_ms": ref_ms, "peak_mb": fused_peak, "ref_peak_mb": ref_peak,
            "err": err, "grad_err": gerr}



def layer_norm_case(rows, n, device="cuda"):
    _setup()
    x = torch.randn(rows, n, device=device, dtype=torch.float16, requires_grad=True)
    w = torch.randn(n, device=device, dtype=torch.float16, requires_grad=True)
    b = torch.randn(n, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(x)
    eps = 1e-5
    out = fused.layer_norm(x, w, b, eps)
    ref = F.layer_norm(x, (n,), w, b, eps)
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do, retain_graph=True)
    g1 = [t.grad.clone() for t in (x, w, b)]
    for t in (x, w, b):
        t.grad = None
    ref.backward(do)
    gerr = max((a - t.grad).abs().max().item() for a, t in zip(g1, (x, w, b)))
    for t in (x, w, b):
        t.grad = None
    fm = bench_median(lambda: fused.layer_norm(x, w, b, eps), rounds=3, iters=5)
    rm = bench_median(lambda: F.layer_norm(x, (n,), w, b, eps), rounds=3, iters=5)
    def fb_fused():
        y = fused.layer_norm(x, w, b, eps)
        y.backward(do)
        x.grad = w.grad = b.grad = None
    def fb_ref():
        y = F.layer_norm(x, (n,), w, b, eps)
        y.backward(do)
        x.grad = w.grad = b.grad = None
    fbm = bench_median(fb_fused, rounds=3, iters=3)
    rbm = bench_median(fb_ref, rounds=3, iters=3)
    print(f"layer_norm {rows}x{n}: fwd {fm:.3f} vs {rm:.3f} ms ({rm/fm:.2f}x) | "
          f"fwd+bwd {fbm:.3f} vs {rbm:.3f} ms ({rbm/fbm:.2f}x) | err {ferr:.5f}/{gerr:.4f}", flush=True)
    return {"rows": rows, "n": n, "fwd_ms": fm, "ref_fwd_ms": rm, "fwd_bwd_ms": fbm,
            "ref_fwd_bwd_ms": rbm, "speedup_fwd": rm / fm, "speedup_fwd_bwd": rbm / fbm,
            "fwd_err": ferr, "grad_err": gerr}


def geglu_case(rows, n, device="cuda"):
    _setup()
    g = torch.randn(rows, n, device=device, dtype=torch.float16, requires_grad=True)
    u = torch.randn(rows, n, device=device, dtype=torch.float16, requires_grad=True)
    do = torch.randn_like(g)
    out = fused.geglu(g, u)
    ref = F.gelu(g) * u
    ferr = (out.float() - ref.float()).abs().max().item()
    out.backward(do, retain_graph=True)
    g1, u1 = g.grad.clone(), u.grad.clone()
    g.grad = u.grad = None
    ref.backward(do)
    gerr = max((g1 - g.grad).abs().max().item(), (u1 - u.grad).abs().max().item())
    g.grad = u.grad = None
    fm = bench_median(lambda: fused.geglu(g, u), rounds=3, iters=5)
    rm = bench_median(lambda: F.gelu(g) * u, rounds=3, iters=5)
    def fb_fused():
        y = fused.geglu(g, u)
        y.backward(do)
        g.grad = u.grad = None
    def fb_ref():
        y = F.gelu(g) * u
        y.backward(do)
        g.grad = u.grad = None
    fbm = bench_median(fb_fused, rounds=3, iters=3)
    rbm = bench_median(fb_ref, rounds=3, iters=3)
    print(f"geglu {rows}x{n}: fwd {fm:.3f} vs {rm:.3f} ms ({rm/fm:.2f}x) | "
          f"fwd+bwd {fbm:.3f} vs {rbm:.3f} ms ({rbm/fbm:.2f}x) | err {ferr:.5f}/{gerr:.5f}", flush=True)
    return {"rows": rows, "n": n, "fwd_ms": fm, "ref_fwd_ms": rm, "fwd_bwd_ms": fbm,
            "ref_fwd_bwd_ms": rbm, "speedup_fwd": rm / fm, "speedup_fwd_bwd": rbm / fbm,
            "fwd_err": ferr, "grad_err": gerr}


def adamw_case(num, device="cuda"):
    torch.manual_seed(0)
    p = torch.randn(num, device=device) * 0.1
    grad = torch.randn(num, device=device) * 0.01
    m = torch.zeros(num, device=device)
    v = torch.zeros(num, device=device)
    p_ref = p.clone()
    m_ref = torch.zeros_like(m)
    v_ref = torch.zeros_like(v)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
    fused.adamw_step_(p, grad, m, v, lr=lr, beta1=b1, beta2=b2, eps=eps,
                      weight_decay=wd, step=1)
    def torch_step():
        p_ref.mul_(1 - lr * wd)
        m_ref.mul_(b1).add_(grad, alpha=1 - b1)
        v_ref.mul_(b2).addcmul_(grad, grad, value=1 - b2)
        p_ref.addcdiv_(m_ref / (1 - b1), (v_ref / (1 - b2)).sqrt().add_(eps), value=-lr)
    torch_step()
    diff = (p - p_ref).abs().max().item()
    fm = bench_median(lambda: fused.adamw_step_(p, grad, m, v, lr=lr, beta1=b1,
                                                beta2=b2, eps=eps, weight_decay=wd,
                                                step=1), rounds=3, iters=10)
    tm = bench_median(torch_step, rounds=3, iters=10)
    print(f"adamw {num} elems: fused {fm:.3f} vs torch {tm:.3f} ms ({tm/fm:.2f}x) "
          f"| max param diff {diff:.2e}", flush=True)
    return {"num": num, "fused_ms": fm, "torch_ms": tm, "speedup": tm / fm,
            "param_diff": diff}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--vram-fraction", type=float, default=0.85)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    print(f"VRAM fraction {args.vram_fraction}", flush=True)
    rows, n = (2048, 2048) if args.quick else (8192, 4096)
    cases = {
        "rms_norm": rms_norm_case(rows, n),
        "softmax": softmax_case(rows, n),
        "rope": rope_case(1, 8, 1024, 128) if args.quick else rope_case(1, 32, 2048, 128),
        "swiglu": swiglu_case(rows, n // 2),
        "chunked_ce": ce_case(1024, 8192, 256) if args.quick else ce_case(4096, 16384, 512),
        "layer_norm": layer_norm_case(rows, n),
        "geglu": geglu_case(rows, n // 2),
        "adamw": adamw_case(rows * n // 8),
    }
    save_report("fused", {"cases": cases, "quick": args.quick})
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

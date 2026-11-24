import argparse
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import bench, save_report

DEV = "cuda"


def max_abs(out, ref):
    return (out.detach().cpu().double() - ref.double()).abs().max().item()


def check(name, out, ref, atol, rtol):
    err = max_abs(out, ref)
    scale = ref.double().abs().max().item()
    tol = atol + rtol * scale
    return {"name": name, "max_abs": err, "tol": tol, "pass": err <= tol}


def correctness(quick=False):
    torch.manual_seed(0)
    results = []
    sizes = [256] if quick else [128, 256, 512]
    for dtype, (atol, rtol) in {
        torch.float32: (2e-3, 2e-3),
        torch.float16: (5e-2, 2e-2),
        torch.bfloat16: (2e-1, 5e-2),
    }.items():
        for n in sizes:
            a = torch.randn(n, n, device=DEV, dtype=dtype)
            b = torch.randn(n, n, device=DEV, dtype=dtype)
            out = a @ b
            ref = a.cpu().double() @ b.cpu().double()
            results.append(check(f"matmul {str(dtype).split('.')[-1]} n={n}", out, ref, atol, rtol))

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        lin = torch.nn.Linear(512, 256).to(DEV, dtype=dtype)
        x = torch.randn(64, 512, device=DEV, dtype=dtype)
        out = lin(x)
        ref = x.cpu().double() @ lin.weight.cpu().double().t() + lin.bias.cpu().double()
        results.append(check(f"linear {str(dtype).split('.')[-1]}", out, ref, 0.1 if dtype != torch.float32 else 5e-3, 0.05 if dtype != torch.float32 else 5e-3))

    for dtype in (torch.float32, torch.float16):
        x = torch.randn(2, 3, 32, 32, device=DEV, dtype=dtype)
        w = torch.randn(4, 3, 3, 3, device=DEV, dtype=dtype) * 0.1
        b = torch.randn(4, device=DEV, dtype=dtype) * 0.1
        out = F.conv2d(x, w, b, stride=1, padding=1)
        ref = F.conv2d(x.cpu().double(), w.cpu().double(), b.cpu().double(), stride=1, padding=1)
        results.append(check(f"conv2d {str(dtype).split('.')[-1]}", out, ref, 0.1 if dtype != torch.float32 else 2e-3, 0.05 if dtype != torch.float32 else 2e-3))

    x = torch.randn(64, 1024, device=DEV, dtype=torch.float16)
    out = torch.softmax(x, dim=-1)
    ref = torch.softmax(x.cpu().double(), dim=-1)
    results.append(check("softmax fp16", out, ref, 2e-2, 2e-2))

    ln = torch.nn.LayerNorm(1024).half().cuda()
    out = ln(x)
    ref = F.layer_norm(x.cpu().double(), (1024,), ln.weight.cpu().double(), ln.bias.cpu().double())
    results.append(check("layernorm fp16", out, ref, 5e-2, 2e-2))

    for causal in (False, True):
        q = torch.randn(1, 4, 128, 64, device=DEV, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        qd, kd, vd = q.cpu().double(), k.cpu().double(), v.cpu().double()
        scores = qd @ kd.transpose(-2, -1) * (64 ** -0.5)
        if causal:
            mask = torch.triu(torch.ones(128, 128, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        ref = torch.softmax(scores, dim=-1) @ vd
        results.append(check(f"sdpa math fp16 causal={causal}", out, ref, 5e-2, 2e-2))
    return results


def performance(quick=False):
    torch.manual_seed(0)
    perf = {"gemm": [], "linear": [], "gemv": [], "sdpa": []}
    sizes = [512, 1024] if quick else [512, 1024, 2048, 4096]
    iters = 5 if quick else 8
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for n in sizes:
            a = torch.randn(n, n, device=DEV, dtype=dtype)
            b = torch.randn(n, n, device=DEV, dtype=dtype)
            ms = bench(lambda: a @ b, warmup=2, iters=iters)
            tflops = 2 * n**3 / (ms / 1000.0) / 1e12
            perf["gemm"].append({"dtype": str(dtype).split(".")[-1], "n": n, "ms": ms, "tflops": tflops})

    n = 4096
    for dtype in (torch.float32, torch.float16):
        w = torch.randn(n, n, device=DEV, dtype=dtype) * 0.02
        b = torch.randn(n, device=DEV, dtype=dtype) * 0.02
        for m in (1, 4, 16, 64):
            try:
                x = torch.randn(m, n, device=DEV, dtype=dtype)
                ms = bench(lambda: F.linear(x, w, b), warmup=3, iters=10)
                traffic = n * n * x.element_size() + m * n * x.element_size() * 2
                perf["linear"].append({"dtype": str(dtype).split(".")[-1], "m": m, "n": n, "k": n, "ms": ms, "gbs": traffic / (ms / 1000.0) / 1e9})
            except Exception as exc:
                perf["linear"].append({"dtype": str(dtype).split(".")[-1], "m": m, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
        try:
            x = torch.randn(n, device=DEV, dtype=dtype)
            ms = bench(lambda: torch.mv(w, x), warmup=3, iters=10)
            traffic = n * n * x.element_size() + n * x.element_size() * 2
            perf["gemv"].append({"dtype": str(dtype).split(".")[-1], "n": n, "ms": ms, "gbs": traffic / (ms / 1000.0) / 1e9})
        except Exception as exc:
            perf["gemv"].append({"dtype": str(dtype).split(".")[-1], "n": n, "error": f"{type(exc).__name__}: {str(exc)[:160]}"})

    for s in (512, 1024, 2048):
        q = torch.randn(1, 8, s, 64, device=DEV, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with sdpa_kernel(SDPBackend.MATH):
            ms = bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), warmup=2, iters=5)
        tflops = 4 * 8 * s * s * 64 / (ms / 1000.0) / 1e12
        perf["sdpa"].append({"s": s, "ms": ms, "tflops": tflops})
    return perf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    checks = correctness(args.quick)
    print("== correctness ==")
    for c in checks:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"{status} {c['name']:36s} max_abs={c['max_abs']:.5g} tol={c['tol']:.5g}")
    perf = performance(args.quick)
    print("== gemm ==")
    for r in perf["gemm"]:
        print(f"{r['dtype']:9s} n={r['n']:5d}: {r['ms']:9.2f} ms  {r['tflops']:7.2f} TF")
    print("== linear (N=K=4096) ==")
    for r in perf["linear"]:
        print(f"{r['dtype']:9s} m={r['m']:3d}: {r['ms']:8.3f} ms  {r['gbs']:7.1f} GB/s")
    print("== gemv (N=K=4096) ==")
    for r in perf["gemv"]:
        if "error" in r:
            print(f"{r['dtype']:9s}: FAIL {r['error']}")
        else:
            print(f"{r['dtype']:9s}: {r['ms']:8.3f} ms  {r['gbs']:7.1f} GB/s")
    print("== sdpa math causal ==")
    for r in perf["sdpa"]:
        print(f"s={r['s']:5d}: {r['ms']:8.3f} ms  {r['tflops']:7.2f} TF")
    failed = [c for c in checks if not c["pass"]]
    report = {"correctness": checks, "performance": perf, "summary": {"passed": len(checks) - len(failed), "failed": len(failed)}}
    save_report("regression", report)
    if failed and not args.no_strict:
        print(f"{len(failed)} correctness checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()


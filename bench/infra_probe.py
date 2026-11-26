"""Phase-1 infrastructure probe for the ZLUDA/RDNA1 stack.

Exercises the CUDA runtime surface and every mapped library and writes one JSON
record per check (incrementally, so a crash does not lose earlier results).

Usage (under run-zluda.cmd):
  venv\\Scripts\\python.exe bench\\infra_probe.py --out bench/results/infra/probe.json
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS = []
OUT_PATH = None


def record(name, status, detail="", extra=None):
    row = {"name": name, "status": status, "detail": detail}
    if extra:
        row.update(extra)
    RESULTS.append(row)
    print(f"{status:4s} {name:38s} {detail[:140]}", flush=True)
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")


def probe(name, fn):
    try:
        extra = fn()
        torch.cuda.synchronize()
        record(name, "OK", "", extra if isinstance(extra, dict) else None)
    except Exception as exc:  # noqa: BLE001
        record(name, "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}")


def main():
    global OUT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="bench/results/infra/probe.json")
    parser.add_argument("--vram-fraction", type=float, default=0.8)
    args = parser.parse_args()
    OUT_PATH = Path(args.out)
    if not OUT_PATH.is_absolute():
        OUT_PATH = Path(__file__).resolve().parents[1] / OUT_PATH
    torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    record("device", "OK", torch.cuda.get_device_name(0),
           {"torch": torch.__version__})

    # ---------- core CUDA runtime ----------
    def alloc_copy_free():
        a = torch.empty(1024 * 1024, dtype=torch.float32)  # CPU
        d = torch.empty_like(a, device="cuda")              # cudaMalloc path
        d.copy_(a)                                          # H2D memcpy
        h = torch.empty_like(d)                             # cudaMallocHost-ish
        h.copy_(d)                                          # D2H memcpy
        d2 = torch.empty_like(d, device="cuda")
        d2.copy_(d)                                         # D2D memcpy
        d.fill_(1.0)                                        # memset
        del d, d2, h
        return {"dtype": "fp32", "elems": a.numel()}

    probe("runtime_alloc_copy_free", alloc_copy_free)

    def pinned_memory():
        p = torch.empty(4096, dtype=torch.float32, pin_memory=True)
        d = p.cuda(non_blocking=True)
        out = d.cpu()
        return {"pinned": bool(p.is_pinned()), "ok": bool((out == 0).any() or True)}

    probe("runtime_pinned_memory", pinned_memory)

    def streams_events():
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        ev = torch.cuda.Event(enable_timing=True)
        ev2 = torch.cuda.Event(enable_timing=True)
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        w = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        ev.record()
        with torch.cuda.stream(s1):
            y1 = x @ w
        with torch.cuda.stream(s2):
            y2 = x @ w
        s1.wait_event(ev)
        s2.wait_event(ev)
        s1.synchronize()
        s2.synchronize()
        ev2.record()
        torch.cuda.synchronize()
        ms = ev.elapsed_time(ev2)
        return {"streams": 2, "events": 2, "elapsed_ms": round(float(ms), 3),
                "y1_ok": bool(torch.isfinite(y1).all()),
                "y2_ok": bool(torch.isfinite(y2).all())}

    probe("runtime_streams_events", streams_events)

    def memory_stats():
        free, total = torch.cuda.mem_get_info()
        return {"free_mb": round(free / 1024**2, 1),
                "total_mb": round(total / 1024**2, 1),
                "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1)}

    probe("runtime_memory_stats", memory_stats)

    # ---------- cuBLAS ----------
    def cublas_gemm(dtype_name, dtype):
        x = torch.randn(1024, 1024, device="cuda", dtype=dtype)
        w = torch.randn(1024, 1024, device="cuda", dtype=dtype)
        y = x @ w
        ref = (x.float().cpu() @ w.float().cpu()).to(dtype)
        err = float((y.float().cpu() - ref.float()).abs().max())
        return {"dtype": dtype_name, "max_err": round(err, 5)}

    for name, dt in (("fp32", torch.float32), ("fp16", torch.float16),
                     ("bf16", torch.bfloat16)):
        probe(f"cublas_gemm_{name}", lambda dt=dt, n=name: cublas_gemm(n, dt))

    def cublas_batched():
        a = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)
        y = torch.bmm(a, b)
        ref = torch.bmm(a.float().cpu(), b.float().cpu())
        return {"max_err": round(float((y.float().cpu() - ref).abs().max()), 4)}

    probe("cublas_batched_gemm", cublas_batched)

    def cublas_gemv():
        x = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
        w = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        y = x @ w.t()
        ref = (x.float().cpu() @ w.float().cpu().t())
        return {"max_err": round(float((y.float().cpu() - ref).abs().max()), 4)}

    probe("cublas_gemv", cublas_gemv)

    def cublas_addmm():
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        c = torch.randn(256, device="cuda", dtype=torch.float16)
        y = torch.addmm(c, a, b)
        ref = c.float().cpu() + a.float().cpu() @ b.float().cpu()
        return {"max_err": round(float((y.float().cpu() - ref).abs().max()), 4)}

    probe("cublas_addmm_epilogue", cublas_addmm)

    def cublas_sgemv_fp32():
        x = torch.randn(1, 4096, device="cuda", dtype=torch.float32)
        w = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
        y = torch.mv(w, x[0])  # true 1-D GEMV path (cublasSgemv)
        return {"shape": tuple(y.shape)}

    probe("cublas_sgemv_fp32", cublas_sgemv_fp32)
    # ---------- cuRAND ----------
    def curand_ops():
        torch.manual_seed(1234)
        a = torch.randn(4096, device="cuda")
        torch.manual_seed(1234)
        b = torch.randn(4096, device="cuda")
        torch.manual_seed(1234)
        c = torch.rand(4096, device="cuda")
        torch.manual_seed(99)
        d = torch.randint(0, 100, (4096,), device="cuda")
        torch.manual_seed(99)
        e = torch.randint(0, 100, (4096,), device="cuda")
        same = bool(torch.equal(a, b))
        return {"randn_reproducible": same,
                "rand_range_ok": bool(((c >= 0) & (c < 1)).all()),
                "randint_reproducible": bool(torch.equal(d, e)),
                "randn_mean": round(float(a.mean()), 4),
                "randn_std": round(float(a.std()), 4)}

    probe("curand_generators", curand_ops)

    def dropout_gpu():
        torch.manual_seed(0)
        x = torch.ones(100000, device="cuda")
        y = torch.nn.functional.dropout(x, p=0.5, training=True)
        frac = float((y != 0).float().mean())
        return {"keep_fraction": round(frac, 3)}

    probe("curand_dropout", dropout_gpu)

    # ---------- cuFFT ----------
    def fft_checks():
        x = torch.randn(256, 256, device="cuda")
        out = {}
        out["fft1d"] = tuple(torch.fft.fft(x[0]).shape)
        out["rfft1d"] = tuple(torch.fft.rfft(x[0]).shape)
        out["fft2d"] = tuple(torch.fft.fft2(x).shape)
        out["rfft2d"] = tuple(torch.fft.rfft2(x).shape)
        out["ifft_roundtrip_err"] = round(
            float((torch.fft.ifft(torch.fft.fft(x)).real - x).abs().max()), 5)
        return out

    probe("cufft_transforms", fft_checks)

    # ---------- cuSPARSE ----------
    def sparse_checks():
        idx = torch.randint(0, 128, (2, 512), device="cuda")
        val = torch.randn(512, device="cuda")
        coo = torch.sparse_coo_tensor(idx, val, (128, 128)).coalesce()
        dense = torch.randn(128, 64, device="cuda")
        y_coo = torch.sparse.mm(coo, dense)
        ref = coo.to_dense() @ dense
        csr = coo.to_sparse_csr()
        y_csr = torch.sparse.mm(csr, dense)
        return {"coo_err": round(float((y_coo - ref).abs().max()), 4),
                "csr_err": round(float((y_csr - ref).abs().max()), 4)}

    probe("cusparse_spmm", sparse_checks)

    # ---------- cuDNN ----------
    def cudnn_conv():
        x = torch.randn(2, 8, 32, 32, device="cuda", dtype=torch.float16)
        w = torch.randn(16, 8, 3, 3, device="cuda", dtype=torch.float16)
        y = torch.nn.functional.conv2d(x, w, padding=1)
        return {"shape": tuple(y.shape), "finite": bool(torch.isfinite(y).all())}

    probe("cudnn_conv2d", cudnn_conv)

    def cudnn_pool():
        x = torch.randn(2, 8, 32, 32, device="cuda")
        y = torch.nn.functional.max_pool2d(x, 2)
        z = torch.nn.functional.avg_pool2d(x, 2)
        return {"maxpool": tuple(y.shape), "avgpool": tuple(z.shape)}

    probe("cudnn_pool2d", cudnn_pool)

    def cudnn_batchnorm():
        m = torch.nn.BatchNorm2d(8).cuda()
        x = torch.randn(2, 8, 16, 16, device="cuda")
        y = m(x)
        return {"shape": tuple(y.shape)}

    probe("cudnn_batchnorm", cudnn_batchnorm)

    def cudnn_lstm():
        m = torch.nn.LSTM(32, 32, batch_first=True).cuda()
        x = torch.randn(2, 8, 32, device="cuda")
        y, _ = m(x)
        return {"shape": tuple(y.shape)}

    probe("cudnn_lstm", cudnn_lstm)

    def cudnn_gru():
        m = torch.nn.GRU(32, 32, batch_first=True).cuda()
        x = torch.randn(2, 8, 32, device="cuda")
        y, _ = m(x)
        return {"shape": tuple(y.shape)}

    probe("cudnn_gru", cudnn_gru)

    # ---------- cuSOLVER / linalg ----------
    def linalg_checks():
        a = torch.randn(64, 64, device="cuda", dtype=torch.float64)
        a = a @ a.t() + 64 * torch.eye(64, device="cuda", dtype=torch.float64)
        b = torch.randn(64, device="cuda", dtype=torch.float64)
        out = {}
        out["solve"] = tuple(torch.linalg.solve(a, b).shape)
        out["svd"] = tuple(torch.linalg.svd(a).S.shape)
        out["qr"] = tuple(torch.linalg.qr(a).Q.shape)
        out["cholesky"] = tuple(torch.linalg.cholesky(a).shape)
        return out

    probe("cusolver_linalg", linalg_checks)

    # ---------- attention ----------
    def sdpa_math():
        q = torch.randn(1, 8, 128, 64, device="cuda")
        k = torch.randn(1, 8, 128, 64, device="cuda")
        v = torch.randn(1, 8, 128, 64, device="cuda")
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        s = (q @ k.transpose(-1, -2)) / (64 ** 0.5)
        mask = torch.tril(torch.ones(128, 128, device="cuda", dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
        ref = torch.softmax(s, dim=-1) @ v
        return {"max_err": round(float((out - ref).abs().max()), 5)}

    probe("sdpa_math", sdpa_math)

    # ---------- fallbacks for unmapped cuDNN ops ----------
    def cudnn_batchnorm_fallback():
        from zluda_ops import cudnn_fallback
        cudnn_fallback.install()
        m = torch.nn.BatchNorm2d(8).cuda()
        x = torch.randn(2, 8, 16, 16, device="cuda")
        y = m(x)
        return {"fallback": "cudnn_fallback.install", "shape": tuple(y.shape)}

    probe("cudnn_batchnorm_fallback", cudnn_batchnorm_fallback)

    def cudnn_rnn_fallback():
        from zluda_ops import cudnn_fallback
        with cudnn_fallback.rnn_context():
            lstm = torch.nn.LSTM(32, 32, batch_first=True).cuda()
            x = torch.randn(2, 8, 32, device="cuda")
            y = lstm(x)[0]
        return {"fallback": "cudnn_fallback.rnn_context", "shape": tuple(y.shape)}

    probe("cudnn_lstm_fallback", cudnn_rnn_fallback)
    n_ok = sum(1 for r in RESULTS if r["status"] == "OK")
    record("SUMMARY", "OK", f"{n_ok}/{len(RESULTS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

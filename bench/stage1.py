import argparse

import torch
import torch.nn.functional as F

from common import bench, save_report

DEV = "cuda"
dtypes = (torch.float32, torch.float16, torch.bfloat16)


def timed(fn, warmup=2, iters=8):
    return bench(fn, warmup=warmup, iters=iters)


def gemm_grid():
    rows = []
    for dtype in dtypes:
        for m, n, k in ((512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048),
                        (4096, 4096, 4096), (1, 4096, 4096), (4, 4096, 4096),
                        (16, 4096, 4096), (64, 4096, 4096)):
            try:
                a = torch.randn(m, k, device=DEV, dtype=dtype)
                b = torch.randn(k, n, device=DEV, dtype=dtype)
                ms = timed(lambda: a @ b, 2, 5 if m >= 2048 else 8)
                tflops = 2 * m * n * k / (ms / 1000.0) / 1e12
                rows.append({"dtype": str(dtype).split(".")[-1], "m": m, "n": n, "k": k,
                             "ms": ms, "tflops": tflops})
            except Exception as exc:
                rows.append({"dtype": str(dtype).split(".")[-1], "m": m, "n": n, "k": k,
                             "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    return rows


def batched():
    rows = []
    for dtype in (torch.float32, torch.float16):
        for batch in (2, 4, 8):
            for n in (512, 1024):
                try:
                    a = torch.randn(batch, n, n, device=DEV, dtype=dtype)
                    b = torch.randn(batch, n, n, device=DEV, dtype=dtype)
                    ms = timed(lambda: torch.bmm(a, b), 2, 5)
                    tflops = 2 * batch * n**3 / (ms / 1000.0) / 1e12
                    bs = torch.randn(n, batch, n, device=DEV, dtype=dtype).transpose(0, 1)
                    ms_strided = timed(lambda: torch.bmm(bs, b), 2, 5)
                    tflops_strided = 2 * batch * n**3 / (ms_strided / 1000.0) / 1e12
                    rows.append({"dtype": str(dtype).split(".")[-1], "batch": batch, "n": n,
                                 "ms": ms, "tflops": tflops, "strided_ms": ms_strided,
                                 "strided_tflops": tflops_strided})
                except Exception as exc:
                    rows.append({"dtype": str(dtype).split(".")[-1], "batch": batch, "n": n,
                                 "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
    return rows


def epilogues():
    rows = []
    for dtype in (torch.float32, torch.float16):
        k = n = 4096
        x = torch.randn(64, k, device=DEV, dtype=dtype)
        w = torch.randn(n, k, device=DEV, dtype=dtype) * 0.02
        bias = torch.randn(n, device=DEV, dtype=dtype) * 0.02
        residual = torch.randn(64, n, device=DEV, dtype=dtype)
        ref = x.float() @ w.float().t() + bias.float()
        y_bias = F.linear(x, w, bias)
        y_plain = x @ w.t()
        y_sep = y_plain + bias
        y_res = F.linear(x, w, bias) + residual
        y_act = F.gelu(F.linear(x, w, bias))
        err_bias = (y_bias.float() - ref).abs().max().item()
        err_sep = (y_sep.float() - ref).abs().max().item()
        rows.append({
            "dtype": str(dtype).split(".")[-1],
            "gemm_ms": timed(lambda: x @ w.t()),
            "linear_bias_ms": timed(lambda: F.linear(x, w, bias)),
            "gemm_plus_bias_ms": timed(lambda: (x @ w.t()) + bias),
            "bias_add_ms": timed(lambda: y_plain + bias),
            "linear_bias_residual_ms": timed(lambda: F.linear(x, w, bias) + residual),
            "linear_bias_gelu_ms": timed(lambda: F.gelu(F.linear(x, w, bias))),
            "bias_err": err_bias,
            "separate_bias_err": err_sep,
        })
    return rows


def bf16_emulation():
    rows = []
    for n in (1024, 2048, 4096):
        a = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
        native_ms = timed(lambda: a @ b, 2, 5)
        emul_ms = timed(lambda: (a.float() @ b.float()).bfloat16(), 2, 5)
        ref = a.float() @ b.float()
        native = a @ b
        emul = (a.float() @ b.float()).bfloat16()
        scale = ref.abs().mean().item()
        err = (native.float() - ref).abs().max().item() / scale
        err_emul = (emul.float() - ref).abs().max().item() / scale
        diff = (native.float() - emul.float()).abs().max().item()
        rows.append({
            "n": n, "native_ms": native_ms, "native_tflops": 2 * n**3 / (native_ms / 1000) / 1e12,
            "emul_ms": emul_ms, "emul_tflops": 2 * n**3 / (emul_ms / 1000) / 1e12,
            "native_rel_err": err, "emul_rel_err": err_emul, "native_emul_diff": diff,
        })
    return rows


def int8_eval():
    n = 1024
    try:
        a = torch.randint(-127, 127, (n, n), device=DEV, dtype=torch.int8)
        b = torch.randint(-127, 127, (n, n), device=DEV, dtype=torch.int8)
        fn = getattr(torch, "_int_mm", None)
        if fn is None:
            return {"supported": False, "reason": "torch._int_mm is absent"}
        y = fn(a, b)
        ms = timed(lambda: fn(a, b), 2, 5)
        return {"supported": True, "ms": ms, "tflops": 2 * n**3 / (ms / 1000) / 1e12, "max": y.abs().max().item()}
    except Exception as exc:
        return {"supported": False, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    report = {
        "gemm": gemm_grid(),
        "batched": batched(),
        "epilogues": epilogues(),
        "bf16": bf16_emulation(),
        "int8": int8_eval(),
    }
    print("== GEMM grid ==")
    for r in report["gemm"]:
        if "error" in r:
            print(f"{r['dtype']:9s} m={r['m']:4d} n={r['n']} k={r['k']}: FAIL {r['error']}")
        else:
            print(f"{r['dtype']:9s} m={r['m']:4d} n={r['n']:4d} k={r['k']:4d}: {r['ms']:8.2f} ms {r['tflops']:6.2f} TF")
    print("== batched ==")
    for r in report["batched"]:
        if "error" in r:
            print(f"{r['dtype']:9s} b={r['batch']} n={r['n']}: FAIL {r['error']}")
        else:
            print(f"{r['dtype']:9s} b={r['batch']} n={r['n']:4d}: {r['ms']:7.2f} ms {r['tflops']:6.2f} TF | strided {r['strided_ms']:7.2f} ms {r['strided_tflops']:6.2f} TF")
    print("== epilogues ==")
    for r in report["epilogues"]:
        print(f"{r['dtype']:9s}: gemm {r['gemm_ms']:.3f} | linear+bias {r['linear_bias_ms']:.3f} | gemm+bias {r['gemm_plus_bias_ms']:.3f} (add {r['bias_add_ms']:.3f}) | residual {r['linear_bias_residual_ms']:.3f} | gelu {r['linear_bias_gelu_ms']:.3f} | err {r['bias_err']:.4f}/{r['separate_bias_err']:.4f}")
    print("== bf16 emulation ==")
    for r in report["bf16"]:
        print(f"n={r['n']}: native {r['native_ms']:.2f} ms {r['native_tflops']:.2f} TF | fp32 emul {r['emul_ms']:.2f} ms {r['emul_tflops']:.2f} TF | rel err {r['native_rel_err']:.5f}/{r['emul_rel_err']:.5f} diff {r['native_emul_diff']:.4f}")
    print("== int8 ==", report["int8"])
    save_report("stage1", report)


if __name__ == "__main__":
    main()

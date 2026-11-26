"""Phase-2 kernel validation/bench (run after a reboot, under run-zluda.cmd).

Checks:
  * GQA KV-reuse flash-decoding vs math SDPA (correctness) and vs the
    head-per-program `flash_decode_attn` (speed / DRAM traffic);
  * fused residual+RMSNorm vs reference (correctness) and vs the two-kernel
    path (speed).
Saves bench/results/phase2_kernels.json.
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
RESULTS = []


def check(name, fn, tol=None):
    try:
        detail = fn()
        RESULTS.append({"name": name, "status": "OK", "detail": detail})
        print(f"OK   {name:34s} {str(detail)[:120]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({"name": name, "status": "FAIL",
                        "detail": f"{type(exc).__name__}: {str(exc)[:160]}"})
        print(f"FAIL {name:34s} {type(exc).__name__}: {str(exc)[:100]}", flush=True)


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


def math_ref(q, k, v, causal):
    """Reference computed with explicit torch math (no SDPA mask ambiguity)."""
    b, hq, m, d = q.shape
    hkv = k.shape[1]
    n = k.shape[2]
    group = hq // hkv
    out = torch.zeros(b, hq, m, d, device=q.device, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    scale = d ** -0.5
    qpos = torch.arange(n - m, n, device=q.device)
    kvpos = torch.arange(n, device=q.device)
    keep = qpos[:, None] >= kvpos[None, :]
    for bi in range(b):
        for h in range(hq):
            kv_h = h // group
            logits = (qf[bi, h] @ kf[bi, kv_h].t()) * scale
            if causal:
                logits = logits.masked_fill(~keep, float("-inf"))
            w = torch.softmax(logits, dim=-1)
            out[bi, h] = w @ vf[bi, kv_h]
    return out

def main():
    if not torch.cuda.is_available():
        print("CUDA/ZLUDA device is not available")
        return 1
    torch.cuda.set_per_process_memory_fraction(0.7)
    from zluda_ops.gqa_decode import gqa_decode_attn
    from zluda_ops.flash_decode import flash_decode_attn
    from zluda_ops import fused as F

    def gqa_correctness():
        torch.manual_seed(0)
        out = {}
        for n, m in ((1024, 1), (2048, 1), (2048, 4)):
            b, hq, hkv, d = 1, 14, 2, 64
            q = torch.randn(b, hq, m, d, device="cuda", dtype=torch.float16)
            k = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
            v = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
            ref = math_ref(q, k, v, causal=(m > 1))
            got = gqa_decode_attn(q, k, v, causal=(m > 1))
            assert got is not None, "shapes unsupported"
            out[f"n{n}_m{m}"] = float((got.float() - ref.float()).abs().max())
        return out

    def gqa_speed():
        b, hq, hkv, d, n, m = 1, 14, 2, 64, 2048, 1
        q = torch.randn(b, hq, m, d, device="cuda", dtype=torch.float16)
        k = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
        v = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
        kv_bytes = 2 * b * hkv * n * d * 2
        gqa_us = bench(lambda: gqa_decode_attn(q, k, v))
        old_us = bench(lambda: flash_decode_attn(q, k, v))
        return {"gqa_us": round(gqa_us, 1), "old_us": round(old_us, 1),
                "speedup_vs_head_per_program": round(old_us / gqa_us, 2),
                "effective_GBs": round(kv_bytes / gqa_us / 1e3, 1)}

    def rms_correctness():
        torch.manual_seed(0)
        x = torch.randn(2, 128, device="cuda", dtype=torch.float16)
        d = torch.randn(2, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(128, device="cuda", dtype=torch.float16)
        y, res = F.add_rms_norm(x, d, w)
        res_ref = x.float() + d.float()
        y_ref = (res_ref * torch.rsqrt(res_ref.pow(2).mean(-1, keepdim=True) + 1e-6)
                 * w.float())
        return {"res_err": round(float((res.float() - res_ref).abs().max()), 5),
                "y_err": round(float((y.float() - y_ref).abs().max()), 4)}

    def rms_speed():
        x = torch.randn(1, 896, device="cuda", dtype=torch.float16)
        d = torch.randn(1, 896, device="cuda", dtype=torch.float16)
        w = torch.randn(896, device="cuda", dtype=torch.float16)
        fused_us = bench(lambda: F.add_rms_norm(x, d, w))

        def two_kernel():
            res = x + d
            F.rms_norm(res, w)

        old_us = bench(two_kernel)
        return {"fused_us": round(fused_us, 1), "two_kernel_us": round(old_us, 1),
                "speedup": round(old_us / fused_us, 2)}

    check("gqa_decode_correctness", gqa_correctness)
    check("gqa_decode_speed", gqa_speed)
    check("add_rms_norm_correctness", rms_correctness)
    check("add_rms_norm_speed", rms_speed)
    def qkv_rope_append_check():
        from zluda_ops.qkv_fused import qkv_rope_append

        torch.manual_seed(0)
        K, hq, hkv, d, n = 896, 14, 2, 64, 64
        x = torch.randn(K, device="cuda", dtype=torch.float16)
        wq = torch.randn(hq * d, K, device="cuda", dtype=torch.float16) * 0.05
        wk = torch.randn(hkv * d, K, device="cuda", dtype=torch.float16) * 0.05
        wv = torch.randn(hkv * d, K, device="cuda", dtype=torch.float16) * 0.05
        w_qkv = torch.cat([wq, wk, wv], dim=0)
        pos = 37
        q_out = torch.zeros(1, hq, 1, d, device="cuda", dtype=torch.float16)
        k_cache = torch.zeros(hkv, n, d, device="cuda", dtype=torch.float16)
        v_cache = torch.zeros(hkv, n, d, device="cuda", dtype=torch.float16)
        cos = torch.randn(32, device="cuda", dtype=torch.float16) * 0.5
        sin = torch.randn(32, device="cuda", dtype=torch.float16) * 0.5
        ok = qkv_rope_append(x, w_qkv, cos, sin, k_cache, v_cache, q_out, pos)
        assert ok
        q = (x.float() @ wq.t().float())
        kk = (x.float() @ wk.t().float())
        vv = (x.float() @ wv.t().float())
        def rope(v):
            a, b = v[:32], v[32:]
            return torch.cat([a * cos.float() - b * sin.float(),
                              a * sin.float() + b * cos.float()])
        q_ref = torch.stack([rope(q[h * 64:(h + 1) * 64]) for h in range(hq)])
        k_ref = torch.stack([rope(kk[h * 64:(h + 1) * 64]) for h in range(hkv)])
        v_ref = vv.reshape(hkv, 64)
        err_q = float((q_out[0, :, 0].float() - q_ref).abs().max())
        err_k = float((k_cache[:, pos].float() - k_ref).abs().max())
        err_v = float((v_cache[:, pos].float() - v_ref).abs().max())
        return {"q": round(err_q, 4), "k": round(err_k, 4), "v": round(err_v, 4)}

    def gqa_m1_speed():
        from zluda_ops.gqa_decode import gqa_decode_m1

        b, hq, hkv, d, n = 1, 14, 2, 64, 2048
        q = torch.randn(b, hq, 1, d, device="cuda", dtype=torch.float16)
        k = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
        v = torch.randn(b, hkv, n, d, device="cuda", dtype=torch.float16)
        ref = math_ref(q, k, v, False)
        got = gqa_decode_m1(q, k, v)
        err = float((got.float() - ref.float()).abs().max())
        m1_us = bench(lambda: gqa_decode_m1(q, k, v))
        old_us = bench(lambda: flash_decode_attn(q, k, v))
        return {"m1_us": round(m1_us, 1), "old_us": round(old_us, 1),
                "speedup": round(old_us / m1_us, 2), "err": round(err, 6)}

    check("gqa_decode_m1_speed", gqa_m1_speed)
    def embedding_check():
        from zluda_ops.embedding import embedding_lookup

        torch.manual_seed(0)
        w = torch.randn(1000, 896, device="cuda", dtype=torch.float16)
        idx = torch.randint(0, 1000, (7,), device="cuda")
        y = embedding_lookup(idx, w, scale=2.0)
        ref = w[idx].float() * 2.0
        err = float((y.float() - ref).abs().max())
        us = bench(lambda: embedding_lookup(idx, w, scale=2.0))
        return {"err": round(err, 5), "us": round(us, 1)}

    check("embedding_gather_scale", embedding_check)
    def sampling_check():
        from zluda_ops.sampling import sample_logits

        torch.manual_seed(0)
        logits = torch.randn(151936, device="cuda") * 2.0
        greedy = sample_logits(logits, temperature=0.0)
        assert int(greedy) == int(logits.argmax()), "greedy mismatch"
        topk = torch.topk(logits, 50).indices
        hits = 0
        for _ in range(40):
            tok = sample_logits(logits, temperature=1.0, top_k=50)
            hits += int(tok.item() in topk.tolist())
        us = bench(lambda: sample_logits(logits, temperature=1.0, top_k=50), n=20)
        return {"greedy_ok": True, "top50_hits": f"{hits}/40", "us": round(us, 1)}

    def int8_matmul_check():
        from zluda_ops import quant as Q

        torch.manual_seed(0)
        n, k, m = 2048, 2048, 64
        w = (torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.05)
        x = torch.randn(m, k, device="cuda", dtype=torch.float16)
        wi8, sc = Q.quantize_int8(w, 128)
        ref = (Q.dequantize_int8(wi8, sc, (n, k), 128) @ x.t()).t()
        got = Q.int8_matmul(x, wi8, sc)
        err = float((got.float() - ref.float()).abs().max())
        us = bench(lambda: Q.int8_matmul(x, wi8, sc), n=10)
        w_deq = Q.dequantize_int8(wi8, sc, (n, k), 128)
        ref_us = bench(lambda: x @ w_deq.t(), n=10)
        return {"err": round(err, 3), "int8_us": round(us, 1),
                "dequant_fp16_us": round(ref_us, 1)}

    check("sampling_kernel", sampling_check)
    check("int8_prefill_matmul", int8_matmul_check)


    check("qkv_rope_append_correctness", qkv_rope_append_check)
    ok = sum(1 for r in RESULTS if r["status"] == "OK")
    print(f"SUMMARY {ok}/{len(RESULTS)} phase-2 kernel checks passed", flush=True)
    out = Path(__file__).resolve().parents[1] / "bench" / "results" / "phase2_kernels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": RESULTS}, indent=1), encoding="utf-8")
    print(f"report: {out}", flush=True)
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

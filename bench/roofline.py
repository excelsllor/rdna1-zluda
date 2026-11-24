import argparse
import sys

import torch
import triton
import triton.language as tl

from common import bench, nvml_sample, save_report

PEAK_FP32_TF = 9.754
PEAK_FP16_TF = 19.51
PEAK_BW_GBS = 448.0

DEV = "cuda"


@triton.jit
def _alu_f32(out_ptr, a, b, ITER: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc0 = tl.load(out_ptr + offs)
    acc1 = acc0 * 0.5
    acc2 = acc0 * 0.25
    acc3 = acc0 * 0.125
    for _ in range(ITER):
        acc0 = acc0 * a + b
        acc1 = acc1 * a + b
        acc2 = acc2 * a + b
        acc3 = acc3 * a + b
    tl.store(out_ptr + offs, acc0 + acc1 + acc2 + acc3)


@triton.jit
def _alu_f16(out_ptr, a_ptr, b_ptr, ITER: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    acc0 = tl.load(out_ptr + offs)
    acc1 = acc0 * 0.5
    acc2 = acc0 * 0.25
    acc3 = acc0 * 0.125
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    for _ in range(ITER):
        acc0 = acc0 * a + b
        acc1 = acc1 * a + b
        acc2 = acc2 * a + b
        acc3 = acc3 * a + b
    tl.store(out_ptr + offs, acc0 + acc1 + acc2 + acc3)


@triton.jit
def _chase(next_ptr, out_ptr, N: tl.constexpr):
    idx = tl.zeros((1,), tl.int32)
    for _ in range(N):
        idx = tl.load(next_ptr + idx)
    tl.store(out_ptr + tl.arange(0, 1), idx)


@triton.jit
def _cache_load(data_ptr, out_ptr, ITER: tl.constexpr, BLOCK: tl.constexpr, SIZE: tl.constexpr):
    pid = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for i in range(ITER):
        offs = (lanes + i * BLOCK) % SIZE
        acc += tl.load(data_ptr + offs)
    tl.store(out_ptr + pid * BLOCK + lanes, acc)


def alu_test(dtype, iters=20):
    block = 256
    grid = 2048
    n = block * grid
    out = torch.randn(n, device=DEV, dtype=dtype) * 1e-3
    if dtype == torch.float16:
        a = torch.tensor(1.0001, device=DEV, dtype=dtype)
        b = torch.tensor(1e-5, device=DEV, dtype=dtype)
        fn = lambda: _alu_f16[(grid,)](out, a, b, ITER=2048, BLOCK=block, num_warps=4)
    else:
        fn = lambda: _alu_f32[(grid,)](out, 1.0001, 1e-5, ITER=2048, BLOCK=block, num_warps=4)
    ms = bench(fn, warmup=2, iters=iters)
    flops = 2.0 * 4 * grid * block * 2048
    tflops = flops / (ms / 1000.0) / 1e12
    return {"ms": ms, "tflops": tflops}


def memory_test(dtype, elems=32 * 1024 * 1024, iters=15):
    itemsize = torch.tensor([], dtype=dtype).element_size()
    src = torch.randn(elems, device=DEV, dtype=dtype)
    dst = torch.empty_like(src)
    other = torch.randn_like(src)
    third = torch.randn_like(src)
    out = torch.empty_like(src)
    cases = []

    def add_case(name, fn, traffic):
        ms = bench(fn, warmup=2, iters=iters)
        gbs = traffic / (ms / 1000.0) / 1e9
        cases.append({"name": name, "ms": ms, "gbs": gbs, "pct_peak": gbs / PEAK_BW_GBS * 100.0})

    add_case("copy", lambda: dst.copy_(src), elems * itemsize * 2)
    add_case("add", lambda: torch.add(src, other, out=out), elems * itemsize * 3)
    add_case("addcmul", lambda: torch.addcmul(src, other, third, out=out), elems * itemsize * 4)
    add_case("sum", lambda: torch.sum(src), elems * itemsize)
    return cases


def latency_test(size=1 << 23, chain=512, iters=20):
    nxt = torch.randperm(size, dtype=torch.int32, device="cpu").to(DEV)
    out = torch.zeros(1, dtype=torch.int32, device=DEV)
    fn = lambda: _chase[(1,)](nxt, out, N=chain, num_warps=1)
    ms = bench(fn, warmup=3, iters=iters)
    return {"ms": ms, "ns_per_load": ms * 1e6 / chain, "chain": chain, "buffer_mb": size * 4 / 1024**2}


def cache_test(iters=256, block=256, grid=1024, size=8192):
    data = torch.randn(size, device=DEV, dtype=torch.float32)
    out = torch.empty(grid * block, device=DEV, dtype=torch.float32)
    fn = lambda: _cache_load[(grid,)](data, out, ITER=iters, BLOCK=block, SIZE=size, num_warps=4)
    ms = bench(fn, warmup=2, iters=15)
    traffic = grid * block * iters * 4
    gbs = traffic / (ms / 1000.0) / 1e9
    return {"ms": ms, "gbs": gbs, "working_set_kb": size * 4 / 1024, "iters": iters}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    report: dict = {"peak": {"fp32_tf": PEAK_FP32_TF, "fp16_tf": PEAK_FP16_TF, "bw_gbs": PEAK_BW_GBS}}

    print("== ALU ==")
    report["alu"] = {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        try:
            r = alu_test(dtype)
            report["alu"][name] = r
            print(f"{name}: {r['ms']:.3f} ms  {r['tflops']:.2f} TFLOPS")
        except Exception as exc:
            report["alu"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"{name}: FAIL {type(exc).__name__}: {str(exc)[:160]}")

    print("== memory ==")
    report["memory"] = {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        try:
            cases = memory_test(dtype)
            report["memory"][name] = cases
            for c in cases:
                print(f"{name} {c['name']:8s}: {c['ms']:8.3f} ms  {c['gbs']:7.1f} GB/s  {c['pct_peak']:5.1f}%")
        except Exception as exc:
            report["memory"][name] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"{name}: FAIL {type(exc).__name__}: {str(exc)[:160]}")

    print("== latency / cache ==")
    try:
        lat = latency_test()
        report["latency"] = lat
        print(f"chase: {lat['ns_per_load']:.1f} ns/load ({lat['ms']:.2f} ms, {lat['buffer_mb']:.0f} MB)")
    except Exception as exc:
        report["latency"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        print(f"chase: FAIL {type(exc).__name__}: {str(exc)[:160]}")
    try:
        cache = cache_test()
        report["cache"] = cache
        print(f"cache load: {cache['gbs']:.1f} GB/s ({cache['working_set_kb']:.0f} KB working set)")
    except Exception as exc:
        report["cache"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        print(f"cache: FAIL {type(exc).__name__}: {str(exc)[:160]}")

    report["monitor"] = nvml_sample()
    path = save_report("roofline", report)
    if args.json:
        print(path)


if __name__ == "__main__":
    main()


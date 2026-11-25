#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
template <int WARPS>
__global__ void gemv_fp16_kernel(const __half* __restrict__ x,
                                 const __half* __restrict__ W,
                                 __half* __restrict__ y,
                                 int N, int K) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * WARPS + warp;
    if (row >= N) return;
    const __half* w = W + (long)row * K;
    float acc = 0.f;
    for (int k = lane * 2; k < K; k += 64) {
        __half2 xh = *reinterpret_cast<const __half2*>(x + k);
        __half2 wh = *reinterpret_cast<const __half2*>(w + k);
        float2 xf = __half22float2(xh);
        float2 wf = __half22float2(wh);
        acc += xf.x * wf.x + xf.y * wf.y;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0) y[row] = __float2half(acc);
}
extern "C" __declspec(dllexport) int hip_gemv_bench(const void* x, const void* W, void* y,
                                                    int N, int K, int iters) {
    const int warps = 4;
    for (int it = 0; it < iters; ++it) {
        gemv_fp16_kernel<4><<<dim3((N + warps - 1) / warps), dim3(32 * warps), 0, 0>>>(
            (const __half*)x, (const __half*)W, (__half*)y, N, K);
    }
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}
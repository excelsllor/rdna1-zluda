"""CUDA-graph decode helper for ZLUDA (Qwen-like HF causal LMs).

The decode step of a HF causal LM is a fixed sequence of ~240 small ops; on
ZLUDA it is bound by CPU dispatch and small-GEMV inefficiency. When
`zluda_ops.auto` is installed, `F.linear` uses the Triton GEMV kernels and
`F.scaled_dot_product_attention` uses Triton FA/flash-decoding, which removes
cuBLAS/cuDNN from the step and makes it graph-capture safe. This class captures
one decode step with `torch.cuda.CUDAGraph` and replays it, updating the static
token and cache-position buffers outside the graph (the self-loop variant that
updates them inside the graph is much slower and not correct on ZLUDA).

Requirements:
  * `import zluda_ops.auto` before creating the decoder (patches F.linear/F.sdpa)
  * fixed maximum sequence length (`StaticCache`)
  * the model must be in eval() and fp16 on CUDA

Usage:
    import zluda_ops.auto  # noqa: F401  (installs the kernels)
    dec = GraphDecoder(model, prompt_ids, max_new_tokens=64)
    tokens = dec.generate(64)          # Python list of token ids
    print(dec.tokens_per_second)       # graph-replay decode throughput
"""

import time

import torch


class GraphDecoder:
    def __init__(self, model, prompt_ids, max_new_tokens=64, dtype=torch.float16):
        from transformers.cache_utils import StaticCache

        self.model = model
        self.dtype = dtype
        self.prompt_len = int(prompt_ids.shape[1])
        self.max_new = int(max_new_tokens)
        max_len = self.prompt_len + self.max_new + 8
        self.cache = StaticCache(config=model.config, max_batch_size=1,
                                 max_cache_len=max_len, device=prompt_ids.device,
                                 dtype=dtype)
        pos = torch.arange(self.prompt_len, device=prompt_ids.device,
                           dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=prompt_ids, past_key_values=self.cache,
                        cache_position=pos, use_cache=True, logits_to_keep=1)
        self.static_tok = out.logits[:, -1:].argmax(-1).clone()
        self.static_pos = torch.tensor([self.prompt_len],
                                       device=prompt_ids.device, dtype=torch.long)
        self._capture()
        self.tokens_per_second = None

    def _step(self):
        with torch.no_grad():
            return self.model(input_ids=self.static_tok,
                              past_key_values=self.cache,
                              cache_position=self.static_pos,
                              use_cache=True, logits_to_keep=1)

    def _capture(self):
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            self._step()  # warmup on the capture stream
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            with torch.cuda.graph(self.graph):
                self.graph_out = self._step()
        torch.cuda.synchronize()
        self.graph.replay()
        torch.cuda.synchronize()

    def generate(self, max_new_tokens=None):
        n = int(max_new_tokens or self.max_new)
        out = []
        t0 = time.perf_counter()
        for i in range(n):
            self.graph.replay()
            nxt = self.graph_out.logits[:, -1:].argmax(-1)
            tok = int(nxt)  # D2H sync
            out.append(tok)
            self.static_tok.copy_(nxt)
            self.static_pos.fill_(self.prompt_len + i + 1)
        torch.cuda.synchronize()
        self.tokens_per_second = n / (time.perf_counter() - t0)
        return out

    def replay_benchmark(self, steps=None):
        """Replay without reading tokens (pure throughput of the graph step)."""
        n = int(steps or self.max_new)
        self.graph.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            self.graph.replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
        self.tokens_per_second = 1000 / ms
        return ms


__all__ = ["GraphDecoder"]

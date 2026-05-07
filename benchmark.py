"""
Benchmark: FlashAttention-1 vs FlashAttention-2 vs naive vs PyTorch SDPA.

Run: python benchmark.py
"""

import math
import time
import torch
import torch.nn.functional as F
from flash_attn import flash_attention, flash_attention_v2, naive_attention_forward

CONFIGS = [
    (1, 12, 512, 64),
    (1, 16, 1024, 64),
]

# 注释掉大配置，先用小的验证；需要测完整时取消注释
# CONFIGS_FULL = [
#     (1, 16, 2048, 64),
#     (1, 16, 4096, 64),
#     (4, 32, 1024, 64),
#     (8, 32, 512, 64),
# ]


def bench(fn, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / 10 * 1000  # ms


def bench_backward(fn, Q, K, V, sm_scale):
    def run():
        Q1 = Q.clone().requires_grad_()
        K1 = K.clone().requires_grad_()
        V1 = V.clone().requires_grad_()
        O = fn(Q1, K1, V1, sm_scale)
        O.sum().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / 10 * 1000  # ms


def main():
    print("=" * 100)
    print("FlashAttention-1 / FlashAttention-2 Benchmark")
    print("=" * 100)
    header = (f"{'Config':>25s}  {'Naive Fwd':>10s}  {'FA1 Fwd':>10s}  "
              f"{'FA2 Fwd':>10s}  {'SDPA Fwd':>10s}  "
              f"{'FA1 Bwd':>10s}  {'FA2 Bwd':>10s}  {'SDPA Bwd':>10s}")
    print(header)
    print("-" * 100)

    for B, H, N, d in CONFIGS:
        cfg = f"B={B} H={H} N={N} d={d}"
        print(f"{cfg:>25s}  ...", end="\r")
        torch.manual_seed(42)
        Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
        sm_scale = 1.0 / math.sqrt(d)

        try:
            t_naive_fwd = bench(naive_attention_forward, Q, K, V, sm_scale)
        except torch.cuda.OutOfMemoryError:
            t_naive_fwd = float("nan")

        try:
            t_fa1_fwd = bench(flash_attention, Q, K, V, sm_scale)
        except Exception:
            t_fa1_fwd = float("nan")

        try:
            t_fa2_fwd = bench(flash_attention_v2, Q, K, V, sm_scale)
        except Exception:
            t_fa2_fwd = float("nan")

        try:
            t_sdpa_fwd = bench(lambda q,k,v: F.scaled_dot_product_attention(q,k,v, scale=sm_scale), Q, K, V)
        except Exception:
            t_sdpa_fwd = float("nan")

        try:
            t_fa1_bwd = bench_backward(flash_attention, Q, K, V, sm_scale)
        except Exception as e:
            print(f"FA1 BWD error: {e}")
            t_fa1_bwd = float("nan")

        try:
            t_fa2_bwd = bench_backward(flash_attention_v2, Q, K, V, sm_scale)
        except Exception as e:
            print(f"FA2 BWD error: {e}")
            t_fa2_bwd = float("nan")

        try:
            t_sdpa_bwd = bench_backward(
                lambda q,k,v,sm: F.scaled_dot_product_attention(q,k,v, scale=sm),
                Q, K, V, sm_scale)
        except Exception as e:
            print(f"SDPA BWD error: {e}")
            t_sdpa_bwd = float("nan")

        cfg = f"B={B} H={H} N={N} d={d}"
        print(f"{cfg:>25s}  {t_naive_fwd:8.1f}ms  {t_fa1_fwd:8.1f}ms  "
              f"{t_fa2_fwd:8.1f}ms  {t_sdpa_fwd:8.1f}ms  "
              f"{t_fa1_bwd:8.1f}ms  {t_fa2_bwd:8.1f}ms  {t_sdpa_bwd:8.1f}ms")

    print("-" * 100)
    print("FA1  = FlashAttention-1 (Q-tile outer forward, Q-tile outer backward)")
    print("FA2  = FlashAttention-2 (Q-tile outer forward, KV-tile outer backward)")
    print("SDPA = torch.nn.functional.scaled_dot_product_attention (CUDA kernel)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""SM120 FlashAttention benchmark — throughput, latency, roofline efficiency.

Compares:
  1. SM120 FMHA BF16 (our HMMA kernel)
  2. SM120 FMHA FP8 (our QMMA.SF kernel)
  3. PyTorch SDPA (torch.nn.functional.scaled_dot_product_attention)

Metrics:
  - Latency (ms)
  - TFLOPS (achieved)
  - % of peak TFLOPS
  - Memory bandwidth utilization (for decode-like configs)

Usage:
  CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_sm120.py
"""

import sys
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Load SM120 extension
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sm120_api", str(Path(__file__).parent.parent / "python" / "fmha_sm100" / "sm120_api.py"))
sm120_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm120_mod)
fmha_sm120 = sm120_mod.fmha_sm120


# ─── Hardware constants for SM120 ───
# RTX 5090: 170 SMs, BF16 tensor = ~209 TFLOPS, FP8 tensor = ~419 TFLOPS
# RTX PRO 6000: 188 SMs, BF16 tensor = ~231 TFLOPS
SM120_BF16_PEAK_TFLOPS = 209.0   # RTX 5090
SM120_FP8_PEAK_TFLOPS = 419.0    # RTX 5090 (QMMA.SF throughput)
SM120_HBM_BW_GBS = 1792.0        # RTX 5090 GDDR7 bandwidth


def attention_flops(seq_q, seq_k, head_dim, num_heads):
    """FLOPs for standard attention: 2*S*K*D (QK) + 2*S*K*D (PV)."""
    return 2 * 2 * seq_q * seq_k * head_dim * num_heads


def attention_bytes(seq_q, seq_k, head_dim, num_heads, elem_bytes):
    """Bytes loaded: Q + K + V + O."""
    return (seq_q + 2 * seq_k + seq_q) * num_heads * head_dim * elem_bytes


def bench_fn(fn, warmup=5, repeats=20):
    """Benchmark function, return median ms."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.median(times)


def bench_config(seq_q, seq_k, num_heads_q, num_heads_kv, head_dim=128,
                 dtype_name="bf16", repeats=20):
    """Benchmark one configuration."""
    scale = 1.0 / math.sqrt(head_dim)

    if dtype_name == "bf16":
        dtype = torch.bfloat16
        elem_bytes = 2
        peak_tflops = SM120_BF16_PEAK_TFLOPS
    elif dtype_name == "fp8":
        dtype = torch.float8_e4m3fn
        elem_bytes = 1
        peak_tflops = SM120_FP8_PEAK_TFLOPS
    else:
        raise ValueError(f"Unknown dtype: {dtype_name}")

    # Allocate tensors
    q = torch.randn(seq_q, num_heads_q, head_dim, device="cuda").to(dtype)
    k = torch.randn(seq_k, num_heads_kv, head_dim, device="cuda").to(dtype)
    v = torch.randn(seq_k, num_heads_kv, head_dim, device="cuda").to(dtype)

    # --- SM120 kernel ---
    sm120_ms = bench_fn(lambda: fmha_sm120(q, k, v, softmax_scale=scale), repeats=repeats)

    # --- PyTorch SDPA (only for BF16, needs 4D input) ---
    sdpa_ms = None
    if dtype_name == "bf16":
        # SDPA expects [B, H, S, D]
        q4d = q.unsqueeze(0).transpose(1, 2)  # [1, Hq, Sq, D]
        k4d = k.unsqueeze(0).transpose(1, 2)  # [1, Hkv, Sk, D]
        v4d = v.unsqueeze(0).transpose(1, 2)
        if num_heads_q != num_heads_kv:
            # GQA: expand KV
            repeat = num_heads_q // num_heads_kv
            k4d = k4d.repeat(1, repeat, 1, 1)
            v4d = v4d.repeat(1, repeat, 1, 1)
        try:
            sdpa_ms = bench_fn(
                lambda: F.scaled_dot_product_attention(q4d, k4d, v4d, scale=scale),
                repeats=repeats)
        except Exception:
            sdpa_ms = None

    # Compute metrics
    flops = attention_flops(seq_q, seq_k, head_dim, num_heads_q)
    tflops = flops / (sm120_ms * 1e-3) / 1e12
    pct_peak = tflops / peak_tflops * 100
    bytes_total = attention_bytes(seq_q, seq_k, head_dim, num_heads_q, elem_bytes)
    bw_gbs = bytes_total / (sm120_ms * 1e-3) / 1e9

    return {
        "sm120_ms": sm120_ms,
        "sdpa_ms": sdpa_ms,
        "tflops": tflops,
        "pct_peak": pct_peak,
        "bw_gbs": bw_gbs,
        "speedup_vs_sdpa": (sdpa_ms / sm120_ms) if sdpa_ms else None,
    }


def main():
    if not torch.cuda.is_available():
        print("No CUDA!"); return

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (sm_{props.major}{props.minor})")
    print(f"SMs: {props.multi_processor_count}")
    print()

    # ─── Prefill configs (compute-bound) ───
    print("=" * 80)
    print(f"{'CONFIG':<30} {'SM120 ms':>9} {'SDPA ms':>9} {'TFLOPS':>8} "
          f"{'%peak':>6} {'BW GB/s':>8} {'speedup':>8}")
    print("=" * 80)

    prefill_configs = [
        # (seq_q, seq_k, hq, hkv, dtype, label)
        (512, 512, 32, 32, "bf16", "prefill 512 bf16"),
        (1024, 1024, 32, 32, "bf16", "prefill 1K bf16"),
        (2048, 2048, 32, 32, "bf16", "prefill 2K bf16"),
        (4096, 4096, 32, 32, "bf16", "prefill 4K bf16"),
        (1024, 1024, 64, 8, "bf16", "GQA 64/8 1K bf16"),
        (512, 512, 32, 32, "fp8", "prefill 512 fp8"),
        (1024, 1024, 32, 32, "fp8", "prefill 1K fp8"),
        (2048, 2048, 32, 32, "fp8", "prefill 2K fp8"),
        (4096, 4096, 32, 32, "fp8", "prefill 4K fp8"),
    ]

    for sq, sk, hq, hkv, dt, label in prefill_configs:
        try:
            r = bench_config(sq, sk, hq, hkv, dtype_name=dt, repeats=15)
            sdpa_str = f"{r['sdpa_ms']:9.3f}" if r['sdpa_ms'] else "      N/A"
            spd_str = f"{r['speedup_vs_sdpa']:8.2f}x" if r['speedup_vs_sdpa'] else "     N/A"
            print(f"{label:<30} {r['sm120_ms']:9.3f} {sdpa_str} {r['tflops']:8.1f} "
                  f"{r['pct_peak']:5.1f}% {r['bw_gbs']:8.1f} {spd_str}")
        except Exception as e:
            print(f"{label:<30} FAILED: {e}")
        torch.cuda.empty_cache()

    # ─── Decode configs (bandwidth-bound) ───
    print()
    print("=" * 80)
    print("Decode-like (batch=1, long KV cache)")
    print("=" * 80)

    decode_configs = [
        (1, 4096, 32, 32, "bf16", "decode 4K bf16"),
        (1, 8192, 32, 32, "bf16", "decode 8K bf16"),
        (1, 16384, 32, 32, "bf16", "decode 16K bf16"),
        (1, 4096, 32, 32, "fp8", "decode 4K fp8"),
        (1, 8192, 32, 32, "fp8", "decode 8K fp8"),
        (1, 16384, 32, 32, "fp8", "decode 16K fp8"),
    ]

    print(f"{'CONFIG':<30} {'SM120 ms':>9} {'SDPA ms':>9} {'BW GB/s':>8} "
          f"{'%HBM':>6} {'speedup':>8}")
    for sq, sk, hq, hkv, dt, label in decode_configs:
        try:
            r = bench_config(sq, sk, hq, hkv, dtype_name=dt, repeats=20)
            sdpa_str = f"{r['sdpa_ms']:9.3f}" if r['sdpa_ms'] else "      N/A"
            pct_hbm = r['bw_gbs'] / SM120_HBM_BW_GBS * 100
            spd_str = f"{r['speedup_vs_sdpa']:8.2f}x" if r['speedup_vs_sdpa'] else "     N/A"
            print(f"{label:<30} {r['sm120_ms']:9.3f} {sdpa_str} {r['bw_gbs']:8.1f} "
                  f"{pct_hbm:5.1f}% {spd_str}")
        except Exception as e:
            print(f"{label:<30} FAILED: {e}")
        torch.cuda.empty_cache()

    print()
    print("Notes:")
    print(f"  SM120 BF16 peak: {SM120_BF16_PEAK_TFLOPS} TFLOPS")
    print(f"  SM120 FP8 peak:  {SM120_FP8_PEAK_TFLOPS} TFLOPS (QMMA.SF)")
    print(f"  SM120 HBM BW:    {SM120_HBM_BW_GBS} GB/s")
    print(f"  SDPA = torch.nn.functional.scaled_dot_product_attention")


if __name__ == "__main__":
    main()

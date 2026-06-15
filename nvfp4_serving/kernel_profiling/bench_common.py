# SPDX-License-Identifier: MIT
"""Shared utilities for the SM120-vs-Triton MSA head-to-head microbenchmarks.

CUDA-event timing helper (warmup + many iters, median), and the M3 shape config.
Run inside vllm/vllm-openai:minimax-m3 on CUDA_VISIBLE_DEVICES=0.
"""
from __future__ import annotations

import statistics
import torch

# ---- Real MiniMax-M3 shapes (from /models/MiniMax-M3-NVFP4/config.json) ----
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
INDEX_N_HEADS = 4
INDEX_HEAD_DIM = 128
INDEX_BLOCK_SIZE = 128       # == SPARSE_BLOCK_SIZE
INDEX_TOPK_BLOCKS = 16
INDEX_LOCAL_BLOCKS = 1
INDEX_INIT_BLOCKS = 0
ROPE_THETA = 5_000_000.0
PARTIAL_ROTARY_FACTOR = 0.5
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)   # 64
RMS_EPS = 1e-6
SCALE = HEAD_DIM ** -0.5     # 1/sqrt(128)
INDEX_SCALE = INDEX_HEAD_DIM ** -0.5

DTYPE = torch.bfloat16


def cuda_time(fn, *, warmup=20, iters=100, sync_every=False):
    """Median kernel time (ms) over `iters`, measured with CUDA events.

    Returns (median_ms, p10_ms, p90_ms). `fn` is called with no args; it must
    enqueue exactly the work to be timed. We measure each iteration with its own
    event pair so launch overhead is included per-call (we report this as the
    'wall-ish' kernel time). For a pure kernel-only measurement we also batch a
    block of launches between a single event pair (see cuda_time_batched).
    """
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    med = statistics.median(times)
    p10 = times[max(0, int(0.10 * len(times)) - 1)]
    p90 = times[min(len(times) - 1, int(0.90 * len(times)))]
    return med, p10, p90


def cuda_time_batched(fn, *, warmup=20, reps=50, batch=20):
    """Kernel-only time (ms/call): batch `batch` launches between one event pair.

    Amortizes per-launch host overhead across `batch` back-to-back enqueues, so
    the result is dominated by on-GPU time. Returns (median_ms_per_call,).
    """
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    per_call = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        start.record()
        for _ in range(batch):
            fn()
        end.record()
        end.synchronize()
        per_call.append(start.elapsed_time(end) / batch)
    per_call.sort()
    return statistics.median(per_call)

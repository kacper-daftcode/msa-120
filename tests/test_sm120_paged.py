"""SM120 block-sparse PAGED-KV FA2 forward — exact equivalence vs contiguous.

The paged kernel gathers KV from a page pool under a random page permutation.
Given identical underlying data (just gathered differently, no extra
quantization), paged output MUST equal contiguous output to rms ~0.
"""
import os
import math
import torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")

_CFLAGS = [
    "-gencode=arch=compute_120f,code=sm_120f",
    "-O3", "-std=c++17", "--expt-relaxed-constexpr",
]
_LDFLAGS = [
    "-L/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib",
    "-lcudart",
]

print("Building contiguous sparse extension...")
contig = load(
    name="sm120_fmha_sparse_contig",
    sources=[os.path.join(_CSRC, "sm120_fmha_sparse.cu")],
    extra_cuda_cflags=_CFLAGS, extra_ldflags=_LDFLAGS, verbose=False,
)
print("Building paged extension...")
paged = load(
    name="sm120_fmha_paged",
    sources=[os.path.join(_CSRC, "sm120_fmha_paged.cu")],
    extra_cuda_cflags=_CFLAGS, extra_ldflags=_LDFLAGS, verbose=False,
)
print("Extensions built.\n")

BLK_N = 64
PAGE_SIZE = 64
DEV = "cuda"


def rms(a, b):
    a = a.float(); b = b.float()
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def maxabs(a, b):
    return (a.float() - b.float()).abs().max().item()


def build_block_ids(num_m_blocks, seq_q, num_kv_blocks, topk, per_query, gen):
    """block_ids of selected LOGICAL blocks; -1 padding allowed."""
    rows = seq_q if per_query else num_m_blocks
    bids = torch.full((rows, topk), -1, dtype=torch.int32, device=DEV)
    for r in range(rows):
        k = min(topk, num_kv_blocks)
        perm = torch.randperm(num_kv_blocks, generator=gen, device=DEV)[:k]
        bids[r, :k] = perm.to(torch.int32)
    return bids


def scatter_to_pages(k, v, num_kv_blocks, num_heads_kv, gen):
    """Scatter contiguous KV blocks into a page pool under a RANDOM permutation.

    Returns (k_cache, v_cache, block_table_row) where
    k_cache: [num_pages, PAGE_SIZE, Hkv, 128]; logical block b lives in
    physical page perm[b].
    """
    seq_k = k.size(0)
    # extra unused pages to make the test non-trivial
    num_pages = num_kv_blocks + 3
    perm = torch.randperm(num_pages, generator=gen, device=DEV)[:num_kv_blocks]
    k_cache = torch.randn(num_pages, PAGE_SIZE, num_heads_kv, 128,
                          dtype=torch.bfloat16, device=DEV)
    v_cache = torch.randn_like(k_cache)
    for b in range(num_kv_blocks):
        page = perm[b].item()
        tok0 = b * BLK_N
        tok1 = min(tok0 + BLK_N, seq_k)
        n = tok1 - tok0
        k_cache[page, :n] = k[tok0:tok1]
        v_cache[page, :n] = v[tok0:tok1]
        # remaining rows of the partial last page stay random (masked out)
    return k_cache, v_cache, perm.to(torch.int32)


def run_case(name, seq_q, seq_k, hq, hkv, topk, causal, per_query, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    scale = 1.0 / math.sqrt(128)

    q = torch.randn(seq_q, hq, 128, dtype=torch.bfloat16, device=DEV, generator=gen)
    k = torch.randn(seq_k, hkv, 128, dtype=torch.bfloat16, device=DEV, generator=gen)
    v = torch.randn(seq_k, hkv, 128, dtype=torch.bfloat16, device=DEV, generator=gen)

    num_m_blocks = (seq_q + 63) // 64
    num_kv_blocks = (seq_k + BLK_N - 1) // BLK_N

    bids = build_block_ids(num_m_blocks, seq_q, num_kv_blocks, topk, per_query, gen)

    # contiguous reference
    o_c, lse_c = contig.forward_sparse(q, k, v, bids, scale, causal, BLK_N)

    # paged: scatter blocks to pages, build block_table
    k_cache, v_cache, perm = scatter_to_pages(k, v, num_kv_blocks, hkv, gen)
    # block_table[m_blk, logical_block] = physical page. Same mapping per row.
    block_table = perm.unsqueeze(0).repeat(num_m_blocks, 1).contiguous()

    o_p, lse_p = paged.forward_sparse_paged(
        q, k_cache, v_cache, block_table, bids, scale, causal, seq_k)

    r_o = rms(o_p, o_c)
    m_o = maxabs(o_p, o_c)
    # LSE only meaningful where rows attend something; compare finite entries
    finite = torch.isfinite(lse_c) & torch.isfinite(lse_p)
    r_l = rms(lse_p[finite], lse_c[finite]) if finite.any() else 0.0

    ok = r_o < 1e-3
    tag = "PASS" if ok else "FAIL"
    pq = "per-query" if per_query else "per-tile "
    print(f"[{tag}] {name:28s} Sq={seq_q:4d} Sk={seq_k:4d} hq={hq} hkv={hkv} "
          f"topk={topk} causal={int(causal)} {pq} | "
          f"O rms={r_o:.3e} maxabs={m_o:.3e}  LSE rms={r_l:.3e}")
    return ok


def main():
    torch.manual_seed(0)
    cases = [
        # name, Sq, Sk, hq, hkv, topk, causal, per_query, seed
        ("mha_basic",        128, 512, 4, 4, 4, False, False, 1),
        ("mha_causal",       128, 512, 4, 4, 4, True,  False, 2),
        ("gqa_basic",        128, 512, 8, 2, 4, False, False, 3),
        ("gqa_causal",       128, 512, 8, 2, 4, True,  False, 4),
        ("perquery_mha",     128, 512, 4, 4, 3, False, True,  5),
        ("perquery_gqa_caus", 128, 512, 8, 2, 3, True,  True,  6),
        ("partial_last_blk", 100, 500, 4, 4, 5, False, False, 7),  # Sk=500 -> last blk 60 tok
        ("partial_causal",    96, 460, 8, 2, 4, True,  False, 8),
        ("partial_perquery",  96, 470, 4, 2, 3, False, True,  9),
        ("single_block",      64,  64, 4, 4, 1, False, False, 10),
        ("multi_mblk",       192, 768, 8, 4, 6, True,  False, 11),
        ("multi_mblk_pq",    192, 768, 4, 4, 4, False, True,  12),
    ]
    results = []
    for c in cases:
        results.append(run_case(*c))
    print()
    n_pass = sum(results)
    print(f"{n_pass}/{len(results)} cases PASS")
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

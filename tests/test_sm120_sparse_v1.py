"""SM120 block-sparse FA2 forward — incremental MSA-faithful tests.

Step 1: causal masking
Step 2: per-query block selection
Step 3: BLK_KV=128
"""
import os, math, sys, torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")
print("Building SM120 sparse FMHA extension v1 (JIT)...")
ext = load(name="sm120_fmha_sparse_v1",
           sources=[os.path.join(_CSRC, "sm120_fmha_sparse.cu")],
           extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f", "-O3", "-std=c++17",
                              "--expt-relaxed-constexpr"],
           verbose=False)
print("Extension built.\n")
BLK = 64


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def ref_sparse(q, k, v, block_ids, scale, causal=False, blk_kv=BLK):
    """Per-TILE block selection. block_ids: [num_m_blocks, topk]."""
    S, H, D = q.shape; Sk = k.shape[0]
    nm, topk = block_ids.shape
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros_like(qf)
    for h in range(H):
        sc = (qf[:, h, :] @ kf[:, h, :].T) * scale          # [S, Sk]
        mask = torch.zeros(S, Sk, dtype=torch.bool, device=q.device)
        for m in range(nm):
            q0, q1 = m * BLK, min((m + 1) * BLK, S)
            for b in block_ids[m].tolist():
                if b < 0: continue
                k0, k1 = b * blk_kv, min((b + 1) * blk_kv, Sk)
                mask[q0:q1, k0:k1] = True
        if causal:
            qpos = torch.arange(S, device=q.device).view(S, 1)
            kpos = torch.arange(Sk, device=q.device).view(1, Sk)
            mask = mask & (kpos <= qpos)
        sc = sc.masked_fill(~mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(sc, dim=-1), 0.0)
        out[:, h, :] = attn @ vf[:, h, :]
    return out


def ref_sparse_perquery(q, k, v, block_ids, scale, causal=False, blk_kv=BLK):
    """Per-QUERY block selection. block_ids: [seq_q, topk]."""
    S, H, D = q.shape; Sk = k.shape[0]
    sq, topk = block_ids.shape
    assert sq == S, "per-query block_ids must have one row per query"
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros_like(qf)
    for h in range(H):
        sc = (qf[:, h, :] @ kf[:, h, :].T) * scale          # [S, Sk]
        mask = torch.zeros(S, Sk, dtype=torch.bool, device=q.device)
        for r in range(S):
            for b in block_ids[r].tolist():
                if b < 0: continue
                k0, k1 = b * blk_kv, min((b + 1) * blk_kv, Sk)
                mask[r, k0:k1] = True
        if causal:
            qpos = torch.arange(S, device=q.device).view(S, 1)
            kpos = torch.arange(Sk, device=q.device).view(1, Sk)
            mask = mask & (kpos <= qpos)
        sc = sc.masked_fill(~mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(sc, dim=-1), 0.0)
        out[:, h, :] = attn @ vf[:, h, :]
    return out


def _mk_qkv(S, Sk, H, Hkv, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, H, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Sk, Hkv, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Sk, Hkv, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    return q, k, v, g


def _rms_maxabs(o, ref):
    diff = (o.float() - ref).abs()
    rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    return rms, diff.max().item()


# ---------------------------------------------------------------------------
# Baseline (existing behavior): per-tile, non-causal
# ---------------------------------------------------------------------------
def run_baseline(name, S, Sk, H, Hkv, topk, seed=0):
    q, k, v, g = _mk_qkv(S, Sk, H, Hkv, seed)
    scale = 1.0 / math.sqrt(128)
    nm = (S + BLK - 1) // BLK
    nkv = (Sk + BLK - 1) // BLK
    bids = torch.full((nm, topk), -1, dtype=torch.int32)
    for m in range(nm):
        perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
        bids[m, :perm.numel()] = perm.to(torch.int32)
    bids = bids.cuda()
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    o, lse = ext.forward_sparse(q, k, v, bids, scale)   # old signature still works
    ref = ref_sparse(q, k_e, v_e, bids, scale, causal=False)
    rms, maxabs = _rms_maxabs(o, ref)
    ok = rms < 0.05
    print(f"  {'PASS' if ok else 'FAIL'} {name:28} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}  "
          f"rms={rms:.4f} maxabs={maxabs:.4f}")
    return ok


# ---------------------------------------------------------------------------
# Step 1: causal, per-tile
# ---------------------------------------------------------------------------
def run_causal(name, S, Sk, H, Hkv, topk, seed=0):
    q, k, v, g = _mk_qkv(S, Sk, H, Hkv, seed)
    scale = 1.0 / math.sqrt(128)
    nm = (S + BLK - 1) // BLK
    nkv = (Sk + BLK - 1) // BLK
    bids = torch.full((nm, topk), -1, dtype=torch.int32)
    for m in range(nm):
        perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
        bids[m, :perm.numel()] = perm.to(torch.int32)
    bids = bids.cuda()
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    o, lse = ext.forward_sparse(q, k, v, bids, scale, True)   # causal=True
    ref = ref_sparse(q, k_e, v_e, bids, scale, causal=True)
    rms, maxabs = _rms_maxabs(o, ref)
    ok = rms < 0.05
    print(f"  {'PASS' if ok else 'FAIL'} {name:28} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}  "
          f"rms={rms:.4f} maxabs={maxabs:.4f}")
    return ok


# ---------------------------------------------------------------------------
# Step 2: per-query block selection
# ---------------------------------------------------------------------------
def run_perquery(name, S, Sk, H, Hkv, topk, seed=0, causal=False):
    q, k, v, g = _mk_qkv(S, Sk, H, Hkv, seed)
    scale = 1.0 / math.sqrt(128)
    nkv = (Sk + BLK - 1) // BLK
    # per-query: each row picks `topk` distinct blocks
    bids = torch.full((S, topk), -1, dtype=torch.int32)
    for r in range(S):
        perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
        bids[r, :perm.numel()] = perm.to(torch.int32)
    bids = bids.cuda()
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    o, lse = ext.forward_sparse(q, k, v, bids, scale, causal)
    ref = ref_sparse_perquery(q, k_e, v_e, bids, scale, causal=causal)
    rms, maxabs = _rms_maxabs(o, ref)
    ok = rms < 0.05
    cflag = "causal" if causal else "non-c "
    print(f"  {'PASS' if ok else 'FAIL'} {name:28} {cflag} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}  "
          f"rms={rms:.4f} maxabs={maxabs:.4f}")
    return ok


# ---------------------------------------------------------------------------
# Step 3: BLK_KV=128
# ---------------------------------------------------------------------------
def run_blk128(name, S, Sk, H, Hkv, topk, seed=0, causal=False, per_query=False):
    q, k, v, g = _mk_qkv(S, Sk, H, Hkv, seed)
    scale = 1.0 / math.sqrt(128)
    BLKV = 128
    nkv = (Sk + BLKV - 1) // BLKV
    if per_query:
        bids = torch.full((S, topk), -1, dtype=torch.int32)
        for r in range(S):
            perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
            bids[r, :perm.numel()] = perm.to(torch.int32)
    else:
        nm = (S + BLK - 1) // BLK
        bids = torch.full((nm, topk), -1, dtype=torch.int32)
        for m in range(nm):
            perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
            bids[m, :perm.numel()] = perm.to(torch.int32)
    bids = bids.cuda()
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    o, lse = ext.forward_sparse(q, k, v, bids, scale, causal, BLKV)
    if per_query:
        ref = ref_sparse_perquery(q, k_e, v_e, bids, scale, causal=causal, blk_kv=BLKV)
    else:
        ref = ref_sparse(q, k_e, v_e, bids, scale, causal=causal, blk_kv=BLKV)
    rms, maxabs = _rms_maxabs(o, ref)
    ok = rms < 0.05
    tag = ("pq " if per_query else "tile") + ("/causal" if causal else "/non-c ")
    print(f"  {'PASS' if ok else 'FAIL'} {name:24} {tag} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}b128  "
          f"rms={rms:.4f} maxabs={maxabs:.4f}")
    return ok


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = []

    if step in ("baseline", "all"):
        print("="*72); print("BASELINE (per-tile, non-causal) — old signature"); print("="*72)
        res.append(run_baseline("4 blocks, pick 3",  S=256, Sk=256, H=4, Hkv=4, topk=3, seed=1))
        res.append(run_baseline("8 blocks, pick 3",  S=512, Sk=512, H=4, Hkv=4, topk=3, seed=2))
        res.append(run_baseline("16 blk, pick 5",    S=256, Sk=1024, H=8, Hkv=8, topk=5, seed=3))
        res.append(run_baseline("GQA 16/4 pick 4",   S=256, Sk=512, H=16, Hkv=4, topk=4, seed=4))
        res.append(run_baseline("dense-equiv (all)", S=256, Sk=256, H=4, Hkv=4, topk=4, seed=5))

    if step in ("causal", "all"):
        print("\n" + "="*72); print("STEP 1: CAUSAL (per-tile)"); print("="*72)
        res.append(run_causal("4 blocks, pick 3",  S=256, Sk=256, H=4, Hkv=4, topk=3, seed=1))
        res.append(run_causal("8 blocks, pick 3",  S=512, Sk=512, H=4, Hkv=4, topk=3, seed=2))
        res.append(run_causal("16 blk, pick 5",    S=256, Sk=1024, H=8, Hkv=8, topk=5, seed=3))
        res.append(run_causal("GQA 16/4 pick 4",   S=256, Sk=512, H=16, Hkv=4, topk=4, seed=4))
        res.append(run_causal("dense-equiv (all)", S=256, Sk=256, H=4, Hkv=4, topk=4, seed=5))
        res.append(run_causal("non-square 320/256",S=320, Sk=256, H=4, Hkv=4, topk=4, seed=6))

    if step in ("perquery", "all"):
        print("\n" + "="*72); print("STEP 2: PER-QUERY block selection"); print("="*72)
        res.append(run_perquery("4 blocks, pick 3",  S=256, Sk=256, H=4, Hkv=4, topk=3, seed=1))
        res.append(run_perquery("8 blocks, pick 3",  S=512, Sk=512, H=4, Hkv=4, topk=3, seed=2))
        res.append(run_perquery("16 blk, pick 5",    S=256, Sk=1024, H=8, Hkv=8, topk=5, seed=3))
        res.append(run_perquery("GQA 16/4 pick 4",   S=256, Sk=512, H=16, Hkv=4, topk=4, seed=4))
        res.append(run_perquery("causal pq pick 3",  S=256, Sk=256, H=4, Hkv=4, topk=3, seed=7, causal=True))
        res.append(run_perquery("causal pq GQA",     S=256, Sk=512, H=16, Hkv=4, topk=4, seed=8, causal=True))

    if step in ("blk128", "all"):
        print("\n" + "="*72); print("STEP 3: BLK_KV=128"); print("="*72)
        res.append(run_blk128("4x128 blk pick2",  S=256, Sk=512, H=4, Hkv=4, topk=2, seed=11))
        res.append(run_blk128("8x128 blk pick3",  S=512, Sk=1024, H=8, Hkv=8, topk=3, seed=12))
        res.append(run_blk128("GQA 16/4 pick2",   S=256, Sk=512, H=16, Hkv=4, topk=2, seed=13))
        res.append(run_blk128("causal tile",      S=256, Sk=512, H=4, Hkv=4, topk=2, seed=14, causal=True))
        res.append(run_blk128("per-query",        S=256, Sk=512, H=4, Hkv=4, topk=2, seed=15, per_query=True))
        res.append(run_blk128("per-query causal", S=256, Sk=512, H=8, Hkv=4, topk=2, seed=16, per_query=True, causal=True))
        res.append(run_blk128("non-mult Sk=640",  S=256, Sk=640, H=4, Hkv=4, topk=3, seed=17))

    print("\n" + "="*72)
    print(">>> ALL PASSED <<<" if all(res) else ">>> FAILURES <<<")
    sys.exit(0 if all(res) else 1)

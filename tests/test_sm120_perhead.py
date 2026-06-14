"""SM120 block-sparse FA2 forward — per-(query,head) block selection.

Validates forward_sparse_perhead with block_ids shaped [seq_q, Hq, topk]
(golden indexer layout) and [Hq, seq_q, topk], against a torch reference that
does per-(query,head) masked attention. Also confirms backward compat:
 * old 2D per-tile / per-query signatures still work through forward_sparse
 * when all heads share the same blocks, the per-head path reduces to per-query.
"""
import os, math, sys, torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")
print("Building SM120 per-head sparse FMHA extension (JIT)...")
ext = load(name="sm120_fmha_perhead",
           sources=[os.path.join(_CSRC, "sm120_fmha_perhead.cu")],
           extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f", "-O3", "-std=c++17",
                              "--expt-relaxed-constexpr"],
           verbose=False)
print("Extension built.\n")
BLK = 64


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
def ref_sparse_perquery(q, k, v, block_ids, scale, causal=False, blk_kv=BLK):
    """Head-agnostic per-query. block_ids: [seq_q, topk]. k/v already GQA-expanded."""
    S, H, D = q.shape; Sk = k.shape[0]
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros_like(qf)
    for h in range(H):
        sc = (qf[:, h, :] @ kf[:, h, :].T) * scale
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


def ref_sparse_perhead(q, k, v, block_ids, scale, causal=False, blk_kv=BLK):
    """Per-(query,head). block_ids: [seq_q, Hq, topk]. k/v already GQA-expanded."""
    S, H, D = q.shape; Sk = k.shape[0]
    sq, hh, topk = block_ids.shape
    assert sq == S and hh == H, "block_ids must be [seq_q, Hq, topk]"
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros_like(qf)
    for h in range(H):
        sc = (qf[:, h, :] @ kf[:, h, :].T) * scale          # [S, Sk]
        mask = torch.zeros(S, Sk, dtype=torch.bool, device=q.device)
        for r in range(S):
            for b in block_ids[r, h].tolist():              # this head's own blocks
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


def _mk_perhead_bids(S, H, nkv, topk, g):
    """[seq_q, Hq, topk] int32, each (row,head) picks `topk` distinct blocks."""
    bids = torch.full((S, H, topk), -1, dtype=torch.int32)
    for r in range(S):
        for h in range(H):
            perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
            bids[r, h, :perm.numel()] = perm.to(torch.int32)
    return bids.cuda()


# ---------------------------------------------------------------------------
# Per-(query,head), golden layout [seq_q, Hq, topk]
# ---------------------------------------------------------------------------
def run_perhead(name, S, Sk, H, Hkv, topk, seed=0, causal=False, blk_kv=BLK):
    q, k, v, g = _mk_qkv(S, Sk, H, Hkv, seed)
    scale = 1.0 / math.sqrt(128)
    nkv = (Sk + blk_kv - 1) // blk_kv
    bids = _mk_perhead_bids(S, H, nkv, topk, g)             # [S, H, topk]
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    o, lse = ext.forward_sparse_perhead(q, k, v, bids, scale, causal, blk_kv)
    ref = ref_sparse_perhead(q, k_e, v_e, bids, scale, causal=causal, blk_kv=blk_kv)
    rms, maxabs = _rms_maxabs(o, ref)
    ok = rms < 0.05
    cflag = "causal" if causal else "non-c "
    bkv = f"b{blk_kv}"
    print(f"  {'PASS' if ok else 'FAIL'} {name:26} {cflag} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}{bkv}  "
          f"rms={rms:.4f} maxabs={maxabs:.4f}")
    return ok


# ---------------------------------------------------------------------------
# [Hq, seq_q, topk] transposed layout
# ---------------------------------------------------------------------------
def run_perhead_hfirst(name, S, Sk, H, Hkv, topk, seed=0, causal=False):
    q, k, v, g = _mk_qkv(S, Sk, H, Hkv, seed)
    scale = 1.0 / math.sqrt(128)
    nkv = (Sk + BLK - 1) // BLK
    bids = _mk_perhead_bids(S, H, nkv, topk, g)             # [S, H, topk]
    bids_hfirst = bids.permute(1, 0, 2).contiguous()        # [H, S, topk]
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    o, lse = ext.forward_sparse_perhead(q, k, v, bids_hfirst, scale, causal)
    ref = ref_sparse_perhead(q, k_e, v_e, bids, scale, causal=causal)   # same semantics
    rms, maxabs = _rms_maxabs(o, ref)
    ok = rms < 0.05
    cflag = "causal" if causal else "non-c "
    print(f"  {'PASS' if ok else 'FAIL'} {name:26} {cflag} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}  "
          f"rms={rms:.4f} maxabs={maxabs:.4f}  [Hq,Sq,topk]")
    return ok


# ---------------------------------------------------------------------------
# Reduction check: all heads share the same blocks => per-head == per-query
# ---------------------------------------------------------------------------
def run_shared(name, S, Sk, H, Hkv, topk, seed=0, causal=False):
    q, k, v, g = _mk_qkv(S, Sk, H, Hkv, seed)
    scale = 1.0 / math.sqrt(128)
    nkv = (Sk + BLK - 1) // BLK
    # per-query blocks, identical across heads
    bq = torch.full((S, topk), -1, dtype=torch.int32)
    for r in range(S):
        perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
        bq[r, :perm.numel()] = perm.to(torch.int32)
    bq = bq.cuda()
    bph = bq.unsqueeze(1).expand(S, H, topk).contiguous()   # [S, H, topk] same per head
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    # per-head path with shared blocks
    o_ph, _ = ext.forward_sparse_perhead(q, k, v, bph, scale, causal)
    # head-agnostic per-query path (old 2D signature)
    o_pq, _ = ext.forward_sparse(q, k, v, bq, scale, causal)
    # reference
    ref = ref_sparse_perquery(q, k_e, v_e, bq, scale, causal=causal)
    rms_ph, ma_ph = _rms_maxabs(o_ph, ref)
    rms_pq, ma_pq = _rms_maxabs(o_pq, ref)
    # per-head vs per-query kernel outputs should be (near) bit-identical
    diff = (o_ph.float() - o_pq.float()).abs()
    rms_xk = (diff.pow(2).mean().sqrt() / o_pq.float().pow(2).mean().sqrt()).item()
    ok = rms_ph < 0.05 and rms_pq < 0.05 and rms_xk < 1e-3
    cflag = "causal" if causal else "non-c "
    print(f"  {'PASS' if ok else 'FAIL'} {name:26} {cflag} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}  "
          f"ph_rms={rms_ph:.4f} pq_rms={rms_pq:.4f} ph_vs_pq_rms={rms_xk:.2e}")
    return ok


# ---------------------------------------------------------------------------
# Backward compat: old 2D signatures unchanged
# ---------------------------------------------------------------------------
def ref_sparse_tile(q, k, v, block_ids, scale, causal=False, blk_kv=BLK):
    S, H, D = q.shape; Sk = k.shape[0]
    nm, topk = block_ids.shape
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros_like(qf)
    for h in range(H):
        sc = (qf[:, h, :] @ kf[:, h, :].T) * scale
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


def run_compat_tile(name, S, Sk, H, Hkv, topk, seed=0, causal=False):
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
    o, _ = ext.forward_sparse(q, k, v, bids, scale, causal)
    ref = ref_sparse_tile(q, k_e, v_e, bids, scale, causal=causal)
    rms, maxabs = _rms_maxabs(o, ref)
    ok = rms < 0.05
    cflag = "causal" if causal else "non-c "
    print(f"  {'PASS' if ok else 'FAIL'} {name:26} {cflag} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}  "
          f"rms={rms:.4f} maxabs={maxabs:.4f}")
    return ok


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    res = []

    print("=" * 78); print("BACKWARD COMPAT: old 2D per-tile signature (forward_sparse)"); print("=" * 78)
    res.append(run_compat_tile("4 blk pick3",      S=256, Sk=256, H=4, Hkv=4,  topk=3, seed=1))
    res.append(run_compat_tile("GQA 16/4 pick4",   S=256, Sk=512, H=16, Hkv=4, topk=4, seed=2))
    res.append(run_compat_tile("causal tile",      S=320, Sk=256, H=4, Hkv=4,  topk=4, seed=3, causal=True))

    print("\n" + "=" * 78); print("PER-(QUERY,HEAD): golden layout [seq_q, Hq, topk]"); print("=" * 78)
    res.append(run_perhead("MHA 4h pick3",     S=256, Sk=256, H=4,  Hkv=4,  topk=3, seed=10))
    res.append(run_perhead("MHA 8h pick5",     S=256, Sk=1024, H=8, Hkv=8,  topk=5, seed=11))
    res.append(run_perhead("MHA 16h pick4",    S=512, Sk=512, H=16, Hkv=16, topk=4, seed=12))
    res.append(run_perhead("GQA 16/4 pick4",   S=256, Sk=512, H=16, Hkv=4,  topk=4, seed=13))
    res.append(run_perhead("GQA 8/2 pick3",    S=256, Sk=512, H=8,  Hkv=2,  topk=3, seed=14))
    res.append(run_perhead("MHA causal pick3", S=256, Sk=256, H=4,  Hkv=4,  topk=3, seed=15, causal=True))
    res.append(run_perhead("GQA causal pick4", S=256, Sk=512, H=16, Hkv=4,  topk=4, seed=16, causal=True))
    res.append(run_perhead("GQA 8/2 causal",   S=320, Sk=512, H=8,  Hkv=2,  topk=3, seed=17, causal=True))
    res.append(run_perhead("blk128 MHA pick2", S=256, Sk=512, H=4,  Hkv=4,  topk=2, seed=18, blk_kv=128))
    res.append(run_perhead("blk128 GQA causal",S=256, Sk=512, H=8,  Hkv=4,  topk=2, seed=19, causal=True, blk_kv=128))

    print("\n" + "=" * 78); print("PER-(QUERY,HEAD): transposed layout [Hq, seq_q, topk]"); print("=" * 78)
    res.append(run_perhead_hfirst("MHA 4h pick3",    S=256, Sk=256, H=4,  Hkv=4, topk=3, seed=20))
    res.append(run_perhead_hfirst("GQA 16/4 causal", S=256, Sk=512, H=16, Hkv=4, topk=4, seed=21, causal=True))

    print("\n" + "=" * 78); print("REDUCTION: all heads share blocks => per-head == per-query"); print("=" * 78)
    res.append(run_shared("MHA 4h pick3",      S=256, Sk=256, H=4,  Hkv=4, topk=3, seed=30))
    res.append(run_shared("GQA 16/4 pick4",    S=256, Sk=512, H=16, Hkv=4, topk=4, seed=31))
    res.append(run_shared("MHA causal pick3",  S=320, Sk=256, H=4,  Hkv=4, topk=3, seed=32, causal=True))

    print("\n" + "=" * 78)
    print(">>> ALL PASSED <<<" if all(res) else ">>> FAILURES <<<")
    sys.exit(0 if all(res) else 1)

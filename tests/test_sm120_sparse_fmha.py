"""SM120 block-sparse FA2 forward correctness test (vs masked torch reference)."""
import os, math, torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")
print("Building SM120 sparse FMHA extension (JIT)...")
ext = load(name="sm120_fmha_sparse",
           sources=[os.path.join(_CSRC, "sm120_fmha_sparse.cu")],
           extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f", "-O3", "-std=c++17",
                              "--expt-relaxed-constexpr"],
           verbose=False)
print("Extension built.\n")
BLK = 64

def ref_sparse(q, k, v, block_ids, scale):
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
                k0, k1 = b * BLK, min((b + 1) * BLK, Sk)
                mask[q0:q1, k0:k1] = True
        sc = sc.masked_fill(~mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(sc, dim=-1), 0.0)
        out[:, h, :] = attn @ vf[:, h, :]
    return out

def run_case(name, S, Sk, H, Hkv, topk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, H, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Sk, Hkv, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Sk, Hkv, 128, generator=g, device="cuda", dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(128)
    nm = (S + BLK - 1) // BLK
    nkv = (Sk + BLK - 1) // BLK
    # per-tile: pick `topk` distinct blocks (always >=1 valid), pad -1
    bids = torch.full((nm, topk), -1, dtype=torch.int32)
    for m in range(nm):
        perm = torch.randperm(nkv, generator=g, device="cuda")[:topk].cpu()
        bids[m, :perm.numel()] = perm.to(torch.int32)
    bids = bids.cuda()
    # GQA ref: expand kv heads to q heads
    rep = H // Hkv
    k_e = k.repeat_interleave(rep, dim=1); v_e = v.repeat_interleave(rep, dim=1)
    o, lse = ext.forward_sparse(q, k, v, bids, scale)
    ref = ref_sparse(q, k_e, v_e, bids, scale)
    diff = (o.float() - ref).abs()
    denom = ref.abs().mean().clamp_min(1e-6)
    rms = (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    # threshold = FP8-PV noise floor: the dense kernel (whose math this reuses
    # verbatim) is itself ~0.035 rms vs an fp32 reference, because PV runs as
    # FP8 E4M3. sparse(all-blocks) is bit-identical to dense-forward (rms 0.0),
    # so this floor measures the shared kernel's precision, not the sparse logic.
    ok = rms < 0.05
    print(f"  {'✓' if ok else '✗'} {name:28} S={S} Sk={Sk} H={H}/{Hkv} topk={topk}/{nkv}  "
          f"rms={rms:.4f} maxabs={diff.max().item():.4f}")
    return ok

print(f"GPU: {torch.cuda.get_device_name(0)}\n")
print("="*64); print("SM120 block-sparse FA2 forward"); print("="*64)
res = []
res.append(run_case("4 blocks, pick 3",  S=256, Sk=256, H=4, Hkv=4, topk=3, seed=1))
res.append(run_case("8 blocks, pick 3",  S=512, Sk=512, H=4, Hkv=4, topk=3, seed=2))
res.append(run_case("16 blk, pick 5",    S=256, Sk=1024, H=8, Hkv=8, topk=5, seed=3))
res.append(run_case("GQA 16/4 pick 4",   S=256, Sk=512, H=16, Hkv=4, topk=4, seed=4))
res.append(run_case("dense-equiv (all)", S=256, Sk=256, H=4, Hkv=4, topk=4, seed=5))
print("\n" + "="*64)
print(">>> ALL PASSED <<<" if all(res) else ">>> FAILURES <<<")

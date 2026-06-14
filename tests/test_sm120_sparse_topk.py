"""SM120 sparse top-16 indexer correctness test (vs numpy reference)."""
import os, numpy as np, torch
from torch.utils.cpp_extension import load

_CSRC = os.path.join(os.path.dirname(__file__), "..", "python", "fmha_sm100", "csrc")
print("Building SM120 sparse_topk extension (JIT)...")
ext = load(name="sm120_sparse_topk",
           sources=[os.path.join(_CSRC, "sm120_sparse_topk.cu")],
           extra_include_paths=[_CSRC],
           extra_cuda_cflags=["-gencode=arch=compute_120f,code=sm_120f", "-O3", "-std=c++17",
                              "--expt-relaxed-constexpr"],
           verbose=False)
print("Extension built.\n")
TOPK = 16

def ref_topk(scores_HKQ, num_valid, fbeg, fend):
    # scores: [H, K, Q]; returns [Q, H, TOPK] ascending block ids (-1 pad)
    H, K, Q = scores_HKQ.shape
    out = np.full((Q, H, TOPK), -1, np.int32)
    fend_start = max(num_valid - fend, 0)
    for q in range(Q):
        for h in range(H):
            sc = scores_HKQ[h, :, q].copy()
            forced = set(range(0, min(fbeg, num_valid))) | set(range(fend_start, num_valid))
            # candidates span ALL K blocks (selection is by score over everything);
            # clamping to -1 happens AFTER selection, no backfill.
            cand = [i for i in range(K) if i not in forced]
            cand.sort(key=lambda i: -sc[i])
            sel = (list(forced) + cand)[:TOPK]
            sel = [i if i < num_valid else -1 for i in sel]
            sel = sorted(sel)
            out[q, h, :len(sel)] = sel
    return out

def run_case(name, H, K, Q, num_valid=None, fbeg=0, fend=0):
    if num_valid is None: num_valid = K
    g = torch.Generator(device="cpu").manual_seed(1234)
    # distinct scores (avoid ties): base + tiny index-dependent perturbation
    sc = torch.rand(H, K, Q, generator=g)
    sc += torch.arange(K).float().view(1, K, 1) * 1e-4
    sc_gpu = sc.cuda().contiguous()
    out = ext.topk_select(sc_gpu, num_valid, fbeg, fend).cpu().numpy()
    ref = ref_topk(sc.numpy(), num_valid, fbeg, fend)
    # compare as sorted sets per row (robust to pad ordering)
    ok = True; nbad = 0
    for q in range(Q):
        for h in range(H):
            a = sorted(int(x) for x in out[q, h])
            b = sorted(int(x) for x in ref[q, h])
            if a != b:
                ok = False; nbad += 1
                if nbad <= 2:
                    print(f"    mismatch q={q} h={h}\n      got={a}\n      ref={b}")
    print(f"  {'✓' if ok else '✗'} {name:32} H={H} K={K} Q={Q} valid={num_valid} fb={fbeg} fe={fend}  bad={nbad}/{Q*H}")
    return ok

print(f"GPU: {torch.cuda.get_device_name(0)}\n")
print("="*60); print("SM120 sparse_topk_select"); print("="*60)
res = []
res.append(run_case("basic top16",        H=4, K=64,  Q=96))
res.append(run_case("more blocks",        H=8, K=256, Q=128))
res.append(run_case("clamp valid<K",      H=4, K=128, Q=64, num_valid=100))
res.append(run_case("force local window", H=4, K=128, Q=64, fbeg=2, fend=4))
res.append(run_case("trivial K<=16",      H=4, K=12,  Q=32))
print("\n" + "="*60)
print(">>> ALL PASSED <<<" if all(res) else ">>> FAILURES <<<")

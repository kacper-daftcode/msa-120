# Capture-safe NVFP4 MoE — notes

Goal: run MiniMax-M3-NVFP4 on SM120 **without `--enforce-eager`** (CUDA graphs /
piecewise cudagraph capture enabled) by removing every data-dependent /
host-syncing op from the un-fused NVFP4 MoE dispatch, while preserving the
validated numerics (per-expert single-group `group_gemm_nvfp4_nt_groupwise` +
torch `swigluoai`).

Files:
- `graphsafe_moe.py` — `graphsafe_swigluoai_nvfp4_moe(...)` (same signature &
  numerics as `unfused_swigluoai_nvfp4_moe`, capture-safe).
- `test_graphsafe.py` — numeric equivalence vs the validated reference.

---

## 1. Every original data-dependent / host-syncing op and how it was made static

Original hot loop (`unfused_moe.py::unfused_swigluoai_nvfp4_moe` +
`_nvfp4_gemm_one`):

| # | Original op | Why it breaks capture | Capture-safe replacement |
|---|-------------|-----------------------|--------------------------|
| 1 | `sel = (topk_ids == e)`; `if not sel.any(): continue` | `.any()` → host bool sync + Python branch on a tensor value | No branch. Every expert always runs on a fixed `[C, …]` buffer; empty experts process all-zero rows that contribute 0 (router weight 0). |
| 2 | `tok, slot = sel.nonzero(as_tuple=True)` | **`nonzero` output shape depends on values** → dynamic shape, capture-invalidating | Fixed-shape routing table `gather_idx [E, C]` built with `arange` + one-hot + `cumsum` + `scatter_` (`_build_routing_table`). No `nonzero`. |
| 3 | `xe = x[tok]` | variable-length gather → dynamic `m` | `xe = x_pad.index_select(0, gather_idx[e])` → always `[C, H]`. Pad row `T` (appended zero row) fills empty/overflow slots. |
| 4 | `m, K = xe.shape; mpad = _pad4(m)` (python int from dynamic shape) | per-expert dynamic `m`/`mpad` → shapes vary per step | `C = pad4(expert_capacity or T*k)` is a **compile-time constant**; the GEMM, activation and 2nd GEMM all see static `[C, …]`. |
| 5 | `if mpad > m: torch.cat([xe, zeros])` | value-dependent branch + dynamic concat | Gone. `C` is already a multiple of 4, so no per-call padding branch. |
| 6 | `m_indptr = torch.tensor([0, mpad], …)` | **host→device memcpy of a Python list in the hot path** → invalidates capture | `m_indptr_full = torch.arange(2, …, device) * C` — pure-device, no host data copy. Built once per call from the constant `C`. |
| 7 | `out[:m]` (dynamic slice) | dynamic length | Always returns the full `[C, …]`; no slice. |
| 8 | `w = topk_weights[tok, slot]`; `out.index_add_(0, tok, …)` | dynamic-length gather + dynamic-length scatter | Per-slot weights gathered into `slot_w [E, C]` (0.0 for pad/overflow). Combine with a **fixed-length** `out_pad.index_add_(0, gather_idx[e], contrib)`; pad slots target throw-away row `T`, discarded by `out_pad[:T]`. |

Notes on correctness equivalence (why rel RMS ≈ 0, not just “close”):
- Row order within each expert bucket is identical: original uses `nonzero`
  (ascending `(token, slot)`); the table uses `cumsum` over the same row-major
  flattened `(token, slot)` axis → same ascending order.
- The quant global scale `gs = (6·448)/amax(|xe|)` is computed over the buffer.
  Pad rows are exact zeros, so `amax` (hence `gs`, `a_q`, `alpha`) is identical
  to the original padded-to-`pad4(n)` buffer. Extra zero rows produce zero GEMM
  output and are weighted by 0 → **bit-for-bit-equivalent contribution**.
- The `for e in range(E)` Python loop is unchanged: `E` is a compile-time
  constant, so it is fully **unrolled** during capture (it never was the
  problem — the dynamic shapes inside the body were).

### Static-capacity (`expert_capacity` / `C`)
- `C` must be **constant across all captured decode steps**. Default
  `C = pad4(T*k)` is the always-correct worst case (every routed slot to one
  expert) — pass a tighter static `expert_capacity` for speed.
- Overflow beyond `C` is dropped (standard capture-safe MoE “token drop”),
  impossible at the default cap. Size `C` to the captured batch so it can’t
  trigger; the equivalence test includes a deliberately-too-small-`C` case to
  show the drop is the only source of divergence.

---

## 2. Launch flags — dropping `--enforce-eager`

Current launch (`m3_patch_unfused/launch.sh`) pins `--enforce-eager`. With the
capture-safe MoE wired in, replace it with cudagraph capture. On this vLLM image
(v1 engine) the piecewise/full-graph capture is controlled by
`--compilation-config`:

Remove:
```
  --enforce-eager
```

Add (recommended starting point — piecewise cudagraphs, which capture the MoE
region while leaving attention in eager-splittable pieces):
```
  --compilation-config '{"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8, 16]}'
```
`cudagraph_capture_sizes` is the set of (padded) decode batch sizes that get
captured; keep it small for batch-1 decode latency. Each captured size needs its
own static `expert_capacity` — see wiring below. If the attention/indexer path
is fully capture-clean you can try `"cudagraph_mode": "FULL"` for the biggest
batch-1 win; fall back to `PIECEWISE` if any non-MoE op rejects capture.

Equivalent older flag form, if `--compilation-config` JSON is not accepted by
this build:
```
  -O '{"level": 3, "cudagraph_capture_sizes": [1,2,4,8,16]}'
```
(`-O`/`--compilation_config` are aliases; `level 3` = piecewise compile +
cudagraphs in v1.)

Keep everything else from `launch.sh` unchanged (TP4, block-size 128, bf16 KV,
`VLLM_M3_UNFUSED_SWIGLUOAI_MOE=1`, gpu-mem 0.95).

---

## 3. Wiring into the override (`modelopt.py::ModelOptNvFp4FusedMoE`)

`_unfused_apply` currently calls `unfused_swigluoai_nvfp4_moe`. To go capture-safe,
call `graphsafe_swigluoai_nvfp4_moe` instead and pass a **static** capacity:

```python
from graphsafe_moe import graphsafe_swigluoai_nvfp4_moe
...
out = graphsafe_swigluoai_nvfp4_moe(
    x, layer.w13_weight, layer.w13_weight_scale, layer.w13_weight_scale_2,
    layer.w2_weight, layer.w2_weight_scale, layer.w2_weight_scale_2,
    topk_ids, topk_weights,
    activation="swigluoai", alpha=float(alpha), limit=float(limit),
    w13_sf_mma=layer._unfused_w13_sf_mma, w2_sf_mma=layer._unfused_w2_sf_mma,
    expert_capacity=getattr(layer, "_graphsafe_expert_capacity", None),
)
```

`expert_capacity` MUST be constant for a given captured size. Two options:
- **Simple/safe:** leave it `None` → `C = pad4(T*k)`. For batch-1 decode
  `T = num_tokens` is the captured (padded) batch size, so `C` is constant per
  captured size → already capture-safe, just wider than necessary.
- **Tuned:** precompute a per-captured-size cap (e.g. from EPLB/observed max
  tokens-per-expert with a safety margin, rounded with `pad4`) and stash it on
  the layer before capture. Smaller `C` = fewer padded GEMM rows = faster.

Because `C` is derived from `T` (a captured constant), the dispatch shapes are
identical on every replay of a given cudagraph — the requirement for capture.

The mount in `launch.sh` must also expose the new module on the path, e.g.:
```
  -v "${GRAPHSAFE}/graphsafe_moe.py:${SITE}/graphsafe_moe.py:ro"
```

---

## 4. Fallback

`graphsafe_swigluoai_nvfp4_moe` is a strict numeric twin, so the slow-but-correct
`unfused_swigluoai_nvfp4_moe` remains a drop-in fallback: if any capture issue
remains elsewhere, re-add `--enforce-eager` and switch the import back — no other
change. Correctness is preserved either way (`test_graphsafe.py` gates this:
rel RMS < 1e-3 between the two implementations at the worst-case capacity).
```
```

---

## 5. Validating (GPU)

Only when all 4 GPUs are <2 GB used (production container down or idle):
```
sudo docker run --rm --gpus all \
  -v /home/kacper/msa-120/nvfp4_serving:/work \
  --entrypoint python3 vllm/vllm-openai:minimax-m3 \
  /work/graphsafe/test_graphsafe.py
```
Expect `RESULT: PASS` (worst-case-capacity rel RMS < 1e-3). The too-small-`C`
case is allowed to diverge (token drop) and is reported separately.

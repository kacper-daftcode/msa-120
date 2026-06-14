# B2a: Un-fused NVFP4 swigluoai MoE for SM120 (RTX PRO 6000)

Correctness-first NVFP4 MoE for MiniMax-M3 on SM120. Replaces the fused
flashinfer b12x MoE (which applies **plain SiLU** -> wrong for M3) with an
**un-fused** path: NVFP4 `gate_up` GEMM -> **swigluoai** (torch) -> NVFP4 `down`
GEMM, per routed expert. No new PTX. Slower than a fused kernel, numerically
correct. This is workstream **B2a** in `docs/NVFP4_SM120_PLAN.md`.

The torch reference of exactly this math is
`python/fmha_sm100/swiglu_moe_ref.py` (`swigluoai_moe_nvfp4`); this doc is the
plan to wire that math into vLLM reusing existing NVFP4 ops.

---

## 1. The activation (what the fused kernel gets wrong)

M3 routed experts use GPT-OSS-style **clamped SwiGLU** (`hidden_act =
"swigluoai"`). From `MiniMax-M3-NVFP4/config.json`:

```
text_config.swiglu_alpha = 1.702
text_config.swiglu_limit = 7.0
text_config.hidden_act   = "swigluoai"
```

Exact form (verified, see the reference's `__main__` and the report):

```python
gate, up = split(gate_up)            # contiguous halves, gate then up (see 1.1)
gate = gate.clamp(max=limit)         # one-sided: min=None
up   = up.clamp(-limit, +limit)      # symmetric
glu  = gate * sigmoid(alpha * gate)
out  = (up + 1) * glu                # beta = +1 shift on up
```

`alpha = 1.702`, `limit = 7.0`. Algebraically identical to the model card's
`(clamp(up,±7)+1) · clamp(gate,max=7) · σ(1.702·gate)`. The fused b12x MoE does
`silu(gate)*up` (`_supports_activation == SILU`) -> no clamp, no `+1`, no `α`
-> corrupts every one of the 57 MoE layers. Measured gap (random experts):
**swigluoai-vs-SiLU MoE rel RMS ≈ 1.7** (see report) -- not a rounding error, a
different function.

### 1.1 Split convention (the one genuinely subtle point)

vLLM's `SwigluOAIAndMul` (`vllm/model_executor/layers/activation.py`) splits
**interleaved**: `gate = x[..., ::2]`, `up = x[..., 1::2]`. That is correct for
**GPT-OSS**, whose checkpoint ships a *pre-interleaved fused* `gate_up_proj`.

The **M3 checkpoint is different**: it stores `gate_proj` and `up_proj` as
**separate** tensors per expert
(`...experts.<e>.gate_proj.weight`, `...up_proj.weight`; verified in the
safetensors index). vLLM's MoE loader stacks them **contiguously** into
`w13 = [gate_proj; up_proj]` (rows `0:I` = gate, `I:2I` = up). Therefore the
fused activation input is **contiguous halves, gate-then-up** -- the model
card's "non-interleaved". **Use `interleaved=False`.** The reference shows the
two conventions give RMS ≈ 14 apart, so this choice is load-bearing; getting it
wrong is as bad as using SiLU. (If you instead reuse vLLM's interleaved
`SwigluOAIAndMul`, you must interleave w13 at load time to match. Cleaner to keep
contiguous and split contiguous -- see §4.)

---

## 2. SM120 NVFP4 GEMM op to reuse (both GEMMs)

The image has flashinfer b12x. The relevant kernel wrapper is in
`vllm/model_executor/kernels/linear/nvfp4/flashinfer.py`:

| Kernel class | backend | SM120? |
|---|---|---|
| `FlashInferCutlassNvFp4LinearKernel` | cutlass | **No** (SM100+) |
| `FlashInferCudnnNvFp4LinearKernel`   | cudnn   | No |
| `FlashInferTrtllmNvFp4LinearKernel`  | trtllm  | SM100 |
| **`FlashInferB12xNvFp4LinearKernel`** | **b12x** | **Yes (SM120+)** |

> Correction to the brief: it is the **b12x** backend variant that runs on
> SM120, not `...Cutlass...`. They share `apply_weights` and the underlying
> `flashinfer_scaled_fp4_mm(..., backend="b12x")`. Select via
> `init_nvfp4_linear_kernel()` which already picks b12x on SM120.

`apply_weights` signature (all variants):

```python
def apply_weights(self, layer: torch.nn.Module,
                  x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor
```

reading from `layer`: `weight` (E2M1 packed, padded), `weight_scale` (E4M3,
swizzled), `input_global_scale_inv`, `alpha` (= `1/(w_global * x_global)`). This
is the same NVFP4 contract our reference uses:
`w = e2m1 * block_scale_E4M3 * global_scale_F32` (`quantize.py`,
`swiglu_moe_ref.dequantize_nvfp4`).

The companion `ModelOptNvFp4LinearMethod.apply(layer, x, bias)` ->
`kernel.apply_weights(...)` is the per-linear entry point we reuse **twice** (one
gate_up, one down) per expert.

---

## 3. Where to insert swigluoai (the un-fused MoE method)

Subclass the modelopt FP4 MoE method
(`vllm/model_executor/layers/quantization/modelopt.py ::
ModelOptNvFp4FusedMoE`). `apply()` there is:

```python
def apply(self, layer: RoutedExperts, x: torch.Tensor,
          topk_weights: torch.Tensor, topk_ids: torch.Tensor,
          shared_experts, shared_experts_input) -> torch.Tensor
```

and it delegates to `self.moe_kernel.apply(..., activation=layer.activation)`.
Our override **does not** call the fused kernel; it runs the un-fused loop:

```python
class UnfusedSwigluOAINvFp4MoE(ModelOptNvFp4FusedMoE):
    """SM120 correctness path: NVFP4 gate_up GEMM -> swigluoai -> NVFP4 down GEMM."""

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        # Build/keep two FlashInferB12xNvFp4LinearKernel-style views over the
        # per-expert w13 and w2 NVFP4 tensors already created by create_weights:
        #   layer.w13_weight / w13_weight_scale / w13_weight_scale_2 / w13_input_scale
        #   layer.w2_weight  / w2_weight_scale  / w2_weight_scale_2  / w2_input_scale
        self.alpha = layer.swiglu_alpha   # 1.702 (plumbed in §5)
        self.limit = layer.swiglu_limit   # 7.0

    def apply(self, layer, x, topk_weights, topk_ids,
              shared_experts=None, shared_experts_input=None):
        T, H = x.shape
        out = x.new_zeros(T, H)
        for e in routed_expert_ids(topk_ids):           # only experts that fire
            sel = (topk_ids == e)
            tok, slot = sel.nonzero(as_tuple=True)
            xe = x[tok]                                   # [n, H]
            # GEMM 1: NVFP4 gate_up  (reuse b12x kernel on a per-expert linear view)
            gate_up = nvfp4_mm_b12x(xe, layer, expert=e, proj="w13")   # [n, 2I]
            # ACT: swigluoai (contiguous halves -> interleaved=False)
            act = swigluoai(gate_up, alpha=self.alpha, limit=self.limit,
                            interleaved=False)            # [n, I]
            # GEMM 2: NVFP4 down
            ye = nvfp4_mm_b12x(act, layer, expert=e, proj="w2")        # [n, H]
            w = topk_weights[tok, slot].unsqueeze(-1)
            out.index_add_(0, tok, ye * w)
        # shared expert (BF16 gate_up in M3) added by base class / caller
        return out
```

`nvfp4_mm_b12x(x, layer, expert, proj)` is a thin adapter that slices the
per-expert NVFP4 weight/scale/global out of the stacked `[E, ...]` MoE tensors
and calls `FlashInferB12xNvFp4LinearKernel.apply_weights` (or directly
`flashinfer_scaled_fp4_mm(x_q, x_sf, w, w_sf, alpha=1/(w_g*x_g), backend="b12x")`
after `flashinfer.fp4_quantize(x, x_global_scale)`). `swigluoai` is imported from
`python/fmha_sm100/swiglu_moe_ref.py` (or a vLLM-local copy of the same 4 lines).

`process_weights_after_loading` reuses
`convert_to_nvfp4_moe_kernel_format()` from the base method so the packed/
swizzled layouts match what b12x expects -- we do **not** re-quantize.

Activation correctness can be asserted at startup by probing `layer.activation`
against `"swigluoai"` (the README's recommended numeric probe).

### Selection / registration

Register the override for SM120 in the modelopt FP4 MoE factory: when
`get_device_capability() == (12, 0)` and `hidden_act == "swigluoai"`, return
`UnfusedSwigluOAINvFp4MoE` instead of the fused path. Gate behind an env flag
(e.g. `VLLM_M3_UNFUSED_SWIGLUOAI_MOE=1`) so B1 (plain-SiLU fast signal) and B2a
(correct) are switchable.

---

## 4. Performance / correctness trade-off

- **Correct:** swigluoai applied exactly between two real NVFP4 GEMMs; weights
  never leave NVFP4 (we reuse b12x, not a bf16 fallback).
- **Slow:** Python per-expert loop + per-expert GEMM launches (no expert
  batching). Acceptable for B2a (beta phase). The fast follow-up (B2b) patches
  the fused b12x CuTe-DSL epilogue to apply the clamp+`α`+`+1` directly, keeping
  one batched grouped-GEMM.
- **Optimization that keeps correctness:** sort tokens by expert and do one
  grouped/batched NVFP4 GEMM per expert population (vLLM already has the
  `moe_align_block_size` machinery); swigluoai is elementwise so it composes with
  any batching of GEMM-1's output.

---

## 5. swiglu_limit / alpha / beta plumbing

1. **Config -> layer.** `MiniMaxM3Config.text_config` already carries
   `swiglu_alpha=1.702`, `swiglu_limit=7.0`. The vLLM M3 model definition must
   pass these into the MoE layer (e.g. `RoutedExperts(..., swiglu_alpha=...,
   swiglu_limit=...)`) and store them on the layer so
   `UnfusedSwigluOAINvFp4MoE.process_weights_after_loading` can read
   `layer.swiglu_alpha/limit`. `beta` (the `+1` on `up`) is fixed by the
   activation definition; expose it only if a future checkpoint changes it.
2. **Activation tag.** Map `hidden_act == "swigluoai"` to our override and to
   `SwigluOAIAndMul(alpha, limit)` for the dense/shared MLPs (those are
   *interleaved*? -- no: M3 dense/shared `gate_up_proj` is a single fused
   `[gate; up]` linear, **contiguous**, and is **BF16** per the card. So use the
   **contiguous** split there too: either `interleaved=False` in our helper, or
   reshape before `SwigluOAIAndMul`. Do not blindly reuse the interleaved op).
3. **Defaults.** If a checkpoint omits the fields, default `alpha=1.702`,
   `limit=7.0` (GPT-OSS / M3 values) -- matches `swiglu_moe_ref.SWIGLU_*`.
4. **Routing constants** (separate from the activation, but needed for parity):
   `num_experts_per_tok=4`, `num_local_experts=128`, `routed_scaling_factor=2.0`,
   softmax scoring + renormalize over the top-k -- already in M3's router; our
   reference's `route_topk` mirrors it for validation.

---

## 6. Validation hook

Diff `UnfusedSwigluOAINvFp4MoE.apply` against
`swiglu_moe_ref.swigluoai_moe_nvfp4` on identical NVFP4 weights/scales/routing
(small T,H,I) at startup or in a unit test. Expected: bit-comparable up to b12x
GEMM accumulation order (rel RMS at the bf16-GEMM noise floor, ~1e-2). The
reference's NVFP4-vs-bf16 number is the quant noise floor to compare the *fused
SiLU* path against -- if B1's plain-SiLU output diverges from this reference by
the ~1.7 rel-RMS swigluoai-vs-SiLU gap, that is the quality the un-fused path
recovers.

---

## Files

- `python/fmha_sm100/swiglu_moe_ref.py` -- torch reference (`swigluoai`,
  `swigluoai_moe`, NVFP4 quant + `swigluoai_moe_nvfp4`) and self-check.
- this doc -- vLLM wiring design.
- reuse: `vllm .../kernels/linear/nvfp4/flashinfer.py`
  (`FlashInferB12xNvFp4LinearKernel`),
  `vllm .../quantization/modelopt.py` (`ModelOptNvFp4FusedMoE`),
  `vllm .../layers/activation.py` (`SwigluOAIAndMul`, for the dense/shared MLPs).

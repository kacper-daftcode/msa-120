# Batched NVFP4 MoE fix (multi-group SFA-offset bug)

Files:
- `ROOTCAUSE.md` — precise root cause, offending CUDA source location + offset math.
- `repro_multigroup_sfa.py` — standalone tiny 4-group reproducer: shows the bug
  (naive concat SFA) blowing up for groups >=2 and the FIX (scatter SFA) at the
  noise floor for all groups. Needs only a few GB of GPU.
- `batched_nvfp4_moe.py` — the fix as an importable Python module:
  `batched_swigluoai_nvfp4_moe(...)` (same signature/semantics as the validated
  `unfused_moe.unfused_swigluoai_nvfp4_moe`) doing ONE grouped GEMM per
  projection over the active experts.

## The bug (one line)
flashinfer 0.6.12's SM120 NVFP4 grouped GEMM offsets the activation scale
factor for group `i` by `((m_indptr[i] + i*127)//128)*128` swizzled rows, which
over-advances by `floor(127*i/128)` extra 128-row blocks for `i>=2`. Single
group (`i==0`) is unaffected -> the per-expert loop is correct, the batched call
is not.

## The fix (one line)
Pre-scatter each group's 128-padded swizzled activation-SF block to exactly the
(buggy but deterministic) row offset the kernel will read. No kernel rebuild.

## Integration point in vLLM
Replace the per-expert loop body of
`/home/kacper/msa-120/nvfp4_serving/unfused_moe/unfused_moe.py::unfused_swigluoai_nvfp4_moe`
(currently called from the custom `ModelOptNvFp4FusedMoE.apply` override per
RECIPE.md "THE FINISH") with `batched_swigluoai_nvfp4_moe` from this module.
Both take the identical arguments:

    out = batched_swigluoai_nvfp4_moe(
        x, w13_packed, w13_scale, w13_scale_2,
        w2_packed, w2_scale, w2_scale_2,
        topk_ids, topk_weights,
        activation="swigluoai", alpha=1.702, limit=7.0,
    )

Weight scales can be precomputed once (process_weights_after_loading) by stacking
the per-expert swizzled blocks and calling
`convert_sf_to_mma_layout(..., num_groups=E)` (see
`build_batched_weight_scale_mma`); pass them via `w13_sf_mma_full`/`w2_sf_mma_full`
to avoid rebuilding per forward. (Current module rebuilds the *active-subset* SF
each call for simplicity; for production, precompute the full-E SF once and index
the active rows, or keep all E groups dense.)

## Correctness stance
The scattered SFA bytes are bit-identical to those the validated single-group
path feeds (same `fp4_quantize(is_sf_swizzled_layout=True)` blocks), just placed
at the kernel's expected rows. So the batched result is numerically equal to the
per-expert loop, not merely "close". The reproducer asserts FIX-vs-REF per-group
rel RMS ~ 0 in addition to FIX-vs-bf16 at the noise floor.

## Run the reproducer
Requires a (mostly) free GPU. Do NOT run while `minimax-m3-nvfp4` holds the GPU.

    nvidia-smi --query-gpu=memory.used --format=csv,noheader   # all <2GB?
    cd /home/kacper/msa-120/nvfp4_serving/batched_fix
    python3 repro_multigroup_sfa.py

Or, if no local flashinfer, via the same image:

    sudo docker run --rm --gpus all \
      -v /home/kacper/msa-120/nvfp4_serving/batched_fix:/w -w /w \
      vllm/vllm-openai:minimax-m3 python3 repro_multigroup_sfa.py

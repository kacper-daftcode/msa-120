# Selector patch — route MiniMax-M3 to the SM120 MSA impls

Two selectors decide which kernel impl the M3 model builds. We patch both to
return the SM120 subclasses on a family-120 (cap 12.x) CUDA device with a bf16
main / bf16 index cache. Everything else (SM100, AMD, fp8) keeps its current
behaviour.

## 1. `select_main_impl_cls` — sparse attend

File (in the image): `…/vllm/models/minimax_m3/common/sparse_attention.py`
Function at `sparse_attention.py:376`.

Insert a family-120 branch BEFORE the SM100 branch (or after — they are
mutually exclusive since a device is in exactly one family):

```python
def select_main_impl_cls(
    *,
    topk_blocks: int,
    kv_cache_dtype: str,
) -> type[MiniMaxM3SparseImpl]:
    # --- SM120 (RTX PRO 6000 / 5090): our MSA attend, bf16 only ---
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and topk_blocks == 16                       # sm120_sparse_topk fixes topk=16
        and not is_quantized_kv_cache(kv_cache_dtype)  # bf16 KV only
    ):
        from vllm_integration.sm120_sparse_impl import MiniMaxM3SparseSm120Impl
        return MiniMaxM3SparseSm120Impl

    # --- SM100 (B200): CuTe MSA attend (unchanged) ---
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and topk_blocks in (4, 8, 16, 32)
        and not is_quantized_kv_cache(kv_cache_dtype)
    ):
        from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
            MiniMaxM3SparseMSAImpl,
        )
        return MiniMaxM3SparseMSAImpl

    return MiniMaxM3SparseTritonImpl
```

Notes:
- `is_device_capability_family(120)` returns True for cap 12.x because the impl
  is `(cap // 10) == (120 // 10)` i.e. `12 == 12` (interface.py:363-375).
- We require `topk_blocks == 16` because `sm120_sparse_topk.topk_select` hard-codes
  topk = 16 (sm120_sparse_topk.cu:24). The Triton path supports other values; if
  the deployed M3 config uses topk != 16, leave it on Triton (do NOT select us).
- We require bf16 KV (`not is_quantized_kv_cache`) — our paged kernel is bf16
  (sm120_fmha_paged.cu:586). fp8 KV stays on Triton.

## 2. `select_indexer_impl_cls` — indexer score + top-k

File: `…/vllm/models/minimax_m3/common/indexer.py`
Function at `indexer.py:436`.

```python
def select_indexer_impl_cls(
    *,
    indexer_kv_dtype: IndexerKVDType = "bf16",
) -> type[MiniMaxM3IndexerImpl]:
    if indexer_kv_dtype in ("mxfp4", "nvfp4"):
        raise NotImplementedError(...)
    if indexer_kv_dtype != "bf16":
        raise NotImplementedError(...)

    # --- SM120: our score + top-k, bf16 index cache only ---
    from vllm.platforms import current_platform
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    ):
        from vllm_integration.sm120_indexer_impl import MiniMaxM3IndexerSm120Impl
        return MiniMaxM3IndexerSm120Impl

    return MiniMaxM3IndexerTritonImpl
```

Notes:
- `select_indexer_impl_cls` does NOT receive `topk_blocks`; the topk==16 guard
  lives in the impl (`MiniMaxM3IndexerSm120Impl` asserts `topk_blocks == 16`).
  If you must hard-gate at selection, plumb `topk_blocks` through the call site
  in `MiniMaxM3Indexer.__init__` (indexer.py:482) — but the assert is simpler.
- The SM120 indexer impl keeps `indexer_backend_cls = MiniMaxM3IndexerBackend`
  (inherited), so the side-cache shape stays `[num_blocks, 128, head_dim]`
  (indexer.py:95), which our score path reads.

## 3. Overlaying onto the `vllm/vllm-openai:minimax-m3` image

Base image: `vllm/vllm-openai:minimax-m3` (vllm 0.1.dev17492, torch 2.11+cu130).
vLLM installs under `dist-packages` in the image; confirm the exact root:

```bash
python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"
# typically /usr/local/lib/python3.12/dist-packages/vllm
```

Overlay steps (Dockerfile `FROM vllm/vllm-openai:minimax-m3`):

1. **Ship the kernels + adapters.** Copy this `vllm_integration/` package and the
   kernel sources to an importable location, e.g.:
   ```
   COPY vllm_integration /opt/sm120/vllm_integration
   COPY python/fmha_sm100/csrc /opt/sm120/python/fmha_sm100/csrc
   ENV PYTHONPATH=/opt/sm120:$PYTHONPATH
   ENV SM120_MSA_CSRC=/opt/sm120/python/fmha_sm100/csrc
   ```
   `vllm_integration._loader` reads `SM120_MSA_CSRC` to find the `.cu` files.

2. **Build the kernels.** Verify nvcc is present:
   ```bash
   which nvcc && nvcc --version    # need CUDA 13 toolchain for compute_120f
   ```
   - If nvcc present: first-import JIT via `torch.utils.cpp_extension.load`
     works (slow first request). Optionally pre-warm at build by importing each
     ext once on a build box with a 12.x GPU (or `TORCH_CUDA_ARCH_LIST=12.0+PTX`
     for PTX-only AOT).
   - If nvcc absent: AOT-build a wheel `sm120_msa_kernels` (modules
     `sm120_indexer`, `sm120_sparse_topk`, `sm120_fmha_perhead`,
     `sm120_fmha_paged`) on a CUDA-13 builder and `pip install` it into the
     image; `_loader._load` prefers `importlib.import_module("sm120_msa_kernels.<m>")`.

3. **Patch the two selectors.** Three options, least to most invasive:
   - **(preferred) vLLM plugin / monkeypatch at startup.** Add a tiny module
     imported via the `vllm.general_plugins` entry point (or
     `VLLM_PLUGINS`) that rebinds the two functions:
     ```python
     # sm120_select_patch.py
     import vllm.models.minimax_m3.common.sparse_attention as sa
     import vllm.models.minimax_m3.common.indexer as ix
     from vllm_integration.patches import (
         patched_select_main_impl_cls, patched_select_indexer_impl_cls,
     )
     sa.select_main_impl_cls = patched_select_main_impl_cls
     ix.select_indexer_impl_cls = patched_select_indexer_impl_cls
     ```
     (Confirm both are looked up as module globals at call time, not imported by
     value into the layer module — if a caller did
     `from ... import select_main_impl_cls`, patch that binding too. As of this
     code, `MiniMaxM3Indexer.__init__` calls `select_indexer_impl_cls(...)` as a
     module-level name in indexer.py, so rebinding the module global suffices.)
   - **In-place file overlay (0xSero style).** `COPY` patched copies of
     `sparse_attention.py` / `indexer.py` over the dist-packages files. Brittle
     across image bumps; pin the base image digest.
   - **`.pth`-injected sitecustomize** that applies the monkeypatch on interp
     start. Equivalent to the plugin but image-global.

4. **Smoke check selection** (on the target GPU):
   ```python
   from vllm.platforms import current_platform
   assert current_platform.is_device_capability_family(120)
   from vllm.models.minimax_m3.common.sparse_attention import select_main_impl_cls
   print(select_main_impl_cls(topk_blocks=16, kv_cache_dtype="auto").__name__)
   # -> MiniMaxM3SparseSm120Impl
   ```

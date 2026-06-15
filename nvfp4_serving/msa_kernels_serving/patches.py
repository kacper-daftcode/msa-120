# SPDX-License-Identifier: MIT
"""Monkeypatch the M3 main-attend selector to route SM120 (family 120) + bf16
to our SM120 decode kernel, while keeping the Triton indexer (Phase 1).

IMPORTANT: ``nvidia/model.py`` does ``from ...sparse_attention import
select_main_impl_cls`` and calls the LOCAL name at build time, so we must rebind
the name in BOTH the source module and the already-imported ``model`` module
namespace. Rebinding only the source module would have no effect.
"""

from __future__ import annotations

from vllm.platforms import current_platform  # type: ignore

try:  # vLLM moved this symbol around across versions; be tolerant.
    from vllm.v1.kv_cache_interface import is_quantized_kv_cache  # type: ignore
except Exception:  # pragma: no cover
    from vllm.attention.backends.abstract import (  # type: ignore
        is_quantized_kv_cache,
    )

from vllm.models.minimax_m3.common.sparse_attention import (  # type: ignore
    MiniMaxM3SparseImpl,
    select_main_impl_cls as _orig_select_main,
)


def patched_select_main_impl_cls(
    *, topk_blocks: int, kv_cache_dtype: str
) -> "type[MiniMaxM3SparseImpl]":
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and topk_blocks == 16
        and not is_quantized_kv_cache(kv_cache_dtype)
    ):
        from .sm120_sparse_impl import MiniMaxM3SparseSm120Impl

        print("[sm120-msa] select_main_impl_cls -> MiniMaxM3SparseSm120Impl "
              f"(family120, bf16, topk={topk_blocks})", flush=True)
        return MiniMaxM3SparseSm120Impl
    return _orig_select_main(topk_blocks=topk_blocks, kv_cache_dtype=kv_cache_dtype)


def apply() -> None:
    """Rebind ``select_main_impl_cls`` everywhere it is referenced. Idempotent."""
    import vllm.models.minimax_m3.common.sparse_attention as sa  # type: ignore

    sa.select_main_impl_cls = patched_select_main_impl_cls

    # model.py imported the name locally -> patch its module namespace too.
    try:
        import vllm.models.minimax_m3.nvidia.model as m  # type: ignore

        if hasattr(m, "select_main_impl_cls"):
            m.select_main_impl_cls = patched_select_main_impl_cls
    except Exception as e:  # pragma: no cover
        print(f"[sm120-msa] WARNING: could not patch nvidia.model: {e!r}", flush=True)

    # Pre-build the JIT kernel NOW (startup), so the first decode -- which may
    # occur during CUDA-graph capture/warmup -- never triggers a cpp_extension
    # compile inside a graph-capture region.
    try:
        from ._loader import decode_serving_ext

        decode_serving_ext()
        print("[sm120-msa] decode kernel JIT-built at startup (graph-safe)",
              flush=True)
    except Exception as e:  # pragma: no cover
        print(f"[sm120-msa] WARNING: kernel pre-build failed: {e!r}", flush=True)

    print("[sm120-msa] selector patch applied (main attend -> SM120 decode; "
          "indexer stays Triton)", flush=True)

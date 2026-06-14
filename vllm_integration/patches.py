# SPDX-License-Identifier: MIT
"""Monkeypatch helpers for the two M3 impl selectors (DRAFT).

Imported by a vLLM startup plugin (see selector_patch.md §3) to route family-120
+ bf16 to the SM120 impls without editing the vendored model files in-place.
"""

from __future__ import annotations

from vllm.platforms import current_platform  # type: ignore
from vllm.v1.kv_cache_interface import is_quantized_kv_cache  # type: ignore

# Re-use the originals as fallbacks.
from vllm.models.minimax_m3.common.sparse_attention import (  # type: ignore
    MiniMaxM3SparseImpl,
    select_main_impl_cls as _orig_select_main,
)
from vllm.models.minimax_m3.common.indexer import (  # type: ignore
    MiniMaxM3IndexerImpl,
    select_indexer_impl_cls as _orig_select_indexer,
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

        return MiniMaxM3SparseSm120Impl
    return _orig_select_main(topk_blocks=topk_blocks, kv_cache_dtype=kv_cache_dtype)


def patched_select_indexer_impl_cls(
    *, indexer_kv_dtype: str = "bf16"
) -> "type[MiniMaxM3IndexerImpl]":
    if (
        indexer_kv_dtype == "bf16"
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    ):
        from .sm120_indexer_impl import MiniMaxM3IndexerSm120Impl

        return MiniMaxM3IndexerSm120Impl
    return _orig_select_indexer(indexer_kv_dtype=indexer_kv_dtype)


def apply() -> None:
    """Rebind both module globals. Call once at interpreter/plugin startup."""
    import vllm.models.minimax_m3.common.indexer as ix  # type: ignore
    import vllm.models.minimax_m3.common.sparse_attention as sa  # type: ignore

    sa.select_main_impl_cls = patched_select_main_impl_cls
    ix.select_indexer_impl_cls = patched_select_indexer_impl_cls

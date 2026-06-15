# SPDX-License-Identifier: MIT
"""SM120 MSA startup hook (runs in EVERY Python process, incl. all TP workers).

Placed on PYTHONPATH as ``sitecustomize.py`` so CPython auto-imports it at
interpreter startup -- before vLLM is imported. It cannot call
``patches.apply()`` yet (vLLM not importable / selector imported lazily), so it
installs a meta-path hook that fires ``patches.apply()`` the moment
``vllm.models.minimax_m3.nvidia.model`` finishes importing. That module's import
is what pulls in ``select_main_impl_cls``, and the impl is selected slightly
later when the layer is constructed -- so applying the rebind at model-module
import time guarantees the patched selector is the one used.

Robust to:
  * the module already being imported (apply immediately),
  * import ordering across the engine + 4 TP worker processes,
  * apply() being idempotent (safe if triggered more than once).
"""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys

_TARGET = "vllm.models.minimax_m3.nvidia.model"
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# Make our overlay package importable as ``sm120_msa_serving``.
_PARENT = os.path.dirname(_PKG_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


def _do_apply():
    try:
        # import the overlay package's patches module
        from msa_kernels_serving import patches  # type: ignore

        patches.apply()
    except Exception as e:  # pragma: no cover
        print(f"[sm120-msa] startup apply FAILED: {e!r}", flush=True)
        import traceback
        traceback.print_exc()


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """A no-op finder whose only job is to detect when _TARGET is imported."""

    def find_spec(self, name, path, target=None):
        if name == _TARGET:
            # Let the real finders load it; schedule apply right after.
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is not None and spec.loader is not None:
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)
                    _do_apply()

                spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec
        return None


if _TARGET in sys.modules:
    _do_apply()
else:
    sys.meta_path.insert(0, _PostImportFinder())

print("[sm120-msa] sitecustomize startup hook installed", flush=True)

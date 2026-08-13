"""Back-compat shim: ``pbg_tellurium`` was renamed to ``viva_tellurium``.

Part of the pbg -> viva rebrand (the GitHub repo is already
``vivarium-collective/viva-tellurium``). This shim keeps every existing consumer
working during the deprecation window:

  * ``import pbg_tellurium`` / ``from pbg_tellurium import X`` works (re-exports
    the new package's top-level ``__all__``);
  * ``import pbg_tellurium.<sub>`` transparently resolves to
    ``viva_tellurium.<sub>`` via a meta-path finder (so
    ``from pbg_tellurium.simulators import ...`` still resolves); and
  * ``python -m pbg_tellurium.<sub>`` still executes.

Importing anything under this package emits a one-time
:class:`DeprecationWarning`. Update imports to ``viva_tellurium``; this shim is
removed in a future major release.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

warnings.warn(
    "pbg_tellurium is renamed to viva_tellurium; update your imports "
    "(the pbg_tellurium alias is removed in a future major release).",
    DeprecationWarning,
    stacklevel=2,
)

_OLD = "pbg_tellurium."
_NEW = "viva_tellurium."


class _Redirect(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Forward ``pbg_tellurium.<sub>`` imports to ``viva_tellurium.<sub>``."""

    def _target(self, name: str) -> str:
        return _NEW + name[len(_OLD):]

    def find_spec(self, name, path=None, target=None):
        if not name.startswith(_OLD):
            return None
        real = importlib.util.find_spec(self._target(name))
        if real is None:
            return None
        spec = importlib.util.spec_from_loader(
            name,
            self,
            origin=real.origin,
            is_package=real.submodule_search_locations is not None,
        )
        if real.submodule_search_locations is not None:
            spec.submodule_search_locations = list(real.submodule_search_locations)
        return spec

    def create_module(self, spec):
        # Alias the fully-initialized new-package module under BOTH names so
        # `import a.b` and identity checks against either name agree.
        mod = importlib.import_module(self._target(spec.name))
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):  # already executed by import_module
        pass

    def get_code(self, name):
        # Support `python -m pbg_tellurium.<sub>`: runpy needs a code object.
        target = self._target(name)
        return importlib.util.find_spec(target).loader.get_code(target)


sys.meta_path.insert(0, _Redirect())

_new = importlib.import_module("viva_tellurium")
__version__ = getattr(_new, "__version__", "0.0.0")
# Re-export the new package's public surface (if any is declared).
globals().update({k: getattr(_new, k) for k in getattr(_new, "__all__", [])})

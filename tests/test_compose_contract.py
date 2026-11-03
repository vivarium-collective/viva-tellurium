"""Regressions for the compose-api onboarding audit of viva-tellurium (fix/onboarding-compose-contract).

1. The default path must not need viva_superpowers (dashboard tooling; git-only; pins bigraph-schema>=1.8.2, which
   conflicts with compose-api's 1.8.1 core). Runs in a subprocess that blocks the import.
2. The shipped composites' emitter address must resolve on build_core(). Before, they used ``local:ram-emitter``, which
   build_core() never registered (only register_tellurium() did).
3. The composite YAML files must be declared as package data, so an installed wheel can list and build them. Checked
   by reading the build config here; the built wheel is also checked by hand (see the PR).
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NO_SUPERPOWERS = textwrap.dedent(
    """
    import sys
    sys.modules["viva_superpowers"] = None  # any import of viva_superpowers now raises ImportError
    from viva_tellurium.composites import build_composite, list_composite_specs

    names = list_composite_specs()
    assert names, "no composite specs shipped"
    # No core passed: build_composite() takes the default path (register_tellurium), which is where the import was.
    composite = build_composite("decay-demo")
    composite.run(1.0)
    print("ok")
    """
)


def test_default_path_loads_without_viva_superpowers():
    result = subprocess.run([sys.executable, "-c", NO_SUPERPOWERS], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    # process-bigraph prints "skipping ... missing optional dep" notices to stdout for the dashboard modules; the
    # signal is the final line.
    assert result.stdout.strip().splitlines()[-1] == "ok", result.stdout


def test_shipped_composite_emitter_address_resolves_on_build_core():
    from viva_tellurium.composites import build_composite
    from viva_tellurium.core import build_core

    composite = build_composite("decay-demo", core=build_core())
    composite.run(1.0)


def test_composite_yaml_is_declared_as_package_data():
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = config["tool"]["setuptools"]["package-data"]
    assert "composites/*.yaml" in package_data.get("viva_tellurium", []), package_data

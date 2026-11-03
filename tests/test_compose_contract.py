"""Regressions for the compose-api onboarding audit of viva-tellurium (fix/onboarding-compose-contract).

1. The default path must not need viva_superpowers (dashboard tooling; git-only; pins bigraph-schema>=1.8.2, which
   conflicts with compose-api's 1.8.1 core). Runs in a subprocess that blocks the import.
2. The shipped composites' emitter address must resolve on build_core(). Before, they used ``local:ram-emitter``, which
   build_core() never registered (only register_tellurium() did).
3. The composite YAML files must be declared as package data, so an installed wheel can list and build them. Checked
   by reading the build config here; the built wheel is also checked by hand (see the PR).
"""
from __future__ import annotations

import re
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


# --- 0.2.1: follow-ups from the adversarial review of 0.2.0 ---------------------------------------------------------

VIZ_WITHOUT_SUPERPOWERS = textwrap.dedent(
    """
    import sys
    sys.modules["viva_superpowers"] = None
    from viva_tellurium.composites import build_composite

    try:
        build_composite("decay-with-viz")
    except ModuleNotFoundError as exc:
        print("raised:", exc.name, "|", exc)
    else:
        print("built")
    """
)


def _pyproject() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def _requirement_names(requirements: list[str]) -> set[str]:
    return {re.split(r"[\s\[<>=!~;]", requirement, maxsplit=1)[0].lower() for requirement in requirements}


def test_dashboard_extra_carries_the_workbench():
    extras = _pyproject()["project"]["optional-dependencies"]
    assert {"viva-superpowers", "vivarium-workbench"} <= _requirement_names(extras["dashboard"]), extras["dashboard"]


def test_image_sync_installs_the_dashboard_extra():
    """The image's ``uv sync`` must install what its CMD serves with.

    0.2.0 moved vivarium-workbench and viva-superpowers out of the base dependencies. Dockerfile and Singularity.def
    (generated from viva-template) run ``uv sync --no-install-project`` and then ``vivarium-workbench serve``. That sync
    installs no extras but does install the default dependency groups, so one of those must pull the dashboard extra.
    """
    config = _pyproject()
    for image_file in ("Dockerfile", "Singularity.def"):
        text = (REPO / image_file).read_text()
        assert "vivarium-workbench" in text and "serve" in text, image_file
        sync_lines = [line for line in text.splitlines() if re.match(r"\s*(RUN\s+)?uv sync\b", line)]
        assert sync_lines, image_file
        for line in sync_lines:
            assert "--no-default-groups" not in line and "--no-dev" not in line, line
    groups = config.get("dependency-groups", {})
    default_groups = config.get("tool", {}).get("uv", {}).get("default-groups", ["dev"])
    if default_groups == "all":
        default_groups = list(groups)
    installed = [requirement for group in default_groups for requirement in groups.get(group, [])]
    assert "viva-tellurium[dashboard]" in installed, (default_groups, groups)


def test_cli_extra_is_kept_for_compatibility():
    config = _pyproject()
    assert _requirement_names(config["project"]["optional-dependencies"]["cli"]) == {"typer", "rich"}
    assert {"typer", "rich"} <= _requirement_names(config["project"]["dependencies"])


def test_viz_composite_without_viva_superpowers_names_the_dashboard_extra():
    result = subprocess.run(
        [sys.executable, "-c", VIZ_WITHOUT_SUPERPOWERS], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    last = result.stdout.strip().splitlines()[-1]
    assert last.startswith("raised: viva_superpowers |"), result.stdout
    assert "SpeciesTimeSeriesPlots" in last and "viva-tellurium[dashboard]" in last, last

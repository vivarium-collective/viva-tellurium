"""Tellurium composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — `make_tellurium_document(model=..., ...)` builds a
   PBG state-dict programmatically for callers that want full control over
   the SBML/Antimony model + wiring. Used by `demo/demo_report.py` for the
   three Antimony model experiments.

2. **Declarative `*.composite.yaml`** — sibling files in this directory follow
   the pbg-superpowers composite-spec convention. `build_composite()` loads
   one by name and instantiates `process_bigraph.Composite` with parameter
   substitution. The dashboard's composite explorer discovers these
   automatically once the package is installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from viva_tellurium.processes import TelluriumProcess, TelluriumUTCStep, TelluriumSteadyStateStep


# ---------------------------------------------------------------------------
# Hand-coded composite factories (legacy / programmatic API)
# ---------------------------------------------------------------------------

def make_tellurium_document(
    model='',
    model_format='antimony',
    model_file='',
    integrator='cvode',
    absolute_tolerance=1e-10,
    relative_tolerance=1e-8,
    seed=-1,
    species_overrides=None,
    parameter_overrides=None,
    interval=1.0,
):
    """Create a composite document for a Tellurium simulation.

    Returns a document dict ready for use with Composite().

    Args:
        model: Antimony or SBML source string.
        model_format: 'antimony' or 'sbml'.
        model_file: Optional path to a local model file.
        integrator: 'cvode' or 'gillespie'.
        absolute_tolerance, relative_tolerance: CVODE tolerances.
        seed: Stochastic RNG seed (-1 to leave unset).
        species_overrides: {species_id: initial_value}.
        parameter_overrides: {parameter_id: value}.
        interval: Time interval between process updates.

    Returns:
        dict: Composite document with tellurium process, stores, and emitter.
    """
    species_overrides = species_overrides or {}
    parameter_overrides = parameter_overrides or {}

    return {
        'tellurium': {
            '_type': 'process',
            'address': 'local:TelluriumProcess',
            'config': {
                'model': model,
                'model_format': model_format,
                'model_file': model_file,
                'integrator': integrator,
                'absolute_tolerance': absolute_tolerance,
                'relative_tolerance': relative_tolerance,
                'seed': seed,
                'species_overrides': species_overrides,
                'parameter_overrides': parameter_overrides,
            },
            'interval': interval,
            'inputs': {},
            'outputs': {
                'species': ['stores', 'species'],
                'rates': ['stores', 'rates'],
                'parameters': ['stores', 'parameters'],
                'time': ['stores', 'time'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'species': 'map[float]',
                    'rates': 'map[float]',
                    'time': 'float',
                    'global_time': 'float',
                },
            },
            'inputs': {
                'species': ['stores', 'species'],
                'rates': ['stores', 'rates'],
                'time': ['stores', 'time'],
                'global_time': ['global_time'],
            },
        },
    }


def make_tellurium_steady_state_document(
    model='',
    model_format='antimony',
    model_file='',
    species_overrides=None,
    parameter_overrides=None,
):
    """Create a composite document for a Tellurium steady-state solve.

    Returns a document dict ready for use with Composite().

    Args:
        model: Antimony or SBML source string.
        model_format: 'antimony' or 'sbml'.
        model_file: Optional path to a local model file.
        species_overrides: {species_id: initial_value}.
        parameter_overrides: {parameter_id: value}.

    Returns:
        dict: Composite document with TelluriumSteadyStateStep, stores, and emitter.
    """
    species_overrides = species_overrides or {}
    parameter_overrides = parameter_overrides or {}

    return {
        'steady_state': {
            '_type': 'step',
            'address': 'local:TelluriumSteadyStateStep',
            'config': {
                'model': model,
                'model_format': model_format,
                'model_file': model_file,
                'species_overrides': species_overrides,
                'parameter_overrides': parameter_overrides,
            },
            'inputs': {},
            'outputs': {
                'steady_state_concentrations': ['ss_results', 'concentrations'],
            },
        },
        'ss_results': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'concentrations': 'map[float]',
                },
            },
            'inputs': {
                'concentrations': ['ss_results', 'concentrations'],
            },
        },
    }


def register_tellurium(core=None):
    """Return a core with TelluriumProcess + TelluriumUTCStep, the RAM emitter,
    and the species-time-series Visualization registered."""
    if core is None:
        core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('TelluriumUTCStep', TelluriumUTCStep)
    core.register_link('TelluriumSteadyStateStep', TelluriumSteadyStateStep)
    core.register_link('ram-emitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    from viva_tellurium.visualizations import SpeciesTimeSeriesPlots
    core.register_link('SpeciesTimeSeriesPlots', SpeciesTimeSeriesPlots)
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_tellurium() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_tellurium()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)

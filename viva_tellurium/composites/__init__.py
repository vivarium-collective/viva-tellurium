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
# t=0 store seeding (issue #7)
# ---------------------------------------------------------------------------

def _resolve_link_class(core, address):
    """Resolve a composite-doc address ('local:TelluriumProcess') to its class."""
    if not isinstance(address, str):
        return None
    key = address.split(":", 1)[-1]  # strip the protocol prefix ('local:')
    return getattr(core, "link_registry", {}).get(key)


def _set_store_path(state, path, value):
    """Set `value` at nested-dict `path` in `state`, creating dicts as needed.

    A non-empty value the document author already placed at the leaf is
    preserved (the author's explicit initial condition wins).
    """
    cursor = state
    for key in path[:-1]:
        nxt = cursor.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[key] = nxt
        cursor = nxt
    leaf = path[-1]
    existing = cursor.get(leaf)
    if isinstance(existing, dict) and len(existing) > 0:
        return
    if isinstance(existing, (int, float)) and not isinstance(existing, bool) and existing != 0:
        return
    cursor[leaf] = value


def seed_initial_stores(state, core):
    """Write each process/step's ``initial_state()`` into its wired output stores.

    process-bigraph projects a process's ``initial_state()`` through its output
    ports, but when those outputs are wired to empty stores (the ``decay-demo``
    pattern) the realized empty store wins the final merge during ``Composite``
    construction, so the first emitter row (``global_time=0``) sees empty maps —
    issue #7. This walks the composite document, instantiates each process/step
    whose outputs are wired to store paths, and writes its ``initial_state()``
    into those stores so the t=0 emitter row carries the model's initial
    species, parameters and time (as an ``OdeProcess`` composite's does).

    Mutates and returns ``state``. Seeding is best-effort and never blocks
    composite construction; numerical behavior after t=0 is unaffected.
    """
    if not isinstance(state, dict):
        return state

    for node in state.values():
        if not isinstance(node, dict) or node.get("_type") not in ("process", "step"):
            continue
        wires = node.get("outputs")
        if not isinstance(wires, dict):
            continue
        cls = _resolve_link_class(core, node.get("address"))
        if cls is None or not hasattr(cls, "initial_state"):
            continue
        try:
            instance = cls(config=node.get("config", {}) or {}, core=core)
            init = instance.initial_state()
        except Exception:
            continue
        if not isinstance(init, dict):
            continue
        for port, path in wires.items():
            if port not in init:
                continue
            value = init[port]
            if value is None or (isinstance(value, (dict, list)) and len(value) == 0):
                continue
            if isinstance(path, (list, tuple)) and path:
                _set_store_path(state, list(path), value)

    return state


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

    doc = {
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
                    'parameters': 'map[float]',
                    'time': 'float',
                    'global_time': 'float',
                },
            },
            'inputs': {
                'species': ['stores', 'species'],
                'rates': ['stores', 'rates'],
                'parameters': ['stores', 'parameters'],
                'time': ['stores', 'time'],
                'global_time': ['global_time'],
            },
        },
    }

    # Seed the wired stores with the model's t=0 state so the first emitter
    # row (global_time=0) carries initial species/parameters/time (issue #7).
    # Use a minimal core (just the Tellurium links) so seeding never pulls in
    # optional viz dependencies.
    seed_core = allocate_core()
    seed_core.register_link('TelluriumProcess', TelluriumProcess)
    seed_core.register_link('TelluriumUTCStep', TelluriumUTCStep)
    seed_core.register_link('TelluriumSteadyStateStep', TelluriumSteadyStateStep)
    seed_initial_stores(doc, seed_core)
    return doc


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
    # Seed wired output stores with each process's t=0 state so the first
    # emitter row carries the model's initial values (issue #7).
    seed_initial_stores(state, core)
    return Composite({"state": state}, core=core)

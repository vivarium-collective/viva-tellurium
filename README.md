# viva-tellurium

[![tests](https://github.com/vivarium-collective/viva-tellurium/actions/workflows/tests.yml/badge.svg)](https://github.com/vivarium-collective/viva-tellurium/actions/workflows/tests.yml)
[![license](https://img.shields.io/github/license/vivarium-collective/viva-tellurium)](./LICENSE)

The SBML / RoadRunner ODE **engine-provider** package for the vivarium
ecosystem. `viva_tellurium` wraps [Tellurium](https://tellurium.analogmachine.org/) /
[libroadrunner](https://libroadrunner.org/), exposing any SBML or Antimony
kinetic model as a `process-bigraph` Process/Step so it can be composed with
other simulators in a bigraph document.

This is the most widely reused simulator in the ecosystem — 9+ downstream
packages import `viva_tellurium` — so the package and all of its dependencies
are kept exactly stable. It ships on PyPI; pin it with:

```bash
pip install viva-tellurium==0.2.4
```

The public engine entry point is `load_roadrunner()`. A legacy `pbg_tellurium`
import shim is also retained (it re-exports `viva_tellurium`; `viva-biomodels`
imports it).

> **Note:** This repo is now a pure engine provider. Its former demo
> studies and workspace (the `decay-with-viz` / `steady-state` studies and the
> `tellurium-sbml-simulation` investigation) have moved to the
> **[viva-showcase](https://github.com/vivarium-collective/viva-showcase)**
> workspace, which pins `viva-tellurium==0.2.4` from PyPI.

## Installation

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```python
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from viva_tellurium import TelluriumProcess, make_tellurium_document

ANTIMONY = """
model decay
  S1 = 10; S2 = 0
  S1 -> S2; k*S1; k = 0.3
end
"""

core = allocate_core()
core.register_link('TelluriumProcess', TelluriumProcess)
core.register_link('ram-emitter', RAMEmitter)

doc = make_tellurium_document(model=ANTIMONY, interval=1.0)
sim = Composite({'state': doc}, core=core)
sim.run(10.0)

print(gather_emitter_results(sim)[('emitter',)][-1]['species'])
```

## API

### `TelluriumProcess` (Process)

Time-driven bridge wrapping a RoadRunner instance. Lazy init; each `update()`
advances by `interval`.

**Species port is a bulk numpy `Array` by default** (`bulk_species=True`) — it's
moved through roadrunner's bulk C API (one buffer copy instead of a per-species
`map[float]` loop), ~2× faster per `update()` and bit-identical. Wire `species`
to an Array-shaped store. The default ports:

| Port            | Dir    | Schema (default, `bulk_species=True`) |
|-----------------|--------|---------------------------------------|
| `species`       | input  | `maybe[array[(n,),float]]`            |
| `parameters_in` | input  | `map[float]`                          |
| `species`       | output | `overwrite[array[(n,),float]]`        |
| `rates`         | output | `overwrite[map[float]]`               |
| `parameters`    | output | `overwrite[map[float]]`               |
| `time`          | output | `overwrite[float]`                    |

`species` is ordered by `getFloatingSpeciesIds()`. For **coupled multi-process**
composites (two+ processes sharing one species store), also set
`coupled_species=True` — the species output becomes an *additive* array so the
arms' deltas sum under reconcile. For the **legacy per-species `map[float]`**
port (slower; only if a consumer needs a dict-shaped species store), set
`bulk_species=False`.

`parameters_in` changes global parameters while the model runs (a rate
constant, or an input parameter another layer drives). Before each interval is
simulated, every `{parameter_id: value}` on the port is set in roadrunner, the
same way the config `parameter_overrides` are set at build. So
`parameter_overrides` gives the initial value, and the port overrides it from
the first update that carries it.
- A value stays in effect until the port changes it; removing a key keeps its
  last value. A process writing to the port's store should declare its output
  `overwrite[map[float]]` (or `overwrite[float]` when wired to one key):
  a plain `map[float]` output *adds* to the store each update.
- The whole map is checked before anything is set. An id that is not a global
  parameter, or is defined by an assignment rule, and a value that is not a
  finite number, raise a `ValueError`.
- Parameters declared `constant="true"` in SBML can be set. A parameter with a
  rate rule is a state variable: setting it restarts its integration from the
  given value.
- An unwired or empty port changes nothing.
- It is a separate port from the `parameters` output, which overwrites every
  global parameter after each update, so wire them to different stores.

Config (all optional, one of `model`/`model_file` required):
`model`, `model_format` (`'antimony'`/`'sbml'`), `model_file`, `integrator`
(`'cvode'`/`'gillespie'`), `absolute_tolerance`, `relative_tolerance`,
`seed`, `species_overrides`, `parameter_overrides`, `reset_on_init`,
`bulk_species` (default `True`), `coupled_species`, `conserved_moiety_analysis`.

### `TelluriumStep` (Step)

One-shot trajectory Step. Extra config: `start_time`, `end_time`, `n_points`,
`output_times`, `selections`. Outputs: `time_series` (list),
`species_trajectories` (map[list]).

`output_times` is an explicit, possibly non-uniform list of output time points
(assumed sorted ascending). When non-empty, the time course is sampled at
exactly those times (roadrunner `simulate(times=[...])`), overriding the
uniform `start_time`/`end_time`/`n_points` grid, and the returned `time_series`
is those points verbatim. Empty (the default) keeps the uniform behavior. It
composes with `selections` and `species_units`. The key name matches the
viva-copasi wrapper (copasi's term is `values`, tellurium's is `times`).

`selections` is a list of roadrunner selection strings (e.g. `'time'`, `'S1'`,
`'[S1]'` for a concentration, a reaction id for a flux). Empty (the default)
keeps roadrunner's built-in output — time + all floating species for a time
course, and floating-species concentrations for a steady state. When set, the
trajectory/steady-state keys are exactly the requested selections (`'time'`,
if requested in a time course, is pulled out into `time_series`). The
steady-state step uses roadrunner's `steadyStateSelections` path. The key name
and semantics match the viva-copasi and viva-biomodels wrappers.

### `make_tellurium_document(...)`

Ready-to-run composite document wiring a `TelluriumProcess` to a stores dict
and a RAM emitter.

## Demo & Tests

```bash
python demo/demo_report.py   # regenerates demo/report.html, opens in Safari
pytest                       # offline package + contract tests
```

# viva-tellurium

Process-bigraph wrapper for [Tellurium](https://tellurium.analogmachine.org/) /
[libroadrunner](https://libroadrunner.org/), exposing any SBML or Antimony
kinetic model as a `process-bigraph` Process so it can be composed with
other simulators in a bigraph document.

**[View Interactive Demo Report](https://vivarium-collective.github.io/viva-tellurium/)** — Lotka-Volterra predator-prey, Elowitz-Leibler repressilator, and Gillespie stochastic dimerization with Plotly time series, phase portraits, reaction-rate charts, and bigraph architecture diagrams.

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

| Port         | Dir    | Schema                  |
|--------------|--------|-------------------------|
| `species`    | input  | `maybe[map[float]]`     |
| `species`    | output | `overwrite[map[float]]` |
| `rates`      | output | `overwrite[map[float]]` |
| `parameters` | output | `overwrite[map[float]]` |
| `time`       | output | `overwrite[float]`      |

Config (all optional, one of `model`/`model_file` required):
`model`, `model_format` (`'antimony'`/`'sbml'`), `model_file`, `integrator`
(`'cvode'`/`'gillespie'`), `absolute_tolerance`, `relative_tolerance`,
`seed`, `species_overrides`, `parameter_overrides`, `reset_on_init`.

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
pytest tests/                # 17 offline tests
```

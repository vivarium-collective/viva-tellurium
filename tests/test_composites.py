"""Integration tests for Tellurium composites."""

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter
from viva_tellurium.processes import TelluriumProcess
from viva_tellurium.composites import make_tellurium_document


MODEL = """
model decay
  S1 = 10; S2 = 0
  S1 -> S2; k*S1; k = 0.3
end
"""


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('TelluriumProcess', TelluriumProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_composite_assembly(core):
    doc = make_tellurium_document(model=MODEL, interval=1.0)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_composite_short_run(core):
    doc = make_tellurium_document(model=MODEL, interval=1.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(5.0)

    stores = sim.state['stores']
    assert 'species' in stores
    assert stores['species']['S1'] < 10.0
    assert stores['time'] == pytest.approx(5.0)


def test_emitter_collects_timeseries(core):
    doc = make_tellurium_document(model=MODEL, interval=0.5)
    sim = Composite({'state': doc}, core=core)
    sim.run(5.0)

    raw = gather_emitter_results(sim)
    rows = raw[('emitter',)]
    assert len(rows) >= 5

    # Energy-style monotonic check: S1 decreases over time
    s1_vals = [r['species']['S1'] for r in rows if r.get('species')]
    assert s1_vals[0] > s1_vals[-1]


def test_document_factory_with_overrides(core):
    doc = make_tellurium_document(
        model=MODEL,
        species_overrides={'S1': 50.0},
        parameter_overrides={'k': 0.1},
        interval=1.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)

    stores = sim.state['stores']
    # Slower decay with k=0.1, starting from 50
    assert stores['species']['S1'] > 30.0
    assert stores['parameters']['k'] == 0.1


def test_initial_state_emitted_at_t0(core):
    """The step-0 emitter row must carry the model's t=0 state.

    Regression for #7: a TelluriumProcess whose outputs are wired to empty
    stores (like decay-demo) used to emit an empty first row
    ({"species": {}, "parameters": {}, time 0}) because initial_state() never
    reached the wired stores. The first real values only appeared at step 1.
    The t=0 row must instead carry the model's initial species, parameters
    and time, the way an OdeProcess composite's does.
    """
    doc = make_tellurium_document(model=MODEL, interval=0.5)
    sim = Composite({'state': doc}, core=core)
    sim.run(2.0)

    rows = gather_emitter_results(sim)[('emitter',)]

    row0 = rows[0]
    assert row0['time'] == 0.0
    assert row0.get('global_time') == 0.0
    # t=0 species/parameters are the model's initial state, not empty maps.
    assert row0['species'] == {'S1': 10.0, 'S2': 0.0}
    assert row0['parameters']['k'] == 0.3

    # Numerical behavior after t=0 is unchanged: S1 has decayed by step 1.
    row1 = rows[1]
    assert row1['time'] == pytest.approx(0.5)
    assert row1['species']['S1'] < 10.0
    assert row1['species']['S2'] > 0.0


def test_roundtrip_species_conservation(core):
    """Mass conservation over a composite run."""
    doc = make_tellurium_document(model=MODEL, interval=1.0)
    sim = Composite({'state': doc}, core=core)
    sim.run(10.0)
    stores = sim.state['stores']
    total = stores['species']['S1'] + stores['species']['S2']
    assert total == pytest.approx(10.0, rel=1e-3)

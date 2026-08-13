"""Unit tests for TelluriumProcess."""

import pytest
from process_bigraph import allocate_core
from viva_tellurium.processes import TelluriumProcess, TelluriumUTCStep, TelluriumSteadyStateStep


MODEL_DECAY = """
model decay
  S1 = 10; S2 = 0
  S1 -> S2; k*S1; k = 0.3
end
"""

MODEL_OSC = """
model lotka
  P = 10; W = 10
  J1: -> P; kg*P
  J2: P -> W; kc*P*W
  J3: W -> ; kd*W
  kg = 1.0; kc = 0.1; kd = 1.0
end
"""


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('TelluriumProcess', TelluriumProcess)
    c.register_link('TelluriumUTCStep', TelluriumUTCStep)
    c.register_link('TelluriumSteadyStateStep', TelluriumSteadyStateStep)
    return c


def test_instantiation(core):
    proc = TelluriumProcess(config={'model': MODEL_DECAY}, core=core)
    assert proc.config['integrator'] == 'cvode'
    assert proc.config['model'].startswith('\nmodel decay')


def test_missing_model_raises(core):
    proc = TelluriumProcess(config={}, core=core)
    with pytest.raises(ValueError):
        proc.initial_state()


def test_initial_state(core):
    proc = TelluriumProcess(config={'model': MODEL_DECAY}, core=core)
    state = proc.initial_state()
    assert 'species' in state
    assert 'rates' in state
    assert 'parameters' in state
    assert 'time' in state
    assert state['species']['S1'] == 10.0
    assert state['species']['S2'] == 0.0
    assert state['parameters']['k'] == 0.3
    assert state['time'] == 0.0


def test_single_update_advances_time(core):
    proc = TelluriumProcess(config={'model': MODEL_DECAY}, core=core)
    proc.initial_state()
    result = proc.update({}, interval=2.0)
    assert result['time'] == pytest.approx(2.0)
    # S1 should have decayed
    assert result['species']['S1'] < 10.0
    assert result['species']['S2'] > 0.0
    # Conservation
    total = result['species']['S1'] + result['species']['S2']
    assert total == pytest.approx(10.0, rel=1e-4)


def test_multiple_updates_accumulate(core):
    proc = TelluriumProcess(config={'model': MODEL_DECAY}, core=core)
    proc.initial_state()
    for _ in range(5):
        proc.update({}, interval=1.0)
    state = proc._read_state()
    assert state['time'] == pytest.approx(5.0)


def test_species_overrides(core):
    proc = TelluriumProcess(
        config={
            'model': MODEL_DECAY,
            'species_overrides': {'S1': 50.0, 'S2': 5.0},
        }, core=core)
    state = proc.initial_state()
    assert state['species']['S1'] == 50.0
    assert state['species']['S2'] == 5.0


def test_parameter_overrides(core):
    proc = TelluriumProcess(
        config={
            'model': MODEL_DECAY,
            'parameter_overrides': {'k': 1.5},
        }, core=core)
    state = proc.initial_state()
    assert state['parameters']['k'] == 1.5


def test_input_coupling(core):
    """Pushed species values should override RR state before simulating."""
    proc = TelluriumProcess(config={'model': MODEL_DECAY}, core=core)
    proc.initial_state()
    # Push S1 back up to 100, then simulate a tiny interval
    result = proc.update({'species': {'S1': 100.0}}, interval=0.01)
    # After a tiny interval with S1=100, rate k*S1 = 30 should have
    # consumed only a small amount, so S1 remains close to 100.
    assert result['species']['S1'] > 90.0


def test_outputs_schema(core):
    proc = TelluriumProcess(config={'model': MODEL_DECAY}, core=core)
    outputs = proc.outputs()
    assert 'species' in outputs
    assert 'rates' in outputs
    assert 'parameters' in outputs
    assert 'time' in outputs


def test_convenience_accessors(core):
    proc = TelluriumProcess(config={'model': MODEL_OSC}, core=core)
    assert set(proc.get_species_ids()) == {'P', 'W'}
    reactions = proc.get_reaction_ids()
    assert len(reactions) == 3
    sbml = proc.get_sbml()
    assert '<sbml' in sbml


def test_gillespie_integrator(core):
    proc = TelluriumProcess(
        config={
            'model': MODEL_DECAY,
            'integrator': 'gillespie',
            'seed': 42,
            'species_overrides': {'S1': 100.0},
        }, core=core)
    state = proc.initial_state()
    assert state['species']['S1'] == 100.0
    result = proc.update({}, interval=5.0)
    # Gillespie advances stochastically; may finish early if no events remain.
    assert result['time'] > 0.0
    assert result['species']['S1'] < 100.0


def test_tellurium_step(core):
    step = TelluriumUTCStep(
        config={
            'model': MODEL_DECAY,
            'start_time': 0.0,
            'end_time': 10.0,
            'n_points': 11,
        }, core=core)
    result = step.update({})
    assert len(result['time_series']) == 11
    assert result['time_series'][0] == 0.0
    assert result['time_series'][-1] == 10.0
    assert 'S1' in result['species_trajectories']
    assert len(result['species_trajectories']['S1']) == 11
    # S1 should decrease monotonically
    s1 = result['species_trajectories']['S1']
    assert s1[0] > s1[-1]


def test_tellurium_steady_state_step(core):
    """SteadyStateStep loads a model and returns species concentrations at equilibrium.

    MODEL_DECAY (S1 -> S2 with first-order decay) has a trivial steady state
    at S1=0, S2=anything, because the only flux is the irreversible decay.
    RoadRunner's steadyState() converges to that equilibrium successfully.
    """
    from viva_tellurium.processes import TelluriumSteadyStateStep

    step = TelluriumSteadyStateStep(
        config={'model': MODEL_DECAY, 'model_format': 'antimony'},
        core=core,
    )
    out = step.update({})
    assert 'steady_state_concentrations' in out
    concs = out['steady_state_concentrations']
    assert isinstance(concs, dict)
    assert len(concs) > 0
    # All values must be finite floats
    import math
    for sid, val in concs.items():
        assert isinstance(val, float), f"{sid} is not float: {type(val)}"
        assert math.isfinite(val), f"{sid} steady-state concentration is not finite: {val}"

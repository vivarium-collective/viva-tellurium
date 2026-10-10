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

# Compartment C has a NON-UNIT volume (5), so amount != concentration and the
# conversion is observable. S1 is a normal (concentration-based) species;
# S2 is substanceOnly (hasOnlySubstanceUnits=true). The issue (#14) is about
# returning these consistently regardless of that flag.
MODEL_UNITS = """
model units_test
  compartment C = 5;
  species S1 in C = 20;
  substanceOnly species S2 in C = 30;
  S1 -> S2; k*S1; k = 0.1;
end
"""
COMPARTMENT_VOLUME = 5.0


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


def test_utc_step_applies_tolerances(core):
    """TelluriumUTCStep should apply absolute/relative tolerances to the integrator."""
    step = TelluriumUTCStep(
        config={
            'model': MODEL_DECAY,
            'absolute_tolerance': 1e-3,
            'relative_tolerance': 1e-2,
        }, core=core)
    step.update({})
    assert step._rr.integrator.absolute_tolerance == pytest.approx(1e-3)
    assert step._rr.integrator.relative_tolerance == pytest.approx(1e-2)


def test_utc_step_applies_seed(core):
    """TelluriumUTCStep should apply the configured seed to a stochastic integrator."""
    step = TelluriumUTCStep(
        config={
            'model': MODEL_DECAY,
            'integrator': 'gillespie',
            'seed': 5,
        }, core=core)
    step.update({})
    assert step._rr.integrator.seed == 5


def test_steady_state_step_applies_tolerances(core):
    """TelluriumSteadyStateStep should apply absolute/relative tolerances to the integrator."""
    step = TelluriumSteadyStateStep(
        config={
            'model': MODEL_DECAY,
            'absolute_tolerance': 1e-3,
            'relative_tolerance': 1e-2,
        }, core=core)
    step.update({})
    assert step._rr.integrator.absolute_tolerance == pytest.approx(1e-3)
    assert step._rr.integrator.relative_tolerance == pytest.approx(1e-2)


def test_step_defaults_unchanged(core):
    """With no tolerance config, the Step uses the schema defaults on the integrator."""
    step = TelluriumUTCStep(config={'model': MODEL_DECAY}, core=core)
    step.update({})
    assert step._rr.integrator.absolute_tolerance == pytest.approx(1e-10)
    assert step._rr.integrator.relative_tolerance == pytest.approx(1e-8)


def test_utc_default_selections_unchanged(core):
    """Without a `selections` key, UTC output is time + all floating species
    (bracket-stripped), exactly the pre-existing behavior."""
    step = TelluriumUTCStep(
        config={'model': MODEL_DECAY, 'start_time': 0.0,
                'end_time': 10.0, 'n_points': 11},
        core=core)
    result = step.update({})
    assert result['time_series'][0] == 0.0
    assert result['time_series'][-1] == 10.0
    assert set(result['species_trajectories'].keys()) == {'S1', 'S2'}


def test_utc_selections_specific_species(core):
    """A `selections` list makes the output columns exactly the requested
    roadrunner selections (here a single concentration `[S1]`)."""
    step = TelluriumUTCStep(
        config={'model': MODEL_DECAY, 'start_time': 0.0,
                'end_time': 10.0, 'n_points': 11,
                'selections': ['time', '[S1]']},
        core=core)
    result = step.update({})
    # time is pulled out into time_series; the rest are the requested columns.
    assert set(result['species_trajectories'].keys()) == {'[S1]'}
    assert len(result['species_trajectories']['[S1]']) == 11
    assert len(result['time_series']) == 11
    assert result['time_series'][0] == 0.0
    # S2 was NOT requested, so it must be absent.
    assert 'S2' not in result['species_trajectories']
    assert '[S2]' not in result['species_trajectories']


def test_utc_selections_reaction_flux(core):
    """Reaction fluxes (not just species) can be selected."""
    step = TelluriumUTCStep(
        config={'model': MODEL_OSC, 'start_time': 0.0,
                'end_time': 5.0, 'n_points': 6,
                'selections': ['time', 'J2']},
        core=core)
    result = step.update({})
    assert set(result['species_trajectories'].keys()) == {'J2'}
    assert len(result['species_trajectories']['J2']) == 6


def test_utc_selections_without_time(core):
    """When `time` is not requested, time_series falls back to the uniform
    grid so the output contract still holds."""
    step = TelluriumUTCStep(
        config={'model': MODEL_DECAY, 'start_time': 0.0,
                'end_time': 10.0, 'n_points': 11,
                'selections': ['[S1]']},
        core=core)
    result = step.update({})
    assert set(result['species_trajectories'].keys()) == {'[S1]'}
    assert len(result['time_series']) == 11
    assert result['time_series'][0] == 0.0
    assert result['time_series'][-1] == 10.0


def test_steady_state_default_selections_unchanged(core):
    """Without `selections`, steady state returns floating-species
    concentrations keyed by species id, as before."""
    step = TelluriumSteadyStateStep(
        config={'model': MODEL_DECAY, 'model_format': 'antimony'},
        core=core)
    out = step.update({})
    assert set(out['steady_state_concentrations'].keys()) == {'S1', 'S2'}


def test_steady_state_selections(core):
    """A `selections` list drives roadrunner's steadyStateSelections so the
    output keys are exactly the requested selections."""
    step = TelluriumSteadyStateStep(
        config={'model': MODEL_DECAY, 'model_format': 'antimony',
                'selections': ['[S1]', '[S2]']},
        core=core)
    out = step.update({})
    concs = out['steady_state_concentrations']
    assert set(concs.keys()) == {'[S1]', '[S2]'}
    import math
    for v in concs.values():
        assert isinstance(v, float) and math.isfinite(v)


# --- species_units (amount vs concentration, issue #14) --------------------

def test_species_units_defaults_to_concentration(core):
    """The species_units config defaults to 'concentration'."""
    step = TelluriumUTCStep(config={'model': MODEL_DECAY}, core=core)
    assert step.config['species_units'] == 'concentration'


def test_utc_default_output_is_concentration(core):
    """With no species_units key, the default UTC output is concentration
    (pre-existing behavior pinned): for MODEL_UNITS, S1 conc=20, S2 conc=6."""
    step = TelluriumUTCStep(
        config={'model': MODEL_UNITS, 'start_time': 0.0,
                'end_time': 5.0, 'n_points': 2},
        core=core)
    out = step.update({})
    traj = out['species_trajectories']
    assert set(traj.keys()) == {'S1', 'S2'}
    assert traj['S1'][0] == pytest.approx(20.0)
    assert traj['S2'][0] == pytest.approx(6.0)


def test_utc_species_units_amount(core):
    """species_units='amount' returns amounts for ALL floating species,
    including the hasOnlySubstanceUnits=true one (S2), converted with the
    compartment volume: amount = concentration * volume."""
    step = TelluriumUTCStep(
        config={'model': MODEL_UNITS, 'start_time': 0.0,
                'end_time': 5.0, 'n_points': 2,
                'species_units': 'amount'},
        core=core)
    out = step.update({})
    traj = out['species_trajectories']
    assert set(traj.keys()) == {'S1', 'S2'}
    # S1 conc 20 * 5 = 100; S2 conc 6 * 5 = 30 (S2 is substance-only)
    assert traj['S1'][0] == pytest.approx(20.0 * COMPARTMENT_VOLUME)
    assert traj['S2'][0] == pytest.approx(6.0 * COMPARTMENT_VOLUME)


def test_utc_amount_vs_concentration_relationship(core):
    """Across the whole trajectory, amount == concentration * volume for every
    species, demonstrating consistent unit handling (not the raw simulator's
    mixed amount/concentration default)."""
    conc_step = TelluriumUTCStep(
        config={'model': MODEL_UNITS, 'start_time': 0.0,
                'end_time': 5.0, 'n_points': 6,
                'species_units': 'concentration'},
        core=core)
    amt_step = TelluriumUTCStep(
        config={'model': MODEL_UNITS, 'start_time': 0.0,
                'end_time': 5.0, 'n_points': 6,
                'species_units': 'amount'},
        core=core)
    conc = conc_step.update({})['species_trajectories']
    amt = amt_step.update({})['species_trajectories']
    for sid in ('S1', 'S2'):
        for c, a in zip(conc[sid], amt[sid]):
            assert a == pytest.approx(c * COMPARTMENT_VOLUME)


def test_species_units_invalid_raises(core):
    """An unknown species_units value fails loud."""
    step = TelluriumUTCStep(
        config={'model': MODEL_UNITS, 'species_units': 'particles'},
        core=core)
    with pytest.raises(ValueError):
        step.update({})


def test_has_only_substance_units_accessor(core):
    """The wrapper notes each species' hasOnlySubstanceUnits flag."""
    step = TelluriumUTCStep(config={'model': MODEL_UNITS}, core=core)
    flags = step.get_has_only_substance_units()
    assert flags['S1'] is False
    assert flags['S2'] is True


def test_initial_state_species_units_amount(core):
    """initial_state honors species_units='amount'."""
    step = TelluriumUTCStep(
        config={'model': MODEL_UNITS, 'species_units': 'amount'}, core=core)
    state = step.initial_state()
    conc = state['species_concentrations']
    assert conc['S1'] == pytest.approx(20.0 * COMPARTMENT_VOLUME)
    assert conc['S2'] == pytest.approx(6.0 * COMPARTMENT_VOLUME)


def test_steady_state_species_units_amount(core):
    """Steady-state default output honors species_units='amount' (converted
    with the compartment volume)."""
    conc_step = TelluriumSteadyStateStep(
        config={'model': MODEL_UNITS, 'species_units': 'concentration'},
        core=core)
    amt_step = TelluriumSteadyStateStep(
        config={'model': MODEL_UNITS, 'species_units': 'amount'},
        core=core)
    conc = conc_step.update({})['steady_state_concentrations']
    amt = amt_step.update({})['steady_state_concentrations']
    assert set(conc.keys()) == {'S1', 'S2'} == set(amt.keys())
    for sid in ('S1', 'S2'):
        assert amt[sid] == pytest.approx(conc[sid] * COMPARTMENT_VOLUME)


def test_explicit_selections_not_normalized_by_species_units(core):
    """Explicit `selections` are roadrunner's vocabulary and are respected
    VERBATIM even when species_units is set: `[S1]` stays concentration and
    bare `S1` stays amount; species_units does not rewrite them."""
    step = TelluriumUTCStep(
        config={'model': MODEL_UNITS, 'start_time': 0.0,
                'end_time': 5.0, 'n_points': 2,
                'species_units': 'amount',
                'selections': ['time', '[S1]', 'S2']},
        core=core)
    out = step.update({})
    traj = out['species_trajectories']
    assert set(traj.keys()) == {'[S1]', 'S2'}
    # [S1] is a concentration (20) despite species_units='amount'
    assert traj['[S1]'][0] == pytest.approx(20.0)
    # bare S2 is an amount (30)
    assert traj['S2'][0] == pytest.approx(30.0)


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

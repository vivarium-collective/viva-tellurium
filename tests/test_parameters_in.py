"""The ``parameters_in`` input port: SBML parameters set while the model runs.

Other process-bigraph layers drive a kinetic model by changing its parameters
between intervals (a rate constant, or a non-constant input parameter such as
BIOMD0000000786's ``TNF_R``). The port carries ``{global_parameter_id: value}``;
TelluriumProcess sets each value in roadrunner before it simulates the interval.

Every test runs the real libroadrunner engine and checks the dynamics against
the analytic solution of first-order decay, S(t + dt) = S(t) * exp(-k * dt).
"""

from itertools import pairwise

import libsbml
import numpy as np
import pytest
from process_bigraph import Composite, Process, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from viva_tellurium import TelluriumProcess, make_tellurium_document

K0 = 0.3
K1 = 1.2
S0 = 10.0
DT = 1.0

# Antimony declares a plain `k = 0.3` as constant="true" in SBML.
MODEL_DECAY = f"""
model decay
  S = {S0}; P = 0
  J: S -> P; k*S
  k = {K0}
end
"""

# The same model with the parameter declared `const` explicitly.
MODEL_CONST = f"""
model decay_const
  S = {S0}; P = 0
  J: S -> P; k*S
  const k = {K0}
end
"""

# Lipniacki-style input (BIOMD0000000786): a boundary species TNF follows a
# non-constant parameter TNF_R through the assignment rule TNF := TNF_R, and
# gates a first-order reaction as a modifier.
K_TNF = 0.5
MODEL_TNF_RULE = f"""
model tnf_rule
  IKKn = 1; IKKa = 0
  var TNF_R = 0
  species $TNF = 0
  TNF := TNF_R
  J1: IKKn -> IKKa; k1*TNF*IKKn
  k1 = {K_TNF}
end
"""

# A global parameter determined by an assignment rule.
MODEL_RULE_PARAM = """
model rule_param
  S = 10; P = 0
  J: S -> P; krule*S
  kv = 0.1
  krule := 2*kv
end
"""

MODES = [
    pytest.param(bulk, coupled, id=f"bulk={bulk}-coupled={coupled}")
    for bulk in (True, False) for coupled in (False, True)]


def _process(model, **config):
    return TelluriumProcess(config={'model': model, **config}, core=allocate_core())


def _parameter_is_constant(proc, pid):
    """The SBML `constant` attribute of a global parameter, read with libSBML."""
    doc = libsbml.readSBMLFromString(proc.get_sbml())
    return doc.getModel().getParameter(pid).getConstant()


class _Store:
    """The species store a composite would hold for this process.

    Single-process mode: the output is absolute and replaces the store.
    Coupled mode: the store is wired to the species input and the output is an
    additive delta, as process-bigraph applies it.
    """

    def __init__(self, proc):
        self.proc = proc
        self.coupled = proc.config['coupled_species']
        self.value = proc.initial_state()['species']

    def step(self, interval, parameters_in=None):
        state = {}
        if parameters_in is not None:
            state['parameters_in'] = parameters_in
        if self.coupled:
            state['species'] = self.value
        out = self.proc.update(state, interval)
        if not self.coupled:
            self.value = out['species']
        elif isinstance(self.value, dict):
            self.value = {sid: self.value[sid] + out['species'][sid] for sid in self.value}
        else:
            self.value = self.value + out['species']
        return out

    def get(self, sid):
        if isinstance(self.value, dict):
            return float(self.value[sid])
        return float(self.value[self.proc.get_species_ids().index(sid)])


def test_port_is_declared_separately_from_parameters_output():
    for bulk, coupled in [(b, c) for b in (True, False) for c in (False, True)]:
        proc = _process(MODEL_DECAY, bulk_species=bulk, coupled_species=coupled)
        assert proc.inputs()['parameters_in'] == 'map[float]'
        assert 'parameters_in' not in proc.outputs()
        assert proc.outputs()['parameters'] == 'overwrite[map[float]]'


@pytest.mark.parametrize('bulk, coupled', MODES)
def test_parameters_in_changes_rate_mid_run(bulk, coupled):
    """One interval at k0, then k set through the port: the second interval
    follows exponential decay at the new k, in every species-port mode."""
    proc = _process(MODEL_DECAY, bulk_species=bulk, coupled_species=coupled)
    store = _Store(proc)

    store.step(DT)
    s1 = store.get('S')
    assert s1 == pytest.approx(S0 * np.exp(-K0 * DT), rel=1e-6)

    out = store.step(DT, parameters_in={'k': K1})
    s2 = store.get('S')
    assert s2 == pytest.approx(s1 * np.exp(-K1 * DT), rel=1e-6)
    assert out['parameters']['k'] == K1
    assert out['time'] == pytest.approx(2 * DT)


def test_parameters_in_through_a_composite():
    """Wired to a store in a real Composite, the port overrides the config's
    parameter_overrides from the first update on."""
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ram-emitter', RAMEmitter)
    doc = make_tellurium_document(
        model=MODEL_DECAY, interval=DT, parameter_overrides={'k': 0.05})
    doc['tellurium']['inputs']['parameters_in'] = ['stores', 'parameters_in']
    doc['stores']['parameters_in'] = {'k': K1}

    sim = Composite({'state': doc}, core=core)
    sim.run(3 * DT)

    proc = sim.state['tellurium']['instance']
    index = proc.get_species_ids().index('S')
    results = gather_emitter_results(sim)[('emitter',)]
    assert len(results) == 4
    for row in results[1:]:
        expected = S0 * np.exp(-K1 * row['time'])
        assert float(np.asarray(row['species'])[index]) == pytest.approx(expected, rel=1e-6)
        assert row['parameters']['k'] == K1


def test_constant_parameter_can_be_set_through_the_port():
    """A parameter declared constant="true" in SBML is still set by roadrunner
    at runtime, and the dynamics follow the new value."""
    proc = _process(MODEL_CONST, bulk_species=False)
    assert _parameter_is_constant(proc, 'k') is True
    proc.initial_state()

    s1 = proc.update({}, DT)['species']['S']
    out = proc.update({'parameters_in': {'k': K1}}, DT)

    assert out['parameters']['k'] == K1
    assert out['species']['S'] == pytest.approx(s1 * np.exp(-K1 * DT), rel=1e-6)


@pytest.mark.parametrize('bulk', [True, False])
def test_non_constant_parameter_drives_an_assignment_rule(bulk):
    """Setting TNF_R through the port moves TNF := TNF_R, and the reaction it
    gates switches on at the new value."""
    proc = _process(MODEL_TNF_RULE, bulk_species=bulk)
    assert _parameter_is_constant(proc, 'TNF_R') is False
    store = _Store(proc)

    store.step(DT)
    assert store.get('IKKn') == pytest.approx(1.0, abs=1e-12)
    assert proc._rr['TNF'] == 0.0

    out = store.step(DT, parameters_in={'TNF_R': 1.0})
    assert out['parameters']['TNF_R'] == 1.0
    assert proc._rr['TNF'] == 1.0
    assert store.get('IKKn') == pytest.approx(np.exp(-K_TNF * 1.0 * DT), rel=1e-6)


@pytest.mark.parametrize('bulk, coupled', MODES)
def test_empty_or_absent_input_changes_nothing(bulk, coupled):
    """No key, an empty map, and None all give bit-identical results, at the
    model's own parameter value."""
    runs = []
    for parameters_in in (None, {}):
        proc = _process(MODEL_DECAY, bulk_species=bulk, coupled_species=coupled)
        store = _Store(proc)
        outs = [store.step(DT, parameters_in) for _ in range(3)]
        runs.append((store.get('S'), [o['parameters'] for o in outs]))

    assert runs[0] == runs[1]
    assert runs[0][0] == pytest.approx(S0 * np.exp(-K0 * 3 * DT), rel=1e-6)
    assert all(p == {'k': K0} for p in runs[0][1])


def test_unknown_id_raises_and_sets_nothing():
    """Ids that are not global parameters (a typo, a species) raise a
    ValueError naming them, before any value is set."""
    proc = _process(MODEL_DECAY)
    proc.initial_state()
    with pytest.raises(ValueError, match=r"\['S', 'kk'\] are not settable global parameters"):
        proc.update({'parameters_in': {'k': K1, 'kk': 1.0, 'S': 99.0}}, DT)
    assert proc._rr['k'] == K0
    assert proc._rr['S'] == S0


def test_rule_determined_parameter_raises_and_sets_nothing():
    """A global parameter defined by an assignment rule cannot be set: the
    update raises before the valid key beside it is applied."""
    proc = _process(MODEL_RULE_PARAM)
    proc.initial_state()
    with pytest.raises(ValueError, match=r"\['krule'\] are not settable"):
        proc.update({'parameters_in': {'kv': 9.0, 'krule': 1.0}}, DT)
    assert proc._rr['kv'] == 0.1
    assert proc._rr.getCurrentTime() == 0.0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, '2.0', None])
def test_non_finite_or_non_numeric_value_raises_and_sets_nothing(bad):
    proc = _process(MODEL_DECAY)
    proc.initial_state()
    with pytest.raises(ValueError, match=r"values for \['k'\] are not finite numbers"):
        proc.update({'parameters_in': {'k': bad}}, DT)
    assert proc._rr['k'] == K0
    assert proc._rr.getCurrentTime() == 0.0


class _ConstantDriver(Process):
    """Writes one fixed {parameter_id: value} map each update, declared
    overwrite so it replaces the store's value rather than adding to it."""

    config_schema = {'values': 'map[float]'}

    def inputs(self):
        return {}

    def outputs(self):
        return {'values': 'overwrite[map[float]]'}

    def update(self, state, interval):
        return {'values': dict(self.config['values'])}


def test_parameters_in_driven_by_another_process():
    """Another process writes k into the port's store. Each interval decays at
    the k the process reports using, and k holds at the driver's value instead
    of accumulating."""
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    core.register_link('ConstantDriver', _ConstantDriver)
    core.register_link('ram-emitter', RAMEmitter)
    doc = make_tellurium_document(model=MODEL_DECAY, interval=DT)
    doc['tellurium']['inputs']['parameters_in'] = ['stores', 'parameters_in']
    doc['stores']['parameters_in'] = {'k': K0}
    doc['driver'] = {
        '_type': 'process',
        'address': 'local:ConstantDriver',
        'config': {'values': {'k': K1}},
        'interval': DT,
        'inputs': {},
        'outputs': {'values': ['stores', 'parameters_in']},
    }

    sim = Composite({'state': doc}, core=core)
    sim.run(4 * DT)

    proc = sim.state['tellurium']['instance']
    index = proc.get_species_ids().index('S')
    results = gather_emitter_results(sim)[('emitter',)]
    assert len(results) == 5
    ks = [row['parameters']['k'] for row in results[1:]]
    assert ks[-1] == K1
    assert set(ks) <= {K0, K1}
    for before, after in pairwise(results):
        s_before = float(np.asarray(before['species'])[index])
        s_after = float(np.asarray(after['species'])[index])
        k_used = after['parameters']['k']
        assert s_after == pytest.approx(s_before * np.exp(-k_used * DT), rel=1e-6)


def test_parameters_in_applies_after_a_model_in_reload():
    """When model_in and parameters_in arrive together, the parameter is set
    on the reloaded model, not lost with the old one."""
    source = _process(MODEL_DECAY, bulk_species=False)
    source.initial_state()
    source.update({}, DT)
    sbml = source.get_sbml()
    s1 = source._rr['S']

    proc = _process(MODEL_DECAY, bulk_species=False)
    proc.initial_state()
    out = proc.update({'model_in': sbml, 'parameters_in': {'k': K1}}, DT)

    assert out['parameters']['k'] == K1
    assert out['species']['S'] == pytest.approx(s1 * np.exp(-K1 * DT), rel=1e-6)

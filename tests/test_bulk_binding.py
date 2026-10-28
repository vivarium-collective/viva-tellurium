"""Bulk-binding mode: species cross the pbg wire as one numpy Array leaf and
are pushed/pulled via roadrunner's bulk C API (model.setFloatingSpeciesConcentrations
/ getFloatingSpeciesConcentrations) instead of a per-species string-keyed loop.

Contract: bulk mode is BIT-IDENTICAL to the default map[float] mode on the
species trajectory; only the representation and the set/get path change.
"""
import numpy as np
import pytest

from process_bigraph import allocate_core
from viva_tellurium.processes import TelluriumProcess

MODEL = """
    S1 = 10; S2 = 2; S3 = 5;
    k1 = 0.3; k2 = 0.15; k3 = 0.4;
    J1: S1 -> S2; k1*S1;
    J2: S2 -> S3; k2*S2;
    J3: S3 -> S1; k3*S3;
"""


@pytest.fixture
def core():
    return allocate_core()


def _run(proc, steps, interval, perturb=None):
    """Drive a process `steps` updates, feeding its own species output back as
    input each step (coupling). Returns the species trajectory as a list of
    {sid: value} dicts, representation-independent."""
    ids = proc._species_ids if proc._species_ids else (
        proc._build() or proc._species_ids)
    traj = []
    incoming = None
    for i in range(steps):
        state = {'species': incoming} if incoming is not None else {}
        if perturb is not None and i == 0:
            state = {'species': perturb}
        out = proc.update(state, interval)
        sp = out['species']
        as_dict = dict(zip(proc._species_ids, np.asarray(sp))) if not isinstance(sp, dict) else dict(sp)
        traj.append(as_dict)
        incoming = sp
    return traj


def test_bulk_ports_are_arrays(core):
    proc = TelluriumProcess(config={'model': MODEL, 'bulk_species': True}, core=core)
    ins, outs = proc.inputs(), proc.outputs()
    assert ins['species'] == 'maybe[array]' or ins['species'].startswith('maybe[array')
    assert outs['species'].startswith('overwrite[array')


def test_bulk_initial_state_is_array(core):
    proc = TelluriumProcess(config={'model': MODEL, 'bulk_species': True}, core=core)
    st = proc.initial_state()
    assert isinstance(st['species'], np.ndarray)
    assert st['species'].shape == (len(proc._species_ids),)


def test_bulk_matches_map_trajectory(core):
    map_proc = TelluriumProcess(config={'model': MODEL}, core=core)
    bulk_proc = TelluriumProcess(config={'model': MODEL, 'bulk_species': True}, core=core)
    map_traj = _run(map_proc, 20, 0.5)
    bulk_traj = _run(bulk_proc, 20, 0.5)
    assert len(map_traj) == len(bulk_traj)
    for a, b in zip(map_traj, bulk_traj):
        assert set(a) == set(b)
        for sid in a:
            assert a[sid] == pytest.approx(b[sid], rel=0, abs=0)


def test_bulk_accepts_array_input(core):
    bulk_proc = TelluriumProcess(config={'model': MODEL, 'bulk_species': True}, core=core)
    bulk_proc._build()
    n = len(bulk_proc._species_ids)
    out = bulk_proc.update({'species': np.ones(n) * 3.0}, 0.5)
    assert isinstance(out['species'], np.ndarray)
    assert out['species'].shape == (n,)


def test_bulk_species_tolerates_dict_input():
    """A map[float]-wired species port must not TypeError in bulk mode."""
    from process_bigraph import allocate_core
    from viva_tellurium.processes import TelluriumProcess
    model = "S1 = 10; S2 = 2; k1 = 0.3; J1: S1 -> S2; k1*S1;"
    for coupled in (False, True):
        p = TelluriumProcess(
            config={'model': model, 'bulk_species': True,
                    'coupled_species': coupled},
            core=allocate_core())
        p._build()
        out = p.update({'species': {'S1': 8.0}}, 0.5)
        assert 'species' in out

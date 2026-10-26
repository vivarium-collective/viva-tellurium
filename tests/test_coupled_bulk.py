"""Coupled (operator-split / co-simulation) bulk-binding.

The single-process ``bulk_species`` path (#24) pushes the species port as one
numpy Array leaf via roadrunner's bulk C API. That path CANNOT be adopted by
COUPLED multi-process composites — two TelluriumProcesses over complementary
reaction subsets sharing one ``species`` store, composed every coupling
interval (a Lie/Jacobi operator split) — for two reasons, both fixed by the
opt-in ``coupled_species`` mode and both verified here:

1. **Additive apply.** Coupled arms share the ``species`` store and each writes
   its delta; reconcile must SUM the arms' contributions (Jacobi). The default
   ``overwrite[...]`` species leaf makes the second arm clobber the first. In
   coupled mode the species leaf is a plain ADDITIVE leaf — ``map[float]`` or
   ``array[(n,)]`` — which sums under reconcile and accumulates under apply.
   (bigraph_schema's built-in ``array`` already does this; only the
   ``overwrite`` wrapper had to be dropped — no new type, no bigraph_schema
   edit.)

2. **Settable-subset bulk setter.** ``setFloatingSpeciesConcentrations(vec)``
   writes the FULL floating vector. Conservation/assignment-determined species
   must be left to roadrunner. In coupled mode update() writes only the
   settable (independent) floating species via the indexed bulk setter.

Contract (verification gate): the ``map[float]`` coupled path and the
additive-Array + settable-subset bulk coupled path are BIT-IDENTICAL on the
species trajectory over ~20 coupling steps.
"""
import numpy as np
import pytest

from process_bigraph import Composite, allocate_core
from viva_tellurium.processes import TelluriumProcess

# Three-species cycle: S1 -> S2 -> S3 -> S1. The cycle conserves S1+S2+S3, so
# with conserved-moiety analysis one floating species becomes dependent — which
# exercises the settable-subset setter (feature 2).
MODEL = """
    S1 = 10; S2 = 2; S3 = 5;
    k1 = 0.3; k2 = 0.15; k3 = 0.4;
    J1: S1 -> S2; k1*S1;
    J2: S2 -> S3; k2*S2;
    J3: S3 -> S1; k3*S3;
"""

INTERVAL = 0.5
STEPS = 20


def _species_ids(moiety):
    p = TelluriumProcess(
        config={'model': MODEL, 'conserved_moiety_analysis': moiety},
        core=allocate_core())
    p._build()
    return list(p._species_ids)


def _initial_array(moiety):
    p = TelluriumProcess(
        config={'model': MODEL, 'bulk_species': True,
                'conserved_moiety_analysis': moiety},
        core=allocate_core())
    return np.asarray(p.initial_state()['species'], dtype=float)


def _build_coupled(bulk, moiety):
    """A 2-arm coupled composite over complementary reaction subsets.

    Arm A runs only J1 (k2=k3=0); arm B runs only J2, J3 (k1=0). Both share
    the ``species`` store and step every INTERVAL (Jacobi operator split).
    """
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)
    ids = _species_ids(moiety)
    init = _initial_array(moiety)
    species_store = init.copy() if bulk else {
        sid: float(init[i]) for i, sid in enumerate(ids)}

    def arm(name, zero_params):
        return {
            '_type': 'process',
            'address': 'local:TelluriumProcess',
            'config': {
                'model': MODEL,
                'coupled_species': True,
                'bulk_species': bulk,
                'conserved_moiety_analysis': moiety,
                'parameter_overrides': zero_params,
            },
            'interval': INTERVAL,
            'inputs': {'species': ['stores', 'species']},
            'outputs': {
                'species': ['stores', 'species'],
                'rates': ['stores', 'rates_%s' % name],
                'parameters': ['stores', 'params_%s' % name],
                'time': ['stores', 'time'],
            },
        }

    doc = {
        'armA': arm('A', {'k2': 0.0, 'k3': 0.0}),
        'armB': arm('B', {'k1': 0.0}),
        'stores': {'species': species_store},
    }
    return Composite({'state': doc}, core=core), ids


def _trajectory(bulk, moiety):
    sim, ids = _build_coupled(bulk, moiety)

    def snap():
        sp = sim.state['stores']['species']
        if isinstance(sp, dict):
            return {k: float(v) for k, v in sp.items()}
        return {sid: float(sp[i]) for i, sid in enumerate(ids)}

    traj = [snap()]
    for _ in range(STEPS):
        sim.run(INTERVAL)
        traj.append(snap())
    return traj


@pytest.mark.parametrize('moiety', [False, True])
def test_coupled_map_and_bulk_are_bit_identical(moiety):
    """The verification gate: map[float] additive path vs additive-Array +
    settable-subset bulk path, bit-identical over 20 coupling steps."""
    map_traj = _trajectory(bulk=False, moiety=moiety)
    bulk_traj = _trajectory(bulk=True, moiety=moiety)

    assert len(map_traj) == len(bulk_traj) == STEPS + 1
    max_abs = 0.0
    for a, b in zip(map_traj, bulk_traj):
        assert set(a) == set(b)
        for sid in a:
            max_abs = max(max_abs, abs(a[sid] - b[sid]))
    # Bit-identical: identical roadrunner ops, only container type differs.
    assert max_abs == 0.0, f"map vs bulk diverged by {max_abs}"


def test_coupled_arms_actually_sum_additively():
    """Guard the additive-apply fix: with two arms writing deltas into one
    store, the species must evolve (not get clobbered to a single arm's value),
    and S1+S2+S3 is conserved by the cycle."""
    traj = _trajectory(bulk=True, moiety=False)
    first, last = traj[0], traj[-1]
    # Something happened (not frozen, not clobbered to zero).
    assert any(abs(last[s] - first[s]) > 1e-6 for s in first)
    # The reaction cycle conserves total mass; additive Jacobi preserves it.
    assert sum(last.values()) == pytest.approx(sum(first.values()), rel=1e-6)


def test_coupled_bulk_species_port_is_additive_not_overwrite():
    """Feature 1: the coupled bulk species leaf is a plain additive array,
    NOT overwrite[array]."""
    proc = TelluriumProcess(
        config={'model': MODEL, 'bulk_species': True, 'coupled_species': True},
        core=allocate_core())
    outs = proc.outputs()
    assert outs['species'].startswith('array[')
    assert not outs['species'].startswith('overwrite')
    # And the input port is a plain (non-maybe) array leaf.
    assert proc.inputs()['species'].startswith('array[')


def test_settable_subset_excludes_dependent_species():
    """Feature 2: with conserved-moiety analysis the dependent floating species
    is excluded from the settable subset (left to roadrunner)."""
    proc = TelluriumProcess(
        config={'model': MODEL, 'bulk_species': True, 'coupled_species': True,
                'conserved_moiety_analysis': True},
        core=allocate_core())
    proc._build()
    dep = set(proc._rr.getDependentFloatingSpeciesIds())
    assert dep, "expected the conserved cycle to produce a dependent species"
    # Settable subset is strictly the independent species.
    assert set(proc._settable_ids).isdisjoint(dep)
    assert set(proc._settable_ids) == set(proc._rr.getIndependentFloatingSpeciesIds())
    assert len(proc._settable_idx) == len(proc._settable_ids) < len(proc._species_ids)


def test_single_process_paths_unchanged():
    """Non-breaking: default and single-process bulk ports keep overwrite
    semantics; coupled mode is strictly opt-in."""
    default = TelluriumProcess(config={'model': MODEL}, core=allocate_core())
    assert default.outputs()['species'] == 'overwrite[map[float]]'
    assert default.inputs()['species'] == 'maybe[map[float]]'

    bulk = TelluriumProcess(
        config={'model': MODEL, 'bulk_species': True}, core=allocate_core())
    assert bulk.outputs()['species'].startswith('overwrite[array[')
    assert bulk.inputs()['species'].startswith('maybe[array[')

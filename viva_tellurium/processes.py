"""Tellurium Process wrapper for process-bigraph.

Wraps a libroadrunner (via tellurium) SBML/Antimony model as a time-driven
Process using the bridge pattern. The RoadRunner instance is lazily
initialized on first update() call.
"""

import os
from pathlib import Path

import numpy as np

from process_bigraph import Process, Step

try:
    # Structured process contract (summary + governing equations + symbol
    # legend) surfaced in the vivarium-workbench bigraph-loom node inspector.
    # Guarded because older bigraph_schema releases predate this module; the
    # loom renderer (vivarium_workbench.lib.process_docs._contract_for_class)
    # duck-types either a ProcessContract (.to_dict()) or a plain dict, so the
    # fallback below renders identically where a newer bigraph_schema is not
    # installed. Purely descriptive metadata — no runtime behavior depends on it.
    from bigraph_schema.contract import ProcessContract
except Exception:  # pragma: no cover - depends on installed bigraph_schema
    ProcessContract = None


def _contract(**fields):
    """A ProcessContract when the installed bigraph_schema provides one, else an
    equivalent plain dict (the loom renderer accepts both)."""
    return ProcessContract(**fields) if ProcessContract is not None else dict(fields)


def _load_roadrunner(model_source, model_format='antimony', model_file=''):
    """Build a RoadRunner instance from an Antimony string, SBML string, or file."""
    if model_file:
        resolved = Path(model_file)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        if not resolved.exists():
            raise FileNotFoundError(
                f"Tellurium model_file {model_file!r} not found. "
                f"Resolved to {str(resolved)!r} (cwd: {str(Path.cwd())!r}). "
                "Pass an absolute path, or a path relative to the current "
                "working directory."
            )
        import tellurium as te
        if model_file.endswith('.ant') or model_file.endswith('.txt'):
            with open(resolved) as f:
                return te.loada(f.read())
        return te.loadSBMLModel(str(resolved))

    import tellurium as te

    if model_format == 'sbml':
        return te.loadSBMLModel(model_source)
    return te.loada(model_source)


class TelluriumProcess(Process):
    """Bridge Process wrapping a Tellurium / libroadrunner SBML simulation.

    Loads an SBML or Antimony model into a RoadRunner instance and advances
    it in chunks of `interval` on each update() call. Returns absolute
    species concentrations, reaction rates, parameter values, and the
    current time as overwrite values.

    SPECIES PORT (important):
      * DEFAULT (``bulk_species=True``): the ``species`` port is a numpy
        ``array[(n,),float]`` ordered by ``getFloatingSpeciesIds`` and moved via
        roadrunner's bulk C API — ~2x faster per update() and bit-identical.
        Wire it to an Array-shaped store, not a dict.
      * COUPLED multi-process composites (two+ processes sharing one species
        store, operator-split): also set ``coupled_species=True`` — the species
        output becomes an ADDITIVE array so the arms' deltas SUM under reconcile
        instead of clobbering, and only the settable subset is written back.
      * LEGACY (``bulk_species=False``): the old per-species ``map[float]`` port;
        slower, kept only for back-compat with dict-shaped species stores.

    Config:
        model: Antimony or SBML source string.
        model_format: 'antimony' (default) or 'sbml'.
        model_file: Optional path to a local .ant/.txt/.xml file
            (takes precedence over `model`).
        integrator: 'cvode' (deterministic) or 'gillespie' (stochastic).
        absolute_tolerance: CVODE absolute tolerance.
        relative_tolerance: CVODE relative tolerance.
        seed: Random seed for stochastic integrators (None = unset).
        species_overrides: Mapping of {species_id: initial_value}
            applied after model load.
        parameter_overrides: Mapping of {parameter_id: value}
            applied after model load.
        reset_on_init: If True, call reset() after applying overrides
            (default True).
    """

    config_schema = {
        'model': {'_type': 'string', '_default': ''},
        'model_format': {'_type': 'string', '_default': 'antimony'},
        'model_file': {'_type': 'string', '_default': ''},
        'integrator': {'_type': 'string', '_default': 'cvode'},
        'absolute_tolerance': {'_type': 'float', '_default': 1e-10},
        'relative_tolerance': {'_type': 'float', '_default': 1e-8},
        'seed': {'_type': 'integer', '_default': -1},
        'species_overrides': {'_type': 'map[float]', '_default': {}},
        'parameter_overrides': {'_type': 'map[float]', '_default': {}},
        'reset_on_init': {'_type': 'boolean', '_default': True},
        # DEFAULT. The `species` port is a single numpy Array leaf (ordered by
        # getFloatingSpeciesIds) pushed/pulled via roadrunner's bulk C API
        # instead of a per-species map[float] loop. Bit-identical; much lower
        # set/get overhead (one buffer copy vs N string-keyed FFI calls) — ~2x
        # faster whole update() at 17 species, more at larger models. This is
        # the recommended path and is now the default; set bulk_species=False
        # only for a legacy map[float] species port (slower; kept for back-compat
        # with composites that wire a dict-shaped species store).
        'bulk_species': {'_type': 'boolean', '_default': True},
        # Opt-in coupled / operator-split mode for COUPLED multi-process
        # composites (two+ TelluriumProcesses over complementary reaction
        # subsets sharing one `species` store, composed every coupling
        # interval — a Lie/Jacobi operator split). Two things change vs the
        # single-process path, both needed for correctness under multi-process
        # reconcile and both bit-identical between the map[float] and the
        # bulk-array representations:
        #   1. The `species` OUTPUT becomes an ADDITIVE (delta) leaf instead of
        #      an `overwrite[...]` leaf, so the two arms' contributions SUM
        #      (Jacobi) under reconcile/apply rather than the second arm
        #      clobbering the first. For map[float] this is a plain additive
        #      `map[float]`; for bulk this is a plain additive `array[(n,)]`
        #      (bigraph_schema's built-in `array` already sums on reconcile and
        #      accumulates on apply — only the `overwrite` wrapper had to go).
        #   2. update() pushes only the SETTABLE floating species back into
        #      roadrunner (see conserved_moiety_analysis); assignment /
        #      conservation-determined species are left to roadrunner.
        'coupled_species': {'_type': 'boolean', '_default': False},
        # When True, enable roadrunner conserved-moiety analysis so that
        # moiety-dependent floating species are determined by their
        # conservation law at runtime. In coupled mode the settable subset is
        # then getIndependentFloatingSpeciesIds (the dependent species are NOT
        # written back — roadrunner recomputes them). When False, every
        # floating species is settable.
        'conserved_moiety_analysis': {'_type': 'boolean', '_default': False},
    }

    # Integration contract. The kinetic model is user-authored Antimony/SBML, so
    # the specific reaction equations are not fixed here; this contract specifies
    # how the ODE system is advanced each coupling interval and what the ports mean.
    contract = _contract(
        summary=(
            "Advances a user-authored Antimony/SBML kinetic model one coupling "
            "interval by integrating its ODE system dX/dt = f(X, p; t) with "
            "libroadrunner (CVODE)."),
        math=[
            r"\frac{dX}{dt} = f(X,\, p;\, t) = S\, v(X,\, p)",
            r"X(t + \Delta t) = \mathrm{CVODE}\big(f,\, X(t),\, [t,\, t + \Delta t]\big)",
            r"v = v(X,\, p) \quad\text{(reaction-rate vector returned on the } \texttt{rates} \text{ port)}",
            r"\text{coupled mode: } X_{n+1} = X_n + \textstyle\sum_i \big(X^{(i)} - X_n\big)",
        ],
        symbols={
            "X": "floating-species state vector (concentrations by default, or amounts), ordered by getFloatingSpeciesIds",
            "p": "global kinetic parameters of the model",
            "t": "simulation time",
            r"\Delta t": "coupling interval advanced per update() call",
            "f": "ODE right-hand side assembled by libroadrunner from the model's reactions",
            "S": "stoichiometry matrix",
            "v": "reaction-rate (flux) vector",
            r"X^{(i)}": "the i-th coupled arm's post-step state in operator-split (coupled_species) mode",
        },
        inputs={
            "species": (
                "optional incoming floating-species state pushed into roadrunner "
                "before the step (numpy array ordered by getFloatingSpeciesIds "
                "when bulk_species=True, else a {species_id: value} map); in "
                "coupled_species mode this is the shared-store baseline the arm "
                "starts from"),
        },
        outputs={
            "species": "post-step floating-species state (overwrite; an ADDITIVE delta in coupled_species mode)",
            "rates": "{reaction_id: rate} reaction velocities v at the new state (getReactionRates)",
            "parameters": "{parameter_id: value} global parameter values",
            "time": "roadrunner current time after the step",
        },
        config={
            "model": "Antimony or SBML source string",
            "model_format": "'antimony' (default) or 'sbml'",
            "model_file": "optional path to a local .ant/.txt/.xml model (takes precedence over model)",
            "integrator": "'cvode' (deterministic, default) or 'gillespie' (stochastic SSA)",
            "absolute_tolerance": "CVODE absolute tolerance",
            "relative_tolerance": "CVODE relative tolerance",
            "seed": "random seed for stochastic integrators (-1 = unset)",
            "species_overrides": "{species_id: initial_value} applied after model load",
            "parameter_overrides": "{parameter_id: value} applied after model load",
            "reset_on_init": "reset() after applying overrides (default True)",
            "bulk_species": "species port as a numpy Array via roadrunner's bulk C API (default True) vs a legacy map[float]",
            "coupled_species": "operator-split mode: additive species deltas that SUM under multi-process reconcile",
            "conserved_moiety_analysis": "let roadrunner slave moiety-dependent species to conservation laws",
        },
        assumptions=[
            "Deterministic CVODE (BDF) integration by default; stiffness is handled internally by CVODE. Set integrator='gillespie' for stochastic SSA.",
            "The reaction network is user-authored (Antimony/SBML) and loaded at build time, so this contract specifies the integration and port semantics, not any specific reaction equations.",
            "Species are reported as concentrations unless the model yields amounts; the state vector is ordered by getFloatingSpeciesIds.",
            "update() is stateful across intervals (roadrunner carries state between calls); the Step variants reset each firing instead.",
            "coupled_species assumes a Lie/Jacobi operator split across processes sharing one species store; each arm emits its own delta and the deltas sum under reconcile.",
        ],
        references=[
            "Somogyi et al. (2015) libRoadRunner: a high performance SBML simulation and analysis library.",
            "Hindmarsh et al. (2005) SUNDIALS: CVODE stiff/non-stiff ODE integration.",
        ],
    )

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rr = None
        self._species_ids = None
        self._reaction_ids = None
        self._parameter_ids = None
        # Settable-subset bookkeeping (coupled mode). The subset is the set of
        # floating species we write back into roadrunner; the complement is
        # left for roadrunner to determine from conservation/assignment rules.
        self._settable_ids = None
        self._settable_idx = None

    def inputs(self):
        coupled = self.config.get('coupled_species')
        if self.config.get('bulk_species'):
            self._build()
            n = len(self._species_ids)
            # Coupled mode always has the species store wired, so the port is a
            # plain (non-maybe) array leaf — this also keeps the shared store's
            # realized schema a plain Array (a maybe[array] store trips
            # bigraph_schema's realize(Maybe) on ndarray defaults).
            if coupled:
                return {'species': f'array[({n}),float]'}
            return {'species': f'maybe[array[({n}),float]]'}
        if coupled:
            return {'species': 'map[float]'}
        return {
            'species': 'maybe[map[float]]',
        }

    def outputs(self):
        coupled = self.config.get('coupled_species')
        if self.config.get('bulk_species'):
            self._build()
            n = len(self._species_ids)
            # Coupled: ADDITIVE array leaf (deltas sum under reconcile/apply).
            # Single-process: overwrite array leaf (absolute, last-write-wins).
            species_type = (f'array[({n}),float]' if coupled
                            else f'overwrite[array[({n}),float]]')
        else:
            # Coupled: plain map[float] (additive). Single: overwrite[map[float]].
            species_type = 'map[float]' if coupled else 'overwrite[map[float]]'
        return {
            'species': species_type,
            'rates': 'overwrite[map[float]]',
            'parameters': 'overwrite[map[float]]',
            'time': 'overwrite[float]',
        }

    def _build(self):
        if self._rr is not None:
            return

        cfg = self.config
        if not cfg['model'] and not cfg['model_file']:
            raise ValueError(
                "TelluriumProcess requires either 'model' or 'model_file'.")

        self._rr = _load_roadrunner(
            cfg['model'],
            model_format=cfg['model_format'],
            model_file=cfg['model_file'])

        # Conserved-moiety analysis must be toggled before caching ids /
        # independent-species sets, since it changes which floating species
        # are reported independent vs dependent.
        if cfg.get('conserved_moiety_analysis'):
            self._rr.conservedMoietyAnalysis = True

        # Apply overrides
        for sid, val in cfg['species_overrides'].items():
            self._rr[sid] = float(val)
        for pid, val in cfg['parameter_overrides'].items():
            self._rr[pid] = float(val)

        # Integrator selection
        integrator = cfg['integrator']
        if integrator and integrator != 'cvode':
            self._rr.setIntegrator(integrator)
        if integrator == 'cvode':
            self._rr.integrator.absolute_tolerance = cfg['absolute_tolerance']
            self._rr.integrator.relative_tolerance = cfg['relative_tolerance']
        if cfg['seed'] >= 0 and hasattr(self._rr.integrator, 'seed'):
            self._rr.integrator.seed = int(cfg['seed'])

        # Cache identifiers
        self._species_ids = list(self._rr.getFloatingSpeciesIds())
        self._reaction_ids = list(self._rr.getReactionIds())
        self._parameter_ids = list(self._rr.getGlobalParameterIds())

        # Settable-species subset (coupled mode). With conserved-moiety
        # analysis on, the dependent floating species are slaved to their
        # conservation law at runtime and must NOT be written back — so the
        # settable subset is getIndependentFloatingSpeciesIds. Otherwise every
        # floating species is settable.
        if cfg.get('conserved_moiety_analysis'):
            indep = set(self._rr.getIndependentFloatingSpeciesIds())
            self._settable_ids = [s for s in self._species_ids if s in indep]
        else:
            self._settable_ids = list(self._species_ids)
        index_of = {sid: i for i, sid in enumerate(self._species_ids)}
        self._settable_idx = np.array(
            [index_of[s] for s in self._settable_ids], dtype=np.int32)

        if cfg['reset_on_init']:
            # Reset AFTER caching IDs; preserve override values by re-applying
            self._rr.reset()
            for sid, val in cfg['species_overrides'].items():
                self._rr[sid] = float(val)
            for pid, val in cfg['parameter_overrides'].items():
                self._rr[pid] = float(val)

    def _read_state(self):
        rr = self._rr
        if self.config.get('bulk_species'):
            # Bulk C getter: one buffer copy, ordered by getFloatingSpeciesIds.
            species = np.asarray(rr.getFloatingSpeciesConcentrations(), dtype=float)
        else:
            species = {sid: float(rr[sid]) for sid in self._species_ids}
        parameters = {pid: float(rr[pid]) for pid in self._parameter_ids}
        rates_arr = rr.getReactionRates()
        rates = {rid: float(rates_arr[i])
                 for i, rid in enumerate(self._reaction_ids)}
        return {
            'species': species,
            'rates': rates,
            'parameters': parameters,
            'time': float(rr.getCurrentTime()),
        }

    def initial_state(self):
        self._build()
        return self._read_state()

    def update(self, state, interval):
        self._build()

        incoming = state.get('species') if state else None

        if self.config.get('coupled_species'):
            return self._coupled_update(incoming, interval)

        # --- Single-process path (unchanged) ---------------------------------
        # Push coupled species if wired
        if incoming is not None:
            if self.config.get('bulk_species'):
                # Bulk C setter: one buffer copy vs N string-keyed FFI calls.
                arr = self._as_species_array(incoming)
                if arr.shape == (len(self._species_ids),):
                    self._rr.model.setFloatingSpeciesConcentrations(arr)
            else:
                for sid, val in incoming.items():
                    if sid in self._species_ids:
                        self._rr[sid] = float(val)

        t0 = self._rr.getCurrentTime()
        self._rr.simulate(t0, t0 + interval, 2)

        return self._read_state()

    def _as_species_array(self, incoming):
        """Coerce species input to an ordered array; tolerate a dict port
        (map[float]) even in bulk mode. Missing species -> 0.0."""
        if isinstance(incoming, dict):
            return np.array(
                [float(incoming.get(sid, 0.0)) for sid in self._species_ids],
                dtype=float)
        return np.asarray(incoming, dtype=float)

    def _coupled_update(self, incoming, interval):
        """Operator-split (Jacobi) coupling step.

        Re-synchronise roadrunner to the shared-store baseline (writing only
        the SETTABLE floating species — feature 2), advance this arm's reaction
        subset over `interval`, and emit the species change as an ADDITIVE
        delta (feature 1). Under multi-process reconcile the arms' deltas SUM
        into the shared store: x_{n+1} = x_n + sum_i (x_i - x_n).

        The map[float] and bulk-array representations perform the identical
        roadrunner operations and emit the identical numbers — only the
        container type (dict vs ndarray) differs — so the two paths are
        bit-identical.
        """
        rr = self._rr
        n = len(self._species_ids)
        bulk = self.config.get('bulk_species')

        # Baseline = shared-store absolute concentrations this arm started from.
        if incoming is not None:
            if bulk:
                baseline = self._as_species_array(incoming)
            else:
                baseline = np.array(
                    [float(incoming.get(sid, rr[sid])) for sid in self._species_ids],
                    dtype=float)
        else:
            baseline = np.asarray(
                rr.getFloatingSpeciesConcentrations(), dtype=float)

        # Settable-subset bulk setter: resync only the settable floating
        # species; leave conservation/assignment-determined species to
        # roadrunner. Index the baseline vector to the settable subset.
        if baseline.shape == (n,) and self._settable_idx.size:
            rr.model.setFloatingSpeciesConcentrations(
                self._settable_idx, baseline[self._settable_idx])

        t0 = rr.getCurrentTime()
        rr.simulate(t0, t0 + interval, 2)

        after = np.asarray(rr.getFloatingSpeciesConcentrations(), dtype=float)
        delta = after - baseline

        parameters = {pid: float(rr[pid]) for pid in self._parameter_ids}
        rates_arr = rr.getReactionRates()
        rates = {rid: float(rates_arr[i])
                 for i, rid in enumerate(self._reaction_ids)}

        species_out = delta if (bulk and not isinstance(incoming, dict)) else {
            sid: float(delta[i]) for i, sid in enumerate(self._species_ids)}

        return {
            'species': species_out,
            'rates': rates,
            'parameters': parameters,
            'time': float(rr.getCurrentTime()),
        }

    # Convenience accessors ------------------------------------------------

    def get_species_ids(self):
        self._build()
        return list(self._species_ids)

    def get_reaction_ids(self):
        self._build()
        return list(self._reaction_ids)

    def get_sbml(self):
        """Return the current SBML serialization of the loaded model."""
        self._build()
        return self._rr.getCurrentSBML()


class BaseTelluriumStep(Step):
    """Abstract base for Tellurium-backed Steps.

    Provides shared SBML/antimony model loading via _load_roadrunner,
    plus species-id caching. Subclasses implement update() with the
    specific simulation they perform (UTC, steady state, etc.).
    """

    # Valid values for the species_units config key.
    _SPECIES_UNITS = ('concentration', 'amount')

    config_schema = {
        **TelluriumProcess.config_schema,
        # Roadrunner output selections. A list of selection strings
        # (e.g. 'time', 'S1', '[S1]' for concentration, a reaction id for a
        # flux). Empty (the default) preserves roadrunner's built-in output:
        # time + all floating species for a time course, and floating-species
        # concentrations for a steady state. Kept identical to the viva-copasi
        # and viva-biomodels wrappers for cross-wrapper consistency.
        'selections': {'_type': 'list[string]', '_default': []},
        # Units for the DEFAULT floating-species output (i.e. when `selections`
        # is empty). 'concentration' (the default) or 'amount'. Raw roadrunner
        # mixes these: bare 'S1' is an amount while '[S1]' is a concentration,
        # and the auto-selected default depends on each species'
        # hasOnlySubstanceUnits flag. This key hides that inconsistency and
        # returns every floating species in ONE unit, converting with the
        # compartment volume (via roadrunner, which respects
        # hasOnlySubstanceUnits). Explicit `selections` are NOT affected — they
        # stay roadrunner's verbatim vocabulary. Identical key name/semantics
        # to the viva-copasi wrapper (issue #14 / viva-copasi#18).
        'species_units': {'_type': 'string', '_default': 'concentration'},
    }

    def _tellurium_initialize(self):
        if hasattr(self, '_rr'):
            return
        cfg = self.config
        units = cfg.get('species_units', 'concentration')
        if units not in self._SPECIES_UNITS:
            raise ValueError(
                f"species_units must be one of {self._SPECIES_UNITS}, "
                f"got {units!r}.")
        if not cfg['model'] and not cfg['model_file']:
            raise ValueError(
                "Tellurium step requires either 'model' or 'model_file'.")
        self._rr = _load_roadrunner(
            cfg['model'],
            model_format=cfg['model_format'],
            model_file=cfg.get('model_file', ''),
        )
        self._species_ids = list(self._rr.getFloatingSpeciesIds())
        self._reaction_ids = list(self._rr.getReactionIds())
        self._species_index = {sid: i for i, sid in enumerate(self._species_ids)}
        # Note each floating species' hasOnlySubstanceUnits flag (issue #14).
        # Informational / for conversion-correctness checks; the actual
        # amount<->concentration conversion is done by roadrunner's
        # getFloatingSpeciesAmounts/Concentrations, which use compartment
        # volume and honor this flag internally.
        self._has_only_substance_units = self._read_substance_unit_flags()

        # Apply species and parameter overrides
        for sid, val in cfg.get('species_overrides', {}).items():
            self._rr[sid] = float(val)
        for pid, val in cfg.get('parameter_overrides', {}).items():
            self._rr[pid] = float(val)

        # Integrator selection
        integrator = cfg.get('integrator', 'cvode')
        if integrator and integrator != 'cvode':
            self._rr.setIntegrator(integrator)
        if integrator == 'cvode':
            self._rr.integrator.absolute_tolerance = cfg['absolute_tolerance']
            self._rr.integrator.relative_tolerance = cfg['relative_tolerance']
        if cfg['seed'] >= 0 and hasattr(self._rr.integrator, 'seed'):
            self._rr.integrator.seed = int(cfg['seed'])

    def _reset_to_initial(self):
        """Reset the roadrunner instance to its configured initial state.

        A zero-time process-bigraph Step must be a pure function of its config
        and inputs, but roadrunner keeps state between simulate() calls. Call
        this at the START of every Step update() so each firing begins from the
        same initial conditions instead of continuing from the previous end
        state (issue #21). reset() returns the model to its SBML/antimony
        initial values, so the configured species/parameter overrides — which
        set only the CURRENT values at init time — are re-applied afterwards.
        The time-coupled *Process* classes deliberately do NOT do this; their
        cross-interval statefulness is intentional.
        """
        self._rr.reset()
        cfg = self.config
        for sid, val in cfg.get('species_overrides', {}).items():
            self._rr[sid] = float(val)
        for pid, val in cfg.get('parameter_overrides', {}).items():
            self._rr[pid] = float(val)

    def _read_substance_unit_flags(self):
        """Return {species_id: hasOnlySubstanceUnits(bool)} for floating
        species, read from the model's SBML via libSBML. Best-effort: returns
        {} if libSBML is unavailable or the model cannot be parsed."""
        flags = {}
        try:
            import libsbml
            doc = libsbml.readSBMLFromString(self._rr.getCurrentSBML())
            model = doc.getModel()
            if model is not None:
                for i in range(model.getNumSpecies()):
                    sp = model.getSpecies(i)
                    flags[sp.getId()] = bool(sp.getHasOnlySubstanceUnits())
        except Exception:
            pass
        return flags

    def get_has_only_substance_units(self):
        """Return the cached {species_id: hasOnlySubstanceUnits} mapping."""
        self._tellurium_initialize()
        return dict(self._has_only_substance_units)

    def _read_default_species(self):
        """Return {species_id: value} for all floating species in the
        configured species_units. Uses roadrunner's amount/concentration
        accessors, which convert using compartment volume and respect each
        species' hasOnlySubstanceUnits flag."""
        rr = self._rr
        if self.config.get('species_units', 'concentration') == 'amount':
            vals = rr.getFloatingSpeciesAmounts()
        else:
            vals = rr.getFloatingSpeciesConcentrations()
        return {sid: float(vals[i]) for i, sid in enumerate(self._species_ids)}

    def initial_state(self):
        self._tellurium_initialize()
        return {'species_concentrations': self._read_default_species()}

    def inputs(self):
        return {}


class TelluriumUTCStep(BaseTelluriumStep):
    """One-shot UTC simulation Step returning a dense trajectory.

    Loads a model, simulates a fixed span start-to-end, and returns
    the full time series as parallel lists. Use when you want a
    static trajectory rather than time-coupled stepping
    (TelluriumProcess covers the incremental case).
    """

    config_schema = {
        **BaseTelluriumStep.config_schema,
        'start_time': {'_type': 'float', '_default': 0.0},
        'end_time': {'_type': 'float', '_default': 10.0},
        'n_points': {'_type': 'integer', '_default': 101},
        # Explicit, possibly non-uniform output time points (assumed sorted
        # ascending). When non-empty, the time course is sampled at EXACTLY
        # these times via roadrunner's `simulate(times=[...])`, overriding the
        # uniform start_time/end_time/n_points grid; the returned time_series is
        # these points verbatim. Empty (the default) preserves the uniform
        # behavior. Same key name/semantics as the viva-copasi wrapper
        # (copasi's term is 'values', tellurium's is 'times'; issue #11 /
        # viva-copasi #16).
        'output_times': {'_type': 'list[float]', '_default': []},
    }

    contract = _contract(
        summary=(
            "One-shot uniform time course: integrates the model from start_time "
            "to end_time with libroadrunner (CVODE) and returns the dense "
            "trajectory as parallel time and per-species lists."),
        math=[
            r"\frac{dX}{dt} = f(X,\, p;\, t)",
            r"X(t_k) = \mathrm{CVODE}\big(f,\, X(t_0),\, [t_0,\, t_k]\big),\quad k = 0 \ldots N-1",
            r"t_k = t_0 + k\,\tfrac{t_{N-1} - t_0}{N-1} \quad\text{(or the explicit } \texttt{output\_times} \text{ grid)}",
        ],
        symbols={
            "X": "floating-species state vector (concentration or amount per species_units)",
            "f": "ODE right-hand side assembled by libroadrunner from the model's reactions",
            "p": "global kinetic parameters",
            "t_0": "start_time",
            "t_{N-1}": "end_time",
            "N": "number of output points (n_points)",
            "t_k": "k-th output sample time",
        },
        inputs={},
        outputs={
            "time_series": "list of sample times t_k",
            "species_trajectories": "{species_id (or selection): [value at each t_k]}",
        },
        config={
            "start_time": "time course start (default 0.0)",
            "end_time": "time course end (default 10.0)",
            "n_points": "number of uniform output points (default 101)",
            "output_times": "explicit (possibly non-uniform) output grid; overrides start/end/n_points when set",
            "selections": "verbatim roadrunner selection strings; empty = time + all floating species",
            "species_units": "'concentration' (default) or 'amount' for the default species output",
        },
        assumptions=[
            "Zero-time pure Step: reset() to the configured initial state at the start of every update(), so the result is idempotent and independent of prior firings.",
            "Deterministic CVODE integration; the reaction network is user-authored (Antimony/SBML).",
            "output_times, when non-empty, samples at exactly those points and overrides the uniform grid.",
        ],
    )

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'time_series': 'overwrite[list]',
            'species_trajectories': 'overwrite[map[list]]',
        }

    def update(self, state):
        self._tellurium_initialize()
        # Steps are zero-time pure functions: start each firing from the
        # configured initial state so update() is idempotent (issue #21).
        self._reset_to_initial()

        selections = list(self.config.get('selections') or [])
        start = self.config['start_time']
        end = self.config['end_time']
        n_points = self.config['n_points']
        output_times = list(self.config.get('output_times') or [])

        def _simulate():
            """Run the time course, honoring explicit output_times when set.

            When output_times is non-empty, roadrunner samples at exactly those
            points via simulate(times=[...]); otherwise the uniform
            start/end/n_points grid is used. The active selections list has
            already been set on self._rr by the caller.
            """
            if output_times:
                return self._rr.simulate(times=[float(t) for t in output_times])
            return self._rr.simulate(start, end, n_points)

        if selections:
            # Honor the user's exact roadrunner selection list. Columns come
            # back named exactly as requested (e.g. '[S1]', a reaction id for
            # a flux), so the trajectory keys are the requested selections
            # verbatim — no bracket stripping. 'time', if requested, is pulled
            # out into time_series; every other selection becomes a column.
            self._rr.selections = selections
            result = _simulate()
            cols = list(result.colnames)
            time_idx = cols.index('time') if 'time' in cols else None
            if time_idx is not None:
                times = [float(x) for x in result[:, time_idx]]
            elif output_times:
                # Caller did not request 'time' but gave explicit output points;
                # those ARE the time grid.
                times = [float(t) for t in output_times]
            else:
                # Caller did not request 'time'; reconstruct the uniform grid
                # so the time_series output contract still holds.
                import numpy as np
                times = [float(x) for x in np.linspace(start, end, n_points)]
            species = {
                col: [float(x) for x in result[:, i]]
                for i, col in enumerate(cols)
                if i != time_idx
            }
            return {
                'time_series': times,
                'species_trajectories': species,
            }

        # Default: time + all floating species, in the configured species_units
        # (issue #14). We set an explicit selection list rather than relying on
        # roadrunner's auto-default, so every species comes back in the SAME
        # unit regardless of its hasOnlySubstanceUnits flag: '[S1]' for
        # concentration, bare 'S1' for amount. Output keys are the plain
        # species ids either way (brackets stripped); only the values' unit
        # changes.
        units = self.config.get('species_units', 'concentration')
        if units == 'amount':
            self._rr.selections = ['time'] + list(self._species_ids)
        else:
            self._rr.selections = ['time'] + [
                f'[{sid}]' for sid in self._species_ids]
        result = _simulate()

        cols = list(result.colnames)
        times = [float(x) for x in result[:, 0]]
        species = {}
        for i, col in enumerate(cols):
            if i == 0:
                continue
            # Column names look like '[S1]' (concentration) or 'S1' (amount) —
            # strip brackets so keys are plain species ids.
            name = col.strip('[]')
            species[name] = [float(x) for x in result[:, i]]

        return {
            'time_series': times,
            'species_trajectories': species,
        }


class TelluriumSteadyStateStep(BaseTelluriumStep):
    """Steady-state solve via Tellurium / roadrunner.

    Loads the model and computes steady-state species concentrations
    rather than a trajectory. Use when you want the equilibrium state
    of an SBML/antimony model.

    Config (in addition to BaseTelluriumStep):
        steady_state_options: Dict of {option_name: value} for roadrunner's
            steady-state solver, applied before steadyState(). Distinct from
            the integrator tolerances/seed (those tune CVODE; these tune the
            steady-state solver). See the config_schema comment and
            get_steady_state_solver_settings() for the available options.
    """

    config_schema = {
        **BaseTelluriumStep.config_schema,
        # Options for roadrunner's steady-state SOLVER
        # (rr.getSteadyStateSolver()), applied before steadyState(). This is
        # distinct from the integrator-level absolute_tolerance/
        # relative_tolerance/seed inherited from BaseTelluriumStep: those tune
        # the TIME integrator (CVODE), while these tune the steady-state solver
        # itself (nleq2 by default). A dict of {option_name: value} using
        # roadrunner's own vocabulary, e.g. 'maximum_iterations',
        # 'relative_tolerance' (solver, not integrator), 'minimum_damping',
        # 'allow_presimulation', 'presimulation_time', 'presimulation_times',
        # 'presimulation_maximum_steps', 'allow_approx', 'approx_tolerance',
        # 'approx_maximum_steps', 'approx_time', 'auto_moiety_analysis',
        # 'broyden_method', 'linearity'. Enumerate the live set via
        # get_steady_state_solver_settings(). Empty (the default) leaves the
        # solver at roadrunner's defaults. Free-form map (not individual typed
        # keys) because the set is solver-specific and values are mixed-type
        # (bool/int/float/list). The copasi sibling (viva-copasi #19) uses the
        # same container key name `steady_state_options` with COPASI's own
        # option names (same `method`-vs-`integrator` precedent).
        'steady_state_options': {'_type': 'map', '_default': {}},
    }

    contract = _contract(
        summary=(
            "Solves for the steady state f(X, p) = 0 of the model via "
            "roadrunner's steady-state solver (nleq2 by default) and returns the "
            "equilibrium species concentrations."),
        math=[
            r"\frac{dX}{dt} = f(X,\, p) = 0",
            r"X^{*} = \mathrm{steadyState}\big(f,\, X_0\big)",
        ],
        symbols={
            "X^{*}": "steady-state (equilibrium) floating-species state vector",
            "f": "ODE right-hand side assembled by libroadrunner from the model's reactions",
            "p": "global kinetic parameters",
            "X_0": "configured initial state the solve starts from (optionally after presimulation)",
        },
        inputs={},
        outputs={
            "steady_state_concentrations": "{species_id (or selection): steady-state value}",
        },
        config={
            "steady_state_options": "map of roadrunner steady-state SOLVER options (e.g. maximum_iterations, allow_presimulation), distinct from the CVODE integrator tolerances",
            "selections": "verbatim roadrunner steady-state selection strings; empty = all floating-species concentrations",
            "species_units": "'concentration' (default) or 'amount' for the default output",
        },
        assumptions=[
            "Zero-time pure Step: reset() to the configured initial state before every solve, so the result is idempotent and independent of prior firings.",
            "A steady state is assumed to exist and the solver to converge; failure raises rather than returning a partial result.",
            "The reaction network is user-authored (Antimony/SBML); the solver is roadrunner's nleq2 unless reconfigured via steady_state_options.",
        ],
    )

    def outputs(self):
        return {
            'steady_state_concentrations': 'overwrite[map[float]]',
        }

    def _apply_steady_state_options(self):
        """Push the configured steady_state_options onto roadrunner's
        steady-state solver. Fails loud on an unknown/unsettable option."""
        options = self.config.get('steady_state_options') or {}
        if not options:
            return
        solver = self._rr.getSteadyStateSolver()
        for name, value in options.items():
            try:
                solver.setValue(name, value)
            except Exception as e:
                valid = ', '.join(solver.getSettings())
                raise ValueError(
                    f"Invalid steady-state solver option {name!r}: {e}. "
                    f"Valid roadrunner steady-state options: {valid}.")

    def get_steady_state_solver_settings(self):
        """Return {option_name: current_value} for roadrunner's steady-state
        solver, with any configured steady_state_options applied."""
        self._tellurium_initialize()
        self._apply_steady_state_options()
        solver = self._rr.getSteadyStateSolver()
        return {name: solver.getValue(name) for name in solver.getSettings()}

    def update(self, state):
        self._tellurium_initialize()
        # Start each firing from the configured initial state so the solve is
        # idempotent and independent of prior firings (issue #21).
        self._reset_to_initial()
        self._apply_steady_state_options()

        selections = list(self.config.get('selections') or [])
        if selections:
            # Steady state has its own selection path: steadyStateSelections +
            # getSteadyStateValues(). Values come back in the order of the
            # requested selections, so the output keys are the selections
            # verbatim (e.g. '[S1]' for a concentration, a reaction id for a
            # flux). Note: 'time' is not a valid steady-state selection.
            self._rr.steadyStateSelections = selections
            try:
                self._rr.steadyState()
            except Exception as e:
                raise RuntimeError(f"Tellurium steadyState() failed: {e}")
            vals = self._rr.getSteadyStateValues()
            species_ss = {
                sel: float(vals[i]) for i, sel in enumerate(selections)
            }
            return {'steady_state_concentrations': species_ss}

        try:
            self._rr.steadyState()
        except Exception as e:
            raise RuntimeError(f"Tellurium steadyState() failed: {e}")

        # Default output in the configured species_units (issue #14).
        return {'steady_state_concentrations': self._read_default_species()}

"""Tellurium Process wrapper for process-bigraph.

Wraps a libroadrunner (via tellurium) SBML/Antimony model as a time-driven
Process using the bridge pattern. The RoadRunner instance is lazily
initialized on first update() call.
"""

import os
from pathlib import Path
from process_bigraph import Process, Step


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
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._rr = None
        self._species_ids = None
        self._reaction_ids = None
        self._parameter_ids = None

    def inputs(self):
        return {
            'species': 'maybe[map[float]]',
        }

    def outputs(self):
        return {
            'species': 'overwrite[map[float]]',
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

        if cfg['reset_on_init']:
            # Reset AFTER caching IDs; preserve override values by re-applying
            self._rr.reset()
            for sid, val in cfg['species_overrides'].items():
                self._rr[sid] = float(val)
            for pid, val in cfg['parameter_overrides'].items():
                self._rr[pid] = float(val)

    def _read_state(self):
        rr = self._rr
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

        # Push coupled species if wired
        incoming = state.get('species') if state else None
        if incoming:
            for sid, val in incoming.items():
                if sid in self._species_ids:
                    self._rr[sid] = float(val)

        t0 = self._rr.getCurrentTime()
        self._rr.simulate(t0, t0 + interval, 2)

        return self._read_state()

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

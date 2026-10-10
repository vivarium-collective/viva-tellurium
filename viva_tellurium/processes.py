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

    config_schema = {
        **TelluriumProcess.config_schema,
        # Roadrunner output selections. A list of selection strings
        # (e.g. 'time', 'S1', '[S1]' for concentration, a reaction id for a
        # flux). Empty (the default) preserves roadrunner's built-in output:
        # time + all floating species for a time course, and floating-species
        # concentrations for a steady state. Kept identical to the viva-copasi
        # and viva-biomodels wrappers for cross-wrapper consistency.
        'selections': {'_type': 'list[string]', '_default': []},
    }

    def _tellurium_initialize(self):
        if hasattr(self, '_rr'):
            return
        cfg = self.config
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

    def initial_state(self):
        self._tellurium_initialize()
        conc = self._rr.getFloatingSpeciesConcentrations()
        return {
            'species_concentrations': {
                sid: float(conc[i]) for i, sid in enumerate(self._species_ids)
            }
        }

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

        selections = list(self.config.get('selections') or [])
        start = self.config['start_time']
        end = self.config['end_time']
        n_points = self.config['n_points']

        if selections:
            # Honor the user's exact roadrunner selection list. Columns come
            # back named exactly as requested (e.g. '[S1]', a reaction id for
            # a flux), so the trajectory keys are the requested selections
            # verbatim — no bracket stripping. 'time', if requested, is pulled
            # out into time_series; every other selection becomes a column.
            self._rr.selections = selections
            result = self._rr.simulate(start, end, n_points)
            cols = list(result.colnames)
            time_idx = cols.index('time') if 'time' in cols else None
            if time_idx is not None:
                times = [float(x) for x in result[:, time_idx]]
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

        # Default: roadrunner emits time + all floating species.
        result = self._rr.simulate(start, end, n_points)

        cols = list(result.colnames)
        times = [float(x) for x in result[:, 0]]
        species = {}
        for i, col in enumerate(cols):
            if i == 0:
                continue
            # Column names look like '[S1]' — strip brackets
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
    """

    config_schema = {
        **BaseTelluriumStep.config_schema,
    }

    def outputs(self):
        return {
            'steady_state_concentrations': 'overwrite[map[float]]',
        }

    def update(self, state):
        self._tellurium_initialize()

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

        conc_ss = self._rr.getFloatingSpeciesConcentrations()
        species_ss = {
            sid: float(conc_ss[i])
            for i, sid in enumerate(self._species_ids)
        }

        return {'steady_state_concentrations': species_ss}

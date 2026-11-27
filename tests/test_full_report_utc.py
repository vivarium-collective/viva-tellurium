"""TelluriumFullReportUTCStep: one-shot full-report UTC on the engine-core contract."""
import pytest
from engine_core import UniformTimeCourseStep
from process_bigraph import allocate_core

from viva_tellurium.processes import TelluriumFullReportUTCStep

_ANTIMONY = """
model reversible
  J0: S1 -> S2; k1*S1
  J1: S2 -> S1; k2*S2
  S1 = 10; S2 = 0; k1 = 0.5; k2 = 0.1
end
"""


@pytest.fixture
def sbml_file(tmp_path):
    import tellurium as te

    rr = te.loada(_ANTIMONY)
    path = tmp_path / "reversible.xml"
    path.write_text(rr.getSBML())
    return str(path)


def _step():
    core = allocate_core()
    core.register_link("TelluriumFullReportUTCStep", TelluriumFullReportUTCStep)
    return TelluriumFullReportUTCStep(core=core)


def test_subclasses_the_engine_core_contract():
    assert issubclass(TelluriumFullReportUTCStep, UniformTimeCourseStep)
    step = _step()
    assert step.inputs() == {
        "model_source": "string",
        "time": "float",
        "n_points": "integer",
    }
    assert step.outputs() == {"result": "tree"}


def test_emits_canonical_full_report_shape(sbml_file):
    out = _step().update({"model_source": sbml_file, "time": 10.0, "n_points": 11})
    r = out["result"]
    assert set(r) == {"time", "columns", "values"}
    assert len(r["time"]) == 11
    assert len(r["values"]) == 11
    assert all(len(row) == len(r["columns"]) for row in r["values"])
    assert all(isinstance(v, float) for row in r["values"] for v in row)


def test_reports_species_and_reaction_rates(sbml_file):
    """The full report carries species concentrations AND reaction rates."""
    r = _step().update({"model_source": sbml_file, "time": 5.0, "n_points": 6})["result"]
    cols = set(r["columns"])
    assert {"S1", "S2"} <= cols                 # species
    assert {"J0", "J1"} <= cols                 # reaction rates
    assert len(r["columns"]) == 4


def test_rejects_degenerate_n_points(sbml_file):
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        _step().update({"model_source": sbml_file, "time": 5.0, "n_points": 1})

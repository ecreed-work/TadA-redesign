"""The screen's gate decides what survives to the expensive stages, so its
threshold logic is asserted directly rather than inferred from a run."""
import numpy as np
import pytest

from tada_redesign import constants, score_folds as sf


def _row(plddt=0.80, rmsd=0.5, clearance=2.5, status="ok"):
    return {"plddt": plddt, "motif_rmsd": rmsd, "cleft_clearance": clearance,
            "status": status}


def test_gate_passes_a_good_design():
    ok, status = sf.gate(_row(), parent_plddt=0.80)
    assert ok is True and status == "ok"


def test_gate_rejects_low_plddt_relative_to_the_parent():
    """Relative, not absolute: reduced sampling depresses pLDDT for everything."""
    ok, status = sf.gate(_row(plddt=0.80 - constants.SCREEN_PLDDT_MARGIN - 0.01),
                         parent_plddt=0.80)
    assert ok is False and status == "low_plddt"


def test_gate_accepts_plddt_exactly_at_the_margin():
    ok, _ = sf.gate(_row(plddt=0.80 - constants.SCREEN_PLDDT_MARGIN), parent_plddt=0.80)
    assert ok is True


def test_gate_rejects_motif_drift_using_the_SCREEN_threshold():
    """The screen uses the looser SCREEN_MOTIF_RMSD_MAX, not the final one,
    because reduced-sampling folds are noisier."""
    assert constants.SCREEN_MOTIF_RMSD_MAX > constants.FINAL_MOTIF_RMSD_MAX
    ok, status = sf.gate(_row(rmsd=constants.SCREEN_MOTIF_RMSD_MAX + 0.01),
                         parent_plddt=0.80)
    assert ok is False and status == "motif_drift"
    ok, _ = sf.gate(_row(rmsd=constants.FINAL_MOTIF_RMSD_MAX + 0.01), parent_plddt=0.80)
    assert ok is True          # tighter final threshold must NOT apply here


def test_gate_propagates_an_upstream_failure_status():
    ok, status = sf.gate(_row(status="fold_missing"), parent_plddt=0.80)
    assert ok is False and status == "fold_missing"


def test_gate_rejects_a_nan_measurement_rather_than_passing_it():
    """A nan comparison is False in Python, so a naive `>` test would let a
    broken measurement through as a pass."""
    ok, status = sf.gate(_row(rmsd=float("nan")), parent_plddt=0.80)
    assert ok is False and status == "unmeasurable"


def test_columns_carry_the_cell_coordinates_and_the_gate_inputs():
    for col in ("design_id", "parent", "arm", "partial_t", "temperature", "bias",
                "plddt", "ptm", "motif_rmsd", "cleft_clearance", "status", "passed"):
        assert col in sf.COLUMNS

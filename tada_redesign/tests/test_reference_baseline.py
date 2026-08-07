"""The baseline must be folded in the SAME mode as the designs it gates, or the
comparison is meaningless."""
import json

import pytest

from tada_redesign import constants, reference_baseline as rb


def test_baseline_covers_both_parents_in_both_modes():
    jobs = rb.baseline_jobs()
    assert {(j["parent"], j["mode"]) for j in jobs} == {
        (p, m) for p in constants.PARENTS for m in ("screen", "full")}


def test_baseline_sequences_are_the_real_parent_sequences():
    for job in rb.baseline_jobs():
        assert job["sequence"] == constants.PARENT_SEQUENCE[job["parent"]]
        assert len(job["sequence"]) == 156


def test_baseline_id_round_trips():
    assert rb.baseline_id("TadA8e", "screen") == "TadA8e__screen"
    assert rb.baseline_id("TadA9", "full") == "TadA9__full"


def test_read_baseline_parses_metrics_files(tmp_path):
    d = tmp_path / "baseline"
    d.mkdir()
    json.dump({"plddt": 0.71}, open(d / "TadA8e__screen.metrics.json", "w"))
    json.dump({"plddt": 0.83}, open(d / "TadA9__full.metrics.json", "w"))
    got = rb.read_baseline(str(tmp_path))
    assert got[("TadA8e", "screen")] == pytest.approx(0.71)
    assert got[("TadA9", "full")] == pytest.approx(0.83)


def test_read_baseline_raises_when_a_required_baseline_is_missing(tmp_path):
    """Gating designs against an absent baseline would silently pass or fail
    everything."""
    (tmp_path / "baseline").mkdir()
    with pytest.raises(FileNotFoundError):
        rb.read_baseline(str(tmp_path), require=[("TadA8e", "screen")])


def test_screen_and_full_baselines_agree_within_noise():
    """MEASURED 2026-08-06 (job 234208): sampling depth moves pLDDT by <0.004 on
    these 156-residue parents, so the two modes are equivalent within noise.
    The plan originally asserted the opposite, inferred from one uncontrolled
    78-mer fold. If a constants change ever makes the modes genuinely diverge,
    this test is where it surfaces."""
    import os
    sub = os.path.dirname(os.path.dirname(os.path.abspath(rb.__file__)))
    run_dir = os.path.join(sub, "outputs", constants.RUN_DIR_NAME)
    got = rb.read_baseline(run_dir)
    if not got:
        pytest.skip("baselines not folded in this checkout")
    for parent in constants.PARENTS:
        if (parent, "screen") in got and (parent, "full") in got:
            assert abs(got[(parent, "screen")] - got[(parent, "full")]) < 0.02, parent

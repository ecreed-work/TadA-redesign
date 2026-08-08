"""The baseline must be folded in the SAME mode as the designs it gates, or the
comparison is meaningless."""
import json

import pytest

from tada_redesign import constants, reference_baseline as rb


def test_baseline_sequences_are_the_real_parent_sequences():
    for job in rb.baseline_jobs():
        assert job["sequence"] == constants.PARENT_SEQUENCE[job["parent"]]
        assert len(job["sequence"]) == 156


def test_baseline_id_round_trips():
    """The two-tier screen/full split is retired; baseline_id no longer takes a
    mode argument, one baseline per parent."""
    assert rb.baseline_id("TadA8e") == "TadA8e__fold"
    assert rb.baseline_id("TadA9") == "TadA9__fold"


def test_read_baseline_parses_metrics_files(tmp_path):
    d = tmp_path / "baseline"
    d.mkdir()
    json.dump({"plddt": 0.71}, open(d / "TadA8e__fold.metrics.json", "w"))
    json.dump({"plddt": 0.83}, open(d / "TadA9__fold.metrics.json", "w"))
    got = rb.read_baseline(str(tmp_path))
    assert got["TadA8e"] == pytest.approx(0.71)
    assert got["TadA9"] == pytest.approx(0.83)


def test_read_baseline_raises_when_a_required_baseline_is_missing(tmp_path):
    """Gating designs against an absent baseline would silently pass or fail
    everything."""
    (tmp_path / "baseline").mkdir()
    with pytest.raises(FileNotFoundError):
        rb.read_baseline(str(tmp_path), require=["TadA8e"])


def test_baseline_is_a_single_fold_per_parent():
    """One sampling mode means one baseline per parent."""
    jobs = rb.baseline_jobs()
    assert {j["parent"] for j in jobs} == set(constants.PARENTS)
    assert len(jobs) == len(constants.PARENTS)

"""A stage that lost most of its inputs must refuse the canonical output path."""
import json

from tada_redesign import constants, provenance


def test_is_degraded_at_the_threshold(tmp_path):
    assert provenance.is_degraded(100, 79) is True      # 21% lost
    assert provenance.is_degraded(100, 80) is False     # exactly 20% lost
    assert provenance.is_degraded(100, 100) is False


def test_is_degraded_treats_zero_inputs_as_not_degraded():
    """No inputs is an upstream problem, not this stage failing."""
    assert provenance.is_degraded(0, 0) is False


def test_output_path_diverts_a_degraded_run(tmp_path):
    canonical = str(tmp_path / "backbones.tsv")
    path, degraded = provenance.output_path(canonical, 100, 10)
    assert degraded is True
    assert path.endswith("backbones.degraded.tsv")
    path, degraded = provenance.output_path(canonical, 100, 95)
    assert (path, degraded) == (canonical, False)


def test_write_records_counts_and_thresholds(tmp_path):
    p = provenance.write(str(tmp_path), "filter_backbones", 512, 480,
                         extra={"cells": 16})
    doc = json.load(open(p))
    assert p.endswith("filter_backbones.provenance.json")
    assert doc["stage"] == "filter_backbones"
    assert doc["n_in"] == 512 and doc["n_out"] == 480
    assert doc["is_degraded"] is False          # always present, never absent
    assert doc["degraded_fraction"] == constants.DEGRADED_FRACTION
    assert doc["extra"]["cells"] == 16
    assert "submodule_sha" in doc


def test_write_marks_a_degraded_run(tmp_path):
    doc = json.load(open(provenance.write(str(tmp_path), "s", 100, 5)))
    assert doc["is_degraded"] is True

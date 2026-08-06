"""Every gate here is a cheap CPU check standing between RFD3's output and the
much more expensive sequence-design and folding stages."""
import json

import numpy as np
import pytest
import yaml

from tada_redesign import constants, filter_backbones as fb, motif, score_structure as ss
from tada_redesign import io as tio

DONORS = ((57, "ND1"), (87, "SG"), (90, "SG"))


def _linear_backbone(n=160, start=5, step=3.8):
    """CA atoms on a straight line at ideal spacing, plus a Zn at ideal donor
    distance from three synthetic donor atoms."""
    atoms = {}
    for i in range(n):
        resnum = start + i
        atoms[(resnum, "CA")] = np.array([i * step, 0.0, 0.0])
    for resnum, name in DONORS:
        atoms[(resnum, name)] = np.array([0.0, 2.2, 0.0])
    atoms[(201, "ZN")] = np.array([0.0, 0.0, 0.0])
    return atoms


def test_max_ca_break_finds_the_largest_consecutive_gap():
    atoms = _linear_backbone(n=5)
    assert fb.max_ca_break(atoms) == pytest.approx(3.8)
    atoms[(9, "CA")] = np.array([100.0, 0.0, 0.0])       # residue 9 flung away
    assert fb.max_ca_break(atoms) > constants.CA_BREAK_MAX


def test_max_ca_break_ignores_numbering_gaps():
    """A missing residue is not a chain break: only CONSECUTIVE numbers count."""
    atoms = {(5, "CA"): np.array([0.0, 0.0, 0.0]),
             (6, "CA"): np.array([3.8, 0.0, 0.0]),
             (20, "CA"): np.array([200.0, 0.0, 0.0])}
    assert fb.max_ca_break(atoms) == pytest.approx(3.8)


def test_zn_donor_distances_measures_all_three_donors():
    d = fb.zn_donor_distances(_linear_backbone())
    assert set(d) == {"zn_57ND1", "zn_87SG", "zn_90SG"}
    assert all(v == pytest.approx(2.2) for v in d.values())


def test_zn_donor_distances_reports_nan_for_a_missing_donor():
    atoms = _linear_backbone()
    del atoms[(87, "SG")]
    assert np.isnan(fb.zn_donor_distances(atoms)["zn_87SG"])


def test_evaluate_rejects_a_length_outlier(tmp_path, monkeypatch):
    short = _linear_backbone(n=40)
    monkeypatch.setattr(fb, "load_backbone", lambda path: short)
    row = fb.evaluate("x.cif.gz", short, (5, 6, 7), "cell")
    assert row["status"] == "length_out_of_range"
    assert row["passed"] == "False"


def test_evaluate_rejects_a_chain_break(monkeypatch):
    broken = _linear_backbone(n=160)
    broken[(9, "CA")] = np.array([100.0, 0.0, 0.0])
    monkeypatch.setattr(fb, "load_backbone", lambda path: broken)
    row = fb.evaluate("x.cif.gz", broken, (5, 6, 7), "cell")
    assert row["status"] == "chain_break"


def test_evaluate_rejects_motif_drift(monkeypatch):
    """Displace a SIDECHAIN atom, not the CA: moving a CA far enough to drift the
    motif also opens a CA-CA gap, and `evaluate` checks chain_break BEFORE
    motif_drift, so a CA-based fixture would assert the wrong status for the
    right reason. Sidechain drift isolates the motif gate cleanly."""
    ref = _linear_backbone()
    ref[(6, "CB")] = ref[(6, "CA")] + np.array([1.5, 0.0, 0.0])
    moved = dict(ref)
    moved[(6, "CB")] = moved[(6, "CB")] + np.array([0.0, 0.0, 2.0])
    monkeypatch.setattr(fb, "load_backbone", lambda path: moved)
    row = fb.evaluate("x.cif.gz", ref, (6,), "cell")
    assert row["status"] == "motif_drift"
    assert float(row["motif_rmsd"]) > constants.BACKBONE_MOTIF_RMSD_MAX
    assert float(row["max_ca_break"]) <= constants.CA_BREAK_MAX   # not a chain break


def test_evaluate_rejects_a_displaced_zn(monkeypatch):
    ref = _linear_backbone()
    bad = dict(ref)
    bad[(201, "ZN")] = np.array([0.0, 8.0, 0.0])
    monkeypatch.setattr(fb, "load_backbone", lambda path: bad)
    row = fb.evaluate("x.cif.gz", ref, (5, 6, 7), "cell")
    assert row["status"] == "zn_displaced"


def test_evaluate_reports_metal_missing_when_no_zn_is_found(monkeypatch):
    """Absence of the Zn is a MEASUREMENT failure, not a geometry verdict --
    it must not read as `zn_displaced`, the same shape as a real rejection."""
    ref = _linear_backbone()
    no_zn = dict(ref)
    del no_zn[(201, "ZN")]
    monkeypatch.setattr(fb, "load_backbone", lambda path: no_zn)
    monkeypatch.setattr(fb.score_structure, "metal_xyz", lambda path: None)
    row = fb.evaluate("x.cif.gz", ref, (5, 6, 7), "cell")
    assert row["status"] == "metal_missing"


def test_evaluate_reports_metal_ambiguous_on_more_than_one_zn(monkeypatch):
    """`metal_xyz` raises ValueError on >1 match; that must not fall through
    to the same nan-driven `zn_displaced` a real geometric rejection uses."""
    ref = _linear_backbone()
    monkeypatch.setattr(fb, "load_backbone", lambda path: dict(ref))

    def boom(path):
        raise ValueError("expected one ZN, found 2")
    monkeypatch.setattr(fb.score_structure, "metal_xyz", boom)
    row = fb.evaluate("x.cif.gz", ref, (5, 6, 7), "cell")
    assert row["status"] == "metal_ambiguous"


def test_evaluate_passes_a_good_backbone(monkeypatch):
    ref = _linear_backbone()
    monkeypatch.setattr(fb, "load_backbone", lambda path: dict(ref))
    row = fb.evaluate("x.cif.gz", ref, (5, 6, 7), "cell")
    assert row["status"] == "ok" and row["passed"] == "True"
    assert float(row["motif_rmsd"]) == pytest.approx(0.0, abs=1e-6)


def test_evaluate_returns_a_failed_row_instead_of_raising(monkeypatch):
    """A single unreadable file must not kill the shard."""
    def boom(path):
        raise OSError("corrupt cif")
    monkeypatch.setattr(fb, "load_backbone", boom)
    row = fb.evaluate("x.cif.gz", _linear_backbone(), (5,), "cell")
    assert row["passed"] == "False"
    assert "corrupt cif" in row["status"]


def test_columns_are_stable_and_include_the_cell():
    for col in ("backbone", "cell", "parent", "arm", "partial_t",
                "n_res", "max_ca_break", "motif_rmsd", "status", "passed"):
        assert col in fb.COLUMNS


def test_zn_donor_distances_honours_an_explicit_metal_position():
    """The real path supplies the metal from a chain-agnostic lookup."""
    atoms = _linear_backbone()
    del atoms[(201, "ZN")]                       # not in the chain-scoped dict
    d = fb.zn_donor_distances(atoms, zn_xyz=np.array([0.0, 0.0, 0.0]))
    assert all(v == pytest.approx(2.2) for v in d.values())


def test_backbone_id_survives_a_float_in_the_cell_name():
    """Two models of one cell must get DISTINCT ids; `split(".")[0]` collided them."""
    a = fb.backbone_id("/x/cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_0.cif.gz")
    b = fb.backbone_id("/x/cell_TadA8e_FULL_pt1.0_TadA8e_FULL_pt1.0_0_model_1.cif.gz")
    assert a.endswith("_model_0") and b.endswith("_model_1")
    assert a != b


def test_main_derives_the_cell_from_the_directory_not_the_filename(tmp_path, monkeypatch):
    run = tmp_path / "run"
    cell = "TadA8e_FULL_pt1.0"
    cell_dir = run / "rfd" / cell
    cell_dir.mkdir(parents=True)
    (cell_dir / f"cell_{cell}_{cell}_0_model_0.cif.gz").write_bytes(b"")
    atoms = _linear_backbone()
    monkeypatch.setattr(fb, "load_backbone", lambda path: dict(atoms))
    monkeypatch.setattr(fb.score_structure, "heavy_atoms_from_pdb",
                        lambda path, chain=None: dict(atoms))
    monkeypatch.setattr(fb.motif, "load_masks", lambda: {})
    monkeypatch.setattr(fb.motif, "arm_residues", lambda arm, masks: (5, 6, 7))
    assert fb.main(["--run-dir", str(run)]) == 0
    rows = tio.read_tsv(str(run / "backbones.tsv"))
    assert len(rows) == 1
    assert rows[0]["cell"] == cell
    assert rows[0]["parent"] == "TadA8e" and rows[0]["arm"] == "FULL"
    assert rows[0]["partial_t"] == "1.0"
    assert rows[0]["status"] == "ok"


def test_main_low_pass_rate_with_every_row_written_is_not_degraded(tmp_path, monkeypatch):
    """A rejection is a measurement, not a stage failure: `partial_t=6`'s
    poor yield is an EXPECTED result of this stage and must not itself trip
    the same refusal meant to catch a broken measurement."""
    run = tmp_path / "run"
    cell = "TadA8e_FULL_pt6.0"
    cell_dir = run / "rfd" / cell
    cell_dir.mkdir(parents=True)
    n = 5
    for i in range(n):
        (cell_dir / f"cell_{cell}_{cell}_0_model_{i}.cif.gz").write_bytes(b"")

    short = _linear_backbone(n=40)     # fails length_out_of_range, every time
    monkeypatch.setattr(fb, "load_backbone", lambda path: dict(short))
    monkeypatch.setattr(fb.score_structure, "heavy_atoms_from_pdb",
                        lambda path, chain=None: dict(short))
    monkeypatch.setattr(fb.motif, "load_masks", lambda: {})
    monkeypatch.setattr(fb.motif, "arm_residues", lambda arm, masks: (5, 6, 7))

    assert fb.main(["--run-dir", str(run)]) == 0
    out_path = run / "backbones.tsv"
    assert out_path.exists()                       # canonical path, NOT renamed
    assert not (run / "backbones.degraded.tsv").exists()
    rows = tio.read_tsv(str(out_path))
    assert len(rows) == n
    assert all(r["status"] == "length_out_of_range" for r in rows)

    doc = json.load(open(run / "filter_backbones.provenance.json"))
    assert doc["is_degraded"] is False
    assert doc["n_in"] == n and doc["n_out"] == n
    assert doc["extra"]["n_passed"] == 0
    assert doc["extra"]["pass_rate"] == 0.0


def test_main_a_genuine_row_loss_still_triggers_degraded(tmp_path, monkeypatch):
    """An input that produced NO row -- a write failure -- is the actual
    signal the degraded refusal exists for, unlike a low pass rate."""
    run = tmp_path / "run"
    cell = "TadA8e_FULL_pt1.0"
    cell_dir = run / "rfd" / cell
    cell_dir.mkdir(parents=True)
    for i in range(2):
        (cell_dir / f"cell_{cell}_{cell}_0_model_{i}.cif.gz").write_bytes(b"")

    atoms = _linear_backbone()
    monkeypatch.setattr(fb, "load_backbone", lambda path: dict(atoms))
    monkeypatch.setattr(fb.score_structure, "heavy_atoms_from_pdb",
                        lambda path, chain=None: dict(atoms))
    monkeypatch.setattr(fb.motif, "load_masks", lambda: {})
    monkeypatch.setattr(fb.motif, "arm_residues", lambda arm, masks: (5, 6, 7))

    real_append = fb.io.append_row

    def flaky_append(path, row, columns):
        if "model_1" in row["backbone"]:
            raise OSError("disk full")
        return real_append(path, row, columns)
    monkeypatch.setattr(fb.io, "append_row", flaky_append)

    assert fb.main(["--run-dir", str(run)]) == 0
    assert not (run / "backbones.tsv").exists()
    degraded_path = run / "backbones.degraded.tsv"
    assert degraded_path.exists()
    rows = tio.read_tsv(str(degraded_path))
    assert len(rows) == 1

    doc = json.load(open(run / "filter_backbones.provenance.json"))
    assert doc["is_degraded"] is True
    assert doc["n_in"] == 2 and doc["n_out"] == 1


def test_main_warns_when_an_expected_cell_produced_no_files_at_all(
        tmp_path, monkeypatch, capsys):
    """A cell whose RFD3 array task died contributes zero files and never
    appears in per_cell -- the ZERO-survivor warning must still fire, driven
    by the EXPECTED cells in `rfd_in/rfd_inputs.yaml`, never by whatever the
    glob happened to find."""
    run = tmp_path / "run"
    present_cell = "TadA8e_FULL_pt1.0"
    missing_cell = "TadA8e_FULL_pt6.0"
    cell_dir = run / "rfd" / present_cell
    cell_dir.mkdir(parents=True)
    (cell_dir / f"cell_{present_cell}_{present_cell}_0_model_0.cif.gz").write_bytes(b"")
    rfd_in = run / "rfd_in"
    rfd_in.mkdir(parents=True)
    yaml.safe_dump({present_cell: {}, missing_cell: {}},
                   open(rfd_in / "rfd_inputs.yaml", "w"))

    atoms = _linear_backbone()
    monkeypatch.setattr(fb, "load_backbone", lambda path: dict(atoms))
    monkeypatch.setattr(fb.score_structure, "heavy_atoms_from_pdb",
                        lambda path, chain=None: dict(atoms))
    monkeypatch.setattr(fb.motif, "load_masks", lambda: {})
    monkeypatch.setattr(fb.motif, "arm_residues", lambda arm, masks: (5, 6, 7))

    assert fb.main(["--run-dir", str(run)]) == 0
    out = capsys.readouterr().out
    assert f"WARNING {missing_cell}: ZERO backbones present" in out

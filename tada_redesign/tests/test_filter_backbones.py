"""Every gate here is a cheap CPU check standing between RFD3's output and the
much more expensive sequence-design and folding stages."""
import numpy as np
import pytest

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

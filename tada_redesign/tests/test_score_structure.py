"""Geometry tests. The two rigid-body tests are non-negotiable: their absence
is exactly how the predecessor's pocket_rmsd shipped with no superposition at
all, which would have emptied a shortlist after real GPU spend."""
import numpy as np
import pytest

from tada_redesign import constants
from tada_redesign import score_structure as ss


def _synthetic_atoms(n_res=6, seed=0):
    """{(resnum, atom): xyz} with CA + one sidechain atom per residue."""
    rng = np.random.default_rng(seed)
    atoms = {}
    for i in range(1, n_res + 1):
        base = rng.normal(size=3) * 5.0
        atoms[(i, "CA")] = base
        atoms[(i, "CB")] = base + np.array([1.5, 0.0, 0.0])
    return atoms


def _rotation(deg=37.0):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _transformed(atoms, R, shift):
    return {k: R @ v + shift for k, v in atoms.items()}


def test_kabsch_recovers_a_known_rigid_transform():
    ref = _synthetic_atoms()
    moved = _transformed(ref, _rotation(), np.array([10.0, -4.0, 7.0]))
    keys = sorted(ref)
    P = np.array([ref[k] for k in keys])
    Q = np.array([moved[k] for k in keys])
    R, P_mean, Q_mean = ss.kabsch(P, Q)
    back = ss.apply_transform(Q, R, P_mean, Q_mean)
    assert np.allclose(back, P, atol=1e-8)


def test_motif_rmsd_is_invariant_to_rigid_body_motion():
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(51.0), np.array([-30.0, 12.0, 3.0]))
    rmsd = ss.motif_rmsd(ref, pred, residues=(2, 3, 4))
    assert rmsd == pytest.approx(0.0, abs=1e-8)


def test_motif_rmsd_reports_a_local_deviation_not_the_global_offset():
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(51.0), np.array([-30.0, 12.0, 3.0]))
    # perturb ONE measured atom by a known 0.6 A in the prediction's own frame
    R = _rotation(51.0)
    pred[(3, "CB")] = pred[(3, "CB")] + R @ np.array([0.6, 0.0, 0.0])
    # 6 measured atoms over residues 2,3,4 -> rmsd = sqrt(0.6^2 / 6)
    rmsd = ss.motif_rmsd(ref, pred, residues=(2, 3, 4))
    assert rmsd == pytest.approx(np.sqrt(0.36 / 6), abs=1e-6)


def test_motif_rmsd_raises_when_a_measured_residue_is_absent():
    ref = _synthetic_atoms()
    pred = _synthetic_atoms()
    del pred[(3, "CB")]
    with pytest.raises(KeyError):
        ss.motif_rmsd(ref, pred, residues=(2, 3, 4))


def test_motif_rmsd_raises_below_three_anchor_points():
    ref = _synthetic_atoms()
    pred = _synthetic_atoms()
    with pytest.raises(ValueError):
        ss.motif_rmsd(ref, pred, residues=(1,), anchor_residues=(1, 2))


def test_cleft_clearance_is_large_when_the_pocket_is_open():
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(20.0), np.array([4.0, 4.0, 4.0]))
    far = np.array([[100.0, 100.0, 100.0]])
    assert ss.cleft_clearance(ref, pred, far) > 50.0


def test_cleft_clearance_detects_a_closed_cleft():
    """A substrate atom placed on top of a design atom (in the ref frame) must
    come back as a sub-Angstrom clearance after superposition."""
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(20.0), np.array([4.0, 4.0, 4.0]))
    on_top = np.array([ref[(3, "CB")] + np.array([0.3, 0.0, 0.0])])
    assert ss.cleft_clearance(ref, pred, on_top) == pytest.approx(0.3, abs=1e-6)


def _substrate_xyz():
    """Delegates to substrate.substrate_xyz -- one extractor, not two."""
    from tada_redesign import substrate
    return substrate.substrate_xyz()


def test_cleft_clearance_excludes_the_catalytic_metal_by_default():
    """The Zn contacts the target base at ~2.12 A as correct catalytic geometry.
    Counting it made a correctly-placed metal read as a collapsed cleft."""
    raw = ss.heavy_atoms_from_pdb(constants.CHAINF_RAW)
    az = _substrate_xyz()
    with_zn = ss.cleft_clearance(raw, raw, az, exclude_atom_names=())
    without_zn = ss.cleft_clearance(raw, raw, az)
    assert with_zn == pytest.approx(2.121, abs=0.01)
    assert without_zn == pytest.approx(2.330, abs=0.01)
    assert without_zn > with_zn


def test_both_relaxed_parents_clear_their_own_cleft_gate():
    """A gate the unmodified parent cannot pass is a broken gate."""
    raw = ss.heavy_atoms_from_pdb(constants.CHAINF_RAW)
    az = _substrate_xyz()
    for parent, pdb in constants.PARENT_PDB.items():
        clearance = ss.cleft_clearance(raw, ss.heavy_atoms_from_pdb(pdb), az)
        assert clearance > constants.CLEFT_CLEARANCE_MARGIN, parent
        assert clearance == pytest.approx({"TadA8e": 2.211, "TadA9": 2.271}[parent],
                                          abs=0.01)


def test_cleft_clearance_still_catches_a_protein_clash():
    """Excluding the metal must not blind the gate to a real collapse."""
    ref = _synthetic_atoms()
    pred = _transformed(ref, _rotation(20.0), np.array([4.0, 4.0, 4.0]))
    on_top = np.array([ref[(3, "CB")] + np.array([0.3, 0.0, 0.0])])
    assert ss.cleft_clearance(ref, pred, on_top) == pytest.approx(0.3, abs=1e-6)


def test_heavy_atoms_from_pdb_skips_hydrogens_and_other_chains(tmp_path):
    # These four lines are COLUMN-EXACT: chain in col 22, resSeq 23-26,
    # coords 31-54, element right-justified in 77-78. Verified by generating
    # them from a column-aware writer -- an eyeballed HETATM line put the Zn's
    # element column at 'N' instead of 'ZN', which made an earlier version of
    # this fixture pass for the wrong reason.
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA F  57       1.000   2.000   3.000  1.00  0.00           C\n"
        "ATOM      2  HA  ALA F  57       1.500   2.000   3.000  1.00  0.00           H\n"
        "ATOM      3  CA  ALA E  57       9.000   9.000   9.000  1.00  0.00           C\n"
        "HETATM    4  ZN   ZN F 201       4.000   5.000   6.000  1.00  0.00          ZN\n"
        "END\n")
    atoms = ss.heavy_atoms_from_pdb(str(pdb), chain="F")
    assert (57, "CA") in atoms
    assert (57, "HA") not in atoms          # hydrogen dropped
    assert (201, "ZN") in atoms             # hetero heavy atom kept
    assert all(k[0] in (57, 201) for k in atoms)   # chain E excluded
    assert np.allclose(atoms[(57, "CA")], [1.0, 2.0, 3.0])


def test_heavy_atoms_from_cif_reads_the_atom_site_loop(tmp_path):
    cif = tmp_path / "mini.cif"
    cif.write_text(
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "ATOM   C  CA ALA F 57 1.000 2.000 3.000\n"
        "ATOM   H  HA ALA F 57 1.500 2.000 3.000\n"
        "ATOM   C  CA ALA E 57 9.000 9.000 9.000\n"
        "HETATM ZN ZN ZN  F 201 4.000 5.000 6.000\n")
    atoms = ss.heavy_atoms_from_cif(str(cif), chain="F")
    assert (57, "CA") in atoms
    assert (57, "HA") not in atoms
    assert (201, "ZN") in atoms
    assert np.allclose(atoms[(57, "CA")], [1.0, 2.0, 3.0])


def test_metal_xyz_finds_the_zn_on_a_foreign_chain(tmp_path):
    """RFD3 returns the Zn on a relabelled chain; a chain-scoped reader misses it
    and every donor distance silently becomes nan."""
    cif = tmp_path / "m.cif"
    cif.write_text(
        "data_t\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM   C  CA ALA F 57 1.000 2.000 3.000\n"
        "HETATM ZN ZN ZN  B 161 4.000 5.000 6.000\n")
    assert ss.heavy_atoms_from_cif(str(cif), chain="F").get((161, "ZN")) is None
    assert np.allclose(ss.metal_xyz(str(cif)), [4.0, 5.0, 6.0])


def test_metal_xyz_returns_none_when_absent(tmp_path):
    cif = tmp_path / "n.cif"
    cif.write_text(
        "data_t\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM   C  CA ALA F 57 1.000 2.000 3.000\n")
    assert ss.metal_xyz(str(cif)) is None


def test_align_numbering_shifts_a_model_numbered_from_one():
    ref = {(5, "CA"): np.zeros(3), (6, "CA"): np.ones(3)}
    pred = {(1, "CA"): np.zeros(3), (2, "CA"): np.ones(3)}
    got = ss.align_numbering(ref, pred)
    assert sorted(ss.ca_map(got)) == [5, 6]


def test_align_numbering_is_a_no_op_when_numbering_already_matches():
    ref = {(5, "CA"): np.zeros(3), (6, "CA"): np.ones(3)}
    assert ss.align_numbering(ref, dict(ref)) == ref


def test_align_numbering_raises_on_a_residue_count_mismatch():
    ref = {(5, "CA"): np.zeros(3), (6, "CA"): np.ones(3)}
    with pytest.raises(ValueError):
        ss.align_numbering(ref, {(1, "CA"): np.zeros(3)})


def test_align_numbering_raises_when_no_single_shift_reconciles_them():
    """A gapped prediction must fail rather than be forced into alignment."""
    ref = {(5, "CA"): np.zeros(3), (6, "CA"): np.ones(3), (7, "CA"): np.ones(3)}
    pred = {(1, "CA"): np.zeros(3), (2, "CA"): np.ones(3), (9, "CA"): np.ones(3)}
    with pytest.raises(ValueError):
        ss.align_numbering(ref, pred)

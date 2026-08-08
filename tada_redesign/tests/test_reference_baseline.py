"""The baseline must be folded in the SAME mode as the designs it gates, or the
comparison is meaningless."""
import json

import numpy as np
import pytest

from tada_redesign import constants, motif, reference_baseline as rb


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


def test_core_motif_residues_matches_the_shared_definition():
    """`score_folds.py` measures `motif.arm_residues(motif.CORE_MOTIF, ...)`;
    `reference_baseline` must measure the identical set, or the parent's own
    number is not comparable to a design's."""
    masks = motif.load_masks()
    assert rb.core_motif_residues() == motif.arm_residues(motif.CORE_MOTIF, masks)
    assert len(rb.core_motif_residues()) == 17


def _pdb_atom_line(serial, atom_name, resnum, chain, xyz, resname="ALA"):
    """One fixed-column ATOM line. Columns are PDB-standard (chain @ 22,
    resSeq @ 23-26, coords @ 31-54), matching `score_structure.heavy_atoms_from_pdb`'s
    own column-exact parser."""
    element = atom_name.strip()[0]
    line = [" "] * 80

    def put(s, start):
        for i, ch in enumerate(str(s)):
            line[start + i] = ch

    put("ATOM".ljust(6), 0)
    put(str(serial).rjust(5), 6)
    name = atom_name if len(atom_name) == 4 else (" " + atom_name).ljust(4)
    put(name, 12)
    put(resname.ljust(3), 17)
    put(chain, 21)
    put(str(resnum).rjust(4), 22)
    put(f"{xyz[0]:8.3f}", 30)
    put(f"{xyz[1]:8.3f}", 38)
    put(f"{xyz[2]:8.3f}", 46)
    put("  1.00", 54)
    put("  0.00", 60)
    put(element.rjust(2), 76)
    return "".join(line) + "\n"


def _write_ref_pdb(path, resnums, chain="F"):
    lines, serial = [], 1
    for resnum in resnums:
        lines.append(_pdb_atom_line(serial, "CA", resnum, chain,
                                    (float(resnum), 0.0, 0.0)))
        serial += 1
        lines.append(_pdb_atom_line(serial, "CB", resnum, chain,
                                    (float(resnum), 1.5, 0.0)))
        serial += 1
    path.write_text("".join(lines) + "END\n")


def _write_pred_cif(path, resnums, resnum_offset=0, shift=(0.0, 0.0, 0.0),
                    perturb=None, chain="F"):
    """A minimal `_atom_site` loop numbered `resnum - resnum_offset` (as a
    folding model, numbered from 1, would be), rigid-shifted by `shift`.
    `perturb` is an optional {(resnum, atom): (dx, dy, dz)} nudge applied on
    top of the shift, to check a real local deviation is measured."""
    perturb = perturb or {}
    header = (
        "data_test\nloop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n")
    rows = []
    for resnum in resnums:
        pred_resnum = resnum - resnum_offset
        for atom, base in (("CA", (float(resnum), 0.0, 0.0)),
                           ("CB", (float(resnum), 1.5, 0.0))):
            dx, dy, dz = perturb.get((resnum, atom), (0.0, 0.0, 0.0))
            x = base[0] + shift[0] + dx
            y = base[1] + shift[1] + dy
            z = base[2] + shift[2] + dz
            rows.append(f"ATOM C {atom} ALA {chain} {pred_resnum} "
                       f"{x:.3f} {y:.3f} {z:.3f}\n")
    path.write_text(header + "".join(rows))


def test_score_baseline_rmsd_returns_nothing_when_no_fold_exists(tmp_path):
    """A parent with no CIF on disk yet must be absent, not a fabricated 0 --
    `read_baseline`'s `require=` is the place that enforces presence."""
    (tmp_path / "baseline").mkdir()
    assert rb.score_baseline_rmsd(str(tmp_path), residues=(57, 58, 59)) == {}


def test_score_baseline_rmsd_wires_align_numbering_before_motif_rmsd(tmp_path, monkeypatch):
    """The whole point of this function: thread a fold-model CIF (numbered
    from 1) through the IDENTICAL heavy_atoms_from_cif -> align_numbering ->
    motif_rmsd sequence `score_folds.score_one` uses for every design. A pure
    rigid transform must round-trip to ~0 A once renumbered and superposed --
    if align_numbering were NOT wired in, this would raise KeyError instead."""
    ref_pdb = tmp_path / "ref.pdb"
    _write_ref_pdb(ref_pdb, resnums=[57, 58, 59])
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    cif = baseline_dir / f"{rb.baseline_id('TadA8e')}.cif"
    _write_pred_cif(cif, resnums=[57, 58, 59], resnum_offset=56,
                    shift=(10.0, -3.0, 4.0))
    monkeypatch.setitem(constants.RMSD_REFERENCE, "TadA8e", str(ref_pdb))
    rmsd = rb.score_baseline_rmsd(str(tmp_path), parents=["TadA8e"],
                                  residues=(57, 58, 59))
    assert rmsd["TadA8e"] == pytest.approx(0.0, abs=1e-6)


def test_score_baseline_rmsd_reflects_a_known_local_perturbation(tmp_path, monkeypatch):
    """A larger anchor (6 residues' CA) than the measured set (3 residues)
    isolates the anchor fit from the one perturbed atom, so the returned
    value must equal the perturbation exactly -- proof this is a real
    measurement, not a value that is ~0 merely because the anchor and the
    measured set coincide."""
    resnums = [57, 58, 59, 60, 61, 62]
    ref_pdb = tmp_path / "ref.pdb"
    _write_ref_pdb(ref_pdb, resnums=resnums)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    cif = baseline_dir / f"{rb.baseline_id('TadA9')}.cif"
    # 3 measured atoms over residues (57, 58, 59); one CB perturbed by 0.6 A.
    _write_pred_cif(cif, resnums=resnums, resnum_offset=56,
                    shift=(5.0, 5.0, 5.0),
                    perturb={(58, "CB"): (0.6, 0.0, 0.0)})
    monkeypatch.setitem(constants.RMSD_REFERENCE, "TadA9", str(ref_pdb))
    rmsd = rb.score_baseline_rmsd(str(tmp_path), parents=["TadA9"],
                                  residues=(57, 58, 59))
    assert rmsd["TadA9"] == pytest.approx(np.sqrt(0.36 / 6), abs=1e-5)


def test_main_score_only_never_invokes_fold_many_and_writes_the_summary(tmp_path, monkeypatch):
    """`--score-only` must not touch `subprocess.run` at all -- no fold, no
    GPU, no SLURM -- and must still produce a readable summary table and
    provenance carrying the CORE motif RMSD."""
    resnums = [57, 58, 59]
    ref_pdb = tmp_path / "ref.pdb"
    _write_ref_pdb(ref_pdb, resnums=resnums)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    for parent in constants.PARENTS:
        cif = baseline_dir / f"{rb.baseline_id(parent)}.cif"
        _write_pred_cif(cif, resnums=resnums, resnum_offset=56)
        json.dump({"plddt": 0.9}, open(baseline_dir / f"{rb.baseline_id(parent)}.metrics.json", "w"))
        monkeypatch.setitem(constants.RMSD_REFERENCE, parent, str(ref_pdb))
    monkeypatch.setattr(rb, "core_motif_residues", lambda: (57, 58, 59))

    calls = []
    real_run = rb.subprocess.run

    def _tracking_run(cmd, *a, **k):
        # provenance.write's own `git rev-parse` call also goes through this
        # SAME shared `subprocess` module object, so assert on CONTENT (no
        # fold_many.py invocation) rather than forbidding subprocess use
        # outright.
        calls.append(cmd)
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(rb.subprocess, "run", _tracking_run)

    rc = rb.main(["--run-dir", str(tmp_path), "--score-only"])
    assert not any("fold_many.py" in " ".join(c) for c in calls), calls
    assert rc == 0
    summary = (baseline_dir / "baseline_summary.tsv").read_text()
    assert "core_motif_rmsd" in summary
    for parent in constants.PARENTS:
        assert parent in summary
    prov = json.load(open(baseline_dir / "reference_baseline.provenance.json"))
    assert set(prov["extra"]["core_motif_rmsd"]) == set(constants.PARENTS)

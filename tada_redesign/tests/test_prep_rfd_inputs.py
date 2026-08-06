"""RFD3 spec semantics are unforgiving and every rule here was learned from a
real failure -- see the citations in prep_rfd_inputs' docstring."""
import os

import pytest
import yaml

from tada_redesign import constants, motif, prep_rfd_inputs as prep


@pytest.fixture
def masks():
    return motif.load_masks()


def test_cell_id_encodes_all_three_axes():
    assert prep.cell_id("TadA8e", "FULL", 1.0) == "TadA8e_FULL_pt1.0"
    assert prep.cell_id("TadA9", "MIN", 6.0) == "TadA9_MIN_pt6.0"


def test_build_specs_covers_the_sixteen_cells(masks):
    specs = prep.build_specs(masks, {p: f"/tmp/{p}.pdb" for p in constants.PARENTS})
    assert len(specs) == 16 == len(constants.PARENTS) * len(constants.ARMS) * len(constants.PARTIAL_T)
    assert prep.cell_id("TadA8e", "FULL", 1.0) in specs


def test_spec_is_partial_diffusion_with_no_contig(masks):
    """Partial diffusion re-noises the input in place; emitting a contig would
    make RFD3 build a new chain instead."""
    spec = prep.build_spec("TadA8e", "FULL", 2.0, masks, "/tmp/TadA8e.pdb")
    assert spec["input"] == "/tmp/TadA8e.pdb"
    assert spec["partial_t"] == 2.0
    assert "contig" not in spec
    assert spec["is_non_loopy"] is True


def test_spec_fixes_the_arm_motif_and_the_zn_by_ccd_name(masks):
    spec = prep.build_spec("TadA8e", "MIN", 1.0, masks, "/tmp/x.pdb")
    fixed = spec["select_fixed_atoms"]
    assert fixed == motif.rfd_select_fixed_atoms("MIN", masks)
    assert fixed[constants.ZN_RESNAME] == "ALL"
    assert "F201" not in fixed
    assert spec["ligand"] == constants.ZN_RESNAME
    # RFD3 does not retain a nucleic-acid chain through partial diffusion
    # (measured -- see build_spec's docstring): no DNA key belongs here.
    assert "unindex" not in spec


def test_spec_aims_hotspots_at_the_retained_flanking_nucleotides(masks):
    spec = prep.build_spec("TadA8e", "FULL", 1.0, masks, "/tmp/x.pdb")
    assert set(spec["select_hotspots"]) == {
        f"{constants.SUBSTRATE_CHAIN}{r}" for r in constants.HOTSPOT_RESIDS}
    assert spec["infer_ori_strategy"] == "hotspots"


def test_partial_t_never_reaches_the_measured_cliff(masks):
    """partial_t=8 was measured to degrade the active site to 2.5-6.75 A RMSD."""
    specs = prep.build_specs(masks, {p: f"/tmp/{p}.pdb" for p in constants.PARENTS})
    assert max(s["partial_t"] for s in specs.values()) < 8.0


def test_write_input_pdb_carries_chain_f_the_zn_and_the_dna_context(tmp_path):
    out = str(tmp_path / "TadA8e.pdb")
    prep.write_input_pdb("TadA8e", out)
    lines = [ln for ln in open(out) if ln.startswith(("ATOM", "HETATM"))]
    chains = {ln[21] for ln in lines}
    assert chains == {constants.SCAFFOLD_CHAIN, constants.SUBSTRATE_CHAIN}
    assert any(ln.startswith("HETATM") and ln[17:20].strip() == "ZN" for ln in lines)
    dna_resids = {int(ln[22:26]) for ln in lines if ln[21] == constants.SUBSTRATE_CHAIN}
    assert dna_resids == set(constants.DNA_CONTEXT_RESIDS)


def test_yaml_round_trips(tmp_path, masks):
    specs = prep.build_specs(masks, {p: f"/tmp/{p}.pdb" for p in constants.PARENTS})
    path = tmp_path / "rfd_inputs.yaml"
    path.write_text(yaml.safe_dump(specs, sort_keys=False))
    assert yaml.safe_load(path.read_text()) == specs

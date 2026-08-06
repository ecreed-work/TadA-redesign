"""LigandMPNN inputs: one multi-JSON set per arm, plus the solubility bias.

LigandMPNN accepts `--pdb_path_multi`, `--fixed_residues_multi` and
`--bias_AA_per_residue_multi` as JSON maps keyed by PDB path, so ONE process
designs every backbone of an arm at one temperature. That turns this stage from
~2,560 model loads into 10 invocations (2 arms x (4 temperatures + 1 control)).

The Zn travels inside each backbone PDB as a heteroatom; LigandMPNN reads it
as ligand context, which is the whole reason this campaign uses LigandMPNN
rather than plain ProteinMPNN (which has no ligand channel and would design
the metal site as if it were empty). The ssDNA substrate does NOT travel the
same way: RFD3 consumes it for `select_hotspots` orientation but does not
emit it from partial diffusion (measured; see `prep_rfd_inputs.py`'s
docstring, and the unindex-based retention attempt logged there as blocked
by the inference pipeline's `UnindexFlaggedTokens` transform). This is a
deliberate design decision, not an oversight: partial diffusion barely
perturbs the motif (`motif_rmsd` 0.036-0.057 A at `partial_t=1.0`, measured),
so the cleft cannot collapse from the diffusion step itself. Where the
missing substrate DOES matter is HERE, at sequence design -- an unoccupied
groove reads to LigandMPNN as unsatisfied surface to pack with hydrophobics
-- so `graft_substrate` places the crystal chain-D context onto each
designed backbone by superposition before the LigandMPNN inputs are written.

An earlier log entry claimed "the DNA/Zn context was correctly read as ligand
context" based on LigandMPNN designs containing no `:` (which would indicate
a second designed chain). That check cannot distinguish "correctly read as
ligand context" from "there was no DNA there to mis-parse in the first
place" -- at the time, there was no DNA in the backbone PDB at all, so the
absence of `:` proved nothing about ligand-context handling. See the
correction in `docs/logs/20260805_tada_redesign_part2.md`.

Solubility bias is applied ONLY at positions that are exposed AND designable
(`EXPOSED & MODELED - arm_residues`) IN THE PARENT. Buried positions are never
biased -- pushing polar residues into the core would trade away the very
stability the campaign is buying -- and frozen positions cannot change
identity at all. The magnitudes are assumptions, which is why a zero-bias
control set is designed alongside and carried through every scoring stage.

Scope limitation, stated rather than fixed here: `bias_positions` derives
exposure ONCE from the parent's own RASA (`masks.json`'s `EXPOSED`), not
per-backbone. At `partial_t=4/6` a backbone can drift enough that a
formerly-exposed position becomes buried in THAT backbone while still being
biased as if it were still exposed -- this module has no way to know, since
it never re-measures RASA on the diffused backbone. Per-backbone
recomputation is deferred to Part 3, not implemented here.

Honesty ceiling: a bias makes a residue more likely to be chosen. It does not
make the protein soluble, and nothing here measures solubility.
"""
import argparse
import json
import os

import numpy as np

from . import constants, io, motif, provenance, score_structure, substrate


def bias_positions(arm, masks):
    """Exposed, designable positions: `EXPOSED & MODELED - arm_residues`."""
    exposed = set(masks["EXPOSED"]) & set(masks["MODELED"])
    return tuple(sorted(exposed - set(motif.arm_residues(arm, masks))))


def bias_json(arm, masks, chain=None):
    """`{"F12": {"L": -1.0, ..., "E": 0.3, ...}}` for LigandMPNN."""
    chain = chain or constants.SCAFFOLD_CHAIN
    per_residue = {aa: constants.HYDROPHOBIC_BIAS
                   for aa in constants.HYDROPHOBIC_SET}
    per_residue.update({aa: constants.POLAR_BIAS for aa in constants.POLAR_SET})
    return {f"{chain}{resnum}": dict(per_residue)
            for resnum in bias_positions(arm, masks)}


def convert_to_pdb(cif_gz, out_pdb):
    """LigandMPNN reads PDB, RFD3 writes .cif.gz."""
    import gzip
    import shutil
    import tempfile

    from Bio.PDB import MMCIFParser, PDBIO

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        with gzip.open(cif_gz, "rb") as src:
            shutil.copyfileobj(src, tmp)
        tmp_path = tmp.name
    try:
        structure = MMCIFParser(QUIET=True).get_structure("bb", tmp_path)
    finally:
        os.unlink(tmp_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(out_pdb)
    return out_pdb


def graft_substrate(designed_atoms, parent, designed_pdb, out_pdb=None):
    """Write `out_pdb` (default: overwrite `designed_pdb`) as the designed
    backbone plus the crystal chain-D substrate context, placed by
    superposition onto the designed backbone.

    RFD3 consumes the ssDNA for `select_hotspots` orientation but does not
    emit it (measured; `prep_rfd_inputs.py`'s docstring). LigandMPNN needs it
    present as ligand context or it designs against an empty cleft. The
    relaxed parent (`constants.RMSD_REFERENCE[parent]`) and the crystal
    (`constants.PDB6VPC`) share the 6VPC coordinate frame, and the motif is
    frozen during diffusion, so superposing the parent's CA onto the
    designed backbone's CA and applying that SAME rigid-body transform to
    the crystal's chain-D atoms places the substrate in the designed frame.

    The anchor is every CA the reference parent and the designed backbone
    share (chain F's largely-unchanged numbering under partial diffusion),
    not just the motif -- the same "fit on the full backbone, not the small
    measured/placed set" principle `score_structure._anchor_arrays` uses.
    """
    from Bio.PDB import PDBIO, PDBParser

    out_pdb = out_pdb or designed_pdb
    ref_atoms = score_structure.heavy_atoms_from_pdb(constants.RMSD_REFERENCE[parent])
    ref_ca = score_structure.ca_map(ref_atoms)
    design_ca = score_structure.ca_map(designed_atoms)
    shared = sorted(set(ref_ca) & set(design_ca))
    if len(shared) < 3:
        raise ValueError(
            f"grafting anchor has {len(shared)} shared CA between the "
            f"reference and the designed backbone; need >= 3")
    # P is the DESIGN, Q is the REFERENCE: `kabsch`'s docstring convention
    # superposes Q onto P, so `apply_transform` below maps REFERENCE-frame
    # coordinates (the crystal DNA) INTO the design's frame.
    P = np.array([design_ca[r] for r in shared])
    Q = np.array([ref_ca[r] for r in shared])
    R, P_mean, Q_mean = score_structure.kabsch(P, Q)

    parser = PDBParser(QUIET=True)
    designed_model = parser.get_structure("designed", designed_pdb)[0]
    ref_model = parser.get_structure("ref", constants.PDB6VPC)[0]

    keep = set(substrate.context_residues())
    dna = ref_model[constants.SUBSTRATE_CHAIN].copy()
    for res in list(dna):
        if res.id[1] not in keep:
            dna.detach_child(res.id)
            continue
        for atom in res:
            atom.set_coord(score_structure.apply_transform(
                atom.get_coord(), R, P_mean, Q_mean))
    designed_model.add(dna)

    os.makedirs(os.path.dirname(os.path.abspath(out_pdb)), exist_ok=True)
    writer = PDBIO()
    writer.set_structure(designed_model)
    writer.save(out_pdb)
    return out_pdb


MANIFEST_COLUMNS = ("backbone", "cell", "parent", "arm", "pdb_path")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--run-dir", default=os.path.join(
        sub_dir, "outputs", constants.RUN_DIR_NAME))
    args = ap.parse_args(argv)

    masks = motif.load_masks()
    rows = [r for r in io.read_tsv(os.path.join(args.run_dir, "backbones.tsv"))
            if r["passed"] == "True"]
    if not rows:
        raise SystemExit("[prep_mpnn_inputs] no passing backbones in backbones.tsv")

    out_dir = os.path.join(args.run_dir, "mpnn_in")
    manifest = []
    per_arm = {arm: [] for arm in constants.ARMS}
    for row in rows:
        pdb = convert_to_pdb(row["path"], os.path.join(
            out_dir, "pdb", f"{row['backbone']}.pdb"))
        designed_atoms = score_structure.heavy_atoms_from_pdb(pdb)
        graft_substrate(designed_atoms, row["parent"], pdb)
        per_arm[row["arm"]].append(pdb)
        manifest.append({"backbone": row["backbone"], "cell": row["cell"],
                         "parent": row["parent"], "arm": row["arm"],
                         "pdb_path": pdb})

    for arm, pdbs in per_arm.items():
        arm_dir = os.path.join(out_dir, arm)
        os.makedirs(arm_dir, exist_ok=True)
        fixed = motif.mpnn_fixed_residues(arm, masks)
        bias = bias_json(arm, masks)
        json.dump({p: "" for p in pdbs},
                  open(os.path.join(arm_dir, "pdb_paths.json"), "w"), indent=1)
        json.dump({p: fixed for p in pdbs},
                  open(os.path.join(arm_dir, "fixed_residues.json"), "w"), indent=1)
        json.dump({p: bias for p in pdbs},
                  open(os.path.join(arm_dir, "bias.json"), "w"), indent=1)
        print(f"[prep_mpnn_inputs] {arm}: {len(pdbs)} backbones, "
              f"{len(bias)} biased positions, fixed='{fixed[:40]}...'")

    io.write_tsv(os.path.join(out_dir, "mpnn_manifest.tsv"), manifest,
                 MANIFEST_COLUMNS)
    provenance.write(out_dir, "prep_mpnn_inputs", len(rows), len(manifest),
                     extra={"per_arm": {a: len(p) for a, p in per_arm.items()},
                            "hydrophobic_bias": constants.HYDROPHOBIC_BIAS,
                            "polar_bias": constants.POLAR_BIAS,
                            "hydrophobic_set": constants.HYDROPHOBIC_SET,
                            "polar_set": constants.POLAR_SET})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

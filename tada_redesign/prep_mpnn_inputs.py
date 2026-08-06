"""LigandMPNN inputs: one multi-JSON set per arm, plus the solubility bias.

LigandMPNN accepts `--pdb_path_multi`, `--fixed_residues_multi` and
`--bias_AA_per_residue_multi` as JSON maps keyed by PDB path, so ONE process
designs every backbone of an arm at one temperature. That turns this stage from
~2,560 model loads into 10 invocations (2 arms x (4 temperatures + 1 control)).

The Zn and the ssDNA context travel inside each backbone PDB: LigandMPNN reads
non-protein atoms as ligand context, which is the whole reason this campaign
uses LigandMPNN rather than plain ProteinMPNN (which has no ligand channel and
would design the metal site as if it were empty).

Solubility bias is applied ONLY at positions that are exposed AND designable
(`EXPOSED & MODELED - arm_residues`). Buried positions are never biased --
pushing polar residues into the core would trade away the very stability the
campaign is buying -- and frozen positions cannot change identity at all. The
magnitudes are assumptions, which is why a zero-bias control set is designed
alongside and carried through every scoring stage.

Honesty ceiling: a bias makes a residue more likely to be chosen. It does not
make the protein soluble, and nothing here measures solubility.
"""
import argparse
import json
import os

from . import constants, io, motif, provenance


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

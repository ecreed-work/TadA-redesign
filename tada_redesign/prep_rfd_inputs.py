"""RFD3 partial-diffusion inputs: one spec per (parent, arm, partial_t) cell.

RFD3 semantics used here, each learned from a real failure in
`domain-insertion/denovo_tada/` (read that package's make_rfd_inputs.py and
make_rfd_inputs_brace.py docstrings before touching this file):

  - PARTIAL diffusion re-noises the `input` structure IN PLACE by `partial_t`
    ANGSTROMS of injected noise (RFD3 recommends <= 15). No `contig` is emitted:
    a contig would make RFD3 build a new chain instead of perturbing this one,
    and residue numbering is preserved in place, which is what lets every
    downstream stage keep using chain F's 5-160 numbering.
  - `select_fixed_atoms` injects ZERO noise at the named atoms, so the motif's
    internal geometry is preserved while the group may still move as a rigid
    body. That is exactly the semantics this campaign wants for the active site.
  - The catalytic Zn is named by CCD NAME ("ZN"), never chain+resid. AtomWorks
    renames a hetero atom's chain when it shares a chain letter with protein
    residues, so an "F201" key raises
    `ValidationError: [component=F201] Residue F201 not found in atom array`.
  - `select_hotspots` + `infer_ori_strategy: hotspots` aim the pocket. The
    hotspots are chain D 25 and 27, the nucleotides FLANKING the target base --
    the 8AZ at D26 is a non-standard residue that AtomWorks drops, so it cannot
    be a hotspot.
  - The DNA context is chain D only. Chain C has nothing within 12 A of the
    target base (measured), and a smaller fixed context matters: an earlier
    campaign OOM'd an 80 GB A100 by handing RFD3 the whole Cas9 context.

Honesty ceiling: this module writes input files. It makes no claim that the
resulting backbones fold, bind, or catalyse anything.
"""
import argparse
import os

import yaml

from . import constants, motif, provenance, substrate


def cell_id(parent, arm, partial_t):
    return f"{parent}_{arm}_pt{partial_t}"


def write_input_pdb(parent, out_path, pdb6vpc=None):
    """Relaxed parent (chain F + Zn) plus the chain-D context residues.

    The relaxed parents share the 6VPC coordinate frame, so the DNA is copied
    across without superposition. Verified by test_substrate's frame check.
    """
    from Bio.PDB import PDBIO, PDBParser, Select

    parser = PDBParser(QUIET=True)
    parent_model = parser.get_structure(parent, constants.PARENT_PDB[parent])[0]
    ref_model = parser.get_structure("ref", pdb6vpc or constants.PDB6VPC)[0]

    keep = set(substrate.context_residues(pdb6vpc))
    dna = ref_model[constants.SUBSTRATE_CHAIN].copy()
    for res in list(dna):
        if res.id[1] not in keep:
            dna.detach_child(res.id)
    parent_model.add(dna)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    io_writer = PDBIO()
    io_writer.set_structure(parent_model)
    io_writer.save(out_path)
    return out_path


def build_spec(parent, arm, partial_t, masks, input_pdb):
    """One RFD3 partial-diffusion InputSpecification."""
    return {
        "input": input_pdb,
        "partial_t": float(partial_t),
        "is_non_loopy": True,
        "select_fixed_atoms": motif.rfd_select_fixed_atoms(arm, masks),
        "ligand": constants.ZN_RESNAME,
        "select_hotspots": {f"{constants.SUBSTRATE_CHAIN}{r}": "ALL"
                            for r in constants.HOTSPOT_RESIDS},
        "infer_ori_strategy": "hotspots",
    }


def build_specs(masks, input_pdbs):
    specs = {}
    for parent in constants.PARENTS:
        for arm in constants.ARMS:
            for partial_t in constants.PARTIAL_T:
                specs[cell_id(parent, arm, partial_t)] = build_spec(
                    parent, arm, partial_t, masks, input_pdbs[parent])
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", constants.RUN_DIR_NAME))
    ap.add_argument("--skip-preflight", action="store_true",
                    help="only for unit-level smoke runs; never for a batch")
    args = ap.parse_args(argv)

    if not args.skip_preflight:
        from . import preflight
        preflight.require_green(with_env_probes=False)

    in_dir = os.path.join(args.run_dir, "rfd_in")
    masks = motif.load_masks()
    input_pdbs = {p: write_input_pdb(p, os.path.join(in_dir, f"{p}.pdb"))
                  for p in constants.PARENTS}
    specs = build_specs(masks, input_pdbs)

    yaml_path = os.path.join(in_dir, "rfd_inputs.yaml")
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(specs, fh, sort_keys=False)
    provenance.write(in_dir, "prep_rfd_inputs", len(constants.PARENTS), len(specs),
                     extra={"cells": sorted(specs),
                            "partial_t": list(constants.PARTIAL_T)})
    print(f"[prep_rfd_inputs] wrote {len(specs)} specs -> {yaml_path}")
    for parent, pdb in input_pdbs.items():
        print(f"[prep_rfd_inputs] input {parent}: {pdb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

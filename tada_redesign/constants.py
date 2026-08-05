"""Paths, sweep axes, and gate thresholds for the TadA redesign campaign.

No logic beyond the two design-count helpers. Several values encode a MEASURED
tool or cluster fact and carry that citation inline -- do not "tidy" them.

Residue numbering is Met = 1 (UniProt P68398) throughout.
"""
import os

# ---------------------------------------------------------------- cross-repo
# This package is a submodule; the shared TadA assets live in the parent
# monorepo. Resolved through an env var, mirroring the `R=` pattern the
# existing SLURM scripts already use.
MONOREPO = os.environ.get(
    "TADA_MONOREPO",
    "/research/rgs01/home/clusterHome/ecreed/claude-proteindesign")

TADA_STABILITY = os.path.join(MONOREPO, "tada-stability")
MASKS_JSON = os.path.join(
    TADA_STABILITY, "outputs/20260728_tada_stability/masks.json")
REFERENCE_DIR = os.path.join(TADA_STABILITY, "reference")

PARENTS = ("TadA8e", "TadA9")
PARENT_PDB = {p: os.path.join(REFERENCE_DIR, f"{p}.pdb") for p in PARENTS}
CHAINF_RAW = os.path.join(REFERENCE_DIR, "chainF_raw.pdb")

PDB6VPC = os.path.join(
    MONOREPO,
    "tada8e-cas9-Interface-design/structural_analysis/structures/"
    "deaminase/pdb6vpc.ent")

# Chains B and E only, with the Zn STRIPPED. Any ddG or MPNN run from this file
# would model an apo, partner-less deaminase. preflight asserts it is
# referenced nowhere in this package.
KNOWN_BAD_PDB = os.path.join(
    MONOREPO,
    "tada8e-cas9-Interface-design/interface_design/abe_tadA_ruvc/inputs/"
    "6vpc_dCas9_TadA8e.pdb")

# ------------------------------------------------------------------ structure
SCAFFOLD_CHAIN = "F"        # 6VPC's catalytic protomer: its Zn is 2.12 A from
                            # the 8AZ mimic vs 17.3 A for chain E
ZN_RESNAME = "ZN"           # RFD3 keys the ion by CCD name, never chain+resid
SUBSTRATE_CHAIN = "D"
SUBSTRATE_RESID = 26
SUBSTRATE_RESNAME = "8AZ"   # 8-azanebularine target-base analogue

# ---------------------------------------------------------------- sweep axes
ARMS = ("FULL", "MIN")

# Angstroms of re-noise (RFD3 partial diffusion; the tool recommends <=15).
# denovo_tada's 2026-08-04 debug gate measured partial_t=8 degrading the TadA
# active site from ~1.1 A to 2.5-6.75 A RMSD, failing a 1.5 A gate on EVERY
# design. 6 is therefore the deliberate high end of a survivable range; its
# yield is expected to be poor, and that is a result to report.
PARTIAL_T = (1.0, 2.0, 4.0, 6.0)

MPNN_TEMPS = (0.1, 0.15, 0.2, 0.3)
SEQS_PER_TEMP = 5
RFD_N_BATCHES = 4
RFD_BATCH_SIZE = 8
BACKBONES_PER_CELL = 32

# Solubility biasing is an assumption, not a known good, so it gets a control:
# one unbiased sequence per backbone at a single mid temperature.
CONTROL_TEMP = 0.15
CONTROL_SEQS_PER_BACKBONE = 1

# ---------------------------------------------------------------- gates
BACKBONE_MOTIF_RMSD_MAX = 1.0    # A; RFD3 does not perfectly honour fixed atoms
CA_BREAK_MAX = 4.2               # A between consecutive CA
LENGTH_RANGE = (150, 175)        # residues
ZN_DONOR_RANGE = (2.0, 2.6)      # A, Zn to each of its three donors

SCREEN_PLDDT_MARGIN = 0.05       # 0-1 scale (ESMFold2 reports 0-1, not 0-100)
SCREEN_MOTIF_RMSD_MAX = 1.5      # A, looser: reduced sampling is noisier
FINAL_MOTIF_RMSD_MAX = 1.0       # A, full sampling / AF3
SCREEN_SURVIVORS = 2000          # compute decision; the dropped count is logged
# Cleft openness is gated RELATIVE to the parent's own measured clearance, not
# against an absolute floor. Measured on the committed references with the Zn
# excluded (8AZ = 6VPC chain D 26): TadA8e 2.211 A, TadA9 2.271 A, and the
# crystal chain F 2.330 A (closest protein atom Arg111:NH1). Native substrate
# H-bonds sit at 2.2-2.4 A, so an absolute 2.2 A floor left 0.011 A of headroom
# on TadA8e and FAILED the crystal parent outright once the catalytic Zn was
# counted -- the Zn contacts the target base at 2.12 A by design. A design
# passes when its clearance is no worse than its parent's by more than this.
CLEFT_CLEARANCE_MARGIN = 0.3     # Angstrom, vs the parent's own measurement

AF3_TOP_N = 200
AF3_CONTROL_N = 100              # stratified across the full Rosetta range,
                                 # rejects included, so the ESMFold2<->AF3
                                 # correlation is not range-restricted
ROSETTA_REPLICATES = 3

# A stage failing on more than this fraction of its inputs refuses the
# canonical output path and writes *.degraded.tsv instead.
DEGRADED_FRACTION = 0.20

# --------------------------------------------------------------- fold budgets
# Screen settings are the ones verified working in the 2026-08-04 debug fold.
# Reduced sampling depresses pLDDT substantially (that fold returned 0.45 on a
# 78-mer), which is why the parent is folded in the IDENTICAL mode and the gate
# is relative to it.
ESMFOLD_SCREEN = {"num_loops": 4, "num_sampling_steps": 20}
ESMFOLD_FULL = {"num_loops": 20, "num_sampling_steps": 100}

# ------------------------------------------------------------------- runtime
RUN_DIR_NAME = "20260805_tada_redesign"

ENV_TEST = "ligandmpnn_env"
ENV_RFD3 = "cas9-pam-design"
ENV_MPNN = "ligandmpnn-sc"
ENV_ESM = "esmfold2"
ENV_ROSETTA = "pyrosetta"

RFD3_CKPT = ("/research_jude/rgs01_jude/groups/tsaigrp/projects/Genomics/"
             "common/claude/foundry_ckpt/rfd3_latest.ckpt")
LIGANDMPNN_CKPT = os.path.join(
    MONOREPO, "design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
AF3_SIF = "/hpcf/authorized_apps/rhel8_apps/alphafold/3.0.2/alphafold.3.0.2.sif"
AF3_DB = "/lustre_scratch/reference/public/alphafold_data/af3"
ESMFOLD_HF_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/models--biohub--ESMFold2")
ESMFOLD_FOLD_PY = os.path.join(MONOREPO, "tools/esmfold2/fold.py")


def n_designs():
    """Biased designs: cells x backbones x temperatures x sequences."""
    return (len(PARENTS) * len(ARMS) * len(PARTIAL_T) * BACKBONES_PER_CELL
            * len(MPNN_TEMPS) * SEQS_PER_TEMP)


def n_control_designs():
    """Zero-bias control designs: one per backbone at CONTROL_TEMP."""
    return (len(PARENTS) * len(ARMS) * len(PARTIAL_T) * BACKBONES_PER_CELL
            * CONTROL_SEQS_PER_BACKBONE)

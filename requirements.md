# Environment requirements — tada-redesign

Split across conda envs on purpose; this cluster has a history of envs being
rebuilt out from under a project. Update this file whenever a stage adds a
dependency.

| Env | Location | Used for | Needs |
|---|---|---|---|
| `ligandmpnn_env` | `~/.conda/envs` | the test suite, all pure-Python modules | numpy, Biopython (`Bio.PDB`), pytest |
| `cas9-pam-design` | shared miniforge3 | RFD3 partial diffusion (`rfd3 design`) | rfd3 / foundry |
| `ligandmpnn-sc` | shared miniforge3 | LigandMPNN sequence design | ml_collections, prody, torch |
| `esmfold2` | shared miniforge3 | ESMFold2 folding | Biohub `transformers` fork, torch |
| `pyrosetta` | shared miniforge3 | Zn-constrained relax + `ref2015_cart` | pyrosetta, Biopython |

AF3 runs from a singularity image, not a conda env:
`/hpcf/authorized_apps/rhel8_apps/alphafold/3.0.2/alphafold.3.0.2.sif`.

`Bio.PDB.ShrakeRupley` is broken in `ligandmpnn_env` (uses the removed `np.int`
alias); `tada_stability.sasa` vendors its own Shrake-Rupley and is used instead.

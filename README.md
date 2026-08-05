# TadA-redesign

Diffusion-based stabilization of the ABE8e/ABE9 TadA deaminase domain:
motif-frozen RFdiffusion3 partial diffusion, Zn-aware LigandMPNN sequence
design, orthogonal ESMFold2 + AlphaFold3 validation, and PyRosetta
absolute-score ranking against the identically-processed parents.

Design spec: `docs/superpowers/specs/2026-08-05-tada-redesign-design.md` in the
parent monorepo.

## Honesty ceiling

Every metric here measures **structural plausibility and energetic ranking
within a fold family** — not function. Rosetta `ref2015_cart` absolute score is
not a physical ΔG of folding and does not convert to Tm. pLDDT measures model
confidence, not stability. No wet-lab validation has been performed, and
adenine deaminase activity is not demonstrated by any number in this
repository.

## Layout

- `tada_redesign/` — the pipeline package
- `outputs/` — run directories, `YYYYMMDD_` prefixed (gitignored)

## Environment

See `requirements.md`. This repo is a submodule of a monorepo and resolves
shared TadA assets (`masks.json`, reference PDBs, `tada_stability` modules)
through the `TADA_MONOREPO` environment variable.

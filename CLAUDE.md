# CLAUDE.md — tada-redesign

Submodule of `claude-proteindesign`. Read the parent repo's `CLAUDE.md` first;
its rules (the pre-job checklist, `sbatch` not `bsub`, dated output dirs,
absolute paths) all apply here — with ONE deliberate exception, below.

## Documentation lives HERE, not in the parent repo

**Deliberate deviation from the parent's "Plans and Logs" rule, decided by the
repo owner on 2026-08-06.** This campaign's spec, plans and logs live in this
repo under `docs/{specs,plans,logs}/`, not in the monorepo's
`docs/plans/` + `docs/logs/`. The campaign is a standalone repository, so its
paper trail belongs with its code: one push publishes both, and the repo is
readable on its own.

Do NOT "restore" these docs to the monorepo to satisfy the parent convention —
that was the previous arrangement and it left the campaign's documentation on a
branch named for an unrelated project while the code lived here. The monorepo
keeps a one-line stub pointing here.

The parent repo still holds two things this campaign depends on, and those stay
there: the vendored `tada-stability` assets (`masks.json`, the three reference
PDBs, the `tada_stability` modules) and the submodule pointer.

## Submodule-specific rules

- Commit **here first**, then update the parent repo's pointer.
- The remote must use `ssh://git@ssh.github.com:443/...` — github.com port 22
  and HTTPS are both blocked from this cluster.
- Cross-repo assets resolve through `TADA_MONOREPO`; never hardcode a path
  into the parent repo outside `constants.py`.
- `masks.json`'s `FROZEN` key is NOT this campaign's motif (it is 36 residues
  and includes `ZN_PROXIMITY`). Use `motif.arm_residues()`.
- Tests run in the `ligandmpnn_env` conda env.

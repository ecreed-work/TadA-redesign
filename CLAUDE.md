# CLAUDE.md — tada-redesign

Submodule of `claude-proteindesign`. Read the parent repo's `CLAUDE.md` first;
its rules (plans/logs, the pre-job checklist, `sbatch` not `bsub`, dated output
dirs, absolute paths) all apply here.

## Submodule-specific rules

- Commit **here first**, then update the parent repo's pointer.
- The remote must use `ssh://git@ssh.github.com:443/...` — github.com port 22
  and HTTPS are both blocked from this cluster.
- Cross-repo assets resolve through `TADA_MONOREPO`; never hardcode a path
  into the parent repo outside `constants.py`.
- `masks.json`'s `FROZEN` key is NOT this campaign's motif (it is 36 residues
  and includes `ZN_PROXIMITY`). Use `motif.arm_residues()`.
- Tests run in the `ligandmpnn_env` conda env.

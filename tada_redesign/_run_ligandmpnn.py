"""VENDORED from domain-insertion/denovo_tada/_run_ligandmpnn.py (2026-08-05).
Copied rather than imported: that package is Cas9/RuvC-coupled throughout, and
this campaign must not depend on its module layout. The numpy-alias monkeypatch
below is the whole point -- see the original for the two environment blockers it
works around.

Run `design/LigandMPNN/run.py` without editing the vendored submodule.

Two environment blockers, found by direct import probes (see
docs/plans/20260720_denovo-tada-ligandmpnn-chimera.md), block a bare
`python design/LigandMPNN/run.py ...` invocation:

1. The `ligand_mpnn` binary on `$PATH` (conda env `cas9-pam-design`) is a
   broken wrapper pointing at a deleted path -- unusable. `cas9-pam-design`
   also lacks `ml_collections`, which `run.py` needs transitively (it
   unconditionally imports `sc_utils`, which imports the vendored `openfold`
   package's `config.py`).
2. `ligandmpnn-sc` (the env that DOES have `ml_collections` + a working
   `prody`) has numpy>=2, and `openfold/np/residue_constants.py` uses the
   removed `np.int` alias at import time, crashing `sc_utils`'s import even
   when side-chain packing (`--pack_side_chains`) is never requested.

Fix: run this wrapper in conda env `ligandmpnn-sc`. It monkey-patches
`numpy.int/float/bool` back onto the numpy module BEFORE anything under
`design/LigandMPNN` is imported, then executes `run.py` via `runpy.run_path`
with `run_name="__main__"` so its own argparse + `if __name__ == "__main__"`
block behaves exactly as a direct invocation would. No submodule edits.

Usage: identical to `design/LigandMPNN/run.py`, e.g.
    python denovo_tada/_run_ligandmpnn.py --model_type ligand_mpnn \\
        --pdb_path X.pdb --out_folder OUT --fixed_residues "A1 A2 ..." \\
        --checkpoint_ligand_mpnn design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
"""
import os
import runpy
import sys

import numpy as np

for _name, _alias in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _alias)

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../domain-insertion/denovo_tada
_DOMAIN_INSERTION = os.path.dirname(_HERE)                   # .../domain-insertion
_REPO_ROOT = os.path.dirname(_DOMAIN_INSERTION)               # .../claude-proteindesign
_LMPNN_DIR = os.path.join(_REPO_ROOT, "design", "LigandMPNN")

if __name__ == "__main__":
    sys.path.insert(0, _LMPNN_DIR)
    sys.argv = [os.path.join(_LMPNN_DIR, "run.py")] + sys.argv[1:]
    runpy.run_path(os.path.join(_LMPNN_DIR, "run.py"), run_name="__main__")

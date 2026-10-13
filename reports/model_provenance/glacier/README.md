# casmi26-glacier -- GLACIER isomer re-scorer (offline, subprocess)

GLACIER (ms-pred `glacier.JointModel`, MassSpecGym checkpoint) predicts each candidate's MS/MS at every covered query
spectrum's adduct / instrument / 5 eV CE bucket; the score `gl_ex_mean` = mean entropy similarity (query vs
prediction) over the molecule's covered spectra ([M+H]+ / [M+Na]+, positive mode) -- the same definition as the
casmi26-iceberg `ice_ex_mean` (only the forward model differs; GLACIER post-processing merges identical m/z by MAX).
Used as a THIRD term of the post-ranker same-formula re-scoring: z(ranker) + 0.5 z(ice) + 0.5 z(gl).

## notebook use (after the casmi26-iceberg step; same runner input `items`)

```python
GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
sys.path.append(GL_PKG)
import gl_fuse
GL_SCORES = gl_fuse.run_gl(GL_PKG, items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=2400,
                           site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'))
order = gl_fuse.rerank_multi(smis, keys, scores, formulas, [ICE_SCORES.get(mid, {}), GL_SCORES.get(mid, {})],
                             [0.5, 0.5], top_n=60)
```

`run_gl` runs `python gl_runner.py <GL_PKG> in.json out.json --device cuda --budget S --site DIR [--wheels DIR]` in a
subprocess and never raises ({} on total failure). The runner reuses the wheel site the ICE runner installed (marker
`.ice_site_ok`), or installs the casmi26-iceberg `wheels/*.whl.ice` itself (`--wheels`, `$GL_WHEELS`, or found under
/kaggle/input); RDKit 2025.03 required (as for the A1 panel scores). `out.json` is written all-null at start and after
every chunk; the budget stops cleanly; any failure -> null scores, exit code 0; `out.json.meta.json` = status / counts
/ timings / errors.

`rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n)`: fuse.rerank generalised to several re-scorers
(same z definition, NaN -> 0, slot-preserving same-formula groups); with one dict it is exactly fuse.rerank.

Uploading: the Kaggle CLI skips sub-directories by default -- push with `--dir-mode zip` (or `tar`), then check that
`ckpt/`, `src/`, `shim/` exist under the mounted dataset.

Runner options: `--bs 32` (max batch; B * N_atoms^2 <= bs * 1600), `--chunk-jobs 3000`, `--memfrac 0`, `--threads 0`,
`--nondeterministic`, `--any-rdkit` (NOT exact), `--ckpt FILE`.

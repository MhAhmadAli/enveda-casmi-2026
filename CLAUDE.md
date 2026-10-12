# Enveda CASMI 2026 — Molecule ID From Mass Spectra

Predict the 2-D structure (SMILES) of small molecules from LC-MS/MS spectra.

## Competition facts (verified via Kaggle API, 2026-10-10)

- **Metric**: MRR@25 on the first 14 chars of the InChIKey after
  `Chem.RemoveStereochemistry` → RDKit tautomer-canonicalise. Up to 25 ranked SMILES per
  molecule, `;`-separated. Official scorer (`reports/public_kernel/metric/`) pins
  **RDKit 2026.03.3** (wheel dataset `metric/rdkit-2026-3-3-wheel`).
- **Kernels-only, synchronous reruns**: models trained locally, shipped as Kaggle
  Datasets; notebook runs offline and is re-executed on the hidden test.
- 540 min runtime cap (CPU and GPU). Public LB = 31% of test (~124 molecules).
  5 submissions/day, 2 count. Deadline 2026-12-14 (entry/merge 2026-12-07). $50k.
- Data licence CC BY-NC 4.0.

## ⚠ The visible `test.parquet` is a placeholder

All 1,213 visible test spectra are exact duplicates of `enveda-180` train rows
(`scripts/verify_leak.py`); staff replace it at rerun. Exact-match lookup scores ~1.0
there and is worthless. **No statistic from the visible file is a valid prior** (mass
band, adduct mix, instrument). Use it only as a schema/runtime check.

## What the hidden test is

"Small molecules detected in complex biological extracts" — natural products with,
in practice, **no close library relative**.

| library | spectra | structures | in COCONUT |
|---|---|---|---|
| `enveda-180` (timsTOF, synthetic med-chem) | 1.15M | 183k | 46 |
| `enveda-np-examples` (timsTOF, NP) | 1,184 | 250 | 236 |
| gnps/riken/mona/massbank/msdial/pluskal | ~1.33M | ~138k | ~56k |

`enveda-180` is the wrong chemical space (generic fragmentation data only). NP spectra
in train are mostly Orbitrap/QTOF, so cross-instrument transfer matters.

The test mixes **Class 1** (structure in the train library; direct spectral search
solves it) and **Class 2** (novel; needs candidate ranking).

## Current state (2026-10-11)

| # | configuration | local nphard | public LB |
|---|---|---|---|
| 1 | channel 2 alone (analog propagation) | 0.361 | 0.265 |
| 2 | + learned ranker (two channels) | 0.421 | 0.270 |
| 3 | + fingerprint channel (three channels) | 0.450 | 0.294 |
| 4 | private fork of public samanyu1808 "ICEBERG where GLACIER is blind" (`notebooks/kernels/fork438`, kernel `ahmadali1243534/casmi26-public-fork-438` v1) | — | **0.438** |

**The fork is the baseline.** Our pipeline is retired as a submission path and is now
a source of extra channels (fingerprint transformer, entropy direct search) to fuse
into the fork. LB top 0.484, median ~0.334. Only deltas ≳0.02-0.03 are LB-resolvable.

## Dev sets and calibration

- **`val_nphard`** (`scripts/build_hard_split.py`): every npx+npbig molecule whose
  nearest *remaining library* structure has Tanimoto ≤ 0.6. 356 structures / 770
  molecules, noise ±0.018. **Tune here.**
- `npx` (518 timsTOF NP pseudo-molecules) and `npbig` (3,496, mostly Orbitrap/QTOF)
  are the easy-case ceiling only: median nearest relative 0.777, and they overstate
  LB by ~0.2. MRR falls monotonically with analog distance; channel 2 in the ≤0.4 band
  (0.2646) matched LB #1 (0.265).
- Regimes: `--regime inlib` = Class 1, `--regime disjoint` = Class 2.
- **Local→LB transfer is ~20-30%** (+0.060 → +0.005; +0.089 → +0.029). To move LB
  +0.04, expect ~+0.15 locally. Local gains < 0.05 are not worth a submission.
  Reproducing a *level* is much weaker evidence than tracking a *delta*.
- Pool coverage of the hidden truth is unmeasurable locally (every dev truth is in the
  pool by construction) and likely part of the attenuation.

## Channels (full dev set, out-of-fold, structure-grouped)

| channel | mechanism | Class 1 | Class 2 | Class 2 hard | band ≤0.4 |
|---|---|---|---|---|---|
| 1 direct | entropy similarity, mass-matched library | 0.654 | 0.112 | 0.127 | - |
| 2 analog | binned cosine + `tanimoto * sim**4` propagation | 0.719 | 0.477 | 0.361 | 0.265 |
| 3 fingerprint | peak transformer → 2,022 Morgan bits | 0.464 | 0.464 | 0.378 | 0.301 |
| **all three** | prior-hedged GBDT, 20 features | **0.752** | **0.568** | **0.450** | **0.360** |

Channel 1 (`scripts/eval_direct.py`, `src/casmi/entropy.py`): search only spectra whose
neutral mass matches (~480 of 2.4M), so exact alignment + entropy similarity are cheap.
npx inlib 0.880 vs binned cosine 0.793, 110x faster. Best config
`--ppm 8.5 --tol 0.005 --min-rel 0.005`, entropy weighting on. The +1.5 ppm timsTOF
offset is real but correcting it doesn't help; 5 ppm loses recall.

Channel 3 never consults the library, so it helps where channel 2 has no relative.

## Measured to fail: don't repeat

- **Rescoring analog propagation with entropy similarity**: Class 1 +0.077, Class 2
  halved (0.475 → 0.226). Entropy is for the *same* compound, not analogs.
- **Hard confidence gating** between channels: Ch1 similarity separates the classes
  poorly (10% of Class 2 exceed the Class-1 median). Combine per candidate in a ranker.
- **Reverse (library-biased) scoring**: worse (0.828 vs 0.880).
- **Co-isolation contamination** is not the LB gap: 120 injected peaks cost 0.022.
- **More public spectra/structures**: LOTUS is 99.8% in COCONUT; GNPS+MassSpecGym
  harmonized moves nearest-relative Tanimoto by +0.005. Enveda already aggregated the
  public libraries.
- On the synthetic `enveda-180` folds analog propagation scored 0.734 vs 0.475 on NPs.
  Never measure in that space.

Still-valid small lessons: neutral-mass constraint +0.18; max-aggregate across a
molecule's spectra beats mean; soft binning beats hard; structure-disjoint retrieval
is exactly 0, so scores must come from ranking a candidate database.

## Public fork lineage (`reports/public_kernel/<owner>_<slug>/`, read only)

The 0.40-0.45 cluster is public forks: prvsiyan 0.335 → seyitkaangunes 0.399 →
huseyinemreaksoy 0.420 → flexonafft v80 0.436 → samanyu1808 0.438. Their LB-measured
ablations:

- **PubChem popularity prior** `z(score) + 0.15-0.25*(log1p(#substances)+log1p(#PubMed))`:
  +0.010-0.015 (dataset `dmitriigluzdov/casmi26-pubchem-popularity-prior`).
- **PubChem tier** (~105M structures outside the pool), fingerprint-scored, merged only
  when `lib_max < 0.7-0.9`: +0.011. So pool coverage *does* pay on the LB.
- **ICEBERG / GLACIER** forward simulators re-rank same-formula groups
  (`z(score)+z(ICEBERG)+z(GLACIER)`), off for library hits; GLACIER [M+H]+ only.
- Main engine v4n: FPNet + feature families + 4-booster LightGBM over ~711k
  structures, fused with a second engine by RRF `1/(3+r1) + 0.6/(3+r2)`.

## Candidate pool

COCONUT 2.0 (CC0) + all train structures = 729,600, Morgan fps in
`data/cache/pool_fp_packed.npy`. Pool recall at 8.5-10 ppm: npx 0.973, npbig 0.899
(npbig misses are mostly precursor imprecision, a proxy artifact).

**TODO**: `casmi.chem.inchikey14` now strips stereo, but `structures.parquet` key14
values predate that and must be rebuilt (`build_structures.py`), along with the Kaggle
dataset.

## Local layout

    scripts/build_cache.py            3 GB parquet -> memmapped peak store (data/cache)
    scripts/build_structures.py       SMILES -> tautomer-canonical key14 + mass
    scripts/build_np_splits.py        npx / npbig dev sets
    scripts/build_hard_split.py       val_nphard  <- tune on this
    scripts/build_fingerprints.py     Morgan r=2 2048-bit packed pool fps
    scripts/build_features.py, train_ranker.py   ranker features + GBDT
    scripts/train_fpmodel.py, eval_fpmodel.py    channel 3 fingerprint model
    scripts/eval_candidates.py        channel 2 candidate ranking, MRR@25
    scripts/eval_direct.py            channel 1 entropy direct search
    scripts/eval_combined.py          channel combination
    scripts/analyze_analog_distance.py nearest-library-relative Tanimoto (GPU)
    scripts/assess_external_spectra.py would a new library shrink analog distance?
    scripts/build_kaggle_dataset.py   payload for casmi26-candidate-pool
    scripts/verify_leak.py            proof the visible test is duplicated
    src/casmi/{chem,binning,search,retrieval,metric,spectra,entropy,channels,fpmodel}.py
    notebooks/submit_ranker.py, submit_analog.py  our own notebooks (retired)
    notebooks/kernels/<name>/         one dir per pushed kernel + kernel-metadata.json

Everything reads `data/cache`. Hardware: RTX 5090 32 GB, i9-14900K, 96 GB RAM.
CPU search at npbig scale took 5.4 h; use the GPU.

## Submission path

- Private dataset `ahmadali1243534/casmi26-candidate-pool`: `pool_index.parquet`,
  `pool_fp_packed.npy`, `structures.parquet`.
- Submit: `kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra
  -k <owner>/<kernel> -v <version> -f submission.csv -m "<msg>"` (code competitions
  take `-k/-v`, not a file upload).

## Conventions and traps

- Score only with `casmi.metric` on stereo-stripped tautomer-canonical key14. **Never**
  use the shipped `inchikey14` column (not tautomer-canonical).
- No feature meaning "candidate is in the spectral library": dev truths always are.
- Log every evaluated config to `reports/*.jsonl` with a `--tag`.
- Flaky network: `curl -C - --retry` for large downloads.
- Kaggle push gotchas:
  - jupytext `.ipynb` lacks `kernelspec` → papermill fails before running. Set
    `nb.metadata['kernelspec'] = {"name": "python3", ...}` before `nbformat.write`.
  - Pre-create the CLI temp upload dir: `mkdir -p "$TMP/.kaggle/uploads/data"`.
  - `kaggle kernels status <id>`; `kaggle kernels output <id> -p <dir>` (log is JSON).
  - Mount paths are nested (`/kaggle/input/datasets/<owner>/<slug>`). Never hard-code;
    walk `/kaggle/input` for a known filename (`require_dir()` in `submit_analog.py`).
- Kaggle forum bodies aren't reachable via WebFetch/API. Unread: "Adding valid
  guesses at ranks 2–25 lowers the public score?" (748029).

### Fifth submission: fusing our pipeline into the fork HURTS (2026-10-12)

| # | configuration | public LB |
|---|---|---|
| 5 | fork + our three-channel list as a third RRF term, `BETA/(3+rank)`, BETA=0.4 (`notebooks/kernels/fork438_ours` v2) | **0.412** (-0.026) |

The score moved, so our list did run (the cell falls back silently on failure, and the
run log could not be downloaded: `ListKernelSessionOutput` times out for this run).
-0.026 is about 1 sigma at n~124, but the sign agrees with expectation: our list is a
weaker version of channels the fork already has (direct library search, analog
propagation, a spectrum->fingerprint net), so it adds noise, not independent evidence.
**Do not fuse our pipeline into the fork.** Improvements have to come from inside the
fork's own components or from genuinely new evidence. The fork (#4, 0.438) stays the
best submission.

Kaggle CLI on macOS: there is no `timeout` command, so `timeout N kaggle ...` fails
instantly and silently inside retry loops.

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

## Provenance of the fork's models (2026-10-13)

Read from the datasets' own READMEs, manifests and training code
(`reports/model_provenance/`). Nothing had to be reverse-engineered.

| model in the fork | trained on | held out | honest on the NP panel? |
|---|---|---|---|
| FPNet A/B (`v3-models`/`v4b` `fpnet_*.pt`, `fe_A/B`) | train via `casmi/traindata.py` | `holdout_folds=('np','nplib','syn','plusk','twin')` | **yes**: panel MRR 0.581 vs 0.934 on seen |
| `fpnet_full1.pt` | name says full; no README | probably nothing | probably not |
| prvsiyan `fp_merged_m1` / `fp_single_s2` | unknown | unknown | assume not |
| megayak `ours_fpnet_*` | train minus **all 3,150 simulation structures** | NP panel + 6 library panels | **yes** |
| dmitriigluzdov `fpnet_fold0.pt` | train minus identity fold 0 (55,102) + NP panel (265) | stated in `split.parquet` | **yes** |
| `fe_D.pt` | DreaMS fine-tuned to fingerprints (MIT) | unknown | unknown |
| main ranker `ranker_v4b.pkl` | 160 features, 4 boosters x 500 rounds, on "sim1x" simulation rows | simulation structures | probably |
| ICEBERG 2.1 / GLACIER | MassSpecGym `msg_all` checkpoints | MassSpecGym's own split | unknown: the panel structures are in public libraries, so likely seen |

Key facts:

- **`enveda-np-examples` (the 250-structure NP panel) is held out by the public FPNet
  training code by default.** Together with the leak-free megayak and fold-safe
  checkpoints, an honest validation build of the fork is possible: swap the possibly
  leaked models for the held-out ones and score the panel. Identity-disjoint folds are
  published as `dmitriigluzdov/casmi26-identity-disjoint-validation-folds`.
- **Natural products are much harder for the fingerprint model**: held-out identities
  0.790 MRR vs the held-out NP panel 0.506.
- **The popularity prior is strongly NP-specific** (fold-safe README, 12k held-out
  queries): on the NP panel, fingerprint only 0.506, popularity alone **0.810**,
  `z + 0.25 pop` 0.818, `z + 0.6 pop` **0.834**. On ordinary library compounds the same
  prior hurts (0.815 -> 0.444 at 0.6). The fork uses 0.15-0.25. Caveat: the panel is
  famous NPs (median pop 13.1); a molecule "newly detected in an extract" may be less
  documented, so the right weight on the hidden test is unknown.

## Fork validation on the NP panel: first results (2026-10-13)

`scripts/build_val_kernel.py` -> kernels `casmi26-val-{disjoint,inlib}` (380 molecules =
250 structures x polarity, ~3.3 h each on a T4). `scripts/replay_val.py` replays the
popularity / simulator stages offline and reproduces the kernel's stage scores exactly.

| stage (MRR@25) | disjoint | inlib |
|---|---|---|
| main engine, raw ranker | 0.913 | 0.873 |
| + pool popularity (mu 0.15) | 0.950 | 0.935 |
| engine 2 alone | **0.010** | 0.836 |
| + gated PubChem merge | 0.892 | 0.884 |
| + RRF fusion with engine 2 + simulator re-rank | **0.543** | 0.929 |
| final | 0.543 | 0.929 |

**The main engine leaks in `disjoint`.** With every spectrum of the answer removed it
still ranks it first 87% of the time; the honest references on this panel are 0.42-0.58
(fingerprint alone) and 0.69-0.71 (megayak's leak-free ranker). Engine 2 drops to 0.010
in the same run, so the spectra really are gone. The fe_v4 families are documented and
grep-verified to read no pool src / train_sid / chan, and the cross-scorer is off. Remaining
suspects: `fe_D` (DreaMS fine-tune, undocumented training set), `fe_A`/`fe_B`, and the
160-feature ranker trained on simulated queries that likely include panel molecules.
Until the leak is found, absolute numbers and anything comparing the main engine with
an honest component (fusion, PubChem merge) are not interpretable.

What survives the leak because it is a re-ranking on the same lists (replay, paired,
bootstrap 95% CI, n=380):

- popularity is worth a lot: mu 0 vs the shipped 0.15 costs -0.037 / -0.062;
- **mu 0.25 beats 0.15**: +0.006 [-0.001, +0.015] disjoint, +0.012 [+0.004, +0.021] inlib;
  0.4 is flat, 0.6+ declines;
- simulators on the main list at the full weight (1.0/1.0) hurt (-0.016 / -0.014),
  consistent with the fork's own b420 finding; that is why it ships `BASE_ICE = False`.
- ICEBERG only covers [M+H]+/[M+Na]+: 216 of 380 molecules got scores.

### Leak hunt: it is not the fingerprint models (2026-10-14)

| `disjoint`, MRR@25 | base | fe_D -> fe_A | all FPNets -> fold-safe `fpnet_fold0` |
|---|---|---|---|
| main engine, raw | 0.913 | 0.906 | 0.889 |
| + popularity | 0.950 | 0.942 | 0.945 |
| final | 0.543 | 0.540 | 0.570 |

Swapping every fingerprint model the main engine uses for documented panel-held-out
checkpoints costs only 0.024. So either the leak is in the ranker (`ranker_v4b.pkl`,
trained on simulated queries that may include panel molecules) or a precomputed table
(`fe_v4/res/fragnet/sets.pkl`, derivation priors, analog_struct tables), or there is no
leak and the panel is simply easy: it is famous natural products whose close relatives
stay in the library (npx median nearest-relative Tanimoto 0.777). The LB (0.438 overall)
says the hidden test is far harder than this panel either way.

**`disjoint` misrepresents engine 2 and therefore fusion.** Engine 2 builds its pool from
COCONUT + ChEBI + structures *of the library file*; removing the answer's spectra removes
the answer from its pool (in-list recall 0.024), apparently because prvsiyan's COCONUT
file excludes train structures. A genuinely new hidden molecule in COCONUT would be in it.
The 0.95 -> 0.54 collapse at fusion in `disjoint` is that artifact, not evidence that
fusion hurts. `inlib` is unaffected (fusion there: 0.935 -> 0.929, neutral).

Next, on the 5090 box (data/cache is there): stratify the validation molecules by
nearest-relative Tanimoto (`scripts/analyze_analog_distance.py`) and read the main
engine's MRR in the <=0.5 band, where the hidden test appears to sit. That separates
"leaky" from "easy" without more GPU runs.

### Sixth submission: the panel validation's popularity call was wrong on the LB (2026-10-14)

| # | configuration | panel replay prediction | public LB |
|---|---|---|---|
| 6 | fork, `POOLPOP_MU` 0.15 -> 0.25 only (`notebooks/kernels/fork438_pop25`) | +0.006 / +0.012 | **0.422** (-0.016) |

About 1 sigma, but the opposite sign to the prediction, and the second fork change in a
row to lose. Consistent with the panel being famous NPs (median pop 13.1) while the
hidden molecules are less documented: the fold-safe study shows a heavier popularity
weight helps famous answers and hurts obscure ones. **Do not tune popularity-related
weights on the NP panel.** The fork's 0.15 stays. Before trusting the validation again,
rebuild it from molecules with no close library relative (nearest Tanimoto <= 0.5) and
check that its deltas track the LB.

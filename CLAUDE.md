# Enveda CASMI 2026 — Molecule ID From Mass Spectra

Predict the 2-D structure (SMILES) of small molecules from LC-MS/MS spectra.

## Competition facts (verified via Kaggle API, 2026-10-10)

- **Metric**: mean reciprocal rank at 25 on the **connectivity block of a
  tautomer-canonical InChIKey**. RDKit tautomer-canonicalise → InChIKey → first 14
  chars. Up to 25 ranked SMILES per molecule, `;`-separated. (The API labels it
  "CASMI: Mean Average Precision"; with one correct answer per molecule MAP@25 and
  MRR@25 coincide.)
- **Kernels-only** (`onlyAllowKernelSubmissions: true`) **with synchronous reruns**
  (`usesSynchronousReruns: true`). Models are trained locally and shipped to the
  notebook as Kaggle Datasets; the notebook runs offline and is **re-executed against
  the hidden test set** at scoring time.
- Runtime cap **540 min** CPU and GPU. Public LB is **31%** of the test set.
  2 submissions count for final scoring. 5 submissions/day. Deadline 2026-12-14
  (merge/entry 2026-12-07). 2,977 teams. $50,000.
- Data licence CC BY-NC 4.0.

## ⚠ The visible `test.parquet` is a placeholder — do not tune to it

Per the data page: *"The file you see is comprised of examples from the training
dataset; it will be replaced by the hidden test set during re-run, and will be
approximately the same size."* Verified independently: **all 1,213 visible test
spectra are exact duplicates of `enveda-180` rows in `train.parquet`** — identical
m/z, intensities, precursor, adduct and collision energy, 400/400 molecules resolving
to one structure each at ~2 ppm mass agreement (`scripts/verify_leak.py`).

Consequences, learned the hard way:

- An exact-match lookup scores ~1.0 on the visible file and **is worth nothing**. The
  leak is public (forum topic 745715) and staff posted "Rerun test dataset has been
  updated" on 2026-10-07.
- **Every statistic derived from the visible test file is invalid as a prior**: its
  246–439 Da mass band, its 81%/15% adduct mix, its all-timsTOF make-up. These
  describe a sample of `enveda-180`, not the hidden test.
- Anything built from it is only a schema and runtime check.

## What the hidden test actually is

The brief is "small molecules detected in complex biological extracts" — natural
products. That inverts the role of the training libraries:

| library | spectra | structures | natural products (in COCONUT) |
|---|---|---|---|
| `enveda-180` | 1,153,785 | 182,941 | **46** (0.0%) |
| `enveda-np-examples` | 1,184 | 250 | **236** (94%) |
| gnps / riken / mona / massbank / msdial / pluskal | ~1.33M | ~138k | ~56k structures |

- `enveda-180` is the timsTOF bulk (99.9% N-containing, 23% F, 28% S, 12% Cl — i.e.
  **synthetic medicinal chemistry**, not natural products). It is the wrong chemical
  space for the real task, useful mainly as generic fragmentation training data.
- **`enveda-np-examples` is the best proxy for the hidden test**: 1,184 spectra,
  250 natural-product structures, **timsTOF**, collision energy in eV — the test
  file's exact format. All 250 structures also occur in the public libraries, so
  their timsTOF spectra can be held out as queries while public-library spectra serve
  as reference. That is precisely the real task.
- Train has 24,285 natural-product structures over 748k spectra, but almost all on
  **Orbitrap/QTOF**, not timsTOF. So the core difficulty is **cross-instrument
  transfer**, and `enveda-np-examples` is how to measure it.

## Local layout

    scripts/build_cache.py       3 GB parquet -> 1.1 GB memmapped flat peak store
    scripts/build_structures.py  unique SMILES -> tautomer-canonical key14 + mass
    scripts/build_splits.py      folds mirroring the *placeholder* test (legacy;
                                 diagnostic only, wrong chemical space)
    scripts/build_np_splits.py   npx / npbig natural-product dev sets  <- use these
    scripts/build_fingerprints.py Morgan r=2 2048-bit packed fps for the pool
    scripts/eval_retrieval.py    library-retrieval baseline, true MRR@25
    scripts/eval_candidates.py   candidate-pool ranking, true MRR@25  <- the real task
    scripts/probe_test_regime.py offline test-vs-validation distribution comparison
    scripts/verify_leak.py       peak-level proof the visible test is duplicated
    src/casmi/{chem,binning,search,retrieval,metric,spectra}.py
    notebooks/submit_retrieval.py  early retrieval-only notebook (schema check only;
                                 scores 0.000 on disjoint structures - do not submit)
    notebooks/submit_analog.py   the real submission notebook (analog propagation)
    notebooks/kernels/<name>/    one dir per pushed kernel + its kernel-metadata.json
    data/kaggle_dataset/         payload for the casmi26-candidate-pool Kaggle dataset

Everything reads `data/cache`, so no script touches the 3 GB original.
Hardware: RTX 5090 (32 GB, 174 TFLOP/s fp16), i9-14900K, 96 GB RAM.

## LEADERBOARD CALIBRATION (first submission, 2026-10-10)

**Submitted analog propagation -> public LB 0.265. Rank 1943 / 3005.**
Field: median 0.333, top 0.484, 993 teams clustered 0.40-0.45, only 10 above 0.45.

Local dev sets overestimated by ~0.17-0.21 (npx 0.475, npbig 0.429 vs LB 0.265).
**Do not trust npx/npbig as absolute predictors** - use them to rank changes, then
re-calibrate against the LB.

### The hidden test is a MIXTURE of two classes

From the public notebook `haideptry/enveda-casmi-2026-fast-spectral-cosine-baseline`
(V17, pulled to `reports/public_kernel/`), which documents its own ablation:

| Version | Channels added | Class-1 MRR | Class-2 MRR | Public LB |
|---|---|---|---|---|
| V10-11 | library cosine + COCONUT fallback | 1.000 | 0.132 | 0.163 |
| V14 | + confidence gating | 1.000 | 0.132 | 0.193 |
| V15 | + fragment/loss dual channel + consensus | 1.000 | 0.170 | 0.205 |
| V16 | + analog propagation + MetFrag-lite + GBDT | 0.873 | 0.545 | **0.266** |
| V17 | + neural fingerprint + seed ensemble | 0.873 | 0.612 | 0.300-0.336 |

- **Class 1** = the structure IS in the training library; direct cross-instrument
  search solves it (their MRR ~1.0).
- **Class 2** = genuinely novel; needs candidate ranking.
- **Our 0.265 matches V16's 0.266 almost exactly** - independent confirmation that
  our pipeline sits at precisely the "analog propagation" capability level.
- Our entire local evaluation was **100% Class 2** (fully structure-disjoint), so it
  never measured the Class-1 half. That was the methodological blind spot.
- Their stated LB tops out at 0.336, yet 993 teams sit at 0.40-0.45. Unexplained -
  either a different public notebook or a remaining exploit. Worth investigating.

### Our measured Class-1 gap (`--regime inlib`, costs no submission)

| regime | what it measures | our MRR@25 | top-1 | theirs |
|---|---|---|---|---|
| `inlib` (Class 1) | timsTOF query vs Orbitrap/QTOF library, structure present | **0.793** | 0.745 | ~1.000 |
| `disjoint` (Class 2) | every spectrum of the answer removed | 0.475 | 0.336 | 0.545-0.612 |

We are behind on **both** channels, and Class 1 is the cheaper fix: 0.793 -> ~1.0 is
pure spectral-matching quality. Their Class-1 recipe differs from ours in ways that
are each testable locally:

- **entropy similarity** (Li et al. 2021) instead of cosine, with `INT_POWER=1.0`
  (linear intensity) rather than sqrt - they state linear+entropy beats sqrt
- **entropy weighting** of low-entropy spectra (S < 3 -> intensities raised to
  0.25 + 0.25*S)
- **8.5 ppm** candidate window, noting a **+1.4 ppm systematic offset on timsTOF**
- peak-list alignment at 0.01 Da tolerance, not binning
- multi-spectrum consensus across collision energies
- Channel 2 searches **mass shifts to +/-200 Da**, far wider than our neutral-loss
  domain alone
- Channels 3-4: MetFrag-lite bond-breaking peak explainability, and a spectrum ->
  6,930-bit fingerprint transformer scored as Bayes log-likelihood `f.z`
- a 31-feature HistGradientBoosting ranker over all channels, bagged over 4 seeds

## Results

### On `npx` — matched natural products, timsTOF, structure-disjoint

This is the dev set that should track the leaderboard. 518 pseudo-molecules,
candidate pool = COCONUT + train (729,600 structures), reference = all libraries
minus every spectrum of a held-out structure.

| scorer | MRR@25 | top-1 | top-5 | top-25 |
|---|---|---|---|---|
| random order (floor) | 0.104 | 0.035 | — | 0.496 |
| **analog propagation** | **0.475** | 0.336 | 0.653 | 0.849 |

Pool recall at 10 ppm is 0.975 (ceiling); mean 90 candidates per molecule.
Public notebooks were reported at ~0.43 LB, so the analog baseline alone is already
in that region — but `npx` structures all have public-library spectra, so genuinely
novel natural products will be harder.

### On `npbig` — 3,496 natural-product molecules, mostly Orbitrap/QTOF

| scorer | MRR@25 | top-1 | top-5 | top-25 | pool recall |
|---|---|---|---|---|---|
| analog propagation | 0.429 | 0.295 | 0.595 | 0.786 | 0.905 |

The two dev sets agree once the ceiling is accounted for: 0.475/0.975 = **0.487** on
`npx` against 0.429/0.905 = **0.474** on `npbig`. So the scorer behaves the same
across instruments, and the headline gap is almost entirely **candidate pool
coverage**, not spectral matching. Raising pool recall is therefore a direct,
measurable lever — 9.5% of `npbig` molecules currently cannot be found at any rank.

Iteration cost warning: `npbig` took **5.4 h** on CPU (2.39M reference spectra x
13,437 queries x 2 domains). Moving the search to the GPU is a prerequisite for
tuning at this scale; `casmi.search.SpectralIndex` exists for that but is unbenchmarked.

### On the placeholder-matched folds (enveda-180 synthetic space)

Diagnostic of the machinery only, **not** predictions of leaderboard score. Kept
because the contrast is instructive: analog propagation scored 0.734 here versus
0.475 on real natural products. A combinatorial library is dense with near-twins, so
measuring in the wrong chemical space overstated the method by ~0.26 MRR.

| setting | MRR@25 | top-1 | top-25 |
|---|---|---|---|
| library retrieval, in-library, no mass filter | 0.243 | 0.195 | 0.385 |
| library retrieval, in-library, max-agg | 0.292 | 0.250 | 0.417 |
| library retrieval, in-library, **mass-ranked** | 0.468 | 0.443 | 0.507 |
| library retrieval, structure-disjoint | 0.000 | 0.000 | 0.000 |
| candidate ranking, disjoint, random order | 0.045 | 0.010 | 0.300 |
| candidate ranking, disjoint, **analog propagation** | **0.734** | 0.580 | 0.960 |

Transferable lessons:

- The precursor-derived neutral mass constraint is worth **+0.18 MRR**.
- Combining a molecule's spectra by **max** beats mean (0.292 vs 0.243): the
  informative collision energy differs per compound.
- Structure-disjoint *retrieval* is exactly 0.000 — confirms no leakage in the split
  and proves retrieval alone cannot identify an unknown. Scores must come from
  **ranking a candidate database**.
- Soft (linear-interpolation) binning beats hard binning: 0.81 vs 0.64 self-cosine
  under a 0.004 Da calibration shift.
- Analog propagation (`tanimoto(candidate, analog) * spectral_sim**4`) is the
  strongest non-neural scorer: 4.6x the random floor on real natural products.

### Class-1 channel rebuilt: entropy similarity on a mass-restricted search

`scripts/eval_direct.py` + `src/casmi/entropy.py`. The architectural point, taken
from the public notebook: a *direct* search never needs to scan the library. Only
spectra whose **neutral mass already matches the query** can be the same compound --
about 480 of 2.4M. At that scale exact peak alignment and entropy similarity are
affordable, where binning was a compromise forced by scanning everything.

On `npx --regime inlib` (518 molecules, Class 1):

| method | MRR@25 | top-1 | top-25 | runtime |
|---|---|---|---|---|
| binned-cosine analog propagation | 0.793 | 0.745 | 0.907 | ~270 s |
| **entropy similarity, mass-restricted** | **0.880** | **0.828** | 0.985 | **2.4 s** |

**+0.087 MRR and ~110x faster.** On 518 molecules the noise floor is ~+/-0.015, so
the headline gain is real (~6 sigma) but the hyperparameter micro-tuning below is
NOT -- do not read those as settled.

Sweep findings (npx inlib, all within ~1 sigma of each other except where noted):

- **entropy weighting helps**: 0.868 vs 0.860 without. Confirms Li et al.
- tolerance 0.005 Da >= 0.01 >= 0.02; `min_rel` 0.005 >= 0.002 >= 0.001;
  `max_peaks` 512 == 256.
- ppm window 8.5-10 is the sweet spot. **5 ppm loses top-25** (0.973 vs 0.987): the
  window starts excluding the truth.
- **The +1.4 ppm timsTOF offset is real but not worth correcting.** Our independent
  measurement of query-vs-truth mass error is median **+1.54 ppm** (p5 -1.50,
  p95 +3.66), matching their note -- but applying the correction slightly *hurt*
  (0.8706 -> 0.8678), because a +/-10 ppm window already absorbs a 1.5 ppm shift.
- Current best config: `--ppm 8.5 --tol 0.005 --min-rel 0.005`, entropy weight on,
  no offset -> 0.8797.

Direct search scores **exactly 0.000** on `--regime disjoint`, confirming the two
channels are strictly complementary: Class 1 needs direct search (0.880), Class 2
needs analog propagation (0.475). A competitive submission must run both and combine
them, which is what V16->V17 did.

### Ruled out: co-isolated contamination does not explain the LB gap

`scripts/probe_contamination.py`. Hypothesis: the hidden test is molecules "detected
in complex biological extracts", so its spectra carry co-isolated peaks from
co-eluting species, while every local dev query is a clean Enveda standard. Since
entropy similarity is symmetric, extra query peaks count against a match.

Tested by injecting random contaminant peaks into `npx` queries (Class 1):

| contaminant peaks | entropy (forward) | reverse (library-biased) |
|---|---|---|
| 0 | 0.8797 | 0.8275 |
| 10 | 0.8724 | 0.8223 |
| 40 (up to 1.5x base intensity) | 0.8737 | 0.8244 |
| 120 (up to 1.5x base intensity) | 0.8576 | 0.8166 |

**Rejected.** Even 120 spurious peaks brighter than the analyte cost only 0.022 MRR.
Two conclusions worth keeping:

- the entropy matcher is intrinsically robust to additive noise, so spectral
  cleanliness is not where the missing 0.2 lives;
- **reverse (library-biased) scoring is worse, not better** (0.828 vs 0.880), despite
  being the textbook remedy for chimeric spectra. Do not adopt it.

Remaining candidate explanations for LB 0.265 vs local 0.475, in order of
plausibility: (1) candidate-pool coverage is worse for real extract molecules than
the 0.905-0.975 our dev sets enjoy; (2) `npx`/`npbig` molecules are well-studied
natural products with many close library relatives, so analog propagation has
material to work with that a genuine unknown does not. (2) is directly measurable by
stratifying Class-2 MRR on max Tanimoto to the nearest remaining library structure.

## Candidate pool

COCONUT 2.0 (Oct 2026, **CC0**, so legal to re-upload as a Kaggle dataset) plus all
train structures: **729,600 unique structures**, Morgan fps precomputed
(`data/cache/pool_fp_packed.npy`, 1,859 informative bits, mean 50 bits set).
Since the hidden test is natural products, COCONUT is the right backbone — its 0.2%
overlap with `enveda-180` is a feature, not a bug.

## Submission path (working as of 2026-10-10)

- Kaggle dataset `ahmadali1243534/casmi26-candidate-pool` (private, 225 MB):
  `pool_index.parquet`, `pool_fp_packed.npy`, `structures.parquet`.
- Kernel `ahmadali1243534/casmi26-analog-candidate-ranking` v5 runs **COMPLETE in
  12.8 min** (766 s) against the 540 min cap, and its `submission.csv` is
  byte-identical to the local run: 400 rows, 25 unique valid SMILES each, no
  fallbacks triggered.
- Submit with:
  `kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra
   -k <owner>/<kernel> -v <version> -f submission.csv -m "<msg>"`
  (code competitions take `-k/-v`, not a bare file upload.)

## Conventions and traps

- Score only with `casmi.metric` on tautomer-canonical key14. **Never** score on the
  shipped `inchikey14` column: it is not tautomer-canonical (274,195 canonical keys
  vs 275,810 shipped) and will flatter results and corrupt held-out splits.
- Validation truths necessarily have train spectra, so any feature meaning "this
  candidate is in the spectral library" scores well locally and need not transfer.
  Do not use one.
- Log every evaluated config to `reports/*.jsonl` with a `--tag`.
- Network here is flaky; large downloads need `curl -C - --retry`.
- **Kaggle notebook push gotchas** (both cost a failed run to learn):
  - jupytext writes `.ipynb` with no `kernelspec`, and papermill then fails with
    "No kernel name found in notebook" *before executing anything*. Always set
    `nb.metadata['kernelspec'] = {"name": "python3", ...}` before `nbformat.write`.
  - the Kaggle CLI needs its temp upload directory to pre-exist, including a
    subdirectory mirroring the `-p` path: `mkdir -p "$TMP/.kaggle/uploads/data"`.
  - check a pushed kernel with `kaggle kernels status <id>`, and fetch the log with
    `kaggle kernels output <id> -p <dir>` (the log is JSON, not text).
  - **mount paths are nested, not slug-only**: datasets land at
    `/kaggle/input/datasets/<owner>/<slug>` and competition data at a similarly
    nested path -- NOT `/kaggle/input/<slug>` as the docs imply. Never hard-code
    either; discover them by walking `/kaggle/input` for a known filename
    (`test.parquet`, `pool_index.parquet`). `submit_analog.py` has `require_dir()`
    for this, which also prints the tree on failure.
- Unread forum threads worth checking: "Metric Update (RemoveStereochemistry)"
  (746878) and "Adding valid guesses at ranks 2–25 lowers the public score?" (748029).
  Kaggle pages are JS-rendered; bodies are not reachable via WebFetch or the API.
  `GetForumTopicById` returns metadata only.

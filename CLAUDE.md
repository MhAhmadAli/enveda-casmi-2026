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
    notebooks/submit_retrieval.py  self-contained Kaggle notebook (schema check only)

Everything reads `data/cache`, so no script touches the 3 GB original.
Hardware: RTX 5090 (32 GB, 174 TFLOP/s fp16), i9-14900K, 96 GB RAM.

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

## Candidate pool

COCONUT 2.0 (Oct 2026, **CC0**, so legal to re-upload as a Kaggle dataset) plus all
train structures: **729,600 unique structures**, Morgan fps precomputed
(`data/cache/pool_fp_packed.npy`, 1,859 informative bits, mean 50 bits set).
Since the hidden test is natural products, COCONUT is the right backbone — its 0.2%
overlap with `enveda-180` is a feature, not a bug.

## Conventions and traps

- Score only with `casmi.metric` on tautomer-canonical key14. **Never** score on the
  shipped `inchikey14` column: it is not tautomer-canonical (274,195 canonical keys
  vs 275,810 shipped) and will flatter results and corrupt held-out splits.
- Validation truths necessarily have train spectra, so any feature meaning "this
  candidate is in the spectral library" scores well locally and need not transfer.
  Do not use one.
- Log every evaluated config to `reports/*.jsonl` with a `--tag`.
- Network here is flaky; large downloads need `curl -C - --retry`.
- Unread forum threads worth checking: "Metric Update (RemoveStereochemistry)"
  (746878) and "Adding valid guesses at ranks 2–25 lowers the public score?" (748029).
  Kaggle pages are JS-rendered; bodies are not reachable via WebFetch or the API.
  `GetForumTopicById` returns metadata only.

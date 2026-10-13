# CASMI26 Fold-Safe FPNet: Training Loop and Held-Out Checkpoint

The public [CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) pipelines rank candidate
structures with **FPNet**, a transformer that maps a peak list to fingerprint logits. Its public packages contain the model and
the data access, and no training loop; the public checkpoints were fitted on (almost) every training molecule, so they
cannot tell how the encoder behaves on a molecule it has never seen.

This dataset supplies the missing pieces:

* `train_fpnet.py`: a complete training loop for the public architecture (mass-window ranking loss, mixed precision,
  multi-process batch building), written against the public `casmi` package;
* `split.parquet`: which training structures were withheld (identity fold 0 of the
  [identity-disjoint folds](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-identity-disjoint-validation-folds) and the 250
  natural-product panel molecules);
* `fpnet_fold0.pt`: a checkpoint trained without those 55,102 + 265 structures (82,855 steps, 5.0 h on one RTX 6000);
* evaluation scripts and results, including the effect of a documentation prior on honest pool-window ranking.

## Quick start

```python
import sys, torch, numpy as np
sys.path.insert(0, '/kaggle/input/casmi26-v3-models/code')      # public casmi package (dataset ahmedberatozer/casmi26-v3-models)
from casmi import fpnet
model = fpnet.load_model('/kaggle/input/casmi26-fold-safe-fpnet/fpnet_fold0.pt', 'cuda')         # same class as FPNet A
item = dict(mz=pm, it=pi, prec=precursor_mz, adduct_ix=adduct_index, ce=30.0, ce_known=1.0, n_merged=1, mode=1.0)
z = model(*fpnet.collate([item], 'cuda'))[0].float()[0]                                            # 10,226 fingerprint logits
score = candidate_fingerprints @ z                                                                 # higher = better candidate
```

`pm, pi = fpnet.prep_peaks(mz, intensity, precursor_mz)`; candidate fingerprints are the rows of `pool_fp.npy` of the public
pool (unpacked with `fpnet.unpack_gpu`). The checkpoint is a drop-in for `fpnet_0.pt` wherever the public code builds a `ModelBank`.

## What it measures

Task: the answer is in the candidate pool (710,701 structures) and is ranked among the pool structures within 10 ppm of its
mass by the fingerprint score alone. A query merges up to eight spectra of one structure and polarity.

| Model and query set | Top-1 | Top-5 | Top-25 | MRR@25 |
|---|---:|---:|---:|---:|
| **This checkpoint, 3,000 held-out identity queries (never seen)** | 68% | 93% | 98% | 0.790 |
| This checkpoint, 499 natural-product panel queries (never seen) | 33% | 71% | 95% | 0.506 |
| This checkpoint, 3,000 training queries (seen) | 97% | 100% | 100% | 0.985 |
| Public FPNet A, natural-product panel (held out by its training code) | 42% | 80% | 95% | 0.581 |
| Public FPNet A, training queries (seen) | 89% | 99% | 100% | 0.934 |

Reading: a model of this architecture places the answer first for about nine in ten molecules it was trained on and for a
much smaller share of new molecules. The held-out rows are the honest numbers for a hidden answer that has a structure in
the pool and no reference spectrum. On the natural-product panel this five-hour checkpoint is below the public FPNet A
(33% against 42% top-1): it is a reproducible baseline to improve on, and the scripts make every improvement measurable.

### A documentation prior on honest windows

`pop = log(1 + PubChem substance records) + log(1 + PubMed articles)` from the
[popularity dataset](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior); `z` standardises the
fingerprint score inside the window.

| Score | Held-out identities: top-1 | MRR@25 | NP panel: top-1 | MRR@25 |
|---|---:|---:|---:|---:|
| fingerprint only | 67% | 0.789 | 33% | 0.506 |
| popularity only | 5% | 0.105 | 69% | 0.810 |
| z + 0.15 · pop | 65% | 0.772 | 68% | 0.792 |
| z + 0.25 · pop | 56% | 0.717 | 71% | 0.818 |
| z + 0.6 · pop | 25% | 0.456 | 73% | 0.834 |
| z + 0.25 · pop + 1.0 · [in LOTUS or NPAtlas] | 53% | 0.689 | 72% | 0.829 |

The prior table uses 12,000 held-out queries. The panel answers are famous natural products (median `pop`
13.1); the held-out identities are an ordinary slice of the training libraries (median `pop`
1.6), and there the same prior does harm. Splitting the held-out queries by the kind of answer shows why
(top-1 / MRR@25):

| Held-out answers | Queries | Median `pop` | Fingerprint only | z + 0.15 · pop | z + 0.25 · pop | z + 0.6 · pop |
|---|---:|---:|---:|---:|---:|---:|
| not in any natural-product catalogue | 10,824 | 1.6 | 71% / 0.815 | 67% / 0.788 | 58% / 0.729 | 23% / 0.444 |
| in COCONUT, LOTUS or NPAtlas | 1,176 | 5.4 | 38% / 0.547 | 46% / 0.622 | 45% / 0.614 | 40% / 0.568 |
| in LOTUS or NPAtlas | 562 | 6.7 | 29% / 0.467 | 41% / 0.582 | 42% / 0.585 | 39% / 0.561 |

Two things follow. Natural products are much harder for the fingerprint model than the other library compounds, even with
the candidate present. And a documentation prior is a natural-product tool: it helps those answers at weights of about
0.15 to 0.25 and hurts the rest, so its weight should be chosen on natural-product answers that the encoder has not seen.

### Training curve

| Step | Hours | Held-out top-1 | MRR@25 | NP panel top-1 | MRR@25 | Training top-1 |
|---|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.6 | 63% | 0.745 | 34% | 0.498 | 76% |
| 20,000 | 1.2 | 65% | 0.766 | 32% | 0.498 | 83% |
| 30,000 | 1.8 | 65% | 0.769 | 34% | 0.517 | 87% |
| 40,000 | 2.4 | 66% | 0.778 | 32% | 0.495 | 92% |
| 50,000 | 3.0 | 66% | 0.781 | 32% | 0.498 | 94% |
| 60,000 | 3.6 | 67% | 0.788 | 32% | 0.502 | 94% |
| 70,000 | 4.2 | 67% | 0.788 | 33% | 0.507 | 96% |
| 80,000 | 4.8 | 68% | 0.790 | 33% | 0.510 | 97% |
| 82,855 | 5.0 | 68% | 0.790 | 33% | 0.506 | 97% |

## Files

| File | Content |
|---|---|
| `fpnet_fold0.pt` | Checkpoint (FPNet, 8 layers, width 512, 10,226 output bits, 49.7 M parameters), loadable with `casmi.fpnet.load_model`. |
| `split.parquet` | `sid` (row of `train_structs.parquet` in the public pool package), `inchikey14`, `key` (scorer key) and `fold`: `hold` or `np`; every other training structure was used for fitting. |
| `train_fpnet.py` | Training loop. |
| `make_split.py` | Builds the work directory and `split.parquet` from the identity folds. |
| `eval_ckpt.py` | Pool-window evaluation of one or several checkpoints. |
| `eval_prior.py` | The same windows with popularity and natural-product catalogue priors. |
| `fold0_history.json` | Evaluation every 5,000 steps. |
| `results_*.json` | Final numbers behind the tables above. |
| `train_fold0.log` | Training log. |

## Method

Loss: cross-entropy over the true structure and 63 pool structures drawn from its 10-ppm mass window, scored by `f·z`
(the ranking objective named in the public model file), plus a light binary cross-entropy on the fingerprint bits and an
element-count head. Half of each batch is sampled uniformly over spectra and half uniformly over structures; 30% of the
items merge two to four spectra of the same structure and polarity; peaks are dropped and jittered. AdamW, peak learning
rate 3e-4 with 2,000 warm-up steps and cosine decay, batch 256, mixed precision, gradient clipping at 1.

Two things that did not work are worth knowing: initialising the output bias to the log-odds of each bit makes the raw
`f·z` scores of candidates differ by hundreds at the start and the ranking loss explodes; a heavy bit-wise loss
(weight 20) slows the ranking objective several-fold.

## Read the numbers with care

* One run, one seed, one fold, a fixed time budget. The public checkpoints were trained with an unpublished recipe and
  probably for longer; this checkpoint is a baseline for honest comparisons, and its absolute accuracy can be improved.
* Held-out identities keep their structures in the candidate pool: the test is "structure known, spectrum never seen".
  Scaffold relatives of held-out molecules remain in training.
* Pool-window ranking is one component. The complete pipelines add library search, analogue generation, a learned
  ranker, a second candidate database and forward-spectrum models.

## Terms

The checkpoint is derived from the competition training file, which includes components under non-commercial licences;
use it under the [competition data terms](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data),
non-commercially and with attribution. The model architecture and data access come from
[Ahmed Berat Özer's public v4 pipeline](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4g-inference); the training loop
and evaluation scripts in this dataset are released under Apache 2.0.

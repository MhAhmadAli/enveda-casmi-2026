"""Learn to combine the two evidence channels per candidate, hedged over class prior.

Why learning is needed at all: no fixed rule works. Entropy-rescoring the analog
channel halves Class 2; hard confidence gating is worse than either channel alone,
because Channel-1 similarity separates the classes only weakly (median 0.821 for
Class 1 vs 0.468 for Class 2, with 10.4% overlap above the Class-1 median).

The hard part is not the model, it is the prior. Channel-1 evidence means opposite
things in the two regimes:

  inlib    the truth has library spectra at its own mass, so high c1 => likely truth
  disjoint the truth has none, so any c1 > 0 belongs to a *wrong* same-mass compound

So a ranker trained only on `inlib` learns "trust c1" and collapses on novel
molecules; trained only on `disjoint` it learns "distrust c1" and throws away free
Class-1 wins. The correct training distribution is a mixture in the hidden test's
Class-1 proportion, which we do not know. We therefore train at several assumed
proportions and average the predictions -- hedging rather than guessing, which is
what the public notebook's two class priors do.

Evaluation is grouped by *structure*, never by molecule: the same structure appears
as several pseudo-molecules (one per adduct), so splitting by molecule would leak.
Scores are reported per regime and on the calibrated `nphard` subset, since only the
last of those tracks the leaderboard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.metric import mrr  # noqa: E402

CACHE = "data/cache"

FEATURES = [
    "c1", "c1_best", "c1_margin", "c1_rank", "c1_nonzero_frac",
    "c2", "c2_best", "c2_margin", "c2_rank", "c2_mean_top5",
    "tanimoto_best_analog", "analog_sim_of_best",
    "ppm_err", "abs_ppm_err", "n_candidates", "n_query_spectra", "n_analogs",
]


def load(val_names: list[str]) -> pd.DataFrame:
    parts = []
    for n in val_names:
        p = f"{CACHE}/features_{n}.parquet"
        if not os.path.exists(p):
            print(f"  (skipping {n}: {p} not built yet)")
            continue
        df = pd.read_parquet(p)
        df["source"] = n
        df["molecule_id"] = n + "_" + df.molecule_id.astype(str)
        parts.append(df)
    if not parts:
        raise SystemExit("no feature files found; run scripts/build_features.py")
    return pd.concat(parts, ignore_index=True)


def truth_key_of_molecule(df: pd.DataFrame) -> pd.Series:
    pos = df[df.label == 1]
    return pos.groupby("molecule_id").key14.first()


def rank_lists(df: pd.DataFrame, score: np.ndarray):
    """Per molecule: candidate keys ordered by score, plus the truth key."""
    tmp = df[["molecule_id", "key14", "label"]].copy()
    tmp["score"] = score
    tmp = tmp.sort_values(["molecule_id", "score"], ascending=[True, False])
    ranked, truths = [], []
    for mol, g in tmp.groupby("molecule_id", sort=False):
        ranked.append(g.key14.tolist())
        t = g.loc[g.label == 1, "key14"]
        truths.append(t.iat[0] if len(t) else None)
    return ranked, truths


def evaluate(df: pd.DataFrame, score: np.ndarray, label: str) -> dict:
    out = {"split": label}
    for regime in ("inlib", "disjoint"):
        m = (df.regime == regime).to_numpy()
        if not m.any():
            continue
        r, t = rank_lists(df[m], score[m])
        keep = [(a, b) for a, b in zip(r, t) if b is not None]
        out[f"mrr_{regime}"] = round(mrr([a for a, _ in keep],
                                         [b for _, b in keep]), 4)
        out[f"n_{regime}"] = len(keep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="npx,npbig")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--priors", default="0.3,0.6",
                    help="assumed Class-1 fractions to train at and average over")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--hard-max-tanimoto", type=float, default=0.6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    df = load(args.features.split(","))
    print(f"{len(df):,} candidate rows, {df.molecule_id.nunique()} molecules, "
          f"{int(df.label.sum())} positives")

    # structure-grouped folds: a structure's every adduct/molecule stays together
    truth_of_mol = truth_key_of_molecule(df)
    df = df[df.molecule_id.isin(truth_of_mol.index)].reset_index(drop=True)
    df["truth_key"] = df.molecule_id.map(truth_of_mol)
    structs = np.sort(df.truth_key.unique())
    rng = np.random.default_rng(0)
    fold_of_struct = pd.Series(
        rng.integers(0, args.folds, len(structs)), index=structs)
    df["fold"] = df.truth_key.map(fold_of_struct)

    # nphard membership: molecules whose nearest library relative is distant
    hard_keys: set[str] = set()
    for n in args.features.split(","):
        p = f"reports/analog_distance_{n}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            hard_keys |= set(d.loc[d.max_tanimoto_to_library
                                   <= args.hard_max_tanimoto, "key14"])
    df["is_hard"] = df.truth_key.isin(hard_keys)
    print(f"{df.loc[df.is_hard, 'molecule_id'].nunique()} molecules in the hard "
          f"(<= {args.hard_max_tanimoto} Tanimoto) subset")

    priors = [float(x) for x in args.priors.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    oof = np.zeros(len(df))
    for fold in range(args.folds):
        tr = df.fold != fold
        te = ~tr
        if not te.any():
            continue
        Xte = df.loc[te, FEATURES].to_numpy(np.float32)
        preds = np.zeros(int(te.sum()))
        n_models = 0
        for prior in priors:
            # weight the two regimes so the training mixture matches `prior`
            w = np.where(df.loc[tr, "regime"].to_numpy() == "inlib",
                         prior, 1.0 - prior).astype(np.float64)
            Xtr = df.loc[tr, FEATURES].to_numpy(np.float32)
            ytr = df.loc[tr, "label"].to_numpy()
            for seed in seeds:
                clf = HistGradientBoostingClassifier(
                    max_depth=6, max_iter=300, learning_rate=0.05,
                    min_samples_leaf=50, l2_regularization=1.0,
                    random_state=seed)
                clf.fit(Xtr, ytr, sample_weight=w)
                preds += clf.predict_proba(Xte)[:, 1]
                n_models += 1
        oof[te.to_numpy()] = preds / n_models
        print(f"  fold {fold}: trained {n_models} models, "
              f"{int(tr.sum()):,} train rows -> {int(te.sum()):,} eval rows")

    results = [evaluate(df, oof, "ranker (all molecules)")]
    hard = df.is_hard.to_numpy()
    if hard.any():
        results.append(evaluate(df[hard].reset_index(drop=True), oof[hard],
                                f"ranker (hard <= {args.hard_max_tanimoto})"))
    # baselines on the same rows, for a like-for-like comparison
    for name, col in [("channel 2 alone", "c2"), ("channel 1 alone", "c1")]:
        results.append(evaluate(df, df[col].to_numpy(), name))
        if hard.any():
            results.append(evaluate(df[hard].reset_index(drop=True),
                                    df.loc[hard, col].to_numpy(),
                                    f"{name} (hard)"))

    res = pd.DataFrame(results)
    print()
    print(res.to_string(index=False))
    os.makedirs("reports", exist_ok=True)
    res.to_csv("reports/ranker_results.csv", index=False)
    json.dump({"features": FEATURES, "priors": priors, "seeds": seeds,
               "folds": args.folds, "tag": args.tag,
               "results": results}, open("reports/ranker_results.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()

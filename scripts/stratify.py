"""Stratify dev-set MRR by how close the nearest library relative is.

Joins per-molecule ranks (eval_combined.py --dump) with the nearest-relative
Tanimoto (analyze_analog_distance.py). If analog propagation depends strongly on
having a close relative, the per-bucket MRR shows it -- and the bucket whose MRR
matches the leaderboard tells us where the hidden test actually sits.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="npx")
    ap.add_argument("--regime", default="disjoint")
    args = ap.parse_args()

    per = pd.read_csv(f"reports/per_molecule_{args.val}_{args.regime}.csv")
    dist = pd.read_csv(f"reports/analog_distance_{args.val}.csv")
    df = per.merge(dist, left_on="truth_key14", right_on="key14", how="left")
    miss = df.max_tanimoto_to_library.isna().sum()
    if miss:
        print(f"note: {miss} molecules without a distance value, dropped")
        df = df.dropna(subset=["max_tanimoto_to_library"])

    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    df["bucket"] = pd.cut(df.max_tanimoto_to_library, bins)
    g = df.groupby("bucket", observed=True).agg(
        molecules=("rr", "size"), mrr=("rr", "mean"),
        top1=("rank", lambda s: float((s == 1).mean())),
        found=("rank", lambda s: float((s > 0).mean())))
    print(f"{args.val} / {args.regime}: MRR@25 by nearest-library-relative Tanimoto")
    print(g.round(4).to_string())
    print(f"\noverall MRR {df.rr.mean():.4f} over {len(df)} molecules")
    lo = df[df.max_tanimoto_to_library <= 0.6]
    print(f"low-similarity subset (<=0.6): MRR {lo.rr.mean():.4f} "
          f"over {len(lo)} molecules  <- the honest proxy for a novel molecule")


if __name__ == "__main__":
    main()

"""Build `nphard`: the calibrated dev set — natural products with no close library relative.

Why this exists. Our first submission scored 0.265 while the same pipeline scored
0.475 on `npx`. Stratifying by "Tanimoto to the nearest remaining library structure"
explained the whole gap: `npx` molecules are well-studied natural products with a
median nearest relative at 0.777, and analog propagation feeds on exactly that. The
hidden test behaves like the <=0.6 bucket, which scored 0.288 locally against the
0.265 we actually got.

So this pools every natural-product dev molecule whose nearest library relative is
distant, from both `npx` and `npbig`, giving a subset big enough to tune against.
Evaluating here should track the leaderboard; evaluating on `npx` overall will not.

Requires scripts/analyze_analog_distance.py to have been run for each source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tanimoto", type=float, default=0.6,
                    help="keep molecules whose nearest library relative is below this")
    ap.add_argument("--sources", default="npx,npbig")
    ap.add_argument("--out", default="nphard")
    args = ap.parse_args()

    parts, stats = [], {}
    for name in args.sources.split(","):
        val = pd.read_parquet(f"{CACHE}/val_{name}.parquet")
        dist_path = f"reports/analog_distance_{name}.csv"
        if not os.path.exists(dist_path):
            raise SystemExit(f"missing {dist_path}; run analyze_analog_distance.py "
                             f"--val {name} first")
        dist = pd.read_csv(dist_path)
        keep = set(dist.loc[dist.max_tanimoto_to_library <= args.max_tanimoto,
                            "key14"])
        sub = val[val.key14.isin(keep)].copy()
        sub["source"] = name
        # namespace molecule ids so the two sources cannot collide
        sub["molecule_id"] = name + "_" + sub.molecule_id.astype(str)
        parts.append(sub)
        stats[name] = {"structures_kept": int(sub.key14.nunique()),
                       "molecules": int(sub.molecule_id.nunique()),
                       "spectra": int(len(sub))}
        print(f"{name}: {sub.key14.nunique()} structures, "
              f"{sub.molecule_id.nunique()} molecules, {len(sub)} spectra")

    out = pd.concat(parts, ignore_index=True)
    # a structure must not appear twice via two sources
    dup = out.key14.duplicated()
    first_src = out.groupby("key14").source.first()
    out = out[out.source == out.key14.map(first_src)].reset_index(drop=True)

    out.to_parquet(f"{CACHE}/val_{args.out}.parquet", index=False)
    print(f"\nval_{args.out}: {out.key14.nunique()} structures, "
          f"{out.molecule_id.nunique()} molecules, {len(out)} spectra")
    n = out.molecule_id.nunique()
    print(f"noise floor on MRR ~ +/-{0.5/np.sqrt(n):.3f} "
          f"(vs +/-{0.5/np.sqrt(68):.3f} for the 68-molecule npx subset)")
    json.dump({"max_tanimoto": args.max_tanimoto, "sources": args.sources,
               "per_source": stats, "molecules": int(n),
               "structures": int(out.key14.nunique())},
              open(f"{CACHE}/val_{args.out}_config.json", "w"), indent=1)


if __name__ == "__main__":
    main()

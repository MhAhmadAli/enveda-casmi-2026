"""Dump per-candidate ranker features for a dev set, in both regimes.

This is the expensive pass (the library search). Run once per dev set; the ranker
then trains and re-trains on the cached features in seconds.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.channels import ChannelConfig, compute_features  # noqa: E402

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="npx")
    ap.add_argument("--regimes", default="inlib,disjoint")
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--sim-power", type=float, default=4.0)
    args = ap.parse_args()

    t0 = time.time()
    cfg = ChannelConfig(ppm=args.ppm, sim_power=args.sim_power)
    val = pd.read_parquet(f"{CACHE}/val_{args.val}.parquet")

    def log(*a):
        print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

    parts = []
    for regime in args.regimes.split(","):
        parts.append(compute_features(val, regime, cfg, CACHE, log))
    out = pd.concat(parts, ignore_index=True)
    path = f"{CACHE}/features_{args.val}.parquet"
    out.to_parquet(path, index=False)
    log(f"wrote {path}: {len(out):,} rows, "
        f"{out.label.sum():,} positives, {out.molecule_id.nunique()} molecules")


if __name__ == "__main__":
    main()

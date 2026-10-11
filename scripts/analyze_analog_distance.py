"""How close is each dev molecule's nearest library relative? (GPU Tanimoto)

Analog propagation can only work if some library structure actually resembles the
unknown. Our dev molecules are well-studied natural products that appear across
GNPS/MassBank/RIKEN, so they are likely surrounded by close relatives -- congeners,
glycosylation variants, oxidation products. A molecule newly detected in an extract
need not be.

If that is what separates local 0.475 from leaderboard 0.265, it shows up as a
strong dependence of Class-2 MRR on "max Tanimoto to the nearest remaining library
structure". This computes that stratifier for every dev molecule, excluding the
molecule's own structure exactly as the disjoint regime does.

730k x 2048-bit Tanimoto is a matmul, so it runs on the GPU in seconds.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.retrieval import CandidatePool, reference_table  # noqa: E402

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="npx")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()
    import torch

    t0 = time.time()
    val = pd.read_parquet(f"{CACHE}/val_{args.val}.parquet")
    truth = val.groupby("molecule_id").key14.first()

    pool = CandidatePool.load(CACHE, with_fp=True)
    row_of_key: dict[str, int] = {}
    for r, k in enumerate(pool.key14):
        if k not in row_of_key:
            row_of_key[k] = r

    # library structures = those with spectra; a Class-2 query may use any of them
    ref = reference_table(CACHE, instrument="all")
    lib_keys = [k for k in ref.key14.unique() if k in row_of_key]
    lib_rows = np.array([row_of_key[k] for k in lib_keys], dtype=np.int64)
    print(f"{len(lib_rows):,} library structures with fingerprints "
          f"({time.time()-t0:.0f}s)", flush=True)

    q_keys = [k for k in truth.unique() if k in row_of_key]
    q_rows = np.array([row_of_key[k] for k in q_keys], dtype=np.int64)

    dev = args.device if torch.cuda.is_available() else "cpu"
    Q = torch.from_numpy(np.unpackbits(pool.packed_fp[q_rows], axis=1)).to(
        dev, torch.float16)
    qsum = Q.sum(1, keepdim=True)
    best = torch.zeros(len(q_rows), device=dev, dtype=torch.float16)
    best_idx = torch.zeros(len(q_rows), device=dev, dtype=torch.long)

    lib_key_arr = np.asarray(lib_keys)
    for s in range(0, len(lib_rows), args.chunk):
        e = min(s + args.chunk, len(lib_rows))
        B = torch.from_numpy(np.unpackbits(pool.packed_fp[lib_rows[s:e]], axis=1)).to(
            dev, torch.float16)
        inter = Q @ B.T
        union = qsum + B.sum(1)[None, :] - inter
        tan = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        # a molecule is never its own analog: zero the diagonal match by key
        same = torch.from_numpy(
            (np.asarray(q_keys)[:, None] == lib_key_arr[None, s:e])).to(dev)
        tan = tan.masked_fill(same, 0.0)
        v, i = tan.max(dim=1)
        upd = v > best
        best = torch.where(upd, v, best)
        best_idx = torch.where(upd, i + s, best_idx)
        del B, inter, union, tan, same
    print(f"tanimoto done ({time.time()-t0:.0f}s)", flush=True)

    bn = best.float().cpu().numpy()
    bi = best_idx.cpu().numpy()
    out = pd.DataFrame({
        "key14": q_keys,
        "max_tanimoto_to_library": np.round(bn, 4),
        "nearest_library_key14": lib_key_arr[bi],
    })
    out.to_csv(f"reports/analog_distance_{args.val}.csv", index=False)

    print(f"\n{args.val}: nearest library relative, Tanimoto distribution over "
          f"{len(out)} structures")
    print(out.max_tanimoto_to_library.describe().round(3).to_string())
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    cnt = pd.cut(out.max_tanimoto_to_library, bins).value_counts().sort_index()
    print("\nhistogram:")
    for k, v in cnt.items():
        print(f"  {k}: {v:5d}  ({v/len(out):6.1%})")
    print(f"\nwrote reports/analog_distance_{args.val}.csv")


if __name__ == "__main__":
    main()

"""Would an external spectral library shrink analog distance? (the measurable question)

Extra *structure* databases cannot be validated locally: every dev truth is a train
structure, so "absent from the pool" is 0.000 and any coverage gain is invisible
here. LOTUS bore that out, adding 245 structures (0.2%) over COCONUT.

Extra *spectra* are different. Class-2 MRR depends strongly on how close the nearest
library structure is (0.27 at Tanimoto <=0.5 rising to 0.65 above 0.9), and adding
spectra for structures that are closer to our dev molecules shrinks that distance
measurably. So this asks two things of a candidate library:

  1. how many of its structures do we NOT already have spectra for;
  2. for the hard dev molecules, does including those structures actually reduce the
     nearest-relative Tanimoto -- the quantity that predicts MRR.

Only (2) justifies the integration work.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CACHE = "data/cache"


def canon(chunk):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    te = rdMolStandardize.TautomerEnumerator()
    out = []
    for smi in chunk:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append((smi, None, np.nan))
            continue
        try:
            can = te.Canonicalize(mol)
        except Exception:
            can = mol
        try:
            key = Chem.MolToInchiKey(can)[:14]
        except Exception:
            key = None
        out.append((smi, key, Descriptors.ExactMolWt(mol)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", default="data/external/gnps_msg_smiles.parquet")
    ap.add_argument("--out", default="data/external/gnps_msg_structures.parquet")
    ap.add_argument("--val", default="nphard")
    args = ap.parse_args()
    t0 = time.time()

    if os.path.exists(args.out):
        df = pd.read_parquet(args.out)
        print(f"reusing {args.out}: {len(df):,} structures")
    else:
        s = pd.read_parquet(args.smiles).smiles.tolist()
        chunks = [s[i:i + 200] for i in range(0, len(s), 200)]
        rows = []
        with Pool(processes=max(os.cpu_count() - 4, 2)) as p:
            for part in p.imap_unordered(canon, chunks, chunksize=1):
                rows.extend(part)
        df = pd.DataFrame(rows, columns=["smiles", "key14", "mass"]).dropna(
            subset=["key14"])
        df.to_parquet(args.out, index=False)
        print(f"canonicalised {len(df):,} structures ({time.time()-t0:.0f}s)")

    tr = pd.read_parquet(f"{CACHE}/structures.parquet", columns=["key14"])
    have_spectra = set(tr.key14.dropna())
    ext = set(df.key14)
    new = ext - have_spectra
    print(f"\nexternal library: {len(ext):,} unique key14")
    print(f"  we already have spectra for: {len(ext & have_spectra):,} "
          f"({len(ext & have_spectra)/len(ext):.1%})")
    print(f"  NEW structures with spectra: {len(new):,} ({len(new)/len(ext):.1%})")
    if not new:
        print("\nnothing new -- integration would not change analog distance")
        return

    # (2) does including the new structures shrink nearest-relative distance?
    import torch
    from casmi.retrieval import CandidatePool

    pool = CandidatePool.load(CACHE, with_fp=True)
    row_of_key = {}
    for r, k in enumerate(pool.key14):
        if k not in row_of_key:
            row_of_key[k] = r

    val = pd.read_parquet(f"{CACHE}/val_{args.val}.parquet")
    q_keys = [k for k in val.key14.unique() if k in row_of_key]
    q_rows = np.array([row_of_key[k] for k in q_keys], dtype=np.int64)

    # new structures are only usable as analogs if we can fingerprint them, which
    # means they must be in the pool; those outside it would need adding too
    new_in_pool = [k for k in new if k in row_of_key]
    print(f"  of the new ones, {len(new_in_pool):,} are already in the candidate "
          f"pool (so fingerprints exist)")
    if not new_in_pool:
        print("\nnew structures are outside the pool; would need pool + fp rebuild")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Q = torch.from_numpy(np.unpackbits(pool.packed_fp[q_rows], axis=1)).to(
        dev, torch.float16)
    qsum = Q.sum(1, keepdim=True)

    def best_tanimoto(keys):
        rows = np.array([row_of_key[k] for k in keys], dtype=np.int64)
        best = torch.zeros(len(q_rows), device=dev, dtype=torch.float16)
        karr = np.asarray(keys)
        for s0 in range(0, len(rows), 20000):
            e = min(s0 + 20000, len(rows))
            B = torch.from_numpy(
                np.unpackbits(pool.packed_fp[rows[s0:e]], axis=1)).to(dev, torch.float16)
            inter = Q @ B.T
            union = qsum + B.sum(1)[None, :] - inter
            tan = torch.where(union > 0, inter / union, torch.zeros_like(inter))
            same = torch.from_numpy(
                np.asarray(q_keys)[:, None] == karr[None, s0:e]).to(dev)
            tan = tan.masked_fill(same, 0.0)
            best = torch.maximum(best, tan.max(dim=1).values)
            del B, inter, union, tan, same
        return best.float().cpu().numpy()

    cur_keys = [k for k in have_spectra if k in row_of_key]
    before = best_tanimoto(cur_keys)
    after = np.maximum(before, best_tanimoto(new_in_pool))
    print(f"\nnearest-library-relative Tanimoto for {len(q_keys)} {args.val} "
          f"structures:")
    print(f"  before: median {np.median(before):.4f}  mean {before.mean():.4f}")
    print(f"  after : median {np.median(after):.4f}  mean {after.mean():.4f}")
    print(f"  improved for {(after > before + 1e-6).sum()} structures "
          f"({(after > before + 1e-6).mean():.1%}), "
          f"mean gain {np.mean(after - before):+.4f}")
    print(f"  structures crossing 0.6 (out of the hard band): "
          f"{int(((before <= 0.6) & (after > 0.6)).sum())}")


if __name__ == "__main__":
    main()

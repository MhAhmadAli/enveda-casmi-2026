"""Morgan fingerprints for every candidate-pool structure, as packed bits.

The downstream model predicts a fingerprint from a spectrum and ranks candidates by
agreement, so every structure that could ever be a candidate needs its fingerprint
precomputed once. Packed uint8 keeps ~730k x 2048 bits at ~190 MB, small enough to
hold in RAM and to ship to a Kaggle notebook as a dataset.

Also records per-bit frequency over the pool: bits that are almost always on or
almost always off carry no ranking information and are excluded from the model's
prediction targets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

N_BITS = 2048
RADIUS = 2


def fp_chunk(smiles_list):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=N_BITS)
    out = np.zeros((len(smiles_list), N_BITS // 8), dtype=np.uint8)
    ok = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        bv = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        out[i] = np.packbits(bv)
        ok[i] = True
    return out, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/external/candidate_pool.parquet")
    ap.add_argument("--out-dir", default="data/cache")
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    pool = pd.read_parquet(args.pool, columns=["key14", "smiles", "mass", "src"])
    pool = pool.reset_index(drop=True)
    print(f"{len(pool):,} structures", flush=True)

    smiles = pool.smiles.tolist()
    chunks = [smiles[i:i + args.chunk] for i in range(0, len(smiles), args.chunk)]
    packed = np.zeros((len(smiles), N_BITS // 8), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)

    t0 = time.time()
    with Pool(processes=max(os.cpu_count() - 4, 2)) as p:
        pos = 0
        for ci, (out, ok) in enumerate(p.imap(fp_chunk, chunks, chunksize=1)):
            packed[pos:pos + len(out)] = out
            valid[pos:pos + len(ok)] = ok
            pos += len(out)
            if (ci + 1) % 200 == 0:
                rate = pos / (time.time() - t0)
                print(f"  {pos:,}/{len(smiles):,}  {rate:.0f}/s  "
                      f"eta {(len(smiles)-pos)/rate/60:.1f} min", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(f"{args.out_dir}/pool_fp_packed.npy", packed)
    pool["fp_ok"] = valid
    pool.to_parquet(f"{args.out_dir}/pool_index.parquet", index=False)

    bits = np.unpackbits(packed[valid], axis=1)
    freq = bits.mean(axis=0)
    np.save(f"{args.out_dir}/pool_bit_freq.npy", freq)
    informative = int(((freq > 0.005) & (freq < 0.995)).sum())
    json.dump({"n_structures": int(len(pool)), "n_valid": int(valid.sum()),
               "n_bits": N_BITS, "radius": RADIUS,
               "informative_bits_0.5pct": informative,
               "mean_bits_set": float(bits.sum(axis=1).mean())},
              open(f"{args.out_dir}/pool_fp.json", "w"), indent=1)
    print(f"valid {valid.sum():,}/{len(valid):,}; mean bits set "
          f"{bits.sum(axis=1).mean():.1f}; informative bits {informative}")


if __name__ == "__main__":
    main()

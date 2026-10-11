"""Score candidates with the predicted fingerprint, and emit it as a ranker feature.

Two questions, in order:

  1. standalone -- how well does "which candidate's fingerprint explains the
     predicted bits" rank candidates on its own? This is the channel's own merit,
     and unlike analog propagation it does not need a close library relative, so it
     is the only channel that can improve the band where the hidden test sits.
  2. as a feature -- does adding it to the existing two channels raise the ranker?
     A channel can be individually weak and still carry information the others lack.

Reported per analog-distance band, because the whole point is the low-similarity
band: a gain concentrated where a close relative already exists would be worthless.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.chem import neutral_masses  # noqa: E402
from casmi.fpmodel import (FPModelConfig, build_model,  # noqa: E402
                           candidate_scores, predict_fingerprints)
from casmi.metric import mrr, topk_accuracy  # noqa: E402
from casmi.retrieval import CandidatePool  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="nphard")
    ap.add_argument("--model", default="data/cache/fpmodel.pt")
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--out-features", default="",
                    help="write per-candidate fp features to this parquet")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=dev, weights_only=False)
    cfg = FPModelConfig(**ckpt["cfg"])
    model = build_model(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    keep_bits = ckpt["keep_bits"]
    prior = ckpt["prior"]
    add_id = ckpt["adduct_id"]
    print(f"model: epoch {ckpt.get('epoch')}, holdout loss "
          f"{ckpt.get('holdout_loss'):.4f}, {cfg.n_bits} bits", flush=True)

    val = pd.read_parquet(f"{CACHE}/val_{args.val}.parquet")
    truth = val.groupby("molecule_id").key14.first()
    mols = list(truth.index)

    store = PeakStore.load("train", CACHE)
    meta = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                           columns=["precursor_mz", "adduct", "ce_hi"])
    rows = val.row.to_numpy()
    prec = meta.precursor_mz.to_numpy(np.float32)[rows]
    adducts = meta.adduct.astype(str).to_numpy()[rows]
    aid = np.array([add_id.get(a, 0) for a in adducts], np.int64)
    ce = np.nan_to_num(meta.ce_hi.to_numpy(np.float32)[rows])

    pred = predict_fingerprints(model, store, rows, prec, aid, ce, cfg, dev)
    print(f"predicted fingerprints for {len(rows)} spectra "
          f"({time.time()-t0:.0f}s)", flush=True)

    # a molecule's spectra are averaged in probability space: different collision
    # energies reveal different substructures, and averaging keeps all the evidence
    pred_by_mol = {}
    for m, g in pd.DataFrame({"m": val.molecule_id.to_numpy(),
                              "i": np.arange(len(rows))}).groupby("m"):
        pred_by_mol[m] = pred[g.i.to_numpy()].mean(axis=0)

    pool = CandidatePool.load(CACHE, with_fp=True)
    q_nm = neutral_masses(meta.precursor_mz.to_numpy()[rows], adducts)
    first_of_mol = {}
    for i, m in enumerate(val.molecule_id.to_numpy()):
        first_of_mol.setdefault(m, i)

    ranked, rows_out = [], []
    for m in mols:
        nm = q_nm[first_of_mol[m]]
        cr = pool.candidates(nm, args.ppm)
        if len(cr) == 0:
            ranked.append([])
            continue
        bits = pool.unpack(cr)[:, keep_bits]
        sc = candidate_scores(pred_by_mol[m], bits, prior)
        best = {}
        for k, v in zip(pool.key14[cr], sc):
            if v > best.get(k, -np.inf):
                best[k] = float(v)
        ranked.append([k for k, _ in sorted(best.items(), key=lambda kv: -kv[1])])
        if args.out_features:
            order = np.argsort(-sc)
            rank = np.empty(len(sc), np.int64)
            rank[order] = np.arange(1, len(sc) + 1)
            rows_out.append(pd.DataFrame({
                "molecule_id": m, "key14": pool.key14[cr],
                "fp_score": sc.astype(np.float32),
                "fp_margin": (sc - sc.max()).astype(np.float32),
                "fp_rank": rank}))

    tr = [truth[m] for m in mols]
    acc = topk_accuracy(ranked, tr)
    res = {"val": args.val, "scorer": "fp_model", "ppm": args.ppm,
           "n_molecules": len(mols), "mrr@25": round(mrr(ranked, tr), 4),
           "top1": round(acc[1], 4), "top5": round(acc[5], 4),
           "top10": round(acc[10], 4), "top25": round(acc[25], 4),
           "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1))

    # per-band, since the low-similarity band is the one that matters
    dpath = {}
    for n in ("npx", "npbig"):
        p = f"reports/analog_distance_{n}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            dpath.update(dict(zip(d.key14, d.max_tanimoto_to_library)))
    near = np.array([dpath.get(truth[m], np.nan) for m in mols])
    rr = np.array([1.0 / (r.index(t) + 1) if t in r[:25] else 0.0
                   for r, t in zip(ranked, tr)])
    print("\nfp-model MRR by nearest-library-relative Tanimoto:")
    for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
        msk = (near > lo) & (near <= hi)
        if msk.sum():
            print(f"  {lo:.2f}-{hi:.2f}  n={int(msk.sum()):4d}  "
                  f"MRR {rr[msk].mean():.4f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/fpmodel_runs.jsonl", "a") as f:
        f.write(json.dumps({**res, "tag": args.tag}) + "\n")
    if args.out_features and rows_out:
        out = pd.concat(rows_out, ignore_index=True)
        out.to_parquet(args.out_features, index=False)
        print(f"\nwrote {args.out_features}: {len(out):,} rows")


if __name__ == "__main__":
    main()

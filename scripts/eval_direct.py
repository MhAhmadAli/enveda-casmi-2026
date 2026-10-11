"""Class-1 direct search: identify a molecule whose structure IS in the library.

The hidden test is a mixture of Class 1 (structure present in the training library)
and Class 2 (genuinely novel). Our earlier evaluation was entirely Class 2, so this
script measures the half that was never measured.

Architecture differs deliberately from eval_candidates.py. Rather than scanning all
2.4M library spectra with a binned cosine, we first restrict to library spectra whose
*neutral mass* already matches the query within a ppm window -- a few hundred spectra
instead of millions. At that scale exact tolerance-based peak alignment and entropy
similarity are affordable, and both are more accurate than binned cosine.

Reported baseline to beat: binned-cosine analog propagation reaches MRR@25 0.793 /
top-1 0.745 on `npx --regime inlib`.
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
from casmi.entropy import prepare, score_against  # noqa: E402
from casmi.metric import mrr, topk_accuracy  # noqa: E402
from casmi.retrieval import CandidatePool, reference_table  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="npx")
    ap.add_argument("--regime", default="inlib", choices=["inlib", "disjoint"])
    ap.add_argument("--ppm", type=float, default=10.0,
                    help="neutral-mass window for library spectra AND pool candidates")
    ap.add_argument("--ppm-offset", type=float, default=0.0,
                    help="systematic ppm correction applied to query neutral mass")
    ap.add_argument("--tol", type=float, default=0.01, help="Da peak-match tolerance")
    ap.add_argument("--min-rel", type=float, default=0.002)
    ap.add_argument("--max-peaks", type=int, default=256)
    ap.add_argument("--no-entropy-weight", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    ew = not args.no_entropy_weight

    val = pd.read_parquet(f"{CACHE}/val_{args.val}.parquet")
    truth = val.groupby("molecule_id").key14.first()
    mols = list(truth.index)

    ref = reference_table(CACHE, instrument="all")
    if args.regime == "disjoint":
        ref = ref[~ref.key14.isin(set(truth.values))]
    ref = ref[~ref.row.isin(set(val.row.tolist()))].reset_index(drop=True)
    print(f"library ({args.regime}): {len(ref):,} spectra / {ref.key14.nunique():,} "
          f"structures", flush=True)

    ref_nm = neutral_masses(ref.precursor_mz.to_numpy(),
                            ref.adduct.reset_index(drop=True))
    ok = np.isfinite(ref_nm)
    ref, ref_nm = ref[ok].reset_index(drop=True), ref_nm[ok]
    order = np.argsort(ref_nm, kind="stable")
    ref_nm_sorted = ref_nm[order]
    ref_rows = ref.row.to_numpy()
    ref_key = ref.key14.to_numpy()
    print(f"library with usable adducts: {len(ref):,} ({time.time()-t0:.0f}s)",
          flush=True)

    store = PeakStore.load("train")
    all_mz = np.asarray(store.mz)
    all_in = np.asarray(store.inten)
    q_off, q_npk = store.off, store.n

    meta = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                           columns=["precursor_mz", "adduct"])
    q_rows = val.row.to_numpy()
    q_nm = neutral_masses(meta.precursor_mz.to_numpy()[q_rows],
                          meta.adduct.iloc[q_rows].reset_index(drop=True))

    # how far is the precursor-derived mass from the true structure's exact mass?
    pool = CandidatePool.load(CACHE, with_fp=False)
    key_mass = pool.df.groupby("key14").mass.first()
    tm = val.key14.map(key_mass).to_numpy()
    sig = (q_nm - tm) / tm * 1e6
    sig = sig[np.isfinite(sig)]
    print(f"query mass error vs truth: median {np.median(sig):+.2f} ppm, "
          f"mean {sig.mean():+.2f}, p5 {np.percentile(sig,5):+.2f}, "
          f"p95 {np.percentile(sig,95):+.2f} ppm", flush=True)

    q_nm_adj = q_nm * (1.0 + args.ppm_offset * 1e-6)
    q_mol = val.molecule_id.to_numpy()

    by_mol: dict[str, dict[str, float]] = {}
    n_cand_lib = []
    for qi in range(len(q_rows)):
        nm = q_nm_adj[qi]
        if not np.isfinite(nm):
            continue
        r = int(q_rows[qi])
        qmz, qp = prepare(all_mz[q_off[r]:q_off[r] + q_npk[r]],
                          all_in[q_off[r]:q_off[r] + q_npk[r]],
                          args.min_rel, args.max_peaks, ew)
        if qmz.shape[0] == 0:
            continue
        tol = nm * args.ppm * 1e-6
        lo = np.searchsorted(ref_nm_sorted, nm - tol)
        hi = np.searchsorted(ref_nm_sorted, nm + tol)
        cand = order[lo:hi]
        n_cand_lib.append(len(cand))
        if len(cand) == 0:
            continue
        sims = score_against(qmz, qp, ref_rows[cand].astype(np.int64),
                             q_off, q_npk, all_mz, all_in, args.tol,
                             args.min_rel, args.max_peaks, ew,
                             np.zeros(len(cand), np.float32))
        tgt = by_mol.setdefault(q_mol[qi], {})
        for k, v in zip(ref_key[cand], sims):
            v = float(v)
            if v > tgt.get(k, -1.0):
                tgt[k] = v
        if (qi + 1) % 300 == 0:
            print(f"  scored {qi+1}/{len(q_rows)} spectra "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # pad with mass-window pool candidates so every molecule can fill 25 slots
    ranked = []
    for m in mols:
        d = by_mol.get(m, {})
        keys = [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])]
        ranked.append(keys)

    tr = [truth[m] for m in mols]
    score = mrr(ranked, tr)
    acc = topk_accuracy(ranked, tr)
    res = {"val": args.val, "regime": args.regime, "scorer": "direct_entropy",
           "ppm": args.ppm, "ppm_offset": args.ppm_offset, "tol": args.tol,
           "min_rel": args.min_rel, "max_peaks": args.max_peaks,
           "entropy_weight": ew, "n_molecules": len(mols),
           "mean_library_candidates": round(float(np.mean(n_cand_lib or [0])), 1),
           "mrr@25": round(score, 4), "top1": round(acc[1], 4),
           "top5": round(acc[5], 4), "top10": round(acc[10], 4),
           "top25": round(acc[25], 4), "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1))
    os.makedirs("reports", exist_ok=True)
    with open("reports/direct_runs.jsonl", "a") as f:
        f.write(json.dumps({**res, "tag": args.tag}) + "\n")


if __name__ == "__main__":
    main()

"""Does co-isolated contamination explain the local-to-leaderboard gap?

Our dev queries are Enveda's clean standards. The hidden test is molecules "detected
in complex biological extracts", where the precursor isolation window admits
co-eluting species, so real query spectra carry peaks that belong to other molecules.
Every local number we have was measured without that.

Entropy similarity (and cosine) are *symmetric*: a peak present in the query but
absent from the library counts against the match. For a chimeric query that is
exactly wrong -- the contaminant peaks are not evidence against the identification.
The standard remedy in library searching is a *reverse* (library-biased) score,
which only considers m/z positions present in the library spectrum.

This measures both, as a function of injected contamination, so we learn:
  - how fast matching degrades with contamination (is it a plausible gap cause?)
  - whether reverse scoring is materially more robust (is it worth adopting?)
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
from casmi.entropy import entropy_similarity, prepare  # noqa: E402
from casmi.metric import mrr, topk_accuracy  # noqa: E402
from casmi.retrieval import reference_table  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def reverse_similarity(qmz, qp, cmz, cp, tol):
    """Library-biased cosine: only query peaks matching a library peak contribute.

    Unmatched *query* peaks are discarded rather than penalised, which is the point:
    a contaminant peak is not evidence against the identification. Unmatched
    *library* peaks still count, so a candidate that should have produced fragments
    the query lacks is still punished.
    """
    if qmz.size == 0 or cmz.size == 0:
        return 0.0
    j = np.searchsorted(cmz, qmz)
    matched_q = np.zeros(qmz.size, dtype=bool)
    paired_c = np.zeros(cmz.size, dtype=bool)
    qv = np.zeros(cmz.size)
    for i in range(qmz.size):
        best, bestd = -1, tol
        for cand in (j[i] - 1, j[i]):
            if 0 <= cand < cmz.size:
                d = abs(cmz[cand] - qmz[i])
                if d <= bestd and not paired_c[cand]:
                    best, bestd = cand, d
        if best >= 0:
            paired_c[best] = True
            matched_q[i] = True
            qv[best] += qp[i]
    if not matched_q.any():
        return 0.0
    a = qv
    b = cp.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def contaminate(mz, inten, rng, n_extra: int, rel: float, mz_lo: float, mz_hi: float):
    """Add n_extra spurious peaks at random m/z, each up to `rel` of the base peak."""
    if n_extra <= 0 or mz.size == 0:
        return mz, inten
    extra_mz = rng.uniform(mz_lo, mz_hi, n_extra).astype(np.float32)
    extra_in = (rng.uniform(0.05, 1.0, n_extra) * rel * float(inten.max())).astype(np.float32)
    out_mz = np.concatenate([mz, extra_mz])
    out_in = np.concatenate([inten, extra_in])
    o = np.argsort(out_mz, kind="stable")
    return out_mz[o], out_in[o]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="npx")
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--min-rel", type=float, default=0.005)
    ap.add_argument("--max-peaks", type=int, default=256)
    ap.add_argument("--levels", default="0,5,15,40",
                    help="numbers of injected contaminant peaks to test")
    ap.add_argument("--rel", type=float, default=0.3,
                    help="contaminant intensity ceiling, relative to the base peak")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    val = pd.read_parquet(f"{CACHE}/val_{args.val}.parquet")
    truth = val.groupby("molecule_id").key14.first()
    mols = list(truth.index)

    # Class 1: the structure is in the library, so a clean query should be ~0.88
    ref = reference_table(CACHE, instrument="all")
    ref = ref[~ref.row.isin(set(val.row.tolist()))].reset_index(drop=True)
    ref_nm = neutral_masses(ref.precursor_mz.to_numpy(),
                            ref.adduct.reset_index(drop=True))
    ok = np.isfinite(ref_nm)
    ref, ref_nm = ref[ok].reset_index(drop=True), ref_nm[ok]
    order = np.argsort(ref_nm, kind="stable")
    nm_sorted = ref_nm[order]
    ref_rows, ref_key = ref.row.to_numpy(), ref.key14.to_numpy()

    store = PeakStore.load("train")
    all_mz, all_in = np.asarray(store.mz), np.asarray(store.inten)
    meta = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                           columns=["precursor_mz", "adduct"])
    q_rows = val.row.to_numpy()
    q_pmz = meta.precursor_mz.to_numpy()[q_rows]
    q_nm = neutral_masses(q_pmz, meta.adduct.iloc[q_rows].reset_index(drop=True))
    q_mol = val.molecule_id.to_numpy()
    print(f"library {len(ref):,} spectra; {len(q_rows)} query spectra", flush=True)

    rows = []
    for n_extra in [int(x) for x in args.levels.split(",")]:
        fwd: dict[str, dict[str, float]] = {}
        rev: dict[str, dict[str, float]] = {}
        for qi in range(len(q_rows)):
            if not np.isfinite(q_nm[qi]):
                continue
            r = int(q_rows[qi])
            raw_mz = all_mz[store.off[r]:store.off[r] + store.n[r]]
            raw_in = all_in[store.off[r]:store.off[r] + store.n[r]]
            cmz, cin = contaminate(raw_mz, raw_in, rng, n_extra, args.rel,
                                   50.0, float(q_pmz[qi]) + 5.0)
            qmz, qp = prepare(cmz, cin, args.min_rel, args.max_peaks, True)
            if qmz.shape[0] == 0:
                continue
            tol = q_nm[qi] * args.ppm * 1e-6
            lo = np.searchsorted(nm_sorted, q_nm[qi] - tol)
            hi = np.searchsorted(nm_sorted, q_nm[qi] + tol)
            cand = order[lo:hi]
            if cand.size == 0:
                continue
            tf = fwd.setdefault(q_mol[qi], {})
            trv = rev.setdefault(q_mol[qi], {})
            for c in cand:
                rr = int(ref_rows[c])
                lmz, lp = prepare(all_mz[store.off[rr]:store.off[rr] + store.n[rr]],
                                  all_in[store.off[rr]:store.off[rr] + store.n[rr]],
                                  args.min_rel, args.max_peaks, True)
                if lmz.shape[0] == 0:
                    continue
                k = ref_key[c]
                f = entropy_similarity(qmz, qp, lmz, lp, args.tol)
                if f > tf.get(k, -1.0):
                    tf[k] = f
                v = reverse_similarity(np.asarray(qmz, np.float64),
                                       np.asarray(qp, np.float64),
                                       np.asarray(lmz, np.float64),
                                       np.asarray(lp, np.float64), args.tol)
                if v > trv.get(k, -1.0):
                    trv[k] = v

        for name, d in (("entropy(forward)", fwd), ("reverse(library-biased)", rev)):
            ranked = [[k for k, _ in sorted(d.get(m, {}).items(), key=lambda kv: -kv[1])]
                      for m in mols]
            tr = [truth[m] for m in mols]
            acc = topk_accuracy(ranked, tr)
            rows.append({"contaminant_peaks": n_extra, "similarity": name,
                         "mrr@25": round(mrr(ranked, tr), 4),
                         "top1": round(acc[1], 4), "top25": round(acc[25], 4)})
            print(f"  {n_extra:3d} extra peaks  {name:24s} "
                  f"MRR {rows[-1]['mrr@25']:.4f}  top1 {rows[-1]['top1']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print()
    print(df.pivot(index="contaminant_peaks", columns="similarity",
                   values="mrr@25").to_string())
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/contamination_probe.csv", index=False)
    json.dump(rows, open("reports/contamination_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()

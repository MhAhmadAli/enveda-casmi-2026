"""Decide offline whether the test structures are present in the spectral library.

No submission required. For a query whose structure IS in the library, the best
cosine against the library is high (you are matching the same compound at another
collision energy). For a query whose structure is absent, the best match is only an
analog and the score is markedly lower.

So we compare three distributions of "best library cosine per molecule":

  val / in-library  - sibling spectra of the true structure present  (upper reference)
  val / disjoint    - every spectrum of the true structure removed   (lower reference)
  test              - the real thing

Whichever reference the test distribution sits on tells us which regime we are in,
and therefore whether the pipeline should be library lookup or candidate ranking.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from casmi.binning import bin_matrix  # noqa: E402
from casmi.retrieval import reference_table  # noqa: E402
from casmi.search import search_cpu  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def best_per_molecule(scores, idx, mol_ids, ref_key=None, exclude_keys=None):
    """Best cosine per molecule, optionally ignoring hits to banned structures."""
    out: dict[str, float] = {}
    for q in range(scores.shape[0]):
        rows, vals = idx[q], scores[q]
        ok = rows >= 0
        rows, vals = rows[ok], vals[ok]
        if exclude_keys is not None and rows.size:
            keep = ~np.isin(ref_key[rows], list(exclude_keys))
            rows, vals = rows[keep], vals[keep]
        v = float(vals.max()) if vals.size else 0.0
        m = mol_ids[q]
        if v > out.get(m, -1.0):
            out[m] = v
    return out


def describe(name, values):
    v = np.array(sorted(values), dtype=np.float64)
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return {"regime": name, "n": int(v.size), "mean": round(float(v.mean()), 4),
            "p10": round(q[0], 4), "p25": round(q[1], 4), "median": round(q[2], 4),
            "p75": round(q[3], 4), "p90": round(q[4], 4),
            "frac_above_0.7": round(float((v > 0.7).mean()), 4),
            "frac_above_0.9": round(float((v > 0.9).mean()), 4)}


def main() -> None:
    t0 = time.time()
    ref = reference_table(CACHE, instrument="tims")
    store = PeakStore.load("train")
    print(f"library: {len(ref):,} timsTOF spectra / {ref.key14.nunique():,} structures",
          flush=True)

    R = bin_matrix(store, ref.row.to_numpy(), ref.precursor_mz.to_numpy())
    ref_key = ref.key14.to_numpy()
    print(f"library binned, nnz {R.nnz:,} ({time.time()-t0:.0f}s)", flush=True)

    rows = []

    # --- validation queries, both regimes, from one search ---
    val = pd.read_parquet(f"{CACHE}/val_fold0.parquet")
    pmz_all = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                              columns=["precursor_mz"]).precursor_mz.to_numpy()
    vq = val.row.to_numpy()
    Qv = bin_matrix(store, vq, pmz_all[vq])
    # the query spectra themselves are in the library; ask for enough hits that
    # non-self, non-sibling matches survive the later filtering
    sv, iv = search_cpu(R, Qv, topk=3000, progress=True)
    print(f"val searched ({time.time()-t0:.0f}s)", flush=True)

    self_rows = set(val.row.tolist())
    ref_row_arr = ref.row.to_numpy()
    is_self = np.isin(ref_row_arr, list(self_rows))
    # drop self-matches: a query must never match its own spectrum
    for q in range(sv.shape[0]):
        bad = is_self[iv[q]] & (iv[q] >= 0)
        sv[q][bad] = -1.0

    mol_ids = val.molecule_id.to_numpy()
    inlib = best_per_molecule(sv, iv, mol_ids)
    truth = val.groupby("molecule_id").key14.first()
    disjoint = best_per_molecule(sv, iv, mol_ids, ref_key=ref_key,
                                 exclude_keys=set(truth.values))
    rows.append(describe("val_inlib", list(inlib.values())))
    rows.append(describe("val_disjoint", list(disjoint.values())))

    # --- test queries ---
    tstore = PeakStore.load("test")
    tmeta = pd.read_parquet(f"{CACHE}/test_meta.parquet",
                            columns=["molecule_id", "precursor_mz"])
    Qt = bin_matrix(tstore, np.arange(len(tmeta)), tmeta.precursor_mz.to_numpy())
    st, it = search_cpu(R, Qt, topk=200, progress=True)
    test_best = best_per_molecule(st, it, tmeta.molecule_id.to_numpy())
    rows.append(describe("test", list(test_best.values())))
    print(f"test searched ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/test_regime_probe.csv", index=False)
    pd.DataFrame({"molecule_id": list(test_best), "best_cos": list(test_best.values())}
                 ).to_csv("reports/test_best_cosine.csv", index=False)

    t = np.array(list(test_best.values()))
    i = np.array(list(inlib.values()))
    d = np.array(list(disjoint.values()))
    verdict = ("in-library" if abs(t.mean() - i.mean()) < abs(t.mean() - d.mean())
               else "unknown/disjoint")
    print(json.dumps({"test_mean": round(float(t.mean()), 4),
                      "val_inlib_mean": round(float(i.mean()), 4),
                      "val_disjoint_mean": round(float(d.mean()), 4),
                      "test_distribution_resembles": verdict}, indent=1))


if __name__ == "__main__":
    main()

"""Verify, peak by peak, whether each test spectrum is literally present in train.

A cosine of 1.0 is not proof on its own: a spectrum reduced to one or two peaks by
filtering matches anything in the same bin. So this checks the raw peak lists --
counts, m/z values, intensities -- plus precursor m/z, adduct and collision energy,
and checks that the several spectra of one test molecule all point at the same
structure. Only an exact peak-level match across many peaks is evidence of a leak.
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


def main() -> None:
    t0 = time.time()
    ref = reference_table(CACHE, instrument="tims", min_peaks=1)
    tr_meta = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                              columns=["precursor_mz", "adduct", "collision_energy_orig",
                                       "ce_hi", "peak_n", "normalized_smiles"])
    store = PeakStore.load("train")
    tstore = PeakStore.load("test")
    tmeta = pd.read_parquet(f"{CACHE}/test_meta.parquet")

    R = bin_matrix(store, ref.row.to_numpy(), ref.precursor_mz.to_numpy())
    Q = bin_matrix(tstore, np.arange(len(tmeta)), tmeta.precursor_mz.to_numpy())
    print(f"binned ({time.time()-t0:.0f}s); query nnz per spectrum "
          f"median {np.median(np.diff(Q.indptr)):.0f}", flush=True)
    s, i = search_cpu(R, Q, topk=5)
    print(f"searched ({time.time()-t0:.0f}s)", flush=True)

    ref_rows = ref.row.to_numpy()
    recs = []
    for q in range(len(tmeta)):
        if i[q, 0] < 0:
            recs.append({"spectrum_idx": q, "status": "no_match"})
            continue
        tr_row = int(ref_rows[i[q, 0]])
        qmz, qit = tstore.get(q)
        rmz, rit = store.get(tr_row)
        same_n = len(qmz) == len(rmz)
        mz_match = bool(same_n and np.allclose(qmz, rmz, atol=1e-4))
        it_match = bool(same_n and np.allclose(qit, rit, atol=1e-4))
        recs.append({
            "spectrum_idx": q,
            "molecule_id": tmeta.molecule_id[q],
            "cosine": float(s[q, 0]),
            "train_row": tr_row,
            "n_peaks_test": len(qmz),
            "n_peaks_train": len(rmz),
            "peaks_same_count": bool(same_n),
            "mz_identical": mz_match,
            "intensity_identical": it_match,
            "pmz_test": float(tmeta.precursor_mz[q]),
            "pmz_train": float(tr_meta.precursor_mz[tr_row]),
            "adduct_test": tmeta.adduct[q],
            "adduct_train": tr_meta.adduct[tr_row],
            "ce_test": tmeta.collision_energy_orig[q],
            "ce_train": tr_meta.collision_energy_orig[tr_row],
            "train_smiles": tr_meta.normalized_smiles[tr_row],
        })
    df = pd.DataFrame(recs)
    df.to_csv("reports/leak_verification.csv", index=False)

    full = df[df.mz_identical & df.intensity_identical]
    print(f"\n{len(df)} test spectra")
    print(f"  cosine == 1.0 (within 1e-6) : {int((df.cosine > 1 - 1e-6).sum())}")
    print(f"  identical peak count        : {int(df.peaks_same_count.sum())}")
    print(f"  identical m/z values        : {int(df.mz_identical.sum())}")
    print(f"  identical m/z + intensities : {len(full)}")
    print(f"  precursor m/z equal (<1e-3) : "
          f"{int((abs(df.pmz_test - df.pmz_train) < 1e-3).sum())}")
    print(f"  adduct equal                : {int((df.adduct_test == df.adduct_train).sum())}")
    print(f"  collision energy equal      : "
          f"{int((df.ce_test.astype(str) == df.ce_train.astype(str)).sum())}")
    print(f"  peaks per matched spectrum  : median {df.n_peaks_test.median():.0f}, "
          f"min {df.n_peaks_test.min()}, max {df.n_peaks_test.max()}")

    # do all spectra of a molecule agree on one structure?
    agree = df.groupby("molecule_id").train_smiles.nunique()
    print(f"\nmolecules whose spectra all map to ONE train structure: "
          f"{int((agree == 1).sum())}/{len(agree)}")

    # how many test molecules get a confident structure assignment
    sure = df[df.mz_identical].groupby("molecule_id").train_smiles.agg(
        lambda s: s.mode().iat[0] if len(s) else None)
    print(f"molecules with >=1 exact peak-level match: {len(sure)}/400")

    json.dump({"n_test_spectra": int(len(df)),
               "exact_peak_matches": int(len(full)),
               "molecules_assigned": int(len(sure)),
               "molecules_single_structure": int((agree == 1).sum())},
              open("reports/leak_verification.json", "w"), indent=1)

    pd.set_option("display.width", 250)
    print("\nsample rows:")
    print(df[["molecule_id", "cosine", "n_peaks_test", "n_peaks_train", "mz_identical",
              "intensity_identical", "pmz_test", "pmz_train", "ce_test", "ce_train"]]
          .head(12).to_string(index=False))


if __name__ == "__main__":
    main()

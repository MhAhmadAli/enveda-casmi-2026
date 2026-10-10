"""Baseline: identify a molecule by spectral library retrieval, scored as MRR@25.

For every validation molecule we search its query spectra against the reference
library, map each hit spectrum to its structure, and rank structures.

Regimes (see scripts/build_splits.py):
  inlib    - sibling spectra of the true structure stay in the reference. This is
             the library-lookup ceiling, not a prediction of leaderboard score.
  disjoint - every spectrum of the true structure is removed. Retrieval alone then
             cannot return the right answer, so this measures what the analog
             signal is worth and is the setting real candidate ranking must solve.
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
from casmi.binning import bin_matrix  # noqa: E402
from casmi.chem import precursor_to_neutral_mass  # noqa: E402
from casmi.metric import mrr, topk_accuracy  # noqa: E402
from casmi.search import search_cpu  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def build_reference(meta, structs, instrument: str):
    ref = meta.merge(structs[["normalized_smiles", "key14"]], on="normalized_smiles",
                     how="left")
    mask = ref.key14.notna() & (ref.peak_n >= 3)
    if instrument == "tims":
        mask &= ref.instrument_type == "timsTOF"
    return ref[mask]


def rank_with_mass(cand_scores: dict[str, float], mol_mass: float,
                   key_mass: dict[str, float], ppm: float, mode: str) -> list[str]:
    """Order candidates, using the precursor-derived neutral mass as a constraint.

    mode='rank' is deliberately soft: mass-consistent candidates are placed above
    inconsistent ones but the latter are kept, so a wrong adduct assignment or an
    in-source fragment costs ranks rather than discarding the right answer outright.
    mode='hard' drops inconsistent candidates entirely.
    """
    if mode == "none" or not np.isfinite(mol_mass):
        return [k for k, _ in sorted(cand_scores.items(), key=lambda kv: -kv[1])]
    tol = mol_mass * ppm * 1e-6
    hit, miss = [], []
    for k, v in cand_scores.items():
        m = key_mass.get(k)
        (hit if (m is not None and abs(m - mol_mass) <= tol) else miss).append((k, v))
    hit.sort(key=lambda kv: -kv[1])
    miss.sort(key=lambda kv: -kv[1])
    if mode == "hard":
        return [k for k, _ in hit]
    return [k for k, _ in hit] + [k for k, _ in miss]


def aggregate(scores, idx, ref_key14, query_mol, agg: str = "max", per_mol_agg: str = "mean"):
    """Spectrum-level hits -> per-molecule ranked list of candidate structures.

    agg         : how several hit spectra of the same structure combine (max is the
                  classic library-search choice; it avoids rewarding libraries that
                  simply contain many collision energies of one compound).
    per_mol_agg : how the several query spectra of one molecule combine. mean uses
                  all the evidence; max trusts the single best-matching energy.
    """
    out: dict[str, dict[str, float]] = {}
    nq_per_mol: dict[str, int] = {}
    for q in range(scores.shape[0]):
        mol = query_mol[q]
        nq_per_mol[mol] = nq_per_mol.get(mol, 0) + 1
        best: dict[str, float] = {}
        rows, vals = idx[q], scores[q]
        ok = rows >= 0
        for r, v in zip(rows[ok], vals[ok]):
            k = ref_key14[r]
            if k is None:
                continue
            if agg == "max":
                if v > best.get(k, -1.0):
                    best[k] = float(v)
            else:
                best[k] = best.get(k, 0.0) + float(v)
        tgt = out.setdefault(mol, {})
        for k, v in best.items():
            if per_mol_agg == "max":
                if v > tgt.get(k, -1.0):
                    tgt[k] = v
            else:
                tgt[k] = tgt.get(k, 0.0) + v
    if per_mol_agg == "mean":
        for mol, d in out.items():
            n = nq_per_mol[mol]
            for k in d:
                d[k] /= n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--regime", default="inlib", choices=["inlib", "disjoint"])
    ap.add_argument("--instrument", default="tims", choices=["tims", "all"])
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--bin-width", type=float, default=0.01)
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--domain", default="direct", choices=["direct", "nl", "both"])
    ap.add_argument("--per-mol-agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--limit-mols", type=int, default=0, help="debug: first N molecules")
    ap.add_argument("--mass-filter", default="rank", choices=["none", "rank", "hard"])
    ap.add_argument("--mass-ppm", type=float, default=20.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    meta = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                           columns=["normalized_smiles", "instrument_type", "adduct",
                                    "precursor_mz", "peak_n"])
    meta["row"] = np.arange(len(meta), dtype=np.int64)
    structs = pd.read_parquet(f"{CACHE}/structures.parquet",
                              columns=["normalized_smiles", "key14"])
    val = pd.read_parquet(f"{CACHE}/val_fold{args.fold}.parquet")
    if args.limit_mols:
        keep = val.molecule_id.unique()[: args.limit_mols]
        val = val[val.molecule_id.isin(keep)]

    ref = build_reference(meta, structs, args.instrument)
    # queries are never part of the reference
    banned_rows = set(val.row.tolist())
    if args.regime == "disjoint":
        truth_keys = set(val.key14.unique())
        ref = ref[~ref.key14.isin(truth_keys)]
    ref = ref[~ref.row.isin(banned_rows)]
    print(f"reference: {len(ref):,} spectra / {ref.key14.nunique():,} structures "
          f"[{args.instrument}, {args.regime}]")

    store = PeakStore.load("train")
    ref_rows = ref.row.to_numpy()
    ref_pmz = ref.precursor_mz.to_numpy()
    ref_key = ref.key14.to_numpy()

    q_rows = val.row.to_numpy()
    q_pmz = meta.precursor_mz.to_numpy()[q_rows]
    q_mol = val.molecule_id.to_numpy()

    domains = {"direct": [False], "nl": [True], "both": [False, True]}[args.domain]
    score_parts, idx_parts = [], []
    for nl in domains:
        t = time.time()
        R = bin_matrix(store, ref_rows, ref_pmz, bin_width=args.bin_width,
                       power=args.power, neutral_loss=nl)
        Q = bin_matrix(store, q_rows, q_pmz, bin_width=args.bin_width,
                       power=args.power, neutral_loss=nl)
        print(f"  binned {'NL' if nl else 'direct'}: ref nnz {R.nnz:,} "
              f"query nnz {Q.nnz:,} in {time.time()-t:.0f}s", flush=True)
        t = time.time()
        s, i = search_cpu(R, Q, topk=args.topk, progress=True)
        print(f"  searched in {time.time()-t:.0f}s", flush=True)
        score_parts.append(s)
        idx_parts.append(i)
        del R, Q

    if len(score_parts) == 1:
        scores, idx = score_parts[0], idx_parts[0]
    else:
        scores = np.concatenate(score_parts, axis=1)
        idx = np.concatenate(idx_parts, axis=1)

    cand = aggregate(scores, idx, ref_key, q_mol, per_mol_agg=args.per_mol_agg)
    truth = val.groupby("molecule_id").key14.first()
    mols = list(truth.index)

    # neutral mass implied by each molecule's precursor/adduct, and the mass of every
    # candidate structure, so ranking can enforce mass consistency
    all_structs = pd.read_parquet(f"{CACHE}/structures.parquet",
                                  columns=["key14", "exact_mass"])
    key_mass = all_structs.groupby("key14").exact_mass.first().to_dict()
    qinfo = val.groupby("molecule_id").agg(row=("row", "first"), adduct=("adduct", "first"))
    pmz_all = meta.precursor_mz.to_numpy()
    mol_mass = {}
    for m, r in zip(qinfo.index, qinfo.row):
        try:
            mol_mass[m] = precursor_to_neutral_mass(float(pmz_all[r]),
                                                    qinfo.adduct[m])
        except Exception:
            mol_mass[m] = float("nan")

    lists = [rank_with_mass(cand.get(m, {}), mol_mass[m], key_mass,
                            args.mass_ppm, args.mass_filter) for m in mols]
    score = mrr(lists, [truth[m] for m in mols])
    acc = topk_accuracy(lists, [truth[m] for m in mols])
    n_cand = np.mean([len(l) for l in lists])

    res = {"fold": args.fold, "regime": args.regime, "instrument": args.instrument,
           "domain": args.domain, "bin_width": args.bin_width, "power": args.power,
           "topk": args.topk, "per_mol_agg": args.per_mol_agg,
           "mass_filter": args.mass_filter, "mass_ppm": args.mass_ppm,
           "n_molecules": len(mols), "mrr@25": round(score, 4),
           "mean_candidates": round(float(n_cand), 1),
           "top1": round(acc[1], 4), "top5": round(acc[5], 4),
           "top10": round(acc[10], 4), "top25": round(acc[25], 4),
           "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1))
    os.makedirs("reports", exist_ok=True)
    with open("reports/retrieval_runs.jsonl", "a") as f:
        f.write(json.dumps({**res, "tag": args.tag}) + "\n")


if __name__ == "__main__":
    main()

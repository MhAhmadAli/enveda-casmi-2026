"""Rank mass-window candidates from the structure pool. The real task, scored as MRR@25.

Unlike scripts/eval_retrieval.py, the answer here never has to be in the spectral
library: candidates come from the 730k-structure pool, so this is the setting that
also works for genuinely novel molecules.

Scorers (all evaluated in the structure-disjoint regime, i.e. every spectrum of the
true structure is removed from the library first):

  random    - order candidates arbitrarily. The floor.
  analog    - propagate spectral neighbours to candidates by fingerprint similarity:
              score(c) = max over analogs a of  tanimoto(c, a) * spectral_sim(a)**p.
              A library analog that fragments like the unknown is evidence for
              candidates that look like that analog.

A ceiling is reported too: "pool recall" is the fraction of molecules whose true
structure is in the mass window at all. No scorer can beat it.

Caveat recorded deliberately: validation truths are necessarily structures that have
train spectra, so any feature meaning "this candidate is in the spectral library"
would score well here and may not transfer to genuinely unknown test molecules. No
scorer below uses such a feature.
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
from casmi.retrieval import (CandidatePool, analogs_per_query,  # noqa: E402
                             reference_table, tanimoto_matrix)
from casmi.search import search_cpu  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="fold0",
                    help="fold0/fold1/fold2 (placeholder-matched) or npx/npbig "
                         "(natural-product dev sets)")
    ap.add_argument("--instrument", default="all", choices=["tims", "all"],
                    help="reference library subset; natural-product spectra are "
                         "mostly Orbitrap/QTOF so 'all' is the right default now")
    ap.add_argument("--scorer", default="analog", choices=["random", "analog"])
    ap.add_argument("--regime", default="disjoint", choices=["disjoint", "inlib"],
                    help="disjoint = every spectrum of the true structure removed "
                         "(Class 2, a novel molecule). inlib = only the query spectra "
                         "are withheld, so the structure's other spectra (typically on "
                         "another instrument) remain (Class 1). The hidden test is a "
                         "MIXTURE of the two, so both must be measured.")
    ap.add_argument("--ppm", type=float, default=10.0, help="candidate mass window")
    ap.add_argument("--topk-spectra", type=int, default=500)
    ap.add_argument("--top-analogs", type=int, default=100)
    ap.add_argument("--sim-power", type=float, default=4.0,
                    help="spectral similarity exponent when weighting analogs")
    ap.add_argument("--domain", default="both", choices=["direct", "nl", "both"])
    ap.add_argument("--limit-mols", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)

    vfile = (f"{CACHE}/val_{args.val}.parquet" if not args.val.startswith("fold")
             else f"{CACHE}/val_fold{args.val[4:]}.parquet")
    val = pd.read_parquet(vfile)
    if args.limit_mols:
        val = val[val.molecule_id.isin(val.molecule_id.unique()[: args.limit_mols])]
    truth = val.groupby("molecule_id").key14.first()
    mols = list(truth.index)

    pool = CandidatePool.load(CACHE, with_fp=(args.scorer != "random"))
    print(f"pool: {len(pool.df):,} structures", flush=True)

    ref = reference_table(CACHE, instrument=args.instrument)
    if args.regime == "disjoint":
        # Class 2: drop every spectrum of every held-out structure
        ref = ref[~ref.key14.isin(set(truth.values))].reset_index(drop=True)
    # the query spectra themselves are never part of the reference
    ref = ref[~ref.row.isin(set(val.row.tolist()))].reset_index(drop=True)
    print(f"reference ({args.regime}): {len(ref):,} spectra / "
          f"{ref.key14.nunique():,} structures", flush=True)

    store = PeakStore.load("train")
    meta_pmz = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                               columns=["precursor_mz"]).precursor_mz.to_numpy()
    q_rows = val.row.to_numpy()
    q_pmz = meta_pmz[q_rows]

    # per-molecule neutral mass and candidate set
    qinfo = val.groupby("molecule_id").agg(row=("row", "first"),
                                           adduct=("adduct", "first"))
    cand_rows, mol_mass = {}, {}
    for m in mols:
        try:
            nm = precursor_to_neutral_mass(float(meta_pmz[qinfo.row[m]]),
                                           qinfo.adduct[m])
        except Exception:
            nm = float("nan")
        mol_mass[m] = nm
        cand_rows[m] = pool.candidates(nm, args.ppm)

    key_sets = {m: pool.key14[cand_rows[m]] for m in mols}
    recall = np.mean([truth[m] in set(key_sets[m]) for m in mols])
    n_cand = np.mean([len(cand_rows[m]) for m in mols])
    print(f"pool recall @{args.ppm} ppm: {recall:.3f}  "
          f"mean candidates {n_cand:.0f}", flush=True)

    ranked: list[list[str]] = []
    if args.scorer == "random":
        for m in mols:
            ks = list(dict.fromkeys(key_sets[m]))
            rng.shuffle(ks)
            ranked.append(ks)
    else:
        domains = {"direct": [False], "nl": [True], "both": [False, True]}[args.domain]
        parts_s, parts_i = [], []
        for nl in domains:
            R = bin_matrix(store, ref.row.to_numpy(), ref.precursor_mz.to_numpy(),
                           neutral_loss=nl)
            Q = bin_matrix(store, q_rows, q_pmz, neutral_loss=nl)
            s, i = search_cpu(R, Q, topk=args.topk_spectra, progress=True)
            parts_s.append(s)
            parts_i.append(i)
            del R, Q
            print(f"  searched {'NL' if nl else 'direct'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        scores = np.concatenate(parts_s, axis=1)
        idx = np.concatenate(parts_i, axis=1)

        per_q = analogs_per_query(scores, idx, ref.key14.to_numpy(),
                                  top_structures=args.top_analogs)
        # merge a molecule's query spectra: an analog keeps its best similarity
        by_mol: dict[str, dict[str, float]] = {}
        for q, mol in enumerate(val.molecule_id.to_numpy()):
            tgt = by_mol.setdefault(mol, {})
            for k, v in per_q[q].items():
                if v > tgt.get(k, -1.0):
                    tgt[k] = v

        # fingerprints of library structures, looked up through the pool
        pool_row_of_key = {}
        for r, k in enumerate(pool.key14):
            if k not in pool_row_of_key:
                pool_row_of_key[k] = r

        for m in mols:
            cr = cand_rows[m]
            analogs = by_mol.get(m, {})
            a_keys = [k for k in analogs if k in pool_row_of_key]
            if len(cr) == 0:
                ranked.append([])
                continue
            if not a_keys:
                ranked.append(list(dict.fromkeys(pool.key14[cr])))
                continue
            a_rows = np.array([pool_row_of_key[k] for k in a_keys], dtype=np.int64)
            w = np.array([analogs[k] for k in a_keys], dtype=np.float32) ** args.sim_power
            T = tanimoto_matrix(pool.unpack(cr), pool.unpack(a_rows))  # (cand x analog)
            sc = (T * w[None, :]).max(axis=1)
            best: dict[str, float] = {}
            for k, v in zip(pool.key14[cr], sc):
                if v > best.get(k, -1.0):
                    best[k] = float(v)
            ranked.append([k for k, _ in sorted(best.items(), key=lambda kv: -kv[1])])

    tr = [truth[m] for m in mols]
    score = mrr(ranked, tr)
    acc = topk_accuracy(ranked, tr)
    res = {"val": args.val, "instrument": args.instrument, "regime": args.regime,
           "scorer": args.scorer, "ppm": args.ppm,
           "domain": args.domain, "sim_power": args.sim_power,
           "top_analogs": args.top_analogs, "topk_spectra": args.topk_spectra,
           "n_molecules": len(mols), "pool_recall": round(float(recall), 4),
           "mean_candidates": round(float(n_cand), 1),
           "mrr@25": round(score, 4), "top1": round(acc[1], 4),
           "top5": round(acc[5], 4), "top10": round(acc[10], 4),
           "top25": round(acc[25], 4), "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1))
    os.makedirs("reports", exist_ok=True)
    with open("reports/candidate_runs.jsonl", "a") as f:
        f.write(json.dumps({**res, "tag": args.tag}) + "\n")


if __name__ == "__main__":
    main()

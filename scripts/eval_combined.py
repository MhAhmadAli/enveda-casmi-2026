"""Both channels in one scorer: entropy-rescored analog propagation + direct match.

Measured separately, the two channels are strictly complementary -- direct search
gets 0.880 on Class 1 and exactly 0.000 on Class 2; binned-cosine analog propagation
gets 0.793 and 0.475. A competitive submission needs both, and the hidden test is a
mixture of the two in unknown proportion.

MEASURED, do not "simplify" this back: rescoring the analog channel with entropy
similarity looks elegant (a Class-1 molecule is its own best analog) but it HALVES
Class 2 -- 0.475 -> 0.226 -- while gaining 0.077 on Class 1. The two channels need
different similarity measures:

  - identifying the *same* compound: entropy similarity is right, because a peak the
    library spectrum lacks really is evidence against the match.
  - matching a structural *analog*: entropy similarity is wrong, because most peaks
    legitimately differ. Binned cosine over the neutral-loss domain is more forgiving
    and captures partial similarity better.

So each channel keeps its own metric, and they are combined by confidence gating:

  Channel 1 (direct)  entropy similarity over library spectra whose neutral mass
                      matches within ppm.           Class 1 0.880 / Class 2 0.000
  Channel 2 (analog)  binned-cosine prefilter, direct + neutral-loss domains, then
                      score(c) = max_a tanimoto(c, a) * cos(a)**p.
                                                    Class 1 0.793 / Class 2 0.475

Gate: if Channel 1's best similarity clears `--gate`, the molecule is probably a
library compound, so Channel 1's ordering leads and Channel 2's follows; otherwise
the reverse. Nothing is discarded, only reordered, so a wrong gate costs ranks
rather than the answer.
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
from casmi.chem import neutral_masses, precursor_to_neutral_mass  # noqa: E402
from casmi.entropy import prepare, score_against  # noqa: E402
from casmi.metric import mrr, topk_accuracy  # noqa: E402
from casmi.retrieval import CandidatePool, reference_table, tanimoto_matrix  # noqa: E402
from casmi.search import search_cpu  # noqa: E402
from casmi.spectra import PeakStore  # noqa: E402

CACHE = "data/cache"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="npx")
    ap.add_argument("--regime", default="inlib", choices=["inlib", "disjoint", "both"])
    ap.add_argument("--ppm", type=float, default=8.5)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--min-rel", type=float, default=0.005)
    ap.add_argument("--max-peaks", type=int, default=256)
    ap.add_argument("--prefilter-k", type=int, default=500,
                    help="library spectra kept per query per domain before rescoring")
    ap.add_argument("--top-analogs", type=int, default=100)
    ap.add_argument("--sim-power", type=float, default=4.0)
    ap.add_argument("--direct-weight", type=float, default=1.0,
                    help="weight on the explicit mass-matched direct channel; 0 disables")
    ap.add_argument("--no-entropy-weight", action="store_true")
    ap.add_argument("--limit-mols", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.60,
                    help="Channel-1 similarity above which the direct "
                         "channel leads the ranking (V14 used 0.60)")
    ap.add_argument("--dump", action="store_true",
                    help="write per-molecule ranks for stratified analysis")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ew = not args.no_entropy_weight

    regimes = ["inlib", "disjoint"] if args.regime == "both" else [args.regime]
    for regime in regimes:
        run(args, regime, ew)


def run(args, regime: str, ew: bool) -> None:
    t0 = time.time()
    val = pd.read_parquet(f"{CACHE}/val_{args.val}.parquet")
    if args.limit_mols:
        val = val[val.molecule_id.isin(val.molecule_id.unique()[: args.limit_mols])]
    truth = val.groupby("molecule_id").key14.first()
    mols = list(truth.index)

    pool = CandidatePool.load(CACHE, with_fp=True)
    pool_row_of_key: dict[str, int] = {}
    for r, k in enumerate(pool.key14):
        if k not in pool_row_of_key:
            pool_row_of_key[k] = r

    ref = reference_table(CACHE, instrument="all")
    if regime == "disjoint":
        ref = ref[~ref.key14.isin(set(truth.values))]
    ref = ref[~ref.row.isin(set(val.row.tolist()))].reset_index(drop=True)

    ref_nm = neutral_masses(ref.precursor_mz.to_numpy(),
                            ref.adduct.reset_index(drop=True))
    ref_rows = ref.row.to_numpy()
    ref_key = ref.key14.to_numpy()
    ref_pmz = ref.precursor_mz.to_numpy()
    nm_ok = np.isfinite(ref_nm)
    nm_order = np.argsort(np.where(nm_ok, ref_nm, np.inf), kind="stable")
    nm_sorted = ref_nm[nm_order]
    print(f"[{regime}] library {len(ref):,} spectra / {ref.key14.nunique():,} "
          f"structures", flush=True)

    store = PeakStore.load("train")
    all_mz = np.asarray(store.mz)
    all_in = np.asarray(store.inten)
    meta = pd.read_parquet(f"{CACHE}/train_meta.parquet",
                           columns=["precursor_mz", "adduct"])
    q_rows = val.row.to_numpy()
    q_pmz = meta.precursor_mz.to_numpy()[q_rows]
    q_nm = neutral_masses(q_pmz, meta.adduct.iloc[q_rows].reset_index(drop=True))
    q_mol = val.molecule_id.to_numpy()

    # ---- step 1: cheap binned prefilter for recall, both domains ----
    pre: list[dict] = [dict() for _ in range(len(q_rows))]
    for nl in (False, True):
        R = bin_matrix(store, ref_rows, ref_pmz, neutral_loss=nl)
        Q = bin_matrix(store, q_rows, q_pmz, neutral_loss=nl)
        sc, idx = search_cpu(R, Q, topk=args.prefilter_k)
        for qi in range(len(q_rows)):
            d = pre[qi]
            for x, v in zip(idx[qi], sc[qi]):
                if x >= 0 and v > d.get(int(x), -1.0):
                    d[int(x)] = float(v)
        del R, Q
        print(f"[{regime}]   prefiltered {'NL' if nl else 'direct'} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- Channel 2: analog propagation on the binned-cosine similarities ----
    analog_by_mol = {}
    for qi in range(len(q_rows)):
        tgt = analog_by_mol.setdefault(q_mol[qi], {})
        for ref_i, sim in pre[qi].items():
            k = ref_key[ref_i]
            if sim > tgt.get(k, -1.0):
                tgt[k] = float(sim)
    for m in list(analog_by_mol):
        d = analog_by_mol[m]
        if len(d) > args.top_analogs:
            analog_by_mol[m] = dict(
                sorted(d.items(), key=lambda kv: -kv[1])[: args.top_analogs])
    print(f"[{regime}]   channel 2 analogs collected ({time.time()-t0:.0f}s)",
          flush=True)

    # ---- Channel 1: entropy similarity over mass-matched library spectra ----
    direct_by_mol = {}
    if args.direct_weight > 0:
        for qi in range(len(q_rows)):
            if not np.isfinite(q_nm[qi]):
                continue
            r = int(q_rows[qi])
            qmz, qp = prepare(all_mz[store.off[r]:store.off[r] + store.n[r]],
                              all_in[store.off[r]:store.off[r] + store.n[r]],
                              args.min_rel, args.max_peaks, ew)
            if qmz.shape[0] == 0:
                continue
            tol = q_nm[qi] * args.ppm * 1e-6
            lo = np.searchsorted(nm_sorted, q_nm[qi] - tol)
            hi = np.searchsorted(nm_sorted, q_nm[qi] + tol)
            dc = nm_order[lo:hi]
            if dc.size == 0:
                continue
            sims = score_against(qmz, qp, ref_rows[dc].astype(np.int64),
                                 store.off, store.n, all_mz, all_in, args.tol,
                                 args.min_rel, args.max_peaks, ew,
                                 np.zeros(dc.size, np.float32))
            tgt = direct_by_mol.setdefault(q_mol[qi], {})
            for k, v in zip(ref_key[dc], sims):
                v = float(v)
                if v > tgt.get(k, -1.0):
                    tgt[k] = v
        print(f"[{regime}]   channel 1 direct scored ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- rank: gate on Channel-1 confidence, never discard a candidate ----
    qinfo = val.groupby("molecule_id").agg(row=("row", "first"),
                                           adduct=("adduct", "first"))
    pmz_all = meta.precursor_mz.to_numpy()
    ranked, n_cand, best_analog, best_direct = [], [], [], []
    for m in mols:
        try:
            nm = precursor_to_neutral_mass(float(pmz_all[qinfo.row[m]]),
                                           qinfo.adduct[m])
        except Exception:
            nm = float("nan")
        cr = pool.candidates(nm, args.ppm)
        n_cand.append(len(cr))

        analogs = {k: v for k, v in analog_by_mol.get(m, {}).items()
                   if k in pool_row_of_key}
        c2 = {}
        if len(cr) and analogs:
            a_keys = list(analogs)
            a_rows = np.array([pool_row_of_key[k] for k in a_keys], dtype=np.int64)
            w = np.array([analogs[k] for k in a_keys], np.float32) ** args.sim_power
            T = tanimoto_matrix(pool.unpack(cr), pool.unpack(a_rows))
            sc2 = (T * w[None, :]).max(axis=1)
            for k, v in zip(pool.key14[cr], sc2):
                if v > c2.get(k, -1.0):
                    c2[k] = float(v)
        elif len(cr):
            for k in pool.key14[cr]:
                c2.setdefault(k, 0.0)

        c1 = direct_by_mol.get(m, {})
        best_analog.append(max(analogs.values()) if analogs else 0.0)
        best_direct.append(max(c1.values()) if c1 else 0.0)

        l1 = [k for k, _ in sorted(c1.items(), key=lambda kv: -kv[1])]
        l2 = [k for k, _ in sorted(c2.items(), key=lambda kv: -kv[1])]
        lead, follow = (l1, l2) if best_direct[-1] >= args.gate else (l2, l1)
        seen, merged = set(), []
        for k in lead + follow:
            if k not in seen:
                seen.add(k)
                merged.append(k)
        ranked.append(merged)

    tr = [truth[m] for m in mols]
    if args.dump:
        recs = []
        for m, lst, t, nc, ba, bd in zip(mols, ranked, tr, n_cand,
                                         best_analog, best_direct):
            rank = lst.index(t) + 1 if t in lst[:25] else 0
            recs.append({"molecule_id": m, "truth_key14": t, "rank": rank,
                         "rr": (1.0 / rank) if rank else 0.0,
                         "n_candidates": nc,
                         "best_analog_sim": round(ba, 4),
                         "best_direct_sim": round(bd, 4)})
        out = f"reports/per_molecule_{args.val}_{regime}.csv"
        pd.DataFrame(recs).to_csv(out, index=False)
        print(f"[{regime}] wrote {out}", flush=True)
    acc = topk_accuracy(ranked, tr)
    res = {"val": args.val, "regime": regime, "scorer": "combined",
           "ppm": args.ppm, "tol": args.tol, "min_rel": args.min_rel,
           "prefilter_k": args.prefilter_k, "sim_power": args.sim_power,
           "direct_weight": args.direct_weight, "gate": args.gate,
           "entropy_weight": ew,
           "n_molecules": len(mols),
           "mean_pool_candidates": round(float(np.mean(n_cand)), 1),
           "mrr@25": round(mrr(ranked, tr), 4), "top1": round(acc[1], 4),
           "top5": round(acc[5], 4), "top10": round(acc[10], 4),
           "top25": round(acc[25], 4), "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(res, indent=1), flush=True)
    os.makedirs("reports", exist_ok=True)
    with open("reports/combined_runs.jsonl", "a") as f:
        f.write(json.dumps({**res, "tag": args.tag}) + "\n")


if __name__ == "__main__":
    main()

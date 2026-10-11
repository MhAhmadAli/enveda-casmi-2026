"""Compute both evidence channels for a dev set, as per-candidate features.

The two channels disagree in a way no fixed rule resolves (see CLAUDE.md: entropy
rescoring halves Class 2, hard gating is worse than either channel alone), so the
combination has to be learned per candidate. This produces the feature matrix that
learning needs, and is the single expensive step: one pass yields features that many
ranker configurations can then be trained and evaluated on cheaply.

A deliberate omission: no feature encodes "this candidate is in the spectral
library". Every validation truth necessarily has train spectra, so such a feature
scores well locally and transfers nowhere. Channel-1 similarity is different in kind
-- it is evidence read off the spectrum, not mere membership -- but note its sign
flips between regimes: in `inlib` the truth has library spectra at its own mass and
scores high, while in `disjoint` the truth scores zero and only *wrong* same-mass
compounds score at all. Any ranker must therefore be trained on both regimes, which
is why `label` and `regime` are both carried here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .binning import bin_matrix
from .chem import neutral_masses, precursor_to_neutral_mass
from .entropy import prepare, score_against
from .retrieval import CandidatePool, reference_table, tanimoto_matrix
from .search import search_cpu
from .spectra import PeakStore

CACHE = "data/cache"


@dataclass
class ChannelConfig:
    ppm: float = 8.5
    tol: float = 0.005
    min_rel: float = 0.005
    max_peaks: int = 256
    prefilter_k: int = 500
    top_analogs: int = 100
    sim_power: float = 4.0
    entropy_weight: bool = True


def compute_features(val: pd.DataFrame, regime: str, cfg: ChannelConfig,
                     cache: str = CACHE, log=print) -> pd.DataFrame:
    """Per-(molecule, candidate) features with a 0/1 label. One row per candidate."""
    truth = val.groupby("molecule_id").key14.first()
    mols = list(truth.index)

    pool = CandidatePool.load(cache, with_fp=True)
    row_of_key: dict[str, int] = {}
    for r, k in enumerate(pool.key14):
        if k not in row_of_key:
            row_of_key[k] = r

    ref = reference_table(cache, instrument="all")
    if regime == "disjoint":
        ref = ref[~ref.key14.isin(set(truth.values))]
    ref = ref[~ref.row.isin(set(val.row.tolist()))].reset_index(drop=True)

    ref_nm = neutral_masses(ref.precursor_mz.to_numpy(),
                            ref.adduct.reset_index(drop=True))
    ref_rows = ref.row.to_numpy()
    ref_key = ref.key14.to_numpy()
    ref_pmz = ref.precursor_mz.to_numpy()
    nm_order = np.argsort(np.where(np.isfinite(ref_nm), ref_nm, np.inf), kind="stable")
    nm_sorted = ref_nm[nm_order]
    log(f"[{regime}] library {len(ref):,} spectra / {ref.key14.nunique():,} structures")

    store = PeakStore.load("train")
    all_mz, all_in = np.asarray(store.mz), np.asarray(store.inten)
    meta = pd.read_parquet(f"{cache}/train_meta.parquet",
                           columns=["precursor_mz", "adduct"])
    q_rows = val.row.to_numpy()
    q_pmz = meta.precursor_mz.to_numpy()[q_rows]
    q_nm = neutral_masses(q_pmz, meta.adduct.iloc[q_rows].reset_index(drop=True))
    q_mol = val.molecule_id.to_numpy()

    # ---- channel 2: binned-cosine prefilter over the whole library ----
    pre: list[dict] = [dict() for _ in range(len(q_rows))]
    for nl in (False, True):
        R = bin_matrix(store, ref_rows, ref_pmz, neutral_loss=nl)
        Q = bin_matrix(store, q_rows, q_pmz, neutral_loss=nl)
        sc, idx = search_cpu(R, Q, topk=cfg.prefilter_k)
        for qi in range(len(q_rows)):
            d = pre[qi]
            for x, v in zip(idx[qi], sc[qi]):
                if x >= 0 and v > d.get(int(x), -1.0):
                    d[int(x)] = float(v)
        del R, Q
        log(f"[{regime}]   prefiltered {'NL' if nl else 'direct'}")

    analog_by_mol: dict[str, dict[str, float]] = {}
    for qi in range(len(q_rows)):
        tgt = analog_by_mol.setdefault(q_mol[qi], {})
        for ref_i, sim in pre[qi].items():
            k = ref_key[ref_i]
            if sim > tgt.get(k, -1.0):
                tgt[k] = sim
    for m in list(analog_by_mol):
        d = analog_by_mol[m]
        if len(d) > cfg.top_analogs:
            analog_by_mol[m] = dict(
                sorted(d.items(), key=lambda kv: -kv[1])[: cfg.top_analogs])

    # ---- channel 1: entropy similarity over mass-matched library spectra ----
    direct_by_mol: dict[str, dict[str, float]] = {}
    n_direct_spectra: dict[str, int] = {}
    for qi in range(len(q_rows)):
        if not np.isfinite(q_nm[qi]):
            continue
        r = int(q_rows[qi])
        qmz, qp = prepare(all_mz[store.off[r]:store.off[r] + store.n[r]],
                          all_in[store.off[r]:store.off[r] + store.n[r]],
                          cfg.min_rel, cfg.max_peaks, cfg.entropy_weight)
        if qmz.shape[0] == 0:
            continue
        tol = q_nm[qi] * cfg.ppm * 1e-6
        lo = np.searchsorted(nm_sorted, q_nm[qi] - tol)
        hi = np.searchsorted(nm_sorted, q_nm[qi] + tol)
        dc = nm_order[lo:hi]
        n_direct_spectra[q_mol[qi]] = n_direct_spectra.get(q_mol[qi], 0) + len(dc)
        if dc.size == 0:
            continue
        sims = score_against(qmz, qp, ref_rows[dc].astype(np.int64),
                             store.off, store.n, all_mz, all_in, cfg.tol,
                             cfg.min_rel, cfg.max_peaks, cfg.entropy_weight,
                             np.zeros(dc.size, np.float32))
        tgt = direct_by_mol.setdefault(q_mol[qi], {})
        for k, v in zip(ref_key[dc], sims):
            v = float(v)
            if v > tgt.get(k, -1.0):
                tgt[k] = v
    log(f"[{regime}]   channels done")

    # ---- assemble per-candidate rows ----
    qinfo = val.groupby("molecule_id").agg(row=("row", "first"),
                                           adduct=("adduct", "first"),
                                           n_spectra=("row", "size"))
    pmz_all = meta.precursor_mz.to_numpy()
    frames = []
    for m in mols:
        try:
            nm = precursor_to_neutral_mass(float(pmz_all[qinfo.row[m]]),
                                           qinfo.adduct[m])
        except Exception:
            nm = float("nan")
        cr = pool.candidates(nm, cfg.ppm)
        if len(cr) == 0:
            continue
        keys = pool.key14[cr]

        analogs = {k: v for k, v in analog_by_mol.get(m, {}).items()
                   if k in row_of_key}
        c2 = np.zeros(len(cr), np.float32)
        tan_best = np.zeros(len(cr), np.float32)
        ana_sim = np.zeros(len(cr), np.float32)
        c2_mean5 = np.zeros(len(cr), np.float32)
        if analogs:
            a_keys = list(analogs)
            a_rows = np.array([row_of_key[k] for k in a_keys], dtype=np.int64)
            a_sim = np.array([analogs[k] for k in a_keys], np.float32)
            T = tanimoto_matrix(pool.unpack(cr), pool.unpack(a_rows))
            W = T * (a_sim[None, :] ** cfg.sim_power)
            c2 = W.max(axis=1).astype(np.float32)
            arg = W.argmax(axis=1)
            tan_best = T[np.arange(len(cr)), arg].astype(np.float32)
            ana_sim = a_sim[arg]
            k5 = min(5, W.shape[1])
            c2_mean5 = np.sort(W, axis=1)[:, -k5:].mean(axis=1).astype(np.float32)

        c1map = direct_by_mol.get(m, {})
        c1 = np.array([c1map.get(k, 0.0) for k in keys], np.float32)

        cand_mass = pool.mass[cr]
        ppm_err = ((cand_mass - nm) / nm * 1e6) if np.isfinite(nm) else np.zeros(len(cr))

        df = pd.DataFrame({
            "molecule_id": m,
            "key14": keys,
            "label": (keys == truth[m]).astype(np.int8),
            # channel 1
            "c1": c1,
            "c1_best": c1.max() if len(c1) else 0.0,
            "c1_margin": c1 - (c1.max() if len(c1) else 0.0),
            "c1_rank": pd.Series(-c1).rank(method="min").to_numpy(),
            "c1_nonzero_frac": float((c1 > 0).mean()) if len(c1) else 0.0,
            # channel 2
            "c2": c2,
            "c2_best": c2.max() if len(c2) else 0.0,
            "c2_margin": c2 - (c2.max() if len(c2) else 0.0),
            "c2_rank": pd.Series(-c2).rank(method="min").to_numpy(),
            "c2_mean_top5": c2_mean5,
            "tanimoto_best_analog": tan_best,
            "analog_sim_of_best": ana_sim,
            # mass / molecule context
            "ppm_err": np.asarray(ppm_err, np.float32),
            "abs_ppm_err": np.abs(np.asarray(ppm_err, np.float32)),
            "n_candidates": len(cr),
            "n_query_spectra": int(qinfo.n_spectra[m]),
            "n_analogs": len(analogs),
            "regime": regime,
        })
        frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log(f"[{regime}] features: {len(out):,} rows over "
        f"{out.molecule_id.nunique() if len(out) else 0} molecules")
    return out

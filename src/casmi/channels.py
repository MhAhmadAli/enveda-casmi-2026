"""Compute both evidence channels as per-candidate features.

The two channels disagree in ways no fixed rule resolves (see CLAUDE.md: entropy
rescoring halves Class 2, hard gating is worse than either channel alone), so the
combination is learned per candidate. This module produces the feature matrix that
learning needs, and is the single expensive step.

Deliberately parameterised over the query and library sources rather than reading
the dev cache directly, so the Kaggle notebook computes *identical* features for the
hidden test as training saw. Divergence between a training feature and its inference
twin is the classic way a ranker silently degrades, and the only robust guard is one
implementation used by both.

A deliberate omission: no feature encodes "this candidate is in the spectral
library". Every validation truth necessarily has train spectra, so such a feature
scores well locally and transfers nowhere. Channel-1 similarity is admissible
because it is evidence read off the spectrum rather than mere membership -- but note
its meaning inverts between regimes (in `inlib` the truth scores high; in `disjoint`
the truth scores zero and only wrong same-mass compounds score at all), which is why
a ranker must train on both.
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


@dataclass
class SpectraSet:
    """Spectra addressed in a PeakStore, with the metadata the channels need."""
    store: PeakStore
    rows: np.ndarray
    precursor_mz: np.ndarray
    adduct: np.ndarray
    group: np.ndarray | None = None      # molecule id, for queries
    key14: np.ndarray | None = None      # structure key, for the library

    def __len__(self) -> int:
        return len(self.rows)


def dev_sources(val: pd.DataFrame, regime: str, cache: str = CACHE):
    """Build (queries, library) for a local dev set in the given regime."""
    truth = val.groupby("molecule_id").key14.first()
    ref = reference_table(cache, instrument="all")
    if regime == "disjoint":
        ref = ref[~ref.key14.isin(set(truth.values))]
    ref = ref[~ref.row.isin(set(val.row.tolist()))].reset_index(drop=True)

    store = PeakStore.load("train", cache)
    meta = pd.read_parquet(f"{cache}/train_meta.parquet",
                           columns=["precursor_mz", "adduct"])
    q_rows = val.row.to_numpy()
    queries = SpectraSet(
        store=store, rows=q_rows,
        precursor_mz=meta.precursor_mz.to_numpy()[q_rows],
        adduct=meta.adduct.to_numpy()[q_rows],
        group=val.molecule_id.to_numpy())
    library = SpectraSet(
        store=store, rows=ref.row.to_numpy(),
        precursor_mz=ref.precursor_mz.to_numpy(),
        adduct=ref.adduct.to_numpy(), key14=ref.key14.to_numpy())
    return queries, library, truth


def compute_features(queries: SpectraSet, library: SpectraSet,
                     pool: CandidatePool, cfg: ChannelConfig,
                     truth: pd.Series | None = None,
                     regime: str = "test", log=print) -> pd.DataFrame:
    """One row per (molecule, candidate), with both channels' features.

    `truth` (molecule_id -> key14) adds a 0/1 `label`; omit it at inference.
    """
    row_of_key: dict[str, int] = {}
    for r, k in enumerate(pool.key14):
        if k not in row_of_key:
            row_of_key[k] = r

    lib_store = library.store
    lib_mz = np.asarray(lib_store.mz)
    lib_in = np.asarray(lib_store.inten)
    lib_nm = neutral_masses(library.precursor_mz, library.adduct)
    nm_order = np.argsort(np.where(np.isfinite(lib_nm), lib_nm, np.inf), kind="stable")
    nm_sorted = lib_nm[nm_order]
    log(f"[{regime}] library {len(library):,} spectra / "
        f"{len(set(library.key14)):,} structures")

    q_store = queries.store
    q_mz = np.asarray(q_store.mz)
    q_in = np.asarray(q_store.inten)
    q_nm = neutral_masses(queries.precursor_mz, queries.adduct)
    q_group = queries.group

    # ---- channel 2: binned-cosine prefilter over the whole library ----
    pre: list[dict] = [dict() for _ in range(len(queries))]
    for nl in (False, True):
        R = bin_matrix(lib_store, library.rows, library.precursor_mz, neutral_loss=nl)
        Q = bin_matrix(q_store, queries.rows, queries.precursor_mz, neutral_loss=nl)
        sc, idx = search_cpu(R, Q, topk=cfg.prefilter_k)
        for qi in range(len(queries)):
            d = pre[qi]
            for x, v in zip(idx[qi], sc[qi]):
                if x >= 0 and v > d.get(int(x), -1.0):
                    d[int(x)] = float(v)
        del R, Q
        log(f"[{regime}]   prefiltered {'neutral-loss' if nl else 'direct'}")

    analog_by_mol: dict[str, dict[str, float]] = {}
    for qi in range(len(queries)):
        tgt = analog_by_mol.setdefault(q_group[qi], {})
        for ref_i, sim in pre[qi].items():
            k = library.key14[ref_i]
            if sim > tgt.get(k, -1.0):
                tgt[k] = sim
    for m in list(analog_by_mol):
        d = analog_by_mol[m]
        if len(d) > cfg.top_analogs:
            analog_by_mol[m] = dict(
                sorted(d.items(), key=lambda kv: -kv[1])[: cfg.top_analogs])

    # ---- channel 1: entropy similarity over mass-matched library spectra ----
    direct_by_mol: dict[str, dict[str, float]] = {}
    for qi in range(len(queries)):
        if not np.isfinite(q_nm[qi]):
            continue
        r = int(queries.rows[qi])
        qm, qp = prepare(q_mz[q_store.off[r]:q_store.off[r] + q_store.n[r]],
                         q_in[q_store.off[r]:q_store.off[r] + q_store.n[r]],
                         cfg.min_rel, cfg.max_peaks, cfg.entropy_weight)
        if qm.shape[0] == 0:
            continue
        tol = q_nm[qi] * cfg.ppm * 1e-6
        lo = np.searchsorted(nm_sorted, q_nm[qi] - tol)
        hi = np.searchsorted(nm_sorted, q_nm[qi] + tol)
        dc = nm_order[lo:hi]
        if dc.size == 0:
            continue
        sims = score_against(qm, qp, library.rows[dc].astype(np.int64),
                             lib_store.off, lib_store.n, lib_mz, lib_in, cfg.tol,
                             cfg.min_rel, cfg.max_peaks, cfg.entropy_weight,
                             np.zeros(dc.size, np.float32))
        tgt = direct_by_mol.setdefault(q_group[qi], {})
        for k, v in zip(library.key14[dc], sims):
            v = float(v)
            if v > tgt.get(k, -1.0):
                tgt[k] = v
    log(f"[{regime}]   channels done")

    # ---- assemble per-candidate rows ----
    order = pd.unique(q_group)
    n_spectra = pd.Series(q_group).value_counts()
    first_qi = {m: i for i, m in reversed(list(enumerate(q_group)))}
    frames = []
    for m in order:
        qi = first_qi[m]
        try:
            nm = precursor_to_neutral_mass(float(queries.precursor_mz[qi]),
                                           str(queries.adduct[qi]))
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
        ppm_err = (((pool.mass[cr] - nm) / nm * 1e6) if np.isfinite(nm)
                   else np.zeros(len(cr)))

        frames.append(pd.DataFrame({
            "molecule_id": m,
            "key14": keys,
            "label": ((keys == truth[m]).astype(np.int8) if truth is not None
                      and m in truth.index else np.zeros(len(cr), np.int8)),
            "c1": c1,
            "c1_best": c1.max() if len(c1) else 0.0,
            "c1_margin": c1 - (c1.max() if len(c1) else 0.0),
            "c1_rank": pd.Series(-c1).rank(method="min").to_numpy(),
            "c1_nonzero_frac": float((c1 > 0).mean()) if len(c1) else 0.0,
            "c2": c2,
            "c2_best": c2.max() if len(c2) else 0.0,
            "c2_margin": c2 - (c2.max() if len(c2) else 0.0),
            "c2_rank": pd.Series(-c2).rank(method="min").to_numpy(),
            "c2_mean_top5": c2_mean5,
            "tanimoto_best_analog": tan_best,
            "analog_sim_of_best": ana_sim,
            "ppm_err": np.asarray(ppm_err, np.float32),
            "abs_ppm_err": np.abs(np.asarray(ppm_err, np.float32)),
            "n_candidates": len(cr),
            "n_query_spectra": int(n_spectra[m]),
            "n_analogs": len(analogs),
            "regime": regime,
        }))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log(f"[{regime}] features: {len(out):,} rows over "
        f"{out.molecule_id.nunique() if len(out) else 0} molecules")
    return out


FEATURES = [
    "c1", "c1_best", "c1_margin", "c1_rank", "c1_nonzero_frac",
    "c2", "c2_best", "c2_margin", "c2_rank", "c2_mean_top5",
    "tanimoto_best_analog", "analog_sim_of_best",
    "ppm_err", "abs_ppm_err", "n_candidates", "n_query_spectra", "n_analogs",
]

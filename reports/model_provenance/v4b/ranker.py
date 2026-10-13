"""LightGBM reranker: training with query-grouped CV, MRR reporting, bagged inference."""
from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd


def mrr_at(pred: np.ndarray, y: np.ndarray, g: np.ndarray, k: int = 25) -> np.ndarray:
    """Reciprocal rank per group (0 if truth absent or beyond k). Groups must be contiguous."""
    order = np.lexsort((-pred, g))
    gs = g[order]; ys = y[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], len(gs)]
    out = np.zeros(len(starts))
    for i, (s, e) in enumerate(zip(starts, ends)):
        pos = np.flatnonzero(ys[s:e] > 0)
        if len(pos) and pos[0] < k:
            out[i] = 1.0 / (pos[0] + 1)
    return out


def group_sizes(g: np.ndarray) -> np.ndarray:
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    return np.diff(np.r_[starts, len(g)])


DEFAULT_PARAMS = dict(
    objective='lambdarank', metric='None', learning_rate=0.03, num_leaves=63, min_data_in_leaf=40,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_depth=-1,
    lambdarank_truncation_level=30, label_gain=[0, 1], verbose=-1, num_threads=20,
)


class RankerBag:
    def __init__(self, features: Sequence[str], params: Optional[dict] = None, rounds: int = 600):
        self.features = list(features); self.params = dict(DEFAULT_PARAMS); self.boosters: List[lgb.Booster] = []
        if params:
            self.params.update(params)
        self.rounds = rounds

    def fit_one(self, X, y, g, w=None, seed=0, valid=None, rounds=None):
        p = dict(self.params); p['seed'] = seed; p['bagging_seed'] = seed; p['feature_fraction_seed'] = seed
        ds = lgb.Dataset(X, label=y, group=group_sizes(g), weight=w, feature_name=self.features, free_raw_data=False)
        if p['objective'] != 'lambdarank':
            ds = lgb.Dataset(X, label=y, weight=w, feature_name=self.features, free_raw_data=False)
        b = lgb.train(p, ds, num_boost_round=rounds or self.rounds)
        return b

    def fit(self, X, y, g, w=None, seeds=(0, 1, 2, 3)):
        self.boosters = [self.fit_one(X, y, g, w, seed=s) for s in seeds]
        return self

    def predict(self, X) -> np.ndarray:
        return np.mean([b.predict(X, num_threads=8) for b in self.boosters], axis=0)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(features=self.features, params=self.params, rounds=self.rounds,
                             boosters=[b.model_to_string() for b in self.boosters]), f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        r = cls(d['features'], d['params'], d['rounds'])
        r.boosters = [lgb.Booster(model_str=s) for s in d['boosters']]
        return r


def cv_report(X, y, g, meta: pd.DataFrame, features, params=None, rounds=600, w=None, nfold=5,
              seeds=(0,), weight_fn=None, verbose=True) -> Dict[str, float]:
    """Structure-grouped CV. meta has one row per group (gid, sid, fold, regime, ...), aligned with
    unique(g). Returns dict of MRRs per (regime, fold) plus a calibrated LB estimate."""
    gids = meta.gid.values
    sid = meta.sid.values
    usid = np.unique(sid)
    rng = np.random.default_rng(0)
    fold_of_sid = dict(zip(usid, rng.integers(0, nfold, len(usid))))
    gfold = np.array([fold_of_sid[s] for s in sid])
    row_fold = gfold[np.searchsorted(gids, g)]
    pred = np.zeros(len(y))
    for f in range(nfold):
        tr = row_fold != f; te = ~tr
        rb = RankerBag(features, params, rounds)
        ws = None if w is None else w[tr]
        rb.fit(X[tr], y[tr], g[tr], ws, seeds=seeds)
        pred[te] = rb.predict(X[te])
        if verbose:
            print(f'    fold {f} done', flush=True)
    rr = mrr_at(pred, y, g)
    meta = meta.copy(); meta['rr'] = rr
    out = {}
    for (reg, fo), grp in meta.groupby(['regime', 'fold']):
        out[f'{reg}_{fo}'] = float(grp.rr.mean())
    for reg, grp in meta.groupby('regime'):
        out[f'{reg}_all'] = float(grp.rr.mean())
    return out, pred

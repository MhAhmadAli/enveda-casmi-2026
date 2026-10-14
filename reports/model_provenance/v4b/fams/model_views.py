"""F1 model views: per-model f.z (A, B, D=DreamsFP), their agreement, per-spectrum consensus, per-polarity views and
per-fingerprint-block f.z of the A+B logits. All order-invariant (average-tie ranks, max-relative gaps)."""
import numpy as np
from ctxlib import _rank_norm, _z

BLOCKS = [("ecfp4", 4096), ("ecfp6", 4096), ("fcfp4", 2048), ("rdk6", 2048), ("ap", 2048), ("tt", 2048), ("maccs", 167)]
_BLK = {}


def prepare(res):
    bits = res.P.bits
    edges = np.cumsum([0] + [n for _, n in BLOCKS])
    _BLK['id'] = np.searchsorted(edges, bits, side='right') - 1


def compute(v):
    out = {}
    n = v.n
    ranks = []
    for m, nm in enumerate(('A', 'B', 'D')):
        if m >= v.z_s.shape[0]:
            continue
        fz = v.fz_model(m, 'avg').astype(np.float64)
        out[f'fz{nm}_gap'] = ((fz - fz.max()) / 100.0).astype(np.float32)
        r = _rank_norm(fz); out[f'fz{nm}_rank'] = r.astype(np.float32); ranks.append(r)
        out[f'fz{nm}_z'] = _z(fz)
    R = np.stack(ranks)
    out['rank_spread'] = (R.max(0) - R.min(0)).astype(np.float32)
    out['rank_worst'] = R.max(0).astype(np.float32)
    # A+B+D ensemble (equal weights)
    fe = np.mean([v.fz_model(m, 'avg').astype(np.float64) for m in range(R.shape[0])], 0)
    out['fzABD_gap'] = ((fe - fe.max()) / 100.0).astype(np.float32); out['fzABD_rank'] = _rank_norm(fe).astype(np.float32)
    # per-spectrum consensus (A+B mean logits of each query spectrum)
    c64 = v.fp.astype(np.float64)
    zsp = v.z_s[:2].astype(np.float64).mean(0)                      # (n_s, nbits)
    if zsp.shape[0]:
        S = c64 @ zsp.T                                               # (n, n_s)
        RS = np.stack([_rank_norm(S[:, j]) for j in range(S.shape[1])], 1)
        pos = np.stack([1 + (S[:, j][None, :] > S[:, j][:, None]).sum(1) for j in range(S.shape[1])], 1) if n <= 3000 else None
        out['sp_rank_mean'] = RS.mean(1).astype(np.float32); out['sp_rank_min'] = RS.min(1).astype(np.float32)
        out['sp_rank_max'] = RS.max(1).astype(np.float32)
        out['sp_top3_frac'] = ((pos <= 3).mean(1) if pos is not None else np.zeros(n)).astype(np.float32)
        gap = S - S.max(0, keepdims=True)
        out['sp_gap_mean'] = (gap.mean(1) / 100.0).astype(np.float32)
    else:
        for k in ('sp_rank_mean', 'sp_rank_min', 'sp_rank_max', 'sp_top3_frac', 'sp_gap_mean'):
            out[k] = np.zeros(n, np.float32)
    # per-polarity merged view (A+B); missing polarity -> 0.5 rank / 0 gap
    for md, nm in ((1, 'pos'), (-1, 'neg')):
        j = np.flatnonzero(v.m_modes == md)
        if len(j):
            zm = v.z_m[:2, j[0]].astype(np.float64).mean(0)
            f = c64 @ zm
            out[f'pol_{nm}_rank'] = _rank_norm(f).astype(np.float32); out[f'pol_{nm}_gap'] = ((f - f.max()) / 100.0).astype(np.float32)
        else:
            out[f'pol_{nm}_rank'] = np.full(n, 0.5, np.float32); out[f'pol_{nm}_gap'] = np.zeros(n, np.float32)
    # per-block f.z (A+B, single+merged average == engine's z_avg)
    zs = v.z_s[:2].astype(np.float64).mean(0).mean(0) if v.z_s.shape[1] else np.zeros(v.fp.shape[1])
    zm = v.z_m[:2].astype(np.float64).mean(0).mean(0) if v.z_m.shape[1] else np.zeros(v.fp.shape[1])
    za = 0.5 * (zs + zm)
    bid = _BLK['id']
    for b, (bn, _) in enumerate(BLOCKS):
        sel = bid == b
        f = c64[:, sel] @ za[sel]
        out[f'blk_{bn}_gap'] = ((f - f.max()) / 100.0).astype(np.float32)
        out[f'blk_{bn}_rank'] = _rank_norm(f).astype(np.float32)
    return out

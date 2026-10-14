"""Notebook-side glue for the GLACIER isomer re-scorer (pure python; no torch / rdkit import). Companion of the
casmi26-iceberg fuse.py (whose build_ice_input / ice_candidates produce the runner input used here unchanged).

    from gl_fuse import run_gl, rerank_multi
    gl = run_gl(GL_PKG, items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=2400,
                site='/kaggle/working/ice_site', wheels=ICE_PKG + '/wheels')     # {mid: {smiles: score | None}}
    order = rerank_multi(smis, keys, scores, formulas, [ice.get(mid, {}), gl.get(mid, {})], [0.5, 0.5], top_n=60)

rerank_multi generalises fuse.rerank to several re-scorers: within a same-formula group among the first top_n
positions, fused = z(ranker score) + sum_k lam_k * z(score_k), z = (x - mean) / std with pandas semantics (sample std
ddof=1 over the non-missing members; std 0 / < 2 values -> z 0; missing -> z 0) -- the panel fusion of the A1 / v4h
evidence (base z(A) + 0.5 z(ice) + 0.5 z(gl)). A group is re-ordered only if at least one re-scorer covers >=
min_covered of its members; members are put back into the SAME positions (slot-preserving). With a single score
dict it is EXACTLY fuse.rerank (tests/test_gl_fuse.py).
"""
import os, sys, json, math, time, subprocess

__all__ = ['run_gl', 'rerank_multi']


# ----------------------------------------------------------------------------------------------------- run -------
def run_gl(pkg_dir, items, workdir='/tmp/gl_work', device='cuda', budget_s=2400.0, site='/tmp/ice_site',
           wheels=None, python=None, extra_args=(), grace_s=300.0, log=print):
    """= fuse.run_ice for gl_runner.py: returns {mid: {smiles: score | None}} ({} on total failure). Never raises.
    The runner stops at budget_s itself (checkpointing out.json after every chunk); the subprocess is killed at
    budget_s + grace_s as a last resort and whatever out.json holds is used."""
    t0 = time.time()
    try:
        os.makedirs(workdir, exist_ok=True)
        fin, fout = os.path.join(workdir, 'gl_in.json'), os.path.join(workdir, 'gl_out.json')
        for f in (fout, fout + '.meta.json'):
            if os.path.exists(f):
                os.remove(f)
        with open(fin, 'w') as f:
            json.dump(items, f)
        cmd = [python or sys.executable, os.path.join(pkg_dir, 'gl_runner.py'), pkg_dir, fin, fout,
               '--device', device, '--budget', str(float(budget_s)), '--site', site] + \
              (['--wheels', wheels] if wheels else []) + list(extra_args)
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               timeout=budget_s + grace_s if budget_s > 0 else None)
            tail = r.stdout[-3000:]
        except subprocess.TimeoutExpired as e:
            tail = f'TIMEOUT after {time.time() - t0:.0f}s ' + str(e.stdout or '')[-2000:]
        if log:
            log(tail)
        out = json.load(open(fout)) if os.path.exists(fout) else {}
        if log and os.path.exists(fout + '.meta.json'):
            m = json.load(open(fout + '.meta.json'))
            log({k: m.get(k) for k in ('status', 'n_mols', 'n_mols_covered', 'n_mols_scored', 'n_cands', 'n_scored',
                                        'seconds', 'pred_per_s', 'device', 'rdkit', 'max_mem_gb')})
        return out
    except Exception as e:                                         # never crash the notebook
        if log:
            log(f'run_gl failed: {e!r}')
        return {}


# -------------------------------------------------------------------------------------------------- rerank -------
def _z(vals):
    """= fuse._z: pandas groupby-transform z: (x - mean) / std(ddof=1) over non-missing; missing / std 0 / n < 2 -> 0."""
    ok = [v is not None and isinstance(v, (int, float)) and math.isfinite(v) for v in vals]
    xs = [float(v) for v, o in zip(vals, ok) if o]
    n = len(xs)
    if n < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    if not sd > 0:
        return [0.0] * len(vals)
    return [((float(v) - mu) / sd) if o else 0.0 for v, o in zip(vals, ok)]


def _getter(smis, keys, sc):
    if isinstance(sc, dict):
        def of(i):
            v = sc.get(smis[i])
            if v is None and keys is not None:
                v = sc.get(keys[i])
            return v
    else:
        def of(i):
            return sc[i] if sc is not None and i < len(sc) else None
    return of


def _clean(v):
    try:
        return float(v) if v is not None and math.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return None


def rerank_multi(smis, keys, scores, formulas, score_dicts, lams, top_n=60, min_covered=2, tie_eps=1e-9):
    """New order (list of indices into smis) after re-scoring same-formula groups inside the top_n with several
    re-scorers. smis / keys / scores / formulas: aligned lists in the CURRENT ranked order (index 0 = best); scores =
    ranker score (higher = better). score_dicts: list of {smiles: score | None} (a key from `keys` is also accepted)
    or lists aligned with smis (None / {} = re-scorer absent); lams: one weight per re-scorer. For each formula group
    among the first top_n positions with >= 2 members of which >= min_covered are covered by at least ONE re-scorer,
    the members are re-sorted by z(score) + sum_k lam_k * z(score_k) (all z within the group; uncovered -> z 0;
    stable on ties) and put back into the SAME positions. Everything else stays put.
    Ties: when >= 2 re-scorers contribute to a group, fused values are compared on a tie_eps grid, so exact-arithmetic
    ties that float rounding would break at random (e.g. a 2-member group where ice and gl both oppose the ranker:
    +-0.7071 * (1 - 0.5 - 0.5) = 0 for both) keep the current order ("stable on ties"); tie_eps=0 disables this.
    One contributing re-scorer -> identical arithmetic to fuse.rerank, so with a single score dict the result is
    exactly fuse.rerank(smis, keys, scores, formulas, score_dicts[0], lams[0], top_n, min_covered)."""
    n = len(smis)
    order = list(range(n))
    if len(score_dicts) != len(lams):
        raise ValueError('one lam per score dict')
    use = [(_getter(smis, keys, sc), float(l)) for sc, l in zip(score_dicts, lams) if sc]
    if n == 0 or not use:
        return order
    m = min(int(top_n), n)
    groups = {}
    for i in range(m):
        groups.setdefault(formulas[i], []).append(i)
    for f, idx in groups.items():
        if len(idx) < 2:
            continue
        cols = []
        for of, lam in use:
            v = [_clean(of(i)) for i in idx]
            cols.append((v, lam, sum(x is not None for x in v)))
        if not any(c >= max(2, min_covered) for _, _, c in cols):
            continue
        fused = _z([scores[i] for i in idx])
        n_act = 0
        for v, lam, c in cols:
            if c >= 2:                                   # n < 2 -> z 0 (fuse._z); skip the no-op addition
                zi = _z(v)
                fused = [a + lam * b for a, b in zip(fused, zi)]
                n_act += 1
        if n_act >= 2 and tie_eps > 0:
            fused = [round(x / tie_eps) for x in fused]  # monotone: never reverses an order, only merges float noise
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):                  # idx is ascending = the group's slots
            order[slot] = src
    return order

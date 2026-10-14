"""Notebook-side glue for the ICEBERG isomer re-scorer (pure python + numpy; no torch / rdkit import).

    from fuse import build_ice_input, ice_candidates, run_ice, rerank
    items = build_ice_input(test_df, {mid: ice_candidates(smis, formulas, N), ...})   # test rows -> runner input
    ice = run_ice(PKG, items, budget_s=2700)                             # {mid: {smiles: score | None}}
    order = rerank(smis, keys, scores, formulas, ice.get(mid, {}), lam=0.5, top_n=N)
    smis = [smis[i] for i in order]                                      # (same permutation for keys / scores)

rerank = the panel fusion (work/analysis/fwd/panel_lib.py::fuse, used for the +0.027 syn / +0.013 nplib / +0.055 np
evidence): within a same-formula group, fused = z(ranker score) + lam * z(ice), z = (x - mean) / std with pandas
semantics (sample std ddof=1 over the non-missing members; std 0 / < 2 values -> z 0; missing -> z 0).
"""
import os, sys, json, math, time, subprocess

__all__ = ['build_ice_input', 'ice_candidates', 'run_ice', 'rerank']


# --------------------------------------------------------------------------------------------------- input -------
def _flist(x):
    if x is None:
        return None
    try:
        return [float(v) for v in x]
    except TypeError:
        return None


def _str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return str(x)


def build_ice_input(test_df, cands_by_mid, covered_only=True):
    """test.parquet rows (molecule_id, ms2_mzs, ms2_normalized_intensities, precursor_mz, adduct, ionization_mode,
    instrument_type, collision_energy_ev, collision_energy_orig, collision_energy_orig_units) + {mid: [smiles]}
    -> list of runner input items (in the order of cands_by_mid: put the molecules you care most about first, the
    runner stops at its time budget). covered_only drops spectra the runner would skip anyway (smaller JSON)."""
    by = {}
    for r in test_df.itertuples(index=False):
        by.setdefault(str(r.molecule_id), []).append(r)
    items = []
    g = lambda r, k: getattr(r, k, None)                    # optional columns may be absent -> None
    for mid, cands in cands_by_mid.items():
        spectra = []
        for r in by.get(str(mid), []):
            try:                                             # a malformed row drops that spectrum, never raises
                sp = dict(mz=_flist(r.ms2_mzs) or [], it=_flist(r.ms2_normalized_intensities) or [],
                          prec=float(r.precursor_mz), adduct=_str(g(r, 'adduct')), mode=_str(g(r, 'ionization_mode')),
                          instrument=_str(g(r, 'instrument_type')), ce_ev=_flist(g(r, 'collision_energy_ev')),
                          ce_orig=_str(g(r, 'collision_energy_orig')), ce_units=_str(g(r, 'collision_energy_orig_units')))
            except Exception:
                continue
            if covered_only and not (sp['mode'] == 'positive' and sp['adduct'] in ('[M+H]+', '[M+Na]+')):
                continue
            spectra.append(sp)
        items.append(dict(mid=str(mid), spectra=spectra, cands=[str(c) for c in cands]))
    return items


# ----------------------------------------------------------------------------------------------------- run -------
def run_ice(pkg_dir, items, workdir='/tmp/ice_work', device='cuda', budget_s=2700.0,
            site='/tmp/ice_site', python=None, extra_args=(), grace_s=300.0, log=print):
    """Run ice_runner.py as a subprocess; returns {mid: {smiles: score | None}} ({} on total failure).
    Never raises. The runner itself stops at budget_s (checkpointing out.json after every chunk); the subprocess is
    killed at budget_s + grace_s as a last resort and whatever out.json holds is used."""
    t0 = time.time()
    try:
        os.makedirs(workdir, exist_ok=True)
        fin, fout = os.path.join(workdir, 'ice_in.json'), os.path.join(workdir, 'ice_out.json')
        for f in (fout, fout + '.meta.json'):
            if os.path.exists(f):
                os.remove(f)
        with open(fin, 'w') as f:
            json.dump(items, f)
        cmd = [python or sys.executable, os.path.join(pkg_dir, 'ice_runner.py'), pkg_dir, fin, fout,
               '--device', device, '--budget', str(float(budget_s)), '--site', site] + list(extra_args)
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
            log(f'run_ice failed: {e!r}')
        return {}


def ice_candidates(smis, formulas, top_n=40):
    """SMILES worth predicting for one molecule: members of same-formula groups with >= 2 members inside the top_n
    (the only candidates rerank can move). Keeps ranked order; duplicates removed."""
    m = min(int(top_n), len(smis))
    cnt = {}
    for f in formulas[:m]:
        cnt[f] = cnt.get(f, 0) + 1
    return list(dict.fromkeys(s for s, f in zip(smis[:m], formulas[:m]) if cnt[f] >= 2))


# -------------------------------------------------------------------------------------------------- rerank -------
def _z(vals):
    """pandas groupby-transform z: (x - mean) / std(ddof=1) over non-missing; missing / std 0 / n < 2 -> 0."""
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


def rerank(smis, keys, scores, formulas, ice_scores, lam=0.5, top_n=60, min_covered=2):
    """New order (list of indices into smis) after ICEBERG re-scoring of same-formula groups inside the top_n.

    smis / keys / scores / formulas: aligned lists in the CURRENT ranked order (index 0 = best); scores = ranker
    score (higher = better). ice_scores: {smiles: score | None} (runner output for this molecule; a key from `keys`
    is also accepted) or a list aligned with smis. For each formula group among the first top_n positions with
    >= min_covered covered members, the group's members are re-sorted by z(score) + lam * z(ice) (both z within the
    group; uncovered -> z 0; stable on ties) and put back into the SAME positions. Everything else stays put."""
    n = len(smis)
    order = list(range(n))
    if n == 0 or not ice_scores:
        return order
    m = min(int(top_n), n)
    if isinstance(ice_scores, dict):
        def ice_of(i):
            v = ice_scores.get(smis[i])
            if v is None and keys is not None:
                v = ice_scores.get(keys[i])
            return v
    else:
        def ice_of(i):
            return ice_scores[i] if i < len(ice_scores) else None
    groups = {}
    for i in range(m):
        groups.setdefault(formulas[i], []).append(i)
    for f, idx in groups.items():
        if len(idx) < 2:
            continue
        ice = [ice_of(i) for i in idx]
        ice = [float(v) if v is not None and math.isfinite(float(v)) else None for v in ice]
        if sum(v is not None for v in ice) < max(2, min_covered):
            continue
        zs = _z([scores[i] for i in idx]); zi = _z(ice)
        fused = [a + lam * b for a, b in zip(zs, zi)]
        new = [idx[k] for k in sorted(range(len(idx)), key=lambda k: (-fused[k], k))]
        for slot, src in zip(idx, new):                      # idx is ascending = the group's slots
            order[slot] = src
    return order

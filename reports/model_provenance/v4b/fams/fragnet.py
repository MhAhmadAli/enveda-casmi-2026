"""fragnet: LEARNED FRAGMENTATION SCORER ("learned MetFrag", ICEBERG-lite without a GPU).

DECISION LOG (kept here so a restarted session can continue; newest last)
  * attempt 1 start (2026-09-24 ~14:40): no previous fragnet files. Framework read (ctxlib, frag2, evalfe, gate).
  * style check: train vs COCONUT SMILES of the same key differ for 2,576 / 25,746 shared keys; causes (400 sampled):
    imidic acid 'C(O)=N' vs amide ~48%, aromatic hydroxy-N 'c(O)n' vs 'c(=O)[nH]' ~20%, amidine / 1,5 shifts / other
    ~30%. RDKit TautomerEnumerator.Canonicalize: 92% of differing pairs converge, 10-15 ms/mol (too slow on the fly
    for generated candidates: mean n_gen 20/query); InChI round trip: 78%, 1 ms/mol.
    -> style-invariant record WITHOUT tautomer enumeration: element graph (no bond orders), carbon H as parsed,
    aromatic / ring flags, heteroatom H re-assigned canonically inside mobile-H groups (heteroatoms linked 1,3 via an
    unsaturated carbon, 1,2 aromatic X-X, 1,5 via three unsaturated carbons; group H total kept; N first, then larger
    capacity, O on aromatic C, then a WL invariant).
    Check (all 2,576 differing train/COCONUT pairs): multiset of (element, canonical H, atom kind) + bond ring types
    identical for 86.4% (raw (element, H) multiset: 17.6%); residual = keto-enol (carbon H moves), aromaticity changes
    of fused N-heterocycle tautomers, isotopes. Anionic O/S/N neutralised (+H). Record 0.9 ms/mol (RDKit parse).
  * fragments (enum_frags): 1-bond cuts (both sides) + depth-2 fragments whose boundary is exactly the 2 cut bonds
    (ring opening = 2 ring bonds, middle piece = 2 chain cuts); mass set == engine frag.fragment_masses depth 2 when
    canonical H == SMILES H (checked). Annotation per cut: directional class (kind inside, kind outside, chain / ring /
    aromatic ring) -> c1 (, c2); plus fragment heavy atoms, N / O counts. ~300 fragments/mol, 0.4 ms/mol.
  * match classes per fragment (spectrum): unobservable (ion outside [max(50, min peak - .5), prec - 2]) / none / s=0 /
    s=-1 / s=+1 / s=-2 / s=+2 (tol max(0.01 Da, 15 ppm), any ion delta of the adduct).
  * EXPLORATION (scratch, train-fold library spectra, truth + <=15 train-derived + <=15 COCONUT same-formula decoys,
    fit on 4,000 structures, eval on 4,000 other structures; within-formula-group MRR, ~21 candidates, random 0.11):
    explained intensity (== fr_int) 0.486, depth-1 cleavage coverage 0.559, count-based LLR by key (mode, ftype) ->
    c1 -> c2 with Dirichlet backoff alpha 80: fragment-centric sum fc 0.691, peak-centric best-explanation pk 0.693,
    pk + 0.02 * unmatched-fragment LLR 0.714. Deeper chains (+size, +frac) and smaller alpha were worse (sparsity).
    Pairwise win rate vs train-derived / COCONUT decoys: expl 0.794 / 0.834, pk_fcu 0.858 / 0.855 -> the gain is
    larger against TRAIN-derived decoys (no provenance signature).
  * pool record table res/fragnet/rec_* (710,701; ok 99.87%, 0.13% exotic elements computed on demand; 216 MB).
  * training sets (res_fragnet --step sets, seed 0): 32,000 train-fold structures x 1 library spectrum (NP libs x2,
    timsTOF x2 sampling weight), decoys mean 23.3 (same formula <= 15 + 15, other formula +-10 ppm <= 3 + 3; decoy
    src 60% train-derived / 40% COCONUT -> stage A reweights to 50/50); parts A 16,144 / B 12,751 / V 3,105.
  * stage A (--step countA, 60 s with 4 procs): class counts per key chain on part A (truth 2.0M observable fragment
    rows, decoys 33M weighted). match-class shares truth / decoy: none .857/.901, s=0 .064/.036, s=-1 .030/.020.
  * matching fix: peak-centric scores must use EVERY (fragment, peak, shift) match (first version kept only each
    fragment's best peak -> 'explained intensity' 0.53 instead of fr_int 0.65); match_frags now returns the full list.
  * V (3,103 held-out structures, within-formula MRR; config comparison on raw stage-A scores): pk_fcu (peak-centric
    best-explanation LLR + 0.02 x unmatched-fragment LLR): A80 .747, A300 .748, E80 (+CE bin) .733, AG80 .745, AH80
    (chain A + chain (mode, ftype) -> size -> mass fraction) .755 <- chosen (stageA.npz: chains A, H, alpha 80; tiny).
    df-weighted (discriminative) peak sums did not help (.735).
  * stage B (--step fitB): LightGBM lambdarank on 31 stage-B features (19 aggregates + 4 ratios + within-group rank /
    gap of pk, fc, pk_fcu, expl), leaves 15, min leaf 200, lr .03, 300 rounds (V plateau .770-.775 for 100-900 rounds
    and 7-31 leaves; +E80 features +.003 = noise -> not used). MAP-based early stopping stopped at 1 round (metric
    artefact) -> fixed rounds.
    V within-formula MRR: fr_int .6465, frag2 mix_disc (group df) .7022, stage-A pk_fcu .7549, FRAGNET .7712
    (pos .777 / neg .744). Win rate vs train-derived / COCONUT decoys: fr_int .855/.874, mix .875/.892, fragnet
    .896/.906 -> gain over fr_int +.041 train / +.032 COCONUT (no COCONUT-specific gain).
  * family (compute): frag set = top 400 by the A+B f.z (tie-inclusive; others get sentinels); records from the pool
    table / generated SMILES (record + fragments cached by SMILES); matching on the unique fragment masses of the
    whole candidate set (rounded 1e-6 Da; 2-3x faster, results unchanged on 400 queries); stage-B score per spectrum
    (relative features within the frag set of that spectrum) -> features (NAMES): fn_score (mean over spectra),
    fn_score_max, fn_rank, fn_gap, fn_grp_rank / fn_grp_gap (formula group), fn_vs_lead (vs the f.z lead, ties
    averaged), fn_raw_* (stage-A pk_fcu / total peak weight, mean over spectra: rank, gap, group gap, vs lead),
    fn_pk_rank, fn_fc_rank (peak-centric / fragment-centric stage-A), fn_sp_rank (mean per-spectrum stage-B rank).
  * unit test (scratch ut_fragnet.py, sim10_s2 queries, within-formula-group MRR incl. generated candidates):
    300 np queries: fn_score .278, fn_raw .245, frag2 mix_rank .226, fr_int .175, fz .550;
    400 random queries: fn_score .417, fn_raw .414, frag2 mix .400, fr_int .340, fz .599.
    Cost 1 core: mean 73 ms/query (without the first-call numba load), p50 37, p95 278, max ~0.9 s (150 large
    generated candidates, cold cache). build_feats: ~55 ms/query/process.
  * gate_fam 120 queries 0 failures; extra gate with the frag-set cap lowered to 30 (40 queries n >= 40): 0 failures.
  * cold-cache cost (generated cache cleared before every query, 300 random queries, loaded machine): mean 88 ms
    (excl. the one-off numba cache load ~8.6 s), p50 41, p95 455 ms. build_feats --procs 2: 23,748 queries, 398 files,
    ~45 min on the loaded machine (incl. view construction).
  * all sim10_s2 queries, within-formula-group MRR (pool candidates of the truth formula, pool truths): np fn_score
    .401 / fn_raw .370 / frag2 mix_rank .335 / fr_int .296 / fz .639; nplib .668 / .650 / .594 / .541 / .739.
  * per-feature provenance probe (probe_fragnet_prov.py, C2FAM 14,445 groups; share of decoys below the truth, train-
    derived / COCONUT): fn_score .926 / .851 (same-formula .909 / .815), fr_int .852 / .769 (.836 / .724), frag2
    mix_rank .866 / .788, fz .964 / .950. Like every fragmentation feature, fragnet separates COCONUT decoys LESS well
    than train decoys; gain over fr_int +.073 train / +.082 COCONUT (np: +.105 / +.106) -> no provenance signature.
  * rec_bond stored as uint8 (max 238 atoms in the pool; > 255 atoms -> on demand): tables 150 MB; table == on-the-fly
    records on 400 random pool entries (also checks cross-process determinism of the WL ordering).
  * rank-average check inside formula groups (np): fz alone .639, fz + 0.1 x fragnet rank .640, + 0.25 .633 (nplib
    .739 -> .748 at 0.25): simple averaging is nearly redundant with f.z; frag2 mix_rank hurts at every weight.
  * EVAL fragnet_v1 (all 14 feats, 18,465 groups, seed 0): ALL +.0063 [+.0033,+.0091], C2FAM +.0052 [+.0018,+.0086],
    C2FAM_wt +.0106 [+.0036,+.0176], C3FAM +.0271 [+.0163,+.0383], c1 +.0036 [-.0006,+.0075]; np ALL +.0163
    [+.0052,+.0284], np C2FAM +.0144 [-.0003,+.0289], np C2FAM_wt +.0148 [-.0063,+.0370], np C3FAM +.0391, np c1 +.0127.
    provenance tr/co .9876/.9561 -> .9880/.9600 (gain +.0004 / +.0039; provgap +.0035, the largest so far ->
    stratified check provstrat_fam.py: COCONUT decoys are more often isomers).
  * EVAL fragnet_v2_core (7 feats): ALL +.0050 [+.0019,+.0079], np ALL +.0099 [-.0007,+.0212] -> worse than v1 on every
    aggregate -> final = all 14 features. EVAL frag2_fragnet_v2 (frag2 RECOMMENDED 13 + fragnet core 7): ALL +.0082
    [+.0051,+.0111] (frag2 alone, frag2_final_v3: +.0045).
  * provstrat_fragnet_v1 (same-formula decoys): base already wins .9765 vs train-derived / .9424 vs COCONUT; fragnet gain
    +.0005 / +.0047. Reading: train-derived decoys own library spectra that the base's library / analog features already
    reject; COCONUT decoys (no spectra) are left to f.z + fragmentation -> the room for any fragmentation feature is on
    COCONUT decoys. The base-free per-feature probe shows no source asymmetry (+.073 / +.082 over fr_int).
  * 17:00 the final bag-3 chain (eval_fragnet_final.sh) died with the previous session (stale lock pid 27188, only the
    header of eval_fragnet_final_b3.log); probe_fragnet_style.py written but never run.
  * attempt 1 resumed 17:20: relaunch eval_fragnet_final.sh (evalfe removes the stale lock), run the style probe.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), 'res', 'fragnet')
H_MASS = 1.00782503207                 # == casmi.chem.MASS['H'] == frag.H_MASS (asserted in prepare)

# ------------------------------------------------------------------------------------------------ chemistry record
# element codes / atom kinds (style-invariant: element graph, carbon H counts, aromatic / ring flags, canonical
# heteroatom H; never bond orders)
EL = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 5, 'Br': 5, 'I': 5}
KINDS = ['C_sp3', 'C_acetal', 'C_unsat', 'C_arom', 'C_acyl', 'N_arom', 'N', 'O_ring', 'O_arom', 'O_term', 'S', 'P',
         'X_other']
NK = len(KINDS)
_VAL = {'N': 3, 'O': 2, 'S': 2}
_ELPRIO = {'N': 0, 'S': 1, 'O': 2}
_RD = {}


def _rd():
    if not _RD:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        _RD['Chem'] = Chem
        from casmi import chem as cchem
        _RD['MASS'] = cchem.MASS
    return _RD


def _wl(el, hc, arom, nbrs, iters=5):
    """Weisfeiler-Lehman atom invariants of the style-invariant graph (element, carbon H, aromatic flag); used only to
    order heteroatoms inside a mobile-H group (ties = graph symmetry -> isomorphic results). Integer tuple hashes are
    deterministic across processes (no str hashing)."""
    n = len(el)
    lab = [hash((int(el[i]), int(hc[i]), int(arom[i]), len(nbrs[i]))) for i in range(n)]
    for _ in range(iters):
        lab = [hash((lab[i], tuple(sorted(lab[j] for j in nbrs[i])))) for i in range(n)]
    return lab


def mol_record(smiles: str):
    """Style-invariant fragmentation record of one structure, or None.
    dict(amass (n,) float64 atom mass incl. canonical H, kind (n,) int8, el (n,) int8, nh (n,) int8 canonical H,
    bonds (E, 2) int32, bring (E,) int8 0 chain / 1 ring / 2 aromatic ring, mass float)."""
    r = _rd(); Chem = r['Chem']; MASS = r['MASS']
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    atoms = list(m.GetAtoms())
    n = len(atoms)
    sym = [a.GetSymbol() for a in atoms]
    if n == 0 or any(s not in MASS for s in sym):
        return None
    H = np.array([a.GetTotalNumHs() for a in atoms], np.int64)
    deg = np.array([a.GetDegree() for a in atoms], np.int64)
    chg = np.array([a.GetFormalCharge() for a in atoms], np.int64)
    # neutralise anionic heteroatoms (carboxylate / phosphate / phenolate styles): +H, charge 0
    for i, s in enumerate(sym):
        if chg[i] == -1 and s in _VAL and H[i] + deg[i] < _VAL[s]:
            H[i] += 1; chg[i] = 0
    arom = np.array([a.GetIsAromatic() for a in atoms], np.int64)
    nbrs = [[x.GetIdx() for x in a.GetNeighbors()] for a in atoms]
    isC = np.array([s == 'C' for s in sym])
    unsat = isC & (4 - deg - H - np.abs(chg) >= 1)
    mob = np.array([(s in _VAL) and chg[i] == 0 for i, s in enumerate(sym)])
    # ---- mobile-H groups (union-find over heteroatoms)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            par[max(a, b)] = min(a, b)
    for c in range(n):                                   # 1,3: X-C-Y, C unsaturated
        if unsat[c]:
            xs = [x for x in nbrs[c] if mob[x]]
            for x in xs[1:]:
                union(xs[0], x)
    bl = []
    apar = list(range(n))                                # aromatic ring systems (components of aromatic bonds)

    def afind(x):
        while apar[x] != x:
            apar[x] = apar[apar[x]]; x = apar[x]
        return x
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bl.append((i, j, (2 if b.GetIsAromatic() else 1) if b.IsInRing() else 0))
        if mob[i] and mob[j] and arom[i] and arom[j]:    # 1,2 aromatic X-Y (azoles)
            union(i, j)
        if b.GetIsAromatic():
            a1, a2 = afind(i), afind(j)
            if a1 != a2:
                apar[max(a1, a2)] = min(a1, a2)
    first_n: Dict[int, int] = {}                          # annular tautomerism: aromatic N of one ring system
    for x in range(n):
        if mob[x] and arom[x] and sym[x] == 'N':
            k = afind(x)
            if k in first_n:
                union(first_n[k], x)
            else:
                first_n[k] = x
    for x in range(n):                                   # 1,5: X-C=C-C=Y
        if not mob[x]:
            continue
        for c1 in nbrs[x]:
            if not unsat[c1]:
                continue
            for c2 in nbrs[c1]:
                if not unsat[c2]:
                    continue
                for c3 in nbrs[c2]:
                    if c3 == c1 or not unsat[c3]:
                        continue
                    for y in nbrs[c3]:
                        if y != x and mob[y]:
                            union(x, y)
    hc = np.where(isC, H, 0)
    hcan = H.copy()
    groups: Dict[int, List[int]] = {}
    for x in range(n):
        if mob[x]:
            groups.setdefault(find(x), []).append(x)
    multi = [g for g in groups.values() if len(g) > 1]
    if multi:
        el_code = np.array([EL.get(s, 6) for s in sym])
        wl = _wl(el_code, hc, arom, nbrs)
        for g in multi:
            tot = int(H[g].sum())
            cap = {x: max(0, _VAL[sym[x]] - int(deg[x])) for x in g}
            key = lambda x: (_ELPRIO[sym[x]], -cap[x], 0 if any(arom[c] and isC[c] for c in nbrs[x]) else 1, wl[x])
            left = tot
            for x in sorted(g, key=key):
                h = min(cap[x], left); hcan[x] = h; left -= h
            if left > 0:                                 # capacity exceeded (unusual valence): keep the SMILES H
                for x in g:
                    hcan[x] = H[x]
    amass = np.array([MASS[s] for s in sym]) + hcan * MASS['H']
    # ---- atom kinds
    kind = np.full(n, 12, np.int8)
    term_o = [(sym[x] in ('O', 'S')) and deg[x] == 1 and hcan[x] == 0 and chg[x] == 0 for x in range(n)]
    for i, s in enumerate(sym):
        if s == 'C':
            if any(term_o[x] for x in nbrs[i]):
                kind[i] = 4
            elif arom[i]:
                kind[i] = 3
            elif unsat[i]:
                kind[i] = 2
            elif sum(1 for x in nbrs[i] if sym[x] == 'O') >= 2:
                kind[i] = 1
            else:
                kind[i] = 0
        elif s == 'N':
            kind[i] = 5 if arom[i] else 6
        elif s == 'O':
            kind[i] = 8 if arom[i] else (9 if deg[i] <= 1 else 7)
        elif s == 'S':
            kind[i] = 10
        elif s == 'P':
            kind[i] = 11
    bonds = np.array([(a, b) for a, b, _ in bl], np.int32).reshape(-1, 2)
    bring = np.array([t for _, _, t in bl], np.int8)
    el = np.array([EL.get(s, 6) for s in sym], np.int8)
    el10 = np.array([EL10.get(s, 9) for s in sym], np.uint8)
    return dict(amass=amass, kind=kind, el=el, nh=hcan.astype(np.int8), bonds=bonds, bring=bring,
                mass=float(amass.sum()), el10=el10)


# ------------------------------------------------------------------------------------------------ record table (pool)
EL10 = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8}
_EL10_SYM = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
_EL10_TO_EL = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 6], np.int8)


def pack_record(r):
    """-> (ok, atoms (n, 3) uint8 [el10, nh, kind], bonds (E, 3) uint8 [a, b, bring]); ok 0 = no record, 2 = element
    outside the table codes or > 255 atoms (computed from the SMILES on demand)."""
    if r is None:
        return 0, np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.uint8)
    if (r['el10'] == 9).any() or len(r['el10']) > 255:
        return 2, np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.uint8)
    A = np.stack([r['el10'], r['nh'].astype(np.uint8), r['kind'].astype(np.uint8)], 1).astype(np.uint8)
    B = np.concatenate([r['bonds'].astype(np.uint8), r['bring'].astype(np.uint8)[:, None]], 1) if len(r['bonds']) \
        else np.zeros((0, 3), np.uint8)
    return 1, A, B


def unpack_record(A, B):
    """Inverse of pack_record (ok == 1) -> the arrays enum_frags needs."""
    r = _rd(); MASS = r['MASS']
    if 'el10_mass' not in _RD:
        _RD['el10_mass'] = np.array([MASS[s] for s in _EL10_SYM] + [0.0])
    el10 = A[:, 0].astype(np.int64)
    amass = _RD['el10_mass'][el10] + A[:, 1].astype(np.float64) * MASS['H']
    return dict(amass=amass, kind=A[:, 2].astype(np.int8), el=_EL10_TO_EL[el10],
                bonds=B[:, :2].astype(np.int32).reshape(-1, 2), bring=B[:, 2].astype(np.int8), mass=float(amass.sum()))


# ------------------------------------------------------------------------------------------------ fragmentation kernels
MAX_FRAG = 30000
FT_D1, FT_RO, FT_MID = 1, 2, 3          # depth-1 cut / depth-2 ring opening / depth-2 middle piece (two chain cuts)


@njit(cache=True)
def _label2(n, start, nbr, eid, d1, d2, comp, stack):
    for i in range(n):
        comp[i] = -1
    ncomp = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        top = 0
        stack[top] = s; top += 1
        comp[s] = ncomp
        while top > 0:
            top -= 1
            u = stack[top]
            for q in range(start[u], start[u + 1]):
                e = eid[q]
                if e == d1 or e == d2:
                    continue
                x = nbr[q]
                if comp[x] < 0:
                    comp[x] = ncomp
                    stack[top] = x; top += 1
        ncomp += 1
    return ncomp


@njit(cache=True)
def enum_frags(amass, kind, el, bonds, bring, nk, max_out):
    """Annotated 1- and 2-bond cleavage fragments of one structure (parent excluded).
    A depth-2 fragment is a component whose boundary is exactly the two cut bonds (a ring opening = two ring bonds, or
    the middle piece between two chain cuts); components that need only one of the cuts are depth-1 fragments (listed
    once). Returns (mass f8, ftype i1, c1 i2, c2 i2 (-1), nheavy i2, nN i2, nO i2); cut class = ((kind_in * nk +
    kind_out) * 3 + ring type)."""
    n = len(amass); E = bonds.shape[0]
    fm = np.empty(max_out, np.float64); ft = np.empty(max_out, np.int8)
    c1 = np.empty(max_out, np.int16); c2 = np.empty(max_out, np.int16)
    nh = np.empty(max_out, np.int16); nN = np.empty(max_out, np.int16); nO = np.empty(max_out, np.int16)
    k = 0
    if E == 0:
        return fm[:0], ft[:0], c1[:0], c2[:0], nh[:0], nN[:0], nO[:0]
    start = np.zeros(n + 1, np.int64)
    for e in range(E):
        start[bonds[e, 0] + 1] += 1; start[bonds[e, 1] + 1] += 1
    for i in range(n):
        start[i + 1] += start[i]
    fill = start[:n].copy()
    nbr = np.empty(2 * E, np.int64); eid = np.empty(2 * E, np.int64)
    for e in range(E):
        a = bonds[e, 0]; b = bonds[e, 1]
        nbr[fill[a]] = b; eid[fill[a]] = e; fill[a] += 1
        nbr[fill[b]] = a; eid[fill[b]] = e; fill[b] += 1
    comp = np.empty(n, np.int32); stack = np.empty(n, np.int64)
    cm = np.empty(n, np.float64); ch = np.empty(n, np.int64); cn = np.empty(n, np.int64); co = np.empty(n, np.int64)
    for e1 in range(E):
        nc = _label2(n, start, nbr, eid, e1, -1, comp, stack)
        a1 = bonds[e1, 0]; b1 = bonds[e1, 1]
        if nc > 1:
            for c in range(nc):
                cm[c] = 0.0; ch[c] = 0; cn[c] = 0; co[c] = 0
            for i in range(n):
                c = comp[i]
                cm[c] += amass[i]; ch[c] += 1
                if el[i] == 1:
                    cn[c] += 1
                elif el[i] == 2:
                    co[c] += 1
            for c in range(nc):
                if k >= max_out:
                    break
                if comp[a1] == c:
                    ki = kind[a1]; ko = kind[b1]
                else:
                    ki = kind[b1]; ko = kind[a1]
                fm[k] = cm[c]; ft[k] = 1; c1[k] = (ki * nk + ko) * 3 + bring[e1]; c2[k] = -1
                nh[k] = ch[c]; nN[k] = cn[c]; nO[k] = co[c]
                k += 1
        for e2 in range(e1 + 1, E):
            if k >= max_out:
                break
            if (bring[e1] > 0) != (bring[e2] > 0):
                continue                                   # ring + chain cut: never a depth-2-only fragment
            nc2 = _label2(n, start, nbr, eid, e1, e2, comp, stack)
            if nc2 < 2:
                continue
            a2 = bonds[e2, 0]; b2 = bonds[e2, 1]
            for c in range(nc2):
                in1 = (comp[a1] == c) != (comp[b1] == c)
                in2 = (comp[a2] == c) != (comp[b2] == c)
                if not (in1 and in2):
                    continue
                s_m = 0.0; s_h = 0; s_n = 0; s_o = 0
                for i in range(n):
                    if comp[i] == c:
                        s_m += amass[i]; s_h += 1
                        if el[i] == 1:
                            s_n += 1
                        elif el[i] == 2:
                            s_o += 1
                if comp[a1] == c:
                    ki1 = kind[a1]; ko1 = kind[b1]
                else:
                    ki1 = kind[b1]; ko1 = kind[a1]
                if comp[a2] == c:
                    ki2 = kind[a2]; ko2 = kind[b2]
                else:
                    ki2 = kind[b2]; ko2 = kind[a2]
                x1 = (ki1 * nk + ko1) * 3 + bring[e1]; x2 = (ki2 * nk + ko2) * 3 + bring[e2]
                if k >= max_out:
                    break
                fm[k] = s_m; ft[k] = 2 if bring[e1] > 0 else 3
                c1[k] = min(x1, x2); c2[k] = max(x1, x2)
                nh[k] = s_h; nN[k] = s_n; nO[k] = s_o
                k += 1
    return fm[:k], ft[:k], c1[:k], c2[:k], nh[:k], nN[:k], nO[:k]


@njit(cache=True)
def match_frags(fm, pk_mz, pk_w, pk_off, sp_del, sp_nd, lo, hi, h, tol_da, tol_ppm):
    """Match every fragment against every spectrum (fragment-major loop -> the match list is ordered by fragment,
    then spectrum, shift, delta, peak: intrinsic to the candidate).
    Returns cls (F, S) int8 best class per fragment and spectrum (-1 unobservable: main ion outside [lo, hi]; 0 none;
    1 s=0; 2 s=-1; 3 s=+1; 4 s=-2; 5 s=+2; smallest |s| wins, then the stronger peak, then the negative shift), the
    best peak (F, S) (global index, -1), and the full match list moff (F + 1,) CSR over fragments, mk (peak), mc (class)
    (every (fragment, peak, shift) match within tolerance, observable fragments only)."""
    F = len(fm); S = len(pk_off) - 1
    cls = np.zeros((F, S), np.int8); pk = np.full((F, S), -1, np.int32)
    cap = 16 * F + 16
    mk = np.empty(cap, np.int32); mc = np.empty(cap, np.int8); moff = np.zeros(F + 1, np.int64)
    nm = 0
    for f in range(F):
        for s in range(S):
            p0 = pk_off[s]; p1 = pk_off[s + 1]
            m = p1 - p0
            ion = fm[f] + sp_del[s, 0]
            if ion < lo[s] or ion > hi[s]:
                cls[f, s] = -1
                continue
            best_c = 0; best_a = 99; best_w = -1.0; best_k = -1
            for a in range(3):
                for sg in range(2):
                    if a == 0 and sg == 1:
                        continue
                    sh = -a if sg == 0 else a
                    c = 1 if a == 0 else (2 * a if sg == 0 else 2 * a + 1)
                    for d in range(sp_nd[s]):
                        x = fm[f] + sp_del[s, d] + sh * h
                        t = x * tol_ppm * 1e-6
                        if t < tol_da:
                            t = tol_da
                        j = np.searchsorted(pk_mz[p0:p1], x - t)
                        while j < m and pk_mz[p0 + j] <= x + t:
                            w = pk_w[p0 + j]
                            if a < best_a or (a == best_a and w > best_w):
                                best_a = a; best_w = w; best_k = p0 + j; best_c = c
                            if nm >= cap:
                                cap2 = cap * 2
                                mk2 = np.empty(cap2, np.int32); mc2 = np.empty(cap2, np.int8)
                                mk2[:nm] = mk[:nm]; mc2[:nm] = mc[:nm]
                                mk = mk2; mc = mc2; cap = cap2
                            mk[nm] = p0 + j; mc[nm] = c; nm += 1
                            j += 1
            cls[f, s] = best_c; pk[f, s] = best_k
        moff[f + 1] = nm
    return cls, pk, moff, mk[:nm], mc[:nm]


# ------------------------------------------------------------------------------------------------ stage A: fragment LLR
NC = 6                                   # match classes: 0 none, 1 s=0, 2 s=-1, 3 s=+1, 4 s=-2, 5 s=+2
SIZE_EDGES = np.array([3, 5, 7, 10, 14, 20, 30])
FRAC_EDGES = np.arange(1, 10) / 10.0
CE_EDGES = np.array([15.0, 30.0, 50.0])
CARD = dict(mode=2, ce=5, ft=4, c1=NK * NK * 3, c2=NK * NK * 3 + 1, size=8, frac=10, nN=4, nO=6)


def ce_bin(ce):
    return 0 if not np.isfinite(ce) else 1 + int(np.searchsorted(CE_EDGES, ce))


def frag_fields(fr, M, mode, ce):
    """Categorical fields of each fragment for one spectrum context (mode +-1, collision energy)."""
    fm, ft, c1, c2, nh, nN, nO = fr
    n = len(fm)
    return dict(mode=np.full(n, int(mode > 0), np.int64), ce=np.full(n, ce_bin(ce), np.int64), ft=ft.astype(np.int64),
                c1=c1.astype(np.int64), c2=c2.astype(np.int64) + 1,
                size=np.searchsorted(SIZE_EDGES, nh, side='right').astype(np.int64),
                frac=np.minimum(np.searchsorted(FRAC_EDGES, fm / np.maximum(M, 1e-9)), 9).astype(np.int64),
                nN=np.minimum(nN, 3).astype(np.int64), nO=np.minimum(nO, 5).astype(np.int64))


def chain_keys(fd, chain):
    ks = []
    k = np.zeros(len(fd['ft']), np.int64)
    for lev in chain:
        for f in lev:
            k = k * CARD[f] + fd[f]
        ks.append(k.copy())
    return ks


def fit_chain(chain, levels_T, levels_D, alpha):
    """Smoothed per-level LLR tables from count tables (list per level of (keys sorted, counts (n, NC))) of the truth
    (T) and decoy (D) fragments. Level l backs off to level l-1 (Dirichlet prior with weight alpha)."""
    out = []
    prevT = prevD = None
    for li, lev in enumerate(chain):
        kT, cT = levels_T[li]; kD, cD = levels_D[li]
        keys = np.union1d(kT, kD)
        CT = np.zeros((len(keys), NC)); CD = np.zeros((len(keys), NC))
        CT[np.searchsorted(keys, kT)] = cT; CD[np.searchsorted(keys, kD)] = cD
        if li == 0:
            gT = CT.sum(0) + 1.0; gD = CD.sum(0) + 1.0
            pT0 = np.broadcast_to(gT / gT.sum(), CT.shape); pD0 = np.broadcast_to(gD / gD.sum(), CD.shape)
        else:
            div = 1
            for f in lev:
                div *= CARD[f]
            par = keys // div
            pk_, pT_, pD_ = prevT
            ix = np.searchsorted(pk_, par)
            assert (pk_[ix] == par).all()
            pT0 = pT_[ix]; pD0 = prevD[1][ix]
        pT = (CT + alpha * pT0) / (CT.sum(1, keepdims=True) + alpha)
        pD = (CD + alpha * pD0) / (CD.sum(1, keepdims=True) + alpha)
        out.append((keys, (np.log(pT) - np.log(pD)).astype(np.float32)))
        prevT = (keys, pT, None); prevD = (keys, pD)
    return out


class StageA:
    """Fragment LLR model: log P(match class | fragment key, truth) - log P(match class | fragment key, same-formula
    decoy), deepest observed level of each key chain; chains are summed minus their shared first level."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.chains = [list(map(list, c)) for c in z['chains']]
        self.tabs = []
        for ci in range(len(self.chains)):
            self.tabs.append([(z[f'k_{ci}_{li}'], z[f'l_{ci}_{li}']) for li in range(len(self.chains[ci]))])

    def llr(self, fd):
        tot = None
        for ci, (chain, tab) in enumerate(zip(self.chains, self.tabs)):
            ks = chain_keys(fd, chain)
            L = np.zeros((len(ks[0]), NC), np.float64)
            todo = np.ones(len(ks[0]), bool)
            for li in range(len(chain) - 1, -1, -1):
                keys, lt = tab[li]
                if not todo.any() or len(keys) == 0:
                    continue
                q = ks[li][todo]
                ix = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
                f = keys[ix] == q
                idx = np.flatnonzero(todo)
                L[idx[f]] = lt[ix[f]]
                todo[idx[f]] = False
            if ci == 0:
                tot = L; base0 = None
            else:
                keys, lt = self.tabs[0][0]                  # shared first level (same fields in every chain)
                q = chain_keys(fd, self.chains[0])[0]
                ix = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
                b = np.where((keys[ix] == q)[:, None], lt[ix], 0.0)
                tot = tot + L - b
        return tot


# ------------------------------------------------------------------------------------------------ candidate aggregates
AGG = ['fc', 'fc_m', 'fc_u', 'n_obs', 'n_match', 'pk', 'pk_pos', 'expl', 'top5', 'top5_n', 'fc_d1', 'fc_ro', 'fc_mid',
       'obs_d1', 'match_d1', 'unexpl_top', 'pkd', 'pkd_pos', 'expl_d']
NAGG = len(AGG)


@njit(cache=True)
def peak_best(coff, inv, LL, moff, mk, mc, pk_spec, K):
    """(n, K) best LLR over every match of the fragments of candidate c to peak k (-1e30 = unexplained); fragment f
    uses the match list of its unique mass inv[f]."""
    n = len(coff) - 1
    best = np.full((n, K), -1e30)
    for c in range(n):
        for f in range(coff[c], coff[c + 1]):
            u = inv[f]
            for q in range(moff[u], moff[u + 1]):
                k = mk[q]
                L = LL[f, pk_spec[k], mc[q]]
                if L > best[c, k]:
                    best[c, k] = L
    return best


@njit(cache=True)
def aggregate(coff, cls, LL, ft, best, pk_w, pk_spec, top, df, S):
    """Per candidate c and spectrum s (fragments coff[c]:coff[c+1], fixed intrinsic order -> row-independent):
    fragment-centric LLR sums over observable fragments (class of the best match) and peak-centric sums of the best
    explanation LLR of each peak (see AGG); df = share of candidates explaining each peak (discriminative weight)."""
    n = len(coff) - 1; K = len(pk_w)
    out = np.zeros((n, S, 19))
    for c in range(n):
        for f in range(coff[c], coff[c + 1]):
            for s in range(S):
                cl = cls[f, s]
                if cl < 0:
                    continue
                L = LL[f, s, cl]
                out[c, s, 0] += L; out[c, s, 3] += 1.0
                if ft[f] == 1:
                    out[c, s, 10] += L; out[c, s, 13] += 1.0
                elif ft[f] == 2:
                    out[c, s, 11] += L
                else:
                    out[c, s, 12] += L
                if cl > 0:
                    out[c, s, 1] += L; out[c, s, 4] += 1.0
                    if ft[f] == 1:
                        out[c, s, 14] += 1.0
                else:
                    out[c, s, 2] += L
        for k in range(K):
            s = pk_spec[k]
            b = best[c, k]
            if b > -1e29:
                out[c, s, 5] += pk_w[k] * b
                out[c, s, 16] += pk_w[k] * (1.0 - df[k]) * b
                if b > 0:
                    out[c, s, 6] += pk_w[k] * b
                    out[c, s, 17] += pk_w[k] * (1.0 - df[k]) * b
                out[c, s, 7] += pk_w[k]
                out[c, s, 18] += pk_w[k] * (1.0 - df[k])
                if top[k] > 0:
                    out[c, s, 8] += b; out[c, s, 9] += 1.0
            elif top[k] > 0:
                out[c, s, 15] += pk_w[k]
    return out


# ------------------------------------------------------------------------------------------------ peaks / fragments
TOL_DA, TOL_PPM = 0.01, 15.0
MAXD = 4
TOPN = 5


def peaks_of(spectra):
    """Engine peak prep of prepared (cleaned) query spectra: prep_query(mz, it, 0.002, 128, 1.0, False), m/z sorted,
    weight sqrt(it / max); ion deltas of each spectrum's adduct; observable window [max(50, min peak - 0.5), prec - 2];
    top-5 peaks by weight (stable on m/z order). Spectra without peaks are dropped."""
    from casmi.spectra import prep_query
    from casmi import frag as fragmod
    mzs, ws, sp = [], [], []
    for s in spectra:
        m, p = prep_query(np.asarray(s['mz'], np.float64), np.asarray(s['it'], np.float64), 0.002, 128, 1.0, False)
        if len(m) == 0 or p.max() <= 0:
            continue
        o = np.argsort(m, kind='mergesort'); m = m[o]; p = p[o]
        mzs.append(m); ws.append(np.sqrt(p / p.max())); sp.append(s)
    S = len(sp)
    if S == 0:
        return None
    pk_off = np.zeros(S + 1, np.int64); pk_off[1:] = np.cumsum([len(m) for m in mzs])
    pk_mz = np.concatenate(mzs); pk_w = np.concatenate(ws)
    pk_spec = np.concatenate([np.full(len(m), i, np.int64) for i, m in enumerate(mzs)])
    sp_del = np.zeros((S, MAXD)); sp_nd = np.zeros(S, np.int64); lo = np.zeros(S); hi = np.zeros(S)
    top = np.zeros(len(pk_mz), np.int64); wtot = np.zeros(S)
    for i, s in enumerate(sp):
        d = fragmod.ion_deltas(s['adduct'])[:MAXD]
        sp_del[i, :len(d)] = d; sp_nd[i] = len(d)
        lo[i] = max(50.0, float(mzs[i].min()) - 0.5); hi[i] = float(s['prec']) - 2.0
        w = ws[i]; o = np.argsort(-w, kind='mergesort')[:TOPN]
        top[pk_off[i] + o] = 1; wtot[i] = w.sum()
    return dict(S=S, pk_mz=pk_mz, pk_w=pk_w, pk_off=pk_off, pk_spec=pk_spec, sp_del=sp_del, sp_nd=sp_nd, lo=lo, hi=hi,
                top=top, wtot=wtot, mode=np.array([int(s['mode']) for s in sp]),
                ce=np.array([float(s.get('ce', np.nan)) for s in sp]))


def frags_of(rec):
    if rec is None or len(rec['bonds']) == 0:
        z = np.zeros(0)
        return (z, z.astype(np.int8), z.astype(np.int16), z.astype(np.int16), z.astype(np.int16), z.astype(np.int16),
                z.astype(np.int16))
    return enum_frags(rec['amass'], rec['kind'], rec['el'], rec['bonds'], rec['bring'], NK, MAX_FRAG)


def prep_cands(recs, pk, frs=None):
    """Fragments of every candidate record (None -> no fragments) matched against the query peaks. Matching runs on
    the unique fragment masses of the whole candidate set (rounded to 1e-6 Da; identical compositions of different
    candidates are matched once)."""
    if frs is None:
        frs = [frags_of(r) for r in recs]
    n = len(recs)
    coff = np.zeros(n + 1, np.int64); coff[1:] = np.cumsum([len(f[0]) for f in frs])
    cat = tuple(np.concatenate([f[j] for f in frs]) if n else np.zeros(0) for j in range(7))
    M = np.repeat(np.array([r['mass'] if r is not None else 1.0 for r in recs]), np.diff(coff))
    fm = np.round(cat[0].astype(np.float64), 6)
    u, inv = np.unique(fm, return_inverse=True)
    cls_u, pk_u, moff, mk, mc = match_frags(u, pk['pk_mz'], pk['pk_w'], pk['pk_off'], pk['sp_del'], pk['sp_nd'],
                                            pk['lo'], pk['hi'], H_MASS, TOL_DA, TOL_PPM)
    inv = inv.astype(np.int64).reshape(-1)
    return dict(cat=cat, coff=coff, M=M, cls=cls_u[inv], pkix=pk_u[inv], inv=inv, moff=moff, mk=mk, mc=mc, n=n)


def agg_cands(model, pc, pk, mask=None):
    """Stage-A aggregates (n, S, NAGG) of prepared candidates; df over the candidates in `mask` (default all)."""
    S = pk['S']; cls = pc['cls']; F_ = len(pc['M'])
    LL = np.zeros((F_, S, NC))
    for s in range(S):
        fd = frag_fields(pc['cat'], pc['M'], pk['mode'][s], pk['ce'][s])
        LL[:, s, :] = model.llr(fd)
    K = len(pk['pk_w'])
    best = peak_best(pc['coff'], pc['inv'], LL, pc['moff'], pc['mk'], pc['mc'], pk['pk_spec'], K)
    ex = best > -1e29
    if mask is None:
        mask = np.ones(pc['n'], bool)
    df = ex[mask].sum(0) / max(int(mask.sum()), 1)
    return aggregate(pc['coff'], cls, LL, pc['cat'][1].astype(np.int64), best, pk['pk_w'], pk['pk_spec'], pk['top'],
                     df.astype(np.float64), S)


def score_cands(model, recs, pk):
    return agg_cands(model, prep_cands(recs, pk), pk)


class PoolRecords:
    """Pool record table (res_fragnet.py --step rec), memory-mapped; ok == 2 -> computed from the SMILES on demand."""

    def __init__(self, smiles=None, path=RES):
        self.ok = np.load(os.path.join(path, 'rec_ok.npy'), mmap_mode='r')
        self.aoff = np.load(os.path.join(path, 'rec_aoff.npy'), mmap_mode='r')
        self.atom = np.load(os.path.join(path, 'rec_atom.npy'), mmap_mode='r')
        self.boff = np.load(os.path.join(path, 'rec_boff.npy'), mmap_mode='r')
        self.bond = np.load(os.path.join(path, 'rec_bond.npy'), mmap_mode='r')
        self.smiles = smiles

    def get(self, pid):
        ok = int(self.ok[pid])
        if ok == 1:
            return unpack_record(np.asarray(self.atom[self.aoff[pid]:self.aoff[pid + 1]]),
                                 np.asarray(self.bond[self.boff[pid]:self.boff[pid + 1]]))
        if ok == 2 and self.smiles is not None:
            return mol_record(self.smiles[pid])
        return None


CHAINS = {
    'A': [['mode', 'ft'], ['c1'], ['c2']],
    'A1': [['mode', 'ft'], ['c1']],
    'E': [['mode', 'ce', 'ft'], ['c1'], ['c2']],
    'G': [['mode', 'ft'], ['size', 'frac'], ['nN', 'nO']],
    'H': [['mode', 'ft'], ['size'], ['frac']],
    'B': [['mode', 'ft'], ['c1'], ['c2'], ['size']],
}


def model_from_counts(counts, chains, alpha):
    """StageA-like object from the countsA tables (res_fragnet --step countA) for the given chain names / alpha."""
    class _M(StageA):
        def __init__(self):
            self.chains = [CHAINS[c] for c in chains]
            self.tabs = []
            for c in chains:
                lv = len(CHAINS[c])
                LT = [(counts[f'{c}_{li}_T_k'], counts[f'{c}_{li}_T_c']) for li in range(lv)]
                LD = [(counts[f'{c}_{li}_D_k'], counts[f'{c}_{li}_D_c']) for li in range(lv)]
                self.tabs.append(fit_chain(CHAINS[c], LT, LD, alpha))
    return _M()


# ------------------------------------------------------------------------------------------------ stage B features
STAGEA_CHAINS, STAGEA_ALPHA = ['A', 'H'], 80.0          # chosen on V (see the decision log)
REL = ('pk', 'fc', 'pk_fcu', 'expl')
BNAMES = AGG + ['fc_n', 'match_r', 'pk_fcu', 'cov1'] + [f'{n}_{t}' for n in REL for t in ('rk', 'gap')]


def _rank_norm_(x):
    from ctxlib import _rank_norm
    return _rank_norm(x)


def stageB_features(a):
    """(n, len(BNAMES)) stage-B features of one candidate group for one spectrum from its stage-A aggregates (n, NAGG);
    relative features (average-tie rank, gap to the group max) are order invariant."""
    a = np.asarray(a, np.float64)
    d = {nm: a[:, i] for i, nm in enumerate(AGG)}
    d['fc_n'] = d['fc'] / np.maximum(d['n_obs'], 1.0)
    d['match_r'] = d['n_match'] / np.maximum(d['n_obs'], 1.0)
    d['pk_fcu'] = d['pk'] + 0.02 * d['fc_u']
    d['cov1'] = d['match_d1'] / np.maximum(d['obs_d1'], 1.0)
    cols = [d[nm] for nm in AGG + ['fc_n', 'match_r', 'pk_fcu', 'cov1']]
    for nm in REL:
        x = d[nm]
        cols.append(_rank_norm_(x) if len(x) else x)
        cols.append(x - x.max() if len(x) else x)
    return np.stack(cols, 1) if len(a) else np.zeros((0, len(BNAMES)))


def save_stageA(path, counts, chains=None, alpha=None):
    chains = chains or STAGEA_CHAINS; alpha = alpha or STAGEA_ALPHA
    m = model_from_counts(counts, chains, alpha)
    z = {'chains': np.array([np.array([np.array(l) for l in CHAINS[c]], dtype=object) for c in chains], dtype=object),
         'alpha': np.array(alpha)}
    for ci, tab in enumerate(m.tabs):
        for li, (k, l) in enumerate(tab):
            z[f'k_{ci}_{li}'] = k; z[f'l_{ci}_{li}'] = l.astype(np.float32)
    np.savez(path, **z)
    return m


# ------------------------------------------------------------------------------------------------ family
FRAG_MAX_CAND = 400          # frag set: top candidates by the A+B f.z (tie-inclusive); others get sentinels
NAMES = ['fn_score', 'fn_score_max', 'fn_rank', 'fn_gap', 'fn_grp_rank', 'fn_grp_gap', 'fn_vs_lead',
         'fn_raw_rank', 'fn_raw_gap', 'fn_raw_grp_gap', 'fn_raw_vs_lead', 'fn_pk_rank', 'fn_fc_rank', 'fn_sp_rank']
_R: Dict[str, object] = {}
_GEN: Dict[str, object] = {}


def prepare(res):
    import lightgbm as lgb
    from casmi import chem as cchem
    assert abs(cchem.MASS['H'] - H_MASS) < 1e-12
    _R['A'] = StageA(os.path.join(RES, 'stageA.npz'))
    _R['B'] = lgb.Booster(model_file=os.path.join(RES, 'stageB.txt'))
    assert list(_R['B'].feature_name()) == BNAMES, 'stageB.txt was trained with another feature list'
    _R['rec'] = PoolRecords(res.P.smiles)


def _frags_of_cand(v, i):
    """Fragments of candidate i: pool -> table record; generated -> record + fragments cached by SMILES."""
    p = int(v.pid[i])
    if p >= 0:
        r = _R['rec'].get(p)
        return r, frags_of(r)
    s = v.smiles[i]
    hit = _GEN.get(s)
    if hit is None:
        if len(_GEN) > 20000:
            _GEN.clear()
        r = mol_record(s)
        hit = (r, frags_of(r))
        _GEN[s] = hit
    return hit


def _group_ids(formulas):
    u = sorted(set(formulas))
    ix = {f: j for j, f in enumerate(u)}
    return np.array([ix[f] for f in formulas], np.int64), len(u)


def _group_rank(x, gid, G):
    from ctxlib import _rank_norm
    out = np.zeros(len(x))
    for g in range(G):
        j = np.flatnonzero(gid == g)
        out[j] = _rank_norm(x[j])
    return out


def _group_max(x, gid, G):
    mx = np.full(G, -np.inf)
    np.maximum.at(mx, gid, x)
    return mx[gid]


def compute(v):
    from ctxlib import _rank_norm, COL
    if not _R:
        prepare(v.res)
    n = v.n
    out = {k: np.zeros(n) for k in NAMES}
    pk = peaks_of(v.spectra) if n else None
    if pk is None:
        return {k: x.astype(np.float32) for k, x in out.items()}
    fz = v.X[:, COL['fz']]
    if n > FRAG_MAX_CAND:
        thr = np.sort(fz)[::-1][FRAG_MAX_CAND - 1]
        sel = np.flatnonzero(fz >= thr)
    else:
        sel = np.arange(n)
    rf = [_frags_of_cand(v, int(i)) for i in sel]
    pc = prep_cands([x[0] for x in rf], pk, [x[1] for x in rf])
    agg = agg_cands(_R['A'], pc, pk)                        # (N, S, NAGG)
    S = pk['S']; N = len(sel)
    bs = np.zeros((N, S)); raw = np.zeros((N, S)); pkn = np.zeros((N, S)); fcn = np.zeros((N, S)); spr = np.zeros((N, S))
    for s in range(S):
        Xb = stageB_features(agg[:, s, :])
        bs[:, s] = _R['B'].predict(Xb.astype(np.float32), num_threads=1)
        raw[:, s] = (agg[:, s, AGG.index('pk')] + 0.02 * agg[:, s, AGG.index('fc_u')]) / max(pk['wtot'][s], 1e-9)
        pkn[:, s] = agg[:, s, AGG.index('pk')] / max(pk['wtot'][s], 1e-9)
        fcn[:, s] = agg[:, s, AGG.index('fc')] / np.maximum(agg[:, s, AGG.index('n_obs')], 1.0)
        spr[:, s] = _rank_norm(bs[:, s])
    sc = bs[:, 0].copy(); sc_mx = bs[:, 0].copy(); rw = raw[:, 0].copy()
    for s in range(1, S):                                   # fixed spectrum order
        sc = sc + bs[:, s]; sc_mx = np.maximum(sc_mx, bs[:, s]); rw = rw + raw[:, s]
    sc = sc / S; rw = rw / S
    pm_ = pkn[:, 0].copy(); fm_ = fcn[:, 0].copy(); sm_ = spr[:, 0].copy()
    for s in range(1, S):
        pm_ = pm_ + pkn[:, s]; fm_ = fm_ + fcn[:, s]; sm_ = sm_ + spr[:, s]
    pm_ = pm_ / S; fm_ = fm_ / S; sm_ = sm_ / S
    full_sc = np.full(n, sc.min() - 1.0); full_sc[sel] = sc
    full_mx = np.full(n, sc_mx.min() - 1.0); full_mx[sel] = sc_mx
    full_rw = np.full(n, rw.min() - 1.0); full_rw[sel] = rw
    gid, G = _group_ids(v.formula)
    fzs = fz[sel]
    lead = np.flatnonzero(fzs == fzs.max())
    out['fn_score'] = full_sc; out['fn_score_max'] = full_mx
    out['fn_rank'] = _rank_norm(full_sc); out['fn_gap'] = full_sc - full_sc.max()
    out['fn_grp_rank'] = _group_rank(full_sc, gid, G); out['fn_grp_gap'] = full_sc - _group_max(full_sc, gid, G)
    out['fn_vs_lead'] = full_sc - float(np.sort(sc[lead]).mean())
    out['fn_raw_rank'] = _rank_norm(full_rw); out['fn_raw_gap'] = full_rw - full_rw.max()
    out['fn_raw_grp_gap'] = full_rw - _group_max(full_rw, gid, G)
    out['fn_raw_vs_lead'] = full_rw - float(np.sort(rw[lead]).mean())
    for nm, x, fill in (('fn_pk_rank', pm_, None), ('fn_fc_rank', fm_, None)):
        f_ = np.full(n, x.min() - 1.0); f_[sel] = x; out[nm] = _rank_norm(f_)
    f_ = np.ones(n); f_[sel] = sm_; out['fn_sp_rank'] = f_
    return {k: np.nan_to_num(np.asarray(x, np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for k, x in out.items()}

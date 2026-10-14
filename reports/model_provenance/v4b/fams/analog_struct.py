"""F-analog_struct: structure relations between each candidate and the top spectral analogs that Tanimoto misses.

The analog channel (v.an_sid / an_sim, sim desc) finds library structures whose spectra resemble the query (mass-shift
aware). The base features propagate FINGERPRINT Tanimoto to them (ap, a1, best_tan, ...). This family adds relations on
the heavy-atom ELEMENT GRAPH (atoms = element only, every bond single, no H / charge / aromaticity / stereo), which is
invariant to the representation style of a SMILES: train SMILES (InChI-derived, e.g. imidic acids 'C(O)=N', hydroxy-
heteroaromatics) and COCONUT SMILES (amides, lactams) of the SAME molecule give the same graph, while their fingerprints
differ a lot (mean full-fp Tanimoto 0.70 over the 2,576 train/COCONUT keys whose SMILES differ). Style is provenance, so
graph-level relations are the provenance-safe way to compare a candidate with (train-style) analogs.

Per structure (graph_record(smiles), precomputed for the pool by pid and for train structures by sid in
res/analog_struct by res_analog_struct.py; generated candidates call the same function on the fly):
  core      = 2-core of the element graph (ring systems + linkers == Murcko scaffold without exocyclic atoms)
  el_scaf   = hash of the core's canonical element-graph SMILES; gen_scaf = same with every atom carbon
  ring systems = connected components of ring bonds; hash of each (element graph), sorted, <= RSMAX kept
  numbers   = cyclomatic number (== SSSR ring count), #ring systems, #core atoms, ring N / O / S, largest ring system
              size, heavy atoms
  formula   = element counts (C H N O P S F Cl Br I) of the pool / train formula (generated: v.formula)

Features for candidate i over the top-K analogs j (K = 15, sim desc; generated candidates exclude their donor analog
exactly like the engine's 'use' mask), w4 = sim^4, w1 = sim:
  sc_el_max / sc_el_wsum / sc_gen_max / sc_gen_wsum    exact / generic core equality (max w4, w4-weighted share)
  sc_el_top                                            core equality with the top usable analog
  rs_jac_max / rs_jac_wmean / rs_cov_max / rs_big_max  ring-system set Jaccard, coverage of the analog's ring systems
                                                       by the candidate, equality of the largest ring system
  sub_ca_max / sub_ac_max                              RDKit substructure (element graphs, top KSUB=5 analogs, max w1):
                                                       candidate contains the analog's core / analog contains the
                                                       candidate's core (non-empty cores only)
  fd_wmean / fd_top / fd_min                           formula L1 edit distance (element counts) to the analogs
  fx_max / fx_top                                      candidate-vs-analog formula difference is 0 or an EXACT element
                                                       delta of <= 2 derive._TRANSFORMS steps (the mass is the same for
                                                       every candidate, the element delta is candidate-specific)
  dr_ring_top / dr_rs_top / dr_core_top / dr_rhet_top  signed differences to the top usable analog (candidate - analog)
  dr_ring_wabs                                         w4-weighted mean |ring-count difference|
  bt_<block>                                           max_j w4 * Tanimoto on one fingerprint block only (maccs, rdk6,
                                                       fcfp4, ap, tt): fingerprint based -> representation-style
                                                       sensitive (see above); evaluated as a separable subset
  (added in session 2)
  da_ring_top / da_rs_top / da_core_top / da_rhet_top  |dr_*_top|: symmetric relation versions of the signed diffs
  subf_ca_max / subf_ac_max                            FULL element graph containment (candidate contains the whole
                                                       analog graph = derivative by addition / analog contains the
                                                       whole candidate), top KSUB analogs, max w1
  st_ap / st_rank / st_gap / st_fgap                   structural composite R_ij = mean(core eq, generic core eq, ring-
                                                       system Jaccard, coverage, 0.5*(core sub both ways), 0.5*(full sub
                                                       both ways)); st_ap = max_j w4 R_ij; window average-tie rank, gap
                                                       to the window max, gap to the max among candidates with the SAME
                                                       element counts (isomer group)
Everything is intrinsic chemistry of the SMILES / formula; no pool src, no train_sid, pid used only as a table index.

RESUME NOTES / DECISIONS (session 2, 2026-09-24)
  * precompute res/analog_struct/{pool,train}_{hash,num,fvec}.npy finished (res_analog_struct.log: train 94 s, pool 239 s;
    empty core 5.0% train / 8.9% pool; bad formula 0.03% / 0.12%).
  * style check (1,500 random train/COCONUT SMILES pairs of the SAME score key whose SMILES strings differ, 2,576 such keys
    of 25,746 shared): element-graph core / generic core / ring-system hashes / numbers equal in 100%; chemfeat Murcko
    scaffold hash equal in only 18.5% (generic 31.7%) -- cause: tautomer style (train 'C(O)=N', 'c(O)n' vs COCONUT
    'C(=O)N', 'c(=O)[nH]'), not stereo. => chemfeat scaffold hashes are provenance-sensitive for ~8% of structures;
    this family deliberately uses the element-graph hashes only.
  * unit test (first ctx file, 60 queries): c1 truth with its own spectra as top analog -> sc_el_top 1, fd_top 0,
    sub_* = sim; mean 24 ms/query (n~69), max 293 ms (n=316, cold RDKit cache). gate_fam: 80 queries, 0 failures.
  * bt_* (per-block Tanimoto) are fingerprint based -> style sensitive; evaluated separately (variant --drop bt_*).
  * v1 (26 feats, 3,541 groups): ALL +0.0009 [-0.0041, 0.0054]; C2FAM -0.0002; C3FAM +0.0172 [0.0014, 0.0337];
    np ALL +0.0075; prov base tr/co 0.9855/0.9643 -> 0.9854/0.9650 (tiny COCONUT-side gain).
    v2 (-bt_*): ALL -0.0012; nplib C2FAM -0.0073 [-0.0126, -0.0028]; C3FAM +0.0220 [0.0015, 0.0441].
    (v1 vs v2 differ by 0.005 on nplib C2FAM from 5 features that barely matter -> training noise ~0.005.)
  * error analysis (base top-1 wrong): C2FAM 95% same-formula decoys (formula features cannot separate them); the winning
    decoy is MORE analog-like than the truth on every relation (sc_el_max truth< 16% vs truth> 8%; full containment
    16% vs 6%) -> analog-relation features mostly re-confirm the base; expected gain is small, mainly class 3.
  * per-feature provenance probe (pairwise truth-vs-decoy win rate, train vs COCONUT decoys, c2fam): relation
    features (sc/rs/sub) have ~equal win rates; the SIGNED dr_*_top (own-descriptor level) show the largest
    COCONUT-side skew (dr_rs_top +0.19) -> v3 replaces them by |diff| (da_*) and drops bt_*.
  * cost: warm cache 62 ms/query, cold cache (Kaggle-like) mean 121 ms, p95 418 ms, max 1.2 s (n=329) on 1 core.
  * v3 (27 relation feats: -bt_*, -signed dr_*; 5,105 groups): ALL +0.0032 [-0.0005, 0.0066]; C2FAM +0.0025;
    C3FAM +0.0156 [0.0019, 0.0302]; prov tr +0.0011 / co +0.0014.
  * v4 compact (13 feats by joint-ranker gain; same 5,105 groups -> mildly optimistic): ALL +0.0076 [0.0036, 0.0113];
    C2FAM +0.0067 [0.0018, 0.0115]; C3FAM +0.0320 [0.0167, 0.0479]; np ALL +0.0124; np C2FAM_wt -0.0164 (wide CI);
    prov tr +0.0017 / co +0.0036. Stratified (same-formula decoys: tr +0.0041 = 13.5% of the base loss, co +0.0059 =
    10.0%; other-formula: tr -0.0006, co -0.0023): COCONUT decoys are 68% isomers vs 50% for train decoys, so the larger
    absolute COCONUT gain is composition; relative to the base loss the gain is LARGER against train decoys.
    RECOMMENDED = V4 = --only st_rank,st_ap,st_gap,st_fgap,subf_ac_max,subf_ca_max,sub_ca_max,sub_ac_max,rs_jac_wmean,
    sc_gen_wsum,fx_max,fd_wmean,dr_ring_wabs.
  * speed-up (values unchanged, verified bit-identical on 600 queries / 10,323 generated candidates and table ==
    graph_all on 2,000 pool entries): substructure tests parse precomputed canonical element-graph / core SMILES
    (res/analog_struct/{pool,train}_{el,core}smi_{buf,off}.npy, res_analog_struct.py --step str); generated
    candidates run graph_all once (hash record + strings). Cold cache mean 92 ms, p95 451 ms (mean n 88).
  * v4 interim (7,857 groups, 12:20): ALL +0.0037 [-0.0001, 0.0078]; C2FAM +0.0025 [-0.0018, 0.0066]; C3FAM +0.0201
    [0.0046, 0.0361]; np ALL +0.0062; np C2FAM_wt -0.0167 [-0.054, 0.021]; prov tr +0.0002 / co +0.0013 -- bootstrap
    over structures: co-minus-tr +0.0012 [-0.0022, +0.0046] (not distinguishable from 0; stratified by same-formula
    decoys the sign now points to COCONUT in both strata, opposite to the 5,105-group run -> noise-dominated).
    RDKit must stay 2026.03.3 on Kaggle (canonical element-graph SMILES / hashes of generated candidates are compared
    with the precomputed tables; the Kaggle wheel dataset is rdkit-2026-3-3).
"""
from __future__ import annotations

import hashlib
import itertools
import os
from typing import Dict, Optional

import numpy as np

from ctxlib import _rank_norm

K = 15
KSUB = 5
RSMAX = 6
NUM_FIELDS = ['n_ring', 'n_rs', 'n_core', 'ring_n', 'ring_o', 'ring_s', 'rs_big_n', 'n_heavy']
NI = {f: i for i, f in enumerate(NUM_FIELDS)}
HASH_W = 2 + RSMAX                      # el_scaf, gen_scaf, rs[0..RSMAX-1]
BLOCKS = [("ecfp4", 4096), ("ecfp6", 4096), ("fcfp4", 2048), ("rdk6", 2048), ("ap", 2048), ("tt", 2048), ("maccs", 167)]
BT_BLOCKS = ['maccs', 'rdk6', 'fcfp4', 'ap', 'tt']
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'res', 'analog_struct')
_ST: Dict[str, object] = {}
_RD: Dict[str, object] = {}
_EL = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']


def _rd():
    if not _RD:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        _RD['Chem'] = Chem
    return _RD['Chem']


def _h(s: str) -> int:
    """62-bit positive hash (0 is reserved for 'none')."""
    return (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little') & ((1 << 62) - 1)) or 1


# ------------------------------------------------------------------------------------------------ per-structure graph
def element_graph(smiles: str):
    """Heavy-atom element graph of a SMILES: RDKit mol (unsanitized) with atoms = atomic number only (no implicit H,
    charge 0, not aromatic), every bond SINGLE; ring info initialised. None on parse failure."""
    Chem = _rd()
    m = Chem.MolFromSmiles(smiles, sanitize=False) if isinstance(smiles, str) and smiles else None
    if m is None:
        return None
    rw = Chem.RWMol()
    idx = {}
    for a in m.GetAtoms():
        z = a.GetAtomicNum()
        if z == 1:
            continue
        na = Chem.Atom(z); na.SetNoImplicit(True)
        idx[a.GetIdx()] = rw.AddAtom(na)
    seen = set()
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            p = (min(idx[i], idx[j]), max(idx[i], idx[j]))
            if p[0] != p[1] and p not in seen:
                seen.add(p); rw.AddBond(p[0], p[1], Chem.BondType.SINGLE)
    g = rw.GetMol()
    g.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(g)
    return g


def _graph_parts(g):
    """(core atoms, ring bonds, ring systems (list of atom lists)) of an element graph, pure graph algorithms."""
    n = g.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in g.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[i].append((j, b.GetIdx())); adj[j].append((i, b.GetIdx()))
    # 2-core: prune degree <= 1 atoms repeatedly
    deg = [len(a) for a in adj]; alive = [True] * n
    stack = [i for i in range(n) if deg[i] <= 1]
    while stack:
        i = stack.pop()
        if not alive[i]:
            continue
        alive[i] = False
        for j, _ in adj[i]:
            if alive[j]:
                deg[j] -= 1
                if deg[j] <= 1:
                    stack.append(j)
    core = [i for i in range(n) if alive[i]]
    # bridges (iterative Tarjan) -> ring bonds = non-bridge bonds
    disc = [-1] * n; low = [0] * n; t = 0; bridges = set()
    for s in range(n):
        if disc[s] >= 0:
            continue
        disc[s] = low[s] = t; t += 1
        st = [(s, -1, iter(adj[s]))]
        while st:
            u, pe, it = st[-1]
            nxt = next(it, None)
            if nxt is None:
                st.pop()
                if st:
                    p = st[-1][0]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        bridges.add(pe)
                continue
            w, be = nxt
            if be == pe:
                continue
            if disc[w] < 0:
                disc[w] = low[w] = t; t += 1
                st.append((w, be, iter(adj[w])))
            else:
                low[u] = min(low[u], disc[w])
    ring_b = [b.GetIdx() for b in g.GetBonds() if b.GetIdx() not in bridges]
    # ring systems: components over ring bonds
    par = list(range(n))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    ring_atoms = set()
    for bi in ring_b:
        b = g.GetBondWithIdx(bi); i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ring_atoms.add(i); ring_atoms.add(j)
        ri, rj = f(i), f(j)
        if ri != rj:
            par[ri] = rj
    comp: Dict[int, list] = {}
    for a in sorted(ring_atoms):
        comp.setdefault(f(a), []).append(a)
    return core, ring_b, list(comp.values())


def _frag_smiles(g, atoms, generic=False) -> str:
    """Canonical SMILES of the induced subgraph on `atoms` built as a STANDALONE molecule (so the canonical form depends
    on the subgraph only, not on the rest of the molecule); generic: every atom carbon."""
    Chem = _rd()
    if not atoms:
        return ''
    rw = Chem.RWMol(); idx = {}
    for a in sorted(atoms):
        na = Chem.Atom(6 if generic else g.GetAtomWithIdx(a).GetAtomicNum()); na.SetNoImplicit(True)
        idx[a] = rw.AddAtom(na)
    for b in g.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            rw.AddBond(idx[i], idx[j], Chem.BondType.SINGLE)
    m = rw.GetMol(); m.UpdatePropertyCache(strict=False); Chem.FastFindRings(m)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


def graph_all(smiles: str):
    """One pass over the element graph: (hashes int64 (HASH_W,), numbers int16 (len(NUM_FIELDS),), core smiles,
    element-graph smiles). graph_record / graph_strings (the precomputed tables) are views of it, so pool entries and
    generated candidates get identical values; each part falls back to its own default on failure."""
    hs = np.zeros(HASH_W, np.int64); nums = np.zeros(len(NUM_FIELDS), np.int16); core_smi = ''; el_smi = ''
    s_core = ''
    try:
        g = element_graph(smiles)
        if g is None or g.GetNumAtoms() == 0:
            return hs, nums, '', '', ''
        core, ring_b, rsys = _graph_parts(g)
    except Exception:
        return hs, nums, '', '', ''
    try:
        n = g.GetNumAtoms(); nb = g.GetNumBonds()
        ncomp = len(set(_rd().GetMolFrags(g, asMols=False))) if n else 0
        core_smi = _frag_smiles(g, core)
        if core:
            hs[0] = _h('E:' + core_smi); hs[1] = _h('G:' + _frag_smiles(g, core, generic=True))
        rh = sorted(((-len(a), _h('R:' + _frag_smiles(g, a))) for a in rsys))
        uniq = []                                              # ring-system TYPES (set semantics), largest first
        for _, hv in rh:
            if hv not in uniq:
                uniq.append(hv)
        for k, hv in enumerate(uniq[:RSMAX]):
            hs[2 + k] = hv
        ring_atoms = sorted(set(a for c in rsys for a in c))
        sym = [g.GetAtomWithIdx(a).GetSymbol() for a in ring_atoms]
        nums[NI['n_ring']] = nb - n + ncomp
        nums[NI['n_rs']] = len(rsys); nums[NI['n_core']] = len(core)
        nums[NI['ring_n']] = sym.count('N'); nums[NI['ring_o']] = sym.count('O'); nums[NI['ring_s']] = sym.count('S')
        nums[NI['rs_big_n']] = max((len(a) for a in rsys), default=0); nums[NI['n_heavy']] = n
    except Exception:
        hs = np.zeros(HASH_W, np.int64); nums = np.zeros(len(NUM_FIELDS), np.int16); core_smi = ''
    try:
        el_smi = _rd().MolToSmiles(g, canonical=True, isomericSmiles=False)
        s_core = core_smi if core_smi else _frag_smiles(g, core)
    except Exception:
        el_smi, s_core = '', ''
    return hs, nums, core_smi, el_smi, s_core


def graph_record(smiles: str):
    """(hashes int64 (HASH_W,), numbers int16 (len(NUM_FIELDS),), core smiles str) of one structure; zeros on failure."""
    hs, nums, core_smi, _, _ = graph_all(smiles)
    return hs, nums, core_smi


def graph_strings(smiles: str):
    """(canonical element-graph SMILES, canonical core SMILES) of one structure ('' on failure). Precomputed for the
    pool / train structures (res_analog_struct.py --step str); generated candidates get them from graph_all on the fly.
    Every substructure test parses these strings, so pool and generated candidates take the identical code path."""
    _, _, _, el_smi, s_core = graph_all(smiles)
    return el_smi, s_core


def pack_strings(strs):
    """list of str -> (uint8 buffer, int64 offsets (len+1))."""
    b = [x.encode() for x in strs]
    off = np.zeros(len(b) + 1, np.int64); off[1:] = np.cumsum([len(x) for x in b])
    return np.frombuffer(b''.join(b), np.uint8).copy() if off[-1] else np.zeros(0, np.uint8), off


def _str_at(tab, i):
    buf, off = tab
    return bytes(buf[off[i]:off[i + 1]]).decode()


def formula_vec(f) -> np.ndarray:
    """Element counts (C H N O P S F Cl Br I) of a formula string; -1 row on failure."""
    from casmi import chem
    c = chem.parse_formula(f) if isinstance(f, str) else None
    if c is None or any(e not in _EL for e in c):
        return np.full(len(_EL), -1, np.int16)
    return np.array([c.get(e, 0) for e in _EL], np.int16)


def _transform_deltas():
    """Exact element deltas of 0 / 1 / 2 derive._TRANSFORMS steps (engine v6.3 generation grammar, on either path)."""
    import derive6 as derive
    vec = {}
    for nm, (_, d) in derive._TRANSFORMS.items():
        vec[nm] = np.array([d.get(e, 0) for e in _EL], np.int64)
    s = {tuple([0] * len(_EL))}
    for nm in vec:
        s.add(tuple(vec[nm]))
    for a, b in itertools.combinations_with_replacement(sorted(vec), 2):
        s.add(tuple(vec[a] + vec[b]))
    return s


_MULT = np.array([81 ** e for e in range(10)], np.int64)
_BIG = np.iinfo(np.int64).max


def _code(D: np.ndarray) -> np.ndarray:
    """Exact integer code of element-delta vectors (..., 10): signed base-81 digits (|delta| <= 40 per element, max
    |code| ~6e18 < int64 max); vectors with any |delta| > 40 (never a <= 2-step transform delta) -> sentinel."""
    D = np.asarray(D, np.int64)
    c = (np.clip(D, -40, 40) * _MULT).sum(-1)
    return np.where((np.abs(D) > 40).any(-1), _BIG, c)


# ------------------------------------------------------------------------------------------------ family
def prepare(res):
    _ST['pool_h'] = np.load(f'{RES}/pool_hash.npy', mmap_mode='r')
    _ST['pool_n'] = np.load(f'{RES}/pool_num.npy', mmap_mode='r')
    _ST['pool_f'] = np.load(f'{RES}/pool_fvec.npy', mmap_mode='r')
    _ST['train_h'] = np.load(f'{RES}/train_hash.npy', mmap_mode='r')
    _ST['train_n'] = np.load(f'{RES}/train_num.npy', mmap_mode='r')
    _ST['train_f'] = np.load(f'{RES}/train_fvec.npy', mmap_mode='r')
    _ST['deltas'] = _transform_deltas()
    _ST['dcode'] = np.unique(_code(np.array(sorted(_ST['deltas']), np.int64)))
    bits = np.asarray(res.P.bits)
    edges = np.cumsum([0] + [n for _, n in BLOCKS])
    bid = np.searchsorted(edges, bits, side='right') - 1
    names = [b for b, _ in BLOCKS]
    _ST['blk'] = {nm: np.flatnonzero(bid == names.index(nm)) for nm in BT_BLOCKS}
    _ST['gen'] = {}; _ST['gmol'] = {}
    for w in ('pool', 'train'):
        for f in ('el', 'core'):
            _ST[f'{w}_{f}'] = (np.load(f'{RES}/{w}_{f}smi_buf.npy', mmap_mode='r'), np.load(f'{RES}/{w}_{f}smi_off.npy', mmap_mode='r'))


def _gen_all(smiles):
    c = _ST['gen']
    if smiles not in c:
        if len(c) > 100000:
            c.clear()
        c[smiles] = graph_all(smiles)
    return c[smiles]


def _mols(el_smi: str, core_smi: str):
    """(element graph, core query mol or None, element counts of both) parsed from the graph strings, cached per
    process by the element-graph SMILES (the core is a function of the graph)."""
    c = _ST['gmol']
    if el_smi not in c:
        if len(c) > 50000:
            c.clear()
        Chem = _rd()
        g = Chem.MolFromSmiles(el_smi, sanitize=False) if el_smi else None
        if g is not None:
            g.UpdatePropertyCache(strict=False); Chem.FastFindRings(g)
        q = Chem.MolFromSmiles(core_smi, sanitize=False) if (g is not None and core_smi) else None
        if q is not None:
            q.UpdatePropertyCache(strict=False); Chem.FastFindRings(q)
        cnt = np.zeros(128, np.int32); qc = np.zeros(128, np.int32)
        if g is not None:
            np.add.at(cnt, np.minimum([a.GetAtomicNum() for a in g.GetAtoms()], 127), 1)
        if q is not None:
            np.add.at(qc, np.minimum([a.GetAtomicNum() for a in q.GetAtoms()], 127), 1)
        c[el_smi] = (g, q, cnt, qc)
    return c[el_smi]


def _cand_strings(v):
    """(el_smi, core_smi) per candidate: pool by pid lookup, generated on the fly (same function)."""
    out = [None] * v.n
    for i in range(v.n):
        p = int(v.pid[i])
        out[i] = (_str_at(_ST['pool_el'], p), _str_at(_ST['pool_core'], p)) if p >= 0 else _gen_all(v.smiles[i])[3:5]
    return out


def _contains(big, small) -> bool:
    """Element graph `big` (g, q, cnt, qc) contains the core query of `small`."""
    g, _, cnt, _ = big; _, q, _, qc = small
    if g is None or q is None or (qc > cnt).any():
        return False
    return bool(g.HasSubstructMatch(q))


def _contains_full(big, small) -> bool:
    """Element graph `big` contains the WHOLE element graph of `small` (derivative-by-addition relation)."""
    g, _, cnt, _ = big; gs, _, cs, _ = small
    if g is None or gs is None or gs.GetNumAtoms() == 0 or (cs > cnt).any() or gs.GetNumBonds() > g.GetNumBonds():
        return False
    return bool(g.HasSubstructMatch(gs))


def compute(v):
    n = v.n
    names = ['sc_el_max', 'sc_el_wsum', 'sc_gen_max', 'sc_gen_wsum', 'sc_el_top', 'rs_jac_max', 'rs_jac_wmean',
             'rs_cov_max', 'rs_big_max', 'sub_ca_max', 'sub_ac_max', 'fd_wmean', 'fd_top', 'fd_min', 'fx_max', 'fx_top',
             'dr_ring_top', 'dr_rs_top', 'dr_core_top', 'dr_rhet_top', 'dr_ring_wabs'] + [f'bt_{b}' for b in BT_BLOCKS] + \
            ['da_ring_top', 'da_rs_top', 'da_core_top', 'da_rhet_top', 'subf_ca_max', 'subf_ac_max',
             'st_ap', 'st_rank', 'st_gap', 'st_fgap']
    out = {k: np.zeros(n, np.float32) for k in names}
    k = min(K, len(v.an_sid))
    if n == 0 or k == 0:
        for nm in ('fd_wmean', 'fd_top', 'fd_min'):
            out[nm][:] = -1.0
        out['st_rank'][:] = 0.5                                  # == _rank_norm of a constant
        return out
    # ---- candidate tables (pool by pid lookup, generated on the fly by the same function)
    ch = np.zeros((n, HASH_W), np.int64); cn = np.zeros((n, len(NUM_FIELDS)), np.float64)
    cf = np.zeros((n, len(_EL)), np.int64)
    pi = np.flatnonzero(v.pid >= 0)
    if len(pi):
        ch[pi] = _ST['pool_h'][v.pid[pi]]; cn[pi] = _ST['pool_n'][v.pid[pi]]; cf[pi] = _ST['pool_f'][v.pid[pi]]
    for i in np.flatnonzero(v.pid < 0):
        h_, n_ = _gen_all(v.smiles[i])[:2]
        ch[i] = h_; cn[i] = n_; cf[i] = formula_vec(v.formula[i])
    sid = np.asarray(v.an_sid[:k], np.int64)
    sim = np.asarray(v.an_sim[:k], np.float64)
    ah = np.asarray(_ST['train_h'][sid]); an = np.asarray(_ST['train_n'][sid], np.float64)
    af = np.asarray(_ST['train_f'][sid], np.int64)
    # ---- donor-free mask (== engine 'use' for generated candidates)
    use = np.ones((n, k), bool)
    for i in np.flatnonzero(v.parent_sid >= 0):
        use[i, sid == int(v.parent_sid[i])] = False
    w4 = sim ** 4; w1 = sim
    W4 = np.where(use, w4[None, :], 0.0); wsum = W4.sum(1)
    anyu = use.any(1)
    top = np.argmax(use, 1)                                     # first usable (highest-sim) analog
    rows = np.arange(n)

    def wmax(M, W=W4):
        return np.where(use, M * W, 0.0).max(1)

    def wmean(M):
        return np.where(wsum > 0, (M * W4).sum(1) / np.maximum(wsum, 1e-12), 0.0)
    # ---- core equality
    for c, nm in ((0, 'el'), (1, 'gen')):
        E = (ch[:, c][:, None] == ah[:, c][None, :]) & (ch[:, c][:, None] != 0)
        out[f'sc_{nm}_max'] = wmax(E.astype(np.float64)); out[f'sc_{nm}_wsum'] = wmean(E.astype(np.float64))
        if c == 0:
            out['sc_el_top'] = np.where(anyu, E[rows, top], False).astype(np.float64)
    # ---- ring-system sets
    cr = ch[:, 2:]; ar = ah[:, 2:]
    inter = np.zeros((n, k));
    for a in range(RSMAX):
        ca = cr[:, a]
        if not (ca != 0).any():
            continue
        # multiset-safe enough: count candidate ring system a if present in the analog's list (duplicates rare)
        inter += ((ca[:, None, None] == ar[None, :, :]).any(2) & (ca[:, None] != 0))
    nc_ = (cr != 0).sum(1).astype(np.float64); na_ = (ar != 0).sum(1).astype(np.float64)
    union = nc_[:, None] + na_[None, :] - inter
    J = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    C = np.where(na_[None, :] > 0, inter / np.maximum(na_[None, :], 1), 0.0)
    big = (cr[:, 0][:, None] == ar[:, 0][None, :]) & (cr[:, 0][:, None] != 0)
    out['rs_jac_max'] = wmax(J); out['rs_jac_wmean'] = wmean(J); out['rs_cov_max'] = wmax(C)
    out['rs_big_max'] = wmax(big.astype(np.float64))
    # ---- substructure containment on element graphs (top KSUB analogs, w = sim)
    ks = min(KSUB, k)
    amol = [_mols(_str_at(_ST['train_el'], int(sid[j])), _str_at(_ST['train_core'], int(sid[j]))) for j in range(ks)]
    cstr = _cand_strings(v)
    sca = np.zeros((n, ks)); sac = np.zeros((n, ks)); fca = np.zeros((n, ks)); fac = np.zeros((n, ks))
    for i in range(n):
        cm = _mols(*cstr[i])
        for j in range(ks):
            if not use[i, j]:
                continue
            sca[i, j] = _contains(cm, amol[j]); sac[i, j] = _contains(amol[j], cm)
            fca[i, j] = _contains_full(cm, amol[j]); fac[i, j] = _contains_full(amol[j], cm)
    W1 = np.where(use[:, :ks], w1[None, :ks], 0.0)
    out['sub_ca_max'] = (sca * W1).max(1); out['sub_ac_max'] = (sac * W1).max(1)
    out['subf_ca_max'] = (fca * W1).max(1); out['subf_ac_max'] = (fac * W1).max(1)
    # ---- formula relations
    okc = (cf >= 0).all(1); oka = (af >= 0).all(1)
    D = cf[:, None, :] - af[None, :, :]                        # (n, k, 10)
    L1 = np.abs(D).sum(2).astype(np.float64)
    L1 = np.where(okc[:, None] & oka[None, :], L1, 50.0)
    out['fd_wmean'] = np.where(anyu, wmean(L1), -1.0)
    out['fd_top'] = np.where(anyu, L1[rows, top], -1.0)
    out['fd_min'] = np.where(anyu, np.where(use, L1, 1e9).min(1), -1.0)
    X = (np.isin(_code(D), _ST['dcode']) & okc[:, None] & oka[None, :] & use).astype(np.float64)
    out['fx_max'] = wmax(X); out['fx_top'] = np.where(anyu, X[rows, top], 0.0)
    # ---- ring / heteroatom differences
    at = an[top]
    out['dr_ring_top'] = np.where(anyu, cn[:, NI['n_ring']] - at[:, NI['n_ring']], 0.0)
    out['dr_rs_top'] = np.where(anyu, cn[:, NI['n_rs']] - at[:, NI['n_rs']], 0.0)
    out['dr_core_top'] = np.where(anyu, cn[:, NI['n_core']] - at[:, NI['n_core']], 0.0)
    rhet = lambda M: M[..., NI['ring_n']] + M[..., NI['ring_o']] + M[..., NI['ring_s']]
    out['dr_rhet_top'] = np.where(anyu, rhet(cn) - rhet(at), 0.0)
    out['dr_ring_wabs'] = wmean(np.abs(cn[:, NI['n_ring']][:, None] - an[:, NI['n_ring']][None, :]))
    for nm in ('ring', 'rs', 'core', 'rhet'):                  # symmetric (relation-only) versions of dr_*_top
        out[f'da_{nm}_top'] = np.abs(out[f'dr_{nm}_top'])
    # ---- structural composite ("structure propagation", graph analogue of the base ap): per (candidate, analog)
    #      relation strength = mean of exact core, generic core, ring-system Jaccard / coverage and (top KSUB analogs)
    #      core / full containment either way; st_ap = max_j w4_j * R_ij over usable analogs; window rank / gap and gap
    #      to the best candidate with the SAME element composition (isomer group, grouped by exact element counts)
    Rel = np.zeros((n, k))
    Rel += (ch[:, 0][:, None] == ah[:, 0][None, :]) & (ch[:, 0][:, None] != 0)
    Rel += (ch[:, 1][:, None] == ah[:, 1][None, :]) & (ch[:, 1][:, None] != 0)
    Rel += J + C
    Rel[:, :ks] += 0.5 * (sca + sac) + 0.5 * (fca + fac)
    Rel /= 6.0
    st = wmax(Rel)
    out['st_ap'] = st; out['st_rank'] = _rank_norm(st); out['st_gap'] = st - st.max()
    fkey = np.where(okc[:, None], cf, -1)                      # exact element counts (bad formula -> one group)
    u_, inv = np.unique(fkey, axis=0, return_inverse=True); inv = np.asarray(inv).reshape(-1)
    gmax = np.full(len(u_), -np.inf); np.maximum.at(gmax, inv, st)
    out['st_fgap'] = st - gmax[inv]
    # ---- per-block Tanimoto (fingerprint-based: style sensitive)
    A = v.fp; B = v.an_fp[:k]
    for nm in BT_BLOCKS:
        sel = _ST['blk'][nm]
        a = A[:, sel].astype(np.float64); b = B[:, sel].astype(np.float64)
        it = a @ b.T
        T = it / np.maximum(a.sum(1)[:, None] + b.sum(1)[None, :] - it, 1e-9)
        out[f'bt_{nm}'] = wmax(T)
    return {nm: np.nan_to_num(np.asarray(x, np.float64), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            for nm, x in out.items()}

"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles: [...], keys: [...]}} (blend order, top 40)."""
import os, sys, json, time
import numpy as np
if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    import casmi_engine as E
    _orig_find = E.find
    def _find(n):                                  # v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan's 6,930
        if n == 'fp_bits.npy':
            return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    import glob
    fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if 'casmi26-fp-models-v2' in p)
    print('engine fp models', fp_models, flush=True)
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz'), fp_models,
                           workers=os.cpu_count(), w_pv=0.88)
    bl = E.blend_scores(S['pv'], S['ours'], 0.88)
    res = {}
    for r in recs:
        mid = r['mid']
        if mid not in bl:
            continue
        order = np.argsort(-bl[mid], kind='mergesort')[:80]
        smis, keys, seen = [], [], set()
        for c in r['cand'][order]:
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k)
            if len(smis) >= 40:
                break
        res[str(mid)] = dict(smiles=smis, keys=keys)
    json.dump(res, open(out, 'w'))
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)

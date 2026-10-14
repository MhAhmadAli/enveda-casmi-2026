"""PubChem-only channel for the final submission (runs in its own process so its casmi package
(S1 snapshot, FPNet A+B) never collides with the v1 engine's casmi package).

Per molecule: z = A+B logits (probe_core2.molecule_logits, identical to the probe that scored 0.037);
best_pool_fz = max f.z over the pool window (+-10 ppm, fallback 30 ppm) with the same z (as P1 logits.py);
PubChem-only list = probe_core2.probe_one (top 25 by f.z, keys not in the pool) with f.z values and score keys.
Writes a json {molecule_id: {pc, pc_fz, pc_keys, best_pool_fz, target}}.
usage: python pc_runner.py <v3_code_dir> <v3_models_dir> <pool_dir> <pubchem_dir> <test_parquet> <out_json> <n_workers> [max_mols]
"""
import os, sys, json, time
import numpy as np, pandas as pd

if __name__ == '__main__':
    V3CODE, V3MOD, POOL_DIR, PC_DIR, TEST, OUT, NW = sys.argv[1:8]
    MAXM = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    NW = int(NW)
    sys.path.insert(0, V3CODE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch, glob
    import probe_core2 as pc
    from casmi import fpnet, chem
    T0 = time.time()
    models = sorted(glob.glob(os.path.join(V3MOD, 'fpnet_*.pt')))
    assert len(models) == 2, models
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bank = fpnet.ModelBank(models, device=dev)
    bits = np.load(os.path.join(POOL_DIR, 'fp_bits.npy')); nbits = len(bits)
    pmass = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['mass']).mass.values.astype(np.float64)
    assert np.all(np.diff(pmass) >= 0), 'pool not mass-sorted'
    pfp = np.load(os.path.join(POOL_DIR, 'pool_fp.npy'), mmap_mode='r')
    te = pd.read_parquet(TEST)
    te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
    tasks, extra = [], {}
    groups = list(te.groupby('molecule_id', sort=False))
    if MAXM:
        groups = groups[:MAXM]
    for mid, sub in groups:
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms):
            continue
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        spectra = spectra[:16]
        try:
            z = pc.molecule_logits(bank, spectra)
        except Exception as e:
            print('ERROR logits', mid, repr(e), flush=True); z = None
        if z is None:
            continue
        tol = target * 10e-6
        a = np.searchsorted(pmass, target - tol, 'left'); b = np.searchsorted(pmass, target + tol, 'right')
        if b <= a:
            tol = target * 30e-6
            a = np.searchsorted(pmass, target - tol, 'left'); b = np.searchsorted(pmass, target + tol, 'right')
        if b > a:
            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)
            best = float((F @ z.astype(np.float64)).max())
        else:
            best = float('nan')
        extra[mid] = dict(best_pool_fz=best, target=target)
        tasks.append((mid, target, z))
    del bank; torch.cuda.empty_cache()
    print(f'logits for {len(tasks)} molecules {time.time()-T0:.0f}s', flush=True)
    res = {}
    import multiprocessing as mp
    ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
    with ctx.Pool(NW, initializer=pc.init_worker, initargs=(PC_DIR, os.path.join(POOL_DIR, 'fp_bits.npy'),
                                                            os.path.join(POOL_DIR, 'pool_meta.parquet'), V3CODE)) as P:
        for k, (mid, smis, fzs, keys, d) in enumerate(P.imap_unordered(pc.probe_one, tasks, chunksize=1)):
            res[mid] = dict(pc=smis, pc_fz=fzs, pc_keys=keys, fz_top=d.get('fz_top'), S=d.get('S'), top_pop=d.get('top_pop', 0.0), **extra[mid])
            if (k + 1) % 25 == 0 or k + 1 == len(tasks):
                print(f'  pubchem {k+1}/{len(tasks)} {time.time()-T0:.0f}s', flush=True)
    json.dump(res, open(OUT, 'w'))
    print('pc_runner done', len(res), f'{time.time()-T0:.0f}s', flush=True)

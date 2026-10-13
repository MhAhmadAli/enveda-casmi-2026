"""Build the cleaned spectrum cache (used by scripts/03 and the Kaggle notebook).

Cleaning: drop peaks > precursor+2 Da, relative floor 0.1% of base peak, keep top-512 by
intensity, sort by m/z, intensities normalised to base peak = 1.
"""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
from numba import njit
from . import chem
from .spectra import clean_peaks

LIBS = ['enveda-180', 'pluskal_ms2', 'riken', 'gnps', 'massbank', 'mona', 'spectraverse',
        'msdial', 'drug_plus', 'enveda-np-examples', 'masaryk']
LIB_IX = {l: i for i, l in enumerate(LIBS)}
INSTR = ['timsTOF', 'Orbitrap', 'QTOF', 'IT', 'QQQ', 'other']


def instr_family(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return 5
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t or 'ft-icr' in t or 'fticr' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qit' in t or 'lit' in t: return 3
    if 'qq' in t or 'quattro' in t or 'triple' in t: return 4
    return 5


@njit(cache=True)
def clean_block(off_in, mz_in, it_in, prec, floor, topk, over):
    n = len(off_in) - 1
    out_off = np.zeros(n + 1, np.int64)
    mz_out = np.empty(len(mz_in), np.float32)
    it_out = np.empty(len(mz_in), np.float32)
    pos = 0
    for i in range(n):
        a = off_in[i]; b = off_in[i + 1]
        m, v = clean_peaks(mz_in[a:b], it_in[a:b], prec[i], floor, topk, over)
        k = len(m)
        mz_out[pos:pos + k] = m
        it_out[pos:pos + k] = v
        pos += k
        out_off[i + 1] = pos
    return out_off, mz_out[:pos], it_out[:pos]


def build_spec_cache(train_parquet='data/train.parquet', workdir='work', structs_parquet=None):
    """Build cleaned CSR peak arrays + metadata from train.parquet into workdir."""
    structs_parquet = structs_parquet or f'{workdir}/train_structs.parquet'
    t0 = time.time()
    st = pd.read_parquet(structs_parquet, columns=['inchikey14', 'sid'])
    k2s = dict(zip(st.inchikey14, st.sid))
    pf = pq.ParquetFile(train_parquet)
    cols = ['inchikey14', 'ingest_lib', 'instrument_type', 'adduct', 'ionization_mode', 'precursor_mz',
            'precursor_error_ppm', 'collision_energy_ev', 'collision_energy_orig_units',
            'base_peak_intensity', 'num_peaks', 'ms2_mzs', 'ms2_normalized_intensities']
    metas, offs, mzs, its = [], [], [], []
    total = 0
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        mzc = t.column('ms2_mzs').combine_chunks()
        itc = t.column('ms2_normalized_intensities').combine_chunks()
        off = mzc.offsets.to_numpy().astype(np.int64)
        mz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float64)
        it = itc.values.to_numpy(zero_copy_only=False).astype(np.float64)
        prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
        o2, m2, i2 = clean_block(off, mz, it, prec, 0.001, 512, 2.0)
        d = t.drop_columns(['ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
        ce = d.collision_energy_ev
        d['ce_mean'] = [float(np.mean(x)) if x is not None and len(x) else np.nan for x in ce]
        d['ce_min'] = [float(np.min(x)) if x is not None and len(x) else np.nan for x in ce]
        d['ce_max'] = [float(np.max(x)) if x is not None and len(x) else np.nan for x in ce]
        d['ce_n'] = [len(x) if x is not None else 0 for x in ce]
        d = d.drop(columns=['collision_energy_ev'])
        d['sid'] = d.inchikey14.map(k2s).fillna(-1).astype(np.int64)
        d['lib'] = d.ingest_lib.map(LIB_IX).astype(np.int8)
        d['instr'] = d.instrument_type.map(instr_family).astype(np.int8)
        d['adduct_ix'] = d.adduct.map(chem.adduct_index).astype(np.int16)
        d['mode'] = np.where(d.ionization_mode == 'positive', 1, -1).astype(np.int8)
        d['nm'] = chem.neutral_mass(d.precursor_mz.values, d.adduct.values)
        d['n_clean'] = np.diff(o2).astype(np.int32)
        d = d.drop(columns=['inchikey14', 'ingest_lib', 'instrument_type', 'ionization_mode'])
        metas.append(d); offs.append(o2[1:] + total); mzs.append(m2); its.append(i2)
        total += len(m2)
        print(f'rg {rg}: {len(d)} spectra, {len(m2)} peaks, total peaks {total}, {time.time()-t0:.0f}s', flush=True)
    meta = pd.concat(metas, ignore_index=True)
    off = np.concatenate([[0]] + offs).astype(np.int64)
    mz = np.concatenate(mzs); it = np.concatenate(its)
    meta.to_parquet(f'{workdir}/spec_meta.parquet', index=False)
    np.save(f'{workdir}/spec_off.npy', off); np.save(f'{workdir}/spec_mz.npy', mz); np.save(f'{workdir}/spec_it.npy', it)
    print('spectra', len(meta), 'peaks', len(mz), 'sid<0:', (meta.sid < 0).sum(), 'nm nan:', meta.nm.isna().sum())
    print('done', f'{time.time()-t0:.0f}s')

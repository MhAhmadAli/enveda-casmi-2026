"""Generate a validation build of the public 0.438 fork, scored on the enveda-np-examples panel.

The fork reads every input from one directory, COMP (test.parquet, train.parquet,
sample_submission.csv). The validation build points COMP at a directory built in the
kernel from train.parquet:

  test.parquet   the panel's spectra in the real test schema; one molecule per
                 (structure, ionization mode)
  train.parquet  train minus the answers' spectra:
                   disjoint  every spectrum of every panel structure (Class 2)
                   inlib     only the panel library's own spectra (Class 1)

Models that may have seen the panel are swapped for documented held-out ones
(reports/model_provenance): fpnet_full1 -> FPNet A (fpnet_0.pt, panel held out by its
training code); engine 2's prvsiyan fp models -> megayak's leak-free retrains.
Residual leaks that cannot be swapped: fe_D (DreaMS), ICEBERG/GLACIER checkpoints,
rankers trained on simulated rows that may include panel molecules.

Every stage is snapshotted, scored with the official metric identity, and dumped to
/kaggle/working/val for offline replay.

usage: python scripts/build_val_kernel.py
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "notebooks/kernels/fork438"
OWNER = "ahmadali1243534"
# kernel name -> (regime, model variant)
#   base      the fork's models, with fpnet_full1 / engine-2 fp models swapped for held-out ones
#   nod       + fe_D (DreaMS fine-tune, training set undocumented) replaced by fe_A
#   foldsafe  + every FPNet the main engine uses (bank, fe_A/B/D) replaced by fpnet_fold0, which
#             is documented to hold out the panel (dmitriigluzdov/casmi26-fold-safe-fpnet)
VARIANTS = {"disjoint": ("disjoint", "base"), "inlib": ("inlib", "base"),
            "disjoint_nod": ("disjoint", "nod"), "disjoint_foldsafe": ("disjoint", "foldsafe")}
FOLDSAFE_DATASET = "dmitriigluzdov/casmi26-fold-safe-fpnet"

VAL_BUILD = r'''# ===== VALIDATION BUILD (scripts/build_val_kernel.py): panel queries, answers removed =====
VAL_REGIME = '__REGIME__'
VAL_VARIANT = '__VARIANT__'
VAL_PANEL_LIB = 'enveda-np-examples'
import pyarrow as pa, pyarrow.parquet as pq, numpy as np, pandas as pd, shutil
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_VT = rdMolStandardize.TautomerEnumerator()


def val_key(smi):
    """The official metric identity: strip stereo, tautomer-canonicalise, InChIKey14."""
    try:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is None:
            return None
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(_VT.Canonicalize(m))[:14]
    except (ValueError, RuntimeError, TypeError):
        return None


def val_mass(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Descriptors.ExactMolWt(m) if m is not None else np.nan


_FOLD0 = sorted(glob.glob('/kaggle/input/**/fpnet_fold0.pt', recursive=True))
VAL_FE_SWAP, VAL_BANK = {}, None
if VAL_VARIANT == 'nod':
    VAL_FE_SWAP = {'fe_D.pt': os.path.join(V4MOD, 'fe_A.pt')}
elif VAL_VARIANT == 'foldsafe':
    assert _FOLD0, 'fpnet_fold0.pt is not mounted'
    VAL_FE_SWAP = {f: _FOLD0[0] for f in ('fe_A.pt', 'fe_B.pt', 'fe_D.pt')}
    VAL_BANK = _FOLD0[0]
print('VALIDATION variant', VAL_VARIANT, VAL_FE_SWAP, VAL_BANK)

_t = time.time()
_REAL_TEST, _REAL_TRAIN = os.path.join(COMP, 'test.parquet'), os.path.join(COMP, 'train.parquet')
VCOMP = '/tmp/vcomp'
os.makedirs(VCOMP, exist_ok=True); os.makedirs('/kaggle/working/val', exist_ok=True)
_test_schema = pq.read_schema(_REAL_TEST)
_pf = pq.ParquetFile(_REAL_TRAIN)
_tcols = _pf.schema_arrow.names
print('train columns:', _tcols); print('test columns:', _test_schema.names)
_SMI = 'normalized_smiles' if 'normalized_smiles' in _tcols else 'smiles'
_light = _pf.read(columns=['ingest_lib', _SMI]).to_pandas()
_is_panel = (_light.ingest_lib == VAL_PANEL_LIB).to_numpy()
_panel_smis = pd.unique(_light[_SMI][_is_panel])
_pkey = {s: val_key(s) for s in _panel_smis}
_panel_keys = {k for k in _pkey.values() if k}
assert len(_panel_keys) > 200, len(_panel_keys)

# Same structure in another library may be drawn differently: compare official keys of
# every train structure within 0.01 Da of a panel structure.
_pm = np.sort(np.array([val_mass(s) for s in _panel_smis])); _pm = _pm[np.isfinite(_pm)]
_u = pd.unique(_light[_SMI])
_um = np.array([val_mass(s) for s in _u])
_ix = np.searchsorted(_pm, _um)
_near = np.zeros(len(_u), bool)
for _d in (-1, 0):
    _j = np.clip(_ix + _d, 0, len(_pm) - 1)
    _near |= np.abs(_pm[_j] - _um) <= 0.01
_same = {s for s, ok in zip(_u, _near) if ok and val_key(s) in _panel_keys}
_is_struct = _light[_SMI].isin(_same).to_numpy() | _is_panel
_drop = _is_struct if VAL_REGIME == 'disjoint' else _is_panel
print(f'panel: {_is_panel.sum()} spectra / {len(_panel_keys)} structures; same-structure spectra in train '
      f'{_is_struct.sum()}; dropping {_drop.sum()} ({VAL_REGIME}); {time.time()-_t:.0f}s', flush=True)

_w, _off, _panel_tabs = None, 0, []
_want = [c for c in _test_schema.names if c in _tcols]
for _rg in range(_pf.num_row_groups):
    _tb = _pf.read_row_group(_rg)
    _n = _tb.num_rows
    _pmask = _is_panel[_off:_off + _n]
    if _pmask.any():
        _panel_tabs.append(_tb.filter(pa.array(_pmask)).select(_want + ([_SMI] if _SMI not in _want else [])))
    _tb = _tb.filter(pa.array(~_drop[_off:_off + _n]))
    _off += _n
    if _w is None:
        _w = pq.ParquetWriter(os.path.join(VCOMP, 'train.parquet'), _tb.schema)
    _w.write_table(_tb)
_w.close()
assert _off == len(_light)
del _light

_pt = pa.concat_tables(_panel_tabs).to_pandas()
_pt['_key'] = _pt[_SMI].map(_pkey)
_pt = _pt[_pt._key.notna()].reset_index(drop=True)
_pt['molecule_id'] = ['v_' + hashlib.md5(f'{k}|{m}'.encode()).hexdigest()[:12]   # no answer in the id
                      for k, m in zip(_pt._key, _pt.ionization_mode)]
if 'spectrum_id' not in _tcols:
    _pt['spectrum_id'] = [f'{m}_{i}' for i, m in enumerate(_pt.molecule_id)]
_missing = [c for c in _test_schema.names if c not in _pt.columns]
assert not _missing, f'test columns absent from train: {_missing}'
_te_tab = pa.Table.from_pandas(_pt[_test_schema.names], preserve_index=False).cast(_test_schema)
pq.write_table(_te_tab, os.path.join(VCOMP, 'test.parquet'))
_truth = _pt.groupby('molecule_id').agg(smiles=(_SMI, 'first'), key=('_key', 'first'))
_truth.reset_index().to_json('/kaggle/working/val/truth.json', orient='records')
_sample = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'), nrows=1)
_ss = pd.DataFrame({'molecule_id': _truth.index})
for _c in _sample.columns:
    if _c != 'molecule_id':
        _ss[_c] = 'CCO'
_ss.to_csv(os.path.join(VCOMP, 'sample_submission.csv'), index=False)
COMP = VCOMP
print(f'VALIDATION {VAL_REGIME}: {len(_pt)} query spectra, {len(_truth)} molecules, '
      f'{_pt._key.nunique()} structures -> COMP={COMP} ({time.time()-_t:.0f}s)', flush=True)
'''

SNAP_POP = '''# VALIDATION: snapshot the base lists after the popularity re-rank (cell 7 overwrites BASE)
import copy
BASE_POP = copy.deepcopy(BASE)
'''
SNAP_SIM = '''# VALIDATION: snapshot the base lists after the ICEBERG/GLACIER re-rank
BASE_SIM = copy.deepcopy(BASE)
'''
SNAP_PC = '''# VALIDATION: snapshot after the gated PubChem merge
shutil.copy('submission.csv', '/kaggle/working/val/stage_pcmerge.csv')
'''

EVAL = r'''# ===== VALIDATION: score every stage with the official metric identity, dump for replay =====
import collections
VAL_TRUTH = {r['molecule_id']: r['key'] for r in json.load(open('/kaggle/working/val/truth.json'))}
_kc, _cc = {}, {}


def _comp(s):
    """Heavy-atom composition: the official metric's cheap gate before canonicalising."""
    if s not in _cc:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        _cc[s] = None if m is None else tuple(sorted(collections.Counter(
            a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() != 'H').items()))
    return _cc[s]


def _vk(s):
    if s not in _kc:
        _kc[s] = val_key(s)
    return _kc[s]


VAL_COMP = {m: _comp(r['smiles']) for m, r in
            ((r['molecule_id'], r) for r in json.load(open('/kaggle/working/val/truth.json')))}


def _rank(smis, key, k=25, comp=None):
    """Position of the first guess with the answer's identity (the metric counts positions)."""
    for r, s in enumerate(smis[:k], 1):
        if comp is not None and _comp(s) != comp:
            continue
        if _vk(s) == key:
            return r
    return 0


def _csv(p):
    d = pd.read_csv(p)
    return {str(m): [x for x in str(s).split(';') if x] for m, s in zip(d.molecule_id, d.smiles)}


STAGES = {
    'raw_main': {m: v[0] for m, v in RAW_BASE.items()},
    'pop_main': {m: v[0] for m, v in BASE_POP.items()},
    'sim_main': {m: v[0] for m, v in BASE_SIM.items()},
    'engine2': {m: v['smiles'] for m, v in ENG.items()},
    'pcmerge': _csv('/kaggle/working/val/stage_pcmerge.csv'),
    'fused': _csv('baseline_submission.csv'),
    'final': _csv('submission.csv'),
}
_rows, RES = [], {}
for _name, _lists in STAGES.items():
    _r = np.array([_rank(_lists.get(m, []), k, 25, VAL_COMP[m]) for m, k in VAL_TRUTH.items()])
    _r80 = np.array([_rank(_lists.get(m, []), k, 80, VAL_COMP[m]) for m, k in VAL_TRUTH.items()])
    RES[_name] = dict(n=len(_r), mrr25=float(np.mean(np.where(_r > 0, 1.0 / np.maximum(_r, 1), 0.0))),
                      top1=float(np.mean(_r == 1)), top5=float(np.mean((_r > 0) & (_r <= 5))),
                      top25=float(np.mean(_r > 0)), in_list=float(np.mean(_r80 > 0)))
    _rows += [dict(stage=_name, molecule_id=m, rank=int(x)) for m, x in zip(VAL_TRUTH, _r)]
_libmax = {m: float(v[2]) for m, v in RAW_BASE.items()}
RES['_meta'] = dict(regime=VAL_REGIME, variant=VAL_VARIANT, molecules=len(VAL_TRUTH), runtime_s=round(time.time() - T0),
                    lib_max_ge_0_9=float(np.mean([_libmax.get(m, 0) >= 0.9 for m in VAL_TRUTH])))
pd.DataFrame(_rows).to_csv('/kaggle/working/val/ranks.csv', index=False)
json.dump(RES, open('/kaggle/working/val/results.json', 'w'), indent=1)
print(json.dumps(RES, indent=1))

_raw = {m: dict(smiles=v[0], keys=v[1], lib_max=v[2], score=v[3], formula=v[4], pid=v[5],
                pop=[float(POOL_POP[p]) if p >= 0 else float('nan') for p in v[5]])
        for m, v in RAW_BASE.items()}
pickle.dump(dict(raw=_raw, base_pop=BASE_POP, base_sim=BASE_SIM, eng=ENG, pc=PC, ice=ICE_SCORES,
                 gl=GL_SCORES, fusion_overflow=FUSION_OVERFLOW, truth=VAL_TRUTH,
                 cfg=dict(TOPN=TOPN, POOLPOP_MU=POOLPOP_MU, ICE_LAM=ICE_LAM, GL_LAM=GL_LAM, ALPHA=ALPHA,
                          KRR=KRR, FUSE_N=FUSE_N, LIB_TAU=LIB_TAU, POP_LAM=POP_LAM)),
            open('/kaggle/working/val/dump.pkl', 'wb'))

# Leave only small outputs: the spectrum caches here are multi-GB and make the output
# listing time out.
for _e in os.listdir('/kaggle/working'):
    _p = os.path.join('/kaggle/working', _e)
    if os.path.isdir(_p) and _e != 'val':
        shutil.rmtree(_p, ignore_errors=True)
print('kept:', sorted(os.listdir('/kaggle/working')))
'''

FULL1 = "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m"
FULL1_VAL = "_FULL = []   # VALIDATION: fpnet_full1 may have seen the panel; FPNet A (fpnet_0) held it out"
ENG_FP = ("fp_models = sorted(p for p in glob.glob(\\'/kaggle/input/**/fp_*.pt\\', recursive=True) "
          "if \\'casmi26-fp-models-v2\\' in p)")
ENG_FP_VAL = ("fp_models = sorted(p for p in glob.glob(\\'/kaggle/input/**/ours_fpnet_*.pt\\', recursive=True) "
              "if \\'casmi26-simulated-ranker-rows\\' in p)  # VALIDATION: leak-free retrains")
BANK = "os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)"
BANK_VAL = "VAL_BANK or os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)"
FE = "V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)"
FE_VAL = ("_FE = [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']]\n"
          "_FE = [VAL_FE_SWAP.get(os.path.basename(p), p) for p in _FE]\n"
          "print('VALIDATION model views:', _FE, flush=True)\n"
          "V = v1engine.V1FE(E, _FE, MAN['families'], device=dev)")
RAW_INIT = "BASE, n_err = {}, 0"
RAW_HOOK = "    if POOLPOP_MU > 0 and len(scs) > 1:"


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def replace_once(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"expected exactly one match for: {old[:80]}"
    return src.replace(old, new)


def build(regime: str, variant: str) -> dict:
    nb = json.loads((FORK / "fork438.ipynb").read_text())
    cells = nb["cells"]
    code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    src = {i: "".join(cells[i]["source"]) for i in code}
    c = dict(enumerate(code))  # code-cell ordinal -> notebook index
    assert src[c[1]].startswith("# SMOKE switch") and "def fuse2" in src[c[9]]

    src[c[2]] = replace_once(src[c[2]], ENG_FP, ENG_FP_VAL)
    src[c[5]] = replace_once(src[c[5]], FULL1, FULL1_VAL)
    src[c[5]] = replace_once(src[c[5]], BANK, BANK_VAL)
    src[c[5]] = replace_once(src[c[5]], FE, FE_VAL)
    s6 = replace_once(src[c[6]], RAW_INIT, RAW_INIT + "\nRAW_BASE = {}")
    src[c[6]] = replace_once(s6, RAW_HOOK, "    RAW_BASE[mid] = (list(smis), list(keys), lib_max, list(scs), "
                                           "list(forms), list(pids_))\n" + RAW_HOOK)
    for i in code:
        cells[i]["source"] = src[i]

    inserts = {c[0]: VAL_BUILD.replace("__REGIME__", regime).replace("__VARIANT__", variant), c[6]: SNAP_POP, c[7]: SNAP_SIM,
               c[8]: SNAP_PC, c[11]: EVAL}
    for after in sorted(inserts, reverse=True):
        cells.insert(after + 1, code_cell(inserts[after]))
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"# CASMI26 fork validation ({regime}, models: {variant})\n\nGenerated by `scripts/build_val_kernel.py`. "
                     "Runs the 0.438 fork on the enveda-np-examples panel with the answers' spectra removed "
                     "and scores every stage.\n"})
    for cell in cells:
        if cell["cell_type"] == "code":
            body = "\n".join(l for l in "".join(cell["source"]).splitlines() if not l.lstrip().startswith(("!", "%")))
            ast.parse(body)
    return nb


def main() -> None:
    meta = json.loads((FORK / "kernel-metadata.json").read_text())
    for name, (regime, variant) in VARIANTS.items():
        out = ROOT / f"notebooks/kernels/val_{name}"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"val_{name}.ipynb").write_text(json.dumps(build(regime, variant), indent=1))
        slug = name.replace("_", "-")
        m = dict(meta, id=f"{OWNER}/casmi26-val-{slug}", title=f"CASMI26 val {slug}",
                 code_file=f"val_{name}.ipynb", is_private=True,
                 dataset_sources=meta["dataset_sources"] + [FOLDSAFE_DATASET])
        (out / "kernel-metadata.json").write_text(json.dumps(m, indent=2))
        print("wrote", out)


if __name__ == "__main__":
    main()

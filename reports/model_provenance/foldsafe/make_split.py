"""Work directory for fold-safe FPNet training: links the public spectrum cache and pool, writes work/split.parquet.

usage: python make_split.py <harness_work_dir> <folds structures.parquet> <run_dir> <held_out_fold>
Held out: every training structure of identity fold <held_out_fold> ('hold') and every NP-example structure ('np').
"""
import os, sys
import numpy as np
import pandas as pd

SRC, FOLDS, RUN, FOLD = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
wd = os.path.join(RUN, 'work'); os.makedirs(wd, exist_ok=True)
for f in os.listdir(SRC):
    dst = os.path.join(wd, f)
    if not os.path.lexists(dst):
        os.symlink(os.path.realpath(os.path.join(SRC, f)), dst)
ts = pd.read_parquet(os.path.join(wd, 'train_structs.parquet'))
st = pd.read_parquet(FOLDS, columns=['inchikey14', 'identity_group', 'fold_identity']).drop_duplicates('inchikey14')
j = ts.merge(st, on='inchikey14', how='left')
assert j.fold_identity.notna().all() and (j.sid.values == np.arange(len(j))).all()
assert j.groupby('key').fold_identity.nunique().max() == 1
is_np = (j.libmask.astype('int64') & (1 << 9)) != 0
np_groups = set(j.identity_group[is_np])
fold = np.where(j.identity_group.isin(np_groups), 'np', np.where(j.fold_identity == FOLD, 'hold', 'train'))
out = pd.DataFrame(dict(sid=j.sid.values, fold=fold))
out[out.fold != 'train'].to_parquet(os.path.join(wd, 'split.parquet'), index=False)
print(out.fold.value_counts().to_dict())

# -*- coding: utf-8 -*-
"""b10_update_pools.py — 把南方裸地层并入 8 个年度训练池
* 输入：data/pools/train_<year>.parquet、data/yearly/topup_bare_south_<year>.parquet
* 输出：同名文件（追加 class_new=201 / src='multisrc_s2_bare_south'；50 m 去重）
"""
import os, sys
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
sys.stdout.reconfigure(encoding='utf-8')
BASE=r'F:/lc_work/bare_sparse_v3'; FEATS=['A%02d'%i for i in range(64)]
def kmxy(lon,lat): return np.column_stack([lon*111.32*np.cos(np.radians(lat)), lat*110.57])
for y in range(2017,2025):
    fp=os.path.join(BASE,'data','pools','train_%d.parquet'%y)
    tf=os.path.join(BASE,'data','yearly','topup_bare_south_%d.parquet'%y)
    if not (os.path.isfile(fp) and os.path.isfile(tf)): print(y,'skip'); continue
    base=pd.read_parquet(fp)
    t=pd.read_parquet(tf)
    # 去重（与池内 class_new=201 的 250m）
    b=base[base.class_new==201]
    tree=cKDTree(kmxy(b.lon.to_numpy(), b.lat.to_numpy()))
    dd,_=tree.query(kmxy(t.lon.to_numpy(), t.lat.to_numpy()), k=1, distance_upper_bound=0.25)
    keep=~np.isfinite(dd); t=t[keep].reset_index(drop=True)
    add=pd.DataFrame({'row_id':30000000+y*100000+np.arange(len(t)),'class_new':201,
                      'lon':t.lon,'lat':t.lat,'year':y,'tier':'strong','src':'multisrc_s2_bare_south',
                      'src_conf':0.95,'train_weight':1.0,'qc_status':'keep'})
    for c in FEATS: add[c]=t[c].astype('float32')
    out=pd.concat([base,add[base.columns]], ignore_index=True, sort=False)
    out.to_parquet(fp, index=False)
    print('%d: +%d（去重剔 %d）→ %d' % (y, len(add), int((~keep).sum()), len(out)), flush=True)

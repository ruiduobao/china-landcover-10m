# -*- coding: utf-8 -*-
"""p4_neighbor.py — 已有点邻域撒点（仅 A 档，每原点 2 点，距离 80-120m）
* 输入 forest6_final.parquet（doc88 交付 97,354 点）
* 规则：只对 quality_tier=A；每原点 2 点；偏移 80-120m 随机方位；标 is_neighbor/nbr_of
* 输出 cand_nbr.parquet
"""
import numpy as np, pandas as pd, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
F=r'F:/lc_work/nb_national_2026/forest6/forest6_final.parquet'
OUT=r'Z:/nb_national_2026/forest6_10w/cand_nbr.parquet'
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
f=pd.read_parquet(F,columns=['uid','lon','lat','cls6','quality_tier','prov','eco','nf20','ever_share20'])
a=f[f.quality_tier=='A'].copy()
emit('A 档 %d 点（总 %d）'%(len(a),len(f)))
rng=np.random.default_rng(20261027)
rows=[]
MLAT=111320.0; MLON=111320.0
for k in range(2):
    dist=rng.uniform(80,120,len(a))
    brg=rng.uniform(0,2*np.pi,len(a))
    lat2=a.lat.values+dist*np.cos(brg)/MLAT
    lon2=a.lon.values+dist*np.sin(brg)/(MLON*np.cos(np.radians(a.lat.values)))
    d=pd.DataFrame({'uid':a.uid.values+k*10_000_000,'lon':lon2,'lat':lat2,'cls6':a.cls6.values,
        'quality_tier':'B','prov':a.prov.values,'eco':a.eco.values,'nf20':a.nf20.values,
        'ever_share20':a.ever_share20.values,'src':'neighbor','is_neighbor':1,'nbr_of':a.uid.values})
    rows.append(d)
n=pd.concat(rows,ignore_index=True)
n['dist_to_parent_m']=80.0  # 记录名义值
n.to_parquet(OUT,index=False)
emit('写 %d 邻域点 → %s'%(len(n),OUT))
print('per class:',n.cls6.value_counts().sort_index().to_dict())

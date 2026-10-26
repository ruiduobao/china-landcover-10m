# -*- coding: utf-8 -*-
"""e4b_merge.py — 重建候选池：41/43/44 从全池取，42 混合扩样，45/46 用 ESA 判定
* 目标：每类 ~110,000（留 10% 余量给清洗剔除）
"""
import os,sys,glob,json,time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT=r'Z:/nb_national_2026/forest6_10w'
V1=r'F:/lc_work/nb_national_2026/forest6/forest6_v1.parquet'
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
TARGET=110_000
def main():
    v=pd.read_parquet(V1,columns=['uid','lon','lat','cls6_train','tier','prov','eco','south_needle_flag'])
    rng=np.random.default_rng(20261027)
    parts=[]
    # --- 41/43/44：从全池取（排除 south_needle_flag）
    for c in (41,43,44):
        g=v[v.cls6_train==c]
        if c==41: g=g[g.south_needle_flag==0]
        # tier 优先 gold>silver>bronze>external
        pri={'gold':0,'silver':1,'bronze':2,'external':3}
        g=g.assign(_p=g.tier.map(pri)).sort_values('_p')
        take=g.head(TARGET).drop(columns='_p').copy()
        take['cls6']=c; take['src']='pool_v1'; take['quality_tier']=take.tier.map(
            {'gold':'A','silver':'A','bronze':'B','external':'C'}).fillna('C')
        parts.append(take[['uid','lon','lat','cls6','quality_tier','prov','eco','src']])
        emit(f'  cls{c}: 池 {len(g):,} → 取 {len(take):,}')
    # --- 42：现有池 58,277 全用 + FCS10 扩样补足
    g42=v[v.cls6_train==42].copy()
    g42['cls6']=42; g42['src']='pool_v1'
    g42['quality_tier']=g42.tier.map({'gold':'A','silver':'A','bronze':'B','external':'C'}).fillna('C')
    need42=TARGET-len(g42)
    d42=pd.read_parquet(os.path.join(OUT,'cand_42_fcs10_0_40.parquet'))
    d42=d42.sample(min(need42,len(d42)),random_state=20261027).copy()
    d42['cls6']=42; d42['src']='fcs10_scan'; d42['quality_tier']='B'
    d42['prov']=''; d42['eco']=''
    d42['uid']=np.arange(len(d42))+1_000_000_000
    parts.append(g42[['uid','lon','lat','cls6','quality_tier','prov','eco','src']])
    parts.append(d42[['uid','lon','lat','cls6','quality_tier','prov','eco','src']])
    emit(f'  cls42: 池 {len(g42):,} + 扩样 {len(d42):,} = {len(g42)+len(d42):,}')
    # --- 45/46：用 ESA 判定（esa_judge.parquet）
    ej=pd.read_parquet(os.path.join(OUT,'esa_judge.parquet'))
    emit(f'  ESA 判定：45={int((ej.cls45_46==45).sum()):,} 46={int((ej.cls45_46==46).sum()):,} 未定={int((ej.cls45_46==0).sum()):,}')
    # 现有 45/46
    for c in (45,46):
        g=v[v.cls6_train==c].copy(); g['cls6']=c; g['src']='pool_v1'
        g['quality_tier']=g.tier.map({'gold':'A','silver':'A','bronze':'B','external':'C'}).fillna('C')
        parts.append(g[['uid','lon','lat','cls6','quality_tier','prov','eco','src']])
    for c in (45,46):
        g=ej[ej.cls45_46==c].copy()
        cap=max(0,TARGET-len(v[v.cls6_train==c]))
        if len(g)>cap: g=g.sample(cap,random_state=20261027)
        g['cls6']=c; g['src']='esa_fcs10'; g['quality_tier']='B'
        g['prov']=''; g['eco']=''
        g['uid']=g.uid+2_000_000_000
        parts.append(g[['uid','lon','lat','cls6','quality_tier','prov','eco','src']])
        emit(f'  cls{c}: 池 {len(v[v.cls6_train==c]):,} + ESA {len(g):,}')
    allp=pd.concat(parts,ignore_index=True)
    allp['k']=allp.lon.round(6).astype(str)+'_'+allp.lat.round(6).astype(str)
    allp=allp.drop_duplicates('k').drop(columns='k')
    emit('去重后 %d'%len(allp))
    print(allp.cls6.value_counts().sort_index().to_dict())
    allp.to_parquet(os.path.join(OUT,'all_candidates.parquet'),index=False)
    allp[['uid','lon','lat']].to_parquet(os.path.join(OUT,'_pending_aef.parquet'),index=False)
    emit('写 _pending_aef.parquet（%d）'%len(allp))
if __name__=='__main__': main()

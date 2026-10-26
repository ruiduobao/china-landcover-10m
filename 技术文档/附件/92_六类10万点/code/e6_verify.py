# -*- coding: utf-8 -*-
"""e6_verify.py — C1-C10 验证"""
import os,sys,json,time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,confusion_matrix
sys.stdout.reconfigure(encoding='utf-8')
OUT=r'Z:/nb_national_2026/forest6_10w'
AEF=['A%02d'%i for i in range(64)]
SEED=20261027
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
def rf(): return RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_features='sqrt',
                                        class_weight='balanced',random_state=SEED,n_jobs=10)
def grp(lon,lat,b): return (np.floor(np.asarray(lon)/b).astype('int64')*100000+np.floor(np.asarray(lat)/b).astype('int64'))
def cv(X,y,g,fold=5):
    uq=np.unique(g); rng=np.random.default_rng(SEED); perm=rng.permutation(uq)
    folds=[list(x) for x in np.array_split(perm,fold)]
    oof=np.full(len(y),-1,dtype=int)
    for f in folds:
        te=np.isin(g,f); tr=~te
        if te.sum()==0 or len(np.unique(y[tr]))<2: continue
        m=rf().fit(X[tr],y[tr]); oof[te]=m.predict(X[te])
    ok=oof>=0
    return dict(n=int(ok.sum()),oa=round(float(accuracy_score(y[ok],oof[ok])),4),
                macro_f1=round(float(f1_score(y[ok],oof[ok],average='macro')),4))
def pair(X,y,g,a,b):
    m=np.isin(y,[a,b]); Xs,ys,gs=X[m],(y[m]==b).astype(int),g[m]
    uq=np.unique(gs); rng=np.random.default_rng(SEED); perm=rng.permutation(uq)
    folds=[list(x) for x in np.array_split(perm,min(5,len(uq)))]
    oof=np.full(len(ys),-1.0)
    for f in folds:
        te=np.isin(gs,f); tr=~te
        if te.sum()==0 or len(np.unique(ys[tr]))<2: continue
        m2=rf().fit(Xs[tr],ys[tr]); oof[te]=m2.predict_proba(Xs[te])[:,1]
    ok=oof>=0
    if ok.sum()<30: return dict(n=int(ok.sum()),auc=None)
    return dict(n=int(ok.sum()),auc=round(float(roc_auc_score(ys[ok],oof[ok])),4),
                acc=round(float(((oof[ok]>=.5)==ys[ok]).mean()),4))
def main():
    d=pd.read_parquet(os.path.join(OUT,'forest6_10w.parquet'))
    emit('交付 %d'%len(d))
    res={}
    d['g10']=grp(d.lon.values,d.lat.values,0.1)
    d['gprov']=d.prov.fillna('NA').values
    # ===== 全体 C1/C6/C7/C8 =====
    res['C1_by_class']={str(k):int(v) for k,v in d.cls6.value_counts().sort_index().items()}
    res['C6_indep_ratio']=round(float((d.spatial_cluster_size==1).mean()),4)
    eco=pd.crosstab(d.eco,d.cls6); res['C7_eco']={k:{str(kk):int(vv) for kk,vv in v.items()} for k,v in eco.to_dict('index').items()}
    res['C8']=dict(dup=int(d.duplicated(['lon','lat']).sum()),
                   nan=int(d[['lon','lat','cls6','layer']+AEF].isna().sum().sum()),
                   oob=int(((d.lon<73)|(d.lon>136)|(d.lat<17)|(d.lat>54)).sum()),
                   aef_min=round(float(d[AEF].values.min()),4),aef_max=round(float(d[AEF].values.max()),4))
    # ===== 分层验证 =====
    for lay in ('L-A','L-B'):
        s=d[d.layer==lay]
        if len(s)<1000: continue
        X=s[AEF].to_numpy('float32'); y=s.cls6.to_numpy()
        emit('%s n=%d'%(lay,len(s)))
        r={}
        r['C4_six_block10km']=cv(X,y,grp(s.lon.values,s.lat.values,0.1))
        r['C4_six_prov']=cv(X,y,s.prov.fillna('NA').values)
        pr={}
        for a,b,ax in [(41,43,'needle'),(42,44,'needle'),(45,46,'needle'),
                       (41,42,'ever'),(43,44,'ever'),(45,46,'ever')]:
            pr[f'{a}vs{b}_{ax}']=pair(X,y,grp(s.lon.values,s.lat.values,0.1),a,b)
        r['C3_C5_pairs']=pr
        # C10 lon/lat 基线（45vs46）
        m=np.isin(y,[45,46])
        if m.sum()>60:
            Xll=s[['lon','lat']].to_numpy('float32')[m]; y2=(y[m]==46).astype(int)
            g2=grp(s.lon.values[m],s.lat.values[m],0.1)
            uq=np.unique(g2); perm=np.random.default_rng(SEED).permutation(uq)
            folds=[list(x) for x in np.array_split(perm,min(5,len(uq)))]
            oof=np.full(len(y2),-1.0)
            for f in folds:
                te=np.isin(g2,f); tr=~te
                if te.sum()==0 or len(np.unique(y2[tr]))<2: continue
                mm=rf().fit(Xll[tr],y2[tr]); oof[te]=mm.predict_proba(Xll[te])[:,1]
            ok=oof>=0
            r['C10_lonlat_baseline_45v46']=round(float(roc_auc_score(y2[ok],oof[ok])),4) if ok.sum()>30 else None
        res[lay]=r
    json.dump(res,open(os.path.join(OUT,'e6_verify.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
    emit('DONE'); print(json.dumps({k:v for k,v in res.items() if k!='C7_eco'},ensure_ascii=False,indent=1)[:2500])
if __name__=='__main__': main()

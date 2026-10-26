# -*- coding: utf-8 -*-
"""e2b_esa.py — ESA CCI 判定（批量行读，快 50 倍）"""
import os,sys,time,json,glob
import numpy as np, pandas as pd, h5py
sys.stdout.reconfigure(encoding='utf-8')
OUT=r'Z:/nb_national_2026/forest6_10w'
CCI=r'E:/landcover100_com_DB/others/1992年-2020年植被类型数据/ESACCI-LC-L4-PFT-Map-300m-P1Y-2020-v2.0.8.hdf'
BANDS=['TREES-NE','TREES-ND','TREES-BE','TREES-BD']
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
def main():
    fs=sorted(glob.glob(os.path.join(OUT,'cand_mix_*.parquet')))
    d=pd.concat([pd.read_parquet(f) for f in fs],ignore_index=True)
    d=d.drop_duplicates(['lon','lat']).reset_index(drop=True)
    d['uid']=np.arange(len(d))+900_000_000
    emit('混交候选 %d'%len(d))
    with h5py.File(CCI,'r') as h:
        lat=h['lat'][:]; lon=h['lon'][:]
        dlat=abs(float(lat[1]-lat[0])); dlon=float(lon[1]-lon[0])
        c=np.floor((d.lon.values-lon[0])/dlon).astype('int64')
        r=np.floor((90.0-d.lat.values)/dlat).astype('int64')
        ok=(c>=0)&(c<len(lon))&(r>=0)&(r<len(lat))
        emit('有效索引 %d / %d'%(ok.sum(),len(d)))
        # 按行分组，每行内批量读一段连续列
        idx=np.flatnonzero(ok)
        res={b:np.full(len(d),np.nan,dtype='float32') for b in BANDS}
        rows=np.unique(r[idx])
        emit('涉及 %d 个唯一行'%len(rows))
        for k,rr in enumerate(rows):
            sel=idx[r[rr if False else idx]==rr] if False else idx[r[idx]==rr]
            cols=c[sel]
            order=np.argsort(cols); sel=sel[order]; cols=cols[order]
            c0,c1=int(cols[0]),int(cols[-1])+1
            for b in BANDS:
                seg=h[b][0,int(rr),c0:c1]
                res[b][sel]=seg[cols-c0]
            if (k+1)%500==0: emit('  行 %d/%d'%(k+1,len(rows)))
    for b in BANDS: d[b.replace('TREES-','')]=res[b]
    d['nf20']=d.NE+d.ND; d['bf20']=d.BE+d.BD
    with np.errstate(divide='ignore',invalid='ignore'):
        d['ever_share20']=np.where(d.nf20>0, d.NE/np.maximum(d.nf20,1), np.nan)
    d['cls45_46']=np.where((d.nf20>=20)&(d.ever_share20>0.60),45,
                  np.where((d.nf20>=20)&(d.ever_share20<0.40),46,0))
    st={'n':int(len(d)),'cls45':int((d.cls45_46==45).sum()),'cls46':int((d.cls45_46==46).sum()),
        'uncertain':int((d.cls45_46==0).sum()),'nf_ge20':int((d.nf20>=20).sum())}
    emit(json.dumps(st,ensure_ascii=False))
    d[['uid','lon','lat','frac_needle','n_needle_win','n_broad_win','n_dc_win',
       'NE','ND','BE','BD','nf20','bf20','ever_share20','cls45_46']].to_parquet(
        os.path.join(OUT,'esa_judge.parquet'),index=False)
    json.dump(st,open(os.path.join(OUT,'e2_stats.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
    emit('DONE')
if __name__=='__main__': main()

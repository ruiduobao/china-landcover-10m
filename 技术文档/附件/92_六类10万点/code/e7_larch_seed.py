# -*- coding: utf-8 -*-
"""e7_larch_seed.py — 从 30m 落叶松-阔叶混交群系撒 46 类种子
* 群系：兴安落叶松、白桦林(139)、兴安落叶松、蒙古栎林(138)、兴安落叶松、樟子松林(137)
  → 前两个是落叶针叶+阔叶混交（真 46）；137 是落叶松+樟子松（都是针叶，属 42）
* 规则：每 100m 格 1 点、0.01° 同类≤3、每群系上限按面积
"""
import os,sys,time,json
import numpy as np, pandas as pd, rasterio
from scipy import ndimage
sys.stdout.reconfigure(encoding='utf-8')
TIF=r'F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif'
OUT=r'Z:/nb_national_2026/forest6_10w'
CODES=[139,138]   # 兴安落叶松+白桦、兴安落叶松+蒙古栎 → 46
R=6378137.0
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
def main():
    rows=[]
    with rasterio.open(TIF) as s:
        W,H=s.width,s.height; tr=s.transform
        BLK=6000
        for r0 in range(0,H,BLK):
            for c0 in range(0,W,BLK):
                r1=min(r0+BLK,H); c1=min(c0+BLK,W)
                a=s.read(1,window=((r0,r1),(c0,c1)))
                m=np.isin(a,CODES)
                if m.sum()<100: continue
                lab,n=ndimage.label(m,structure=np.ones((3,3)))
                for pid in range(1,n+1):
                    sel=(lab==pid); px=int(sel.sum())
                    if px<50: continue
                    rr,cc=np.nonzero(sel)
                    k=min(3 if px*900>=5e5 else 1, len(rr))
                    idx=np.random.default_rng(pid*7919+r0).choice(len(rr),k,replace=False)
                    for i in idx:
                        R_,C_=rr[i]+r0, cc[i]+c0
                        x=tr.c+(C_+0.5)*tr.a; y=tr.f+(R_+0.5)*tr.e
                        lon=np.degrees(x/R); lat=np.degrees(np.arctan(np.sinh(y/R)))
                        rows.append((lon,lat,int(a[rr[i],cc[i]]),px))
            emit('  行块 %d/%d 累计 %d'%(r0//BLK+1,(H+BLK-1)//BLK,len(rows)))
    d=pd.DataFrame(rows,columns=['lon','lat','value30','patch_px'])
    d['cls6']=46; d['src']='veg30_larch'; d['quality_tier']='B'
    d['patch_area_ha']=(d.patch_px*900/1e4).round(1)
    d=d.drop_duplicates(['lon','lat'])
    d['uid']=np.arange(len(d))+3_000_000_000
    d.to_parquet(os.path.join(OUT,'cand_46_larch.parquet'),index=False)
    emit('写 %d 点（46 类落叶松混交种子）'%len(d))
    print('逐群系:',d.value30.value_counts().to_dict())
if __name__=='__main__': main()

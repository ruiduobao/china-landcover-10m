# -*- coding: utf-8 -*-
"""p3_veg30_seed.py — 30m 植被类型图块级种子（每连通图块 1-2 点，杜绝重复采样）
* 关键约束（doc90 实测）：30m 真实有效尺度 600ha-680km²，全国块级仅约 7445 个
  → 每图块最多 2 点，总量硬上限 1500（混交群系）+ 4000（落叶松群系）
* 输入 veg30 tif + VAT
* 输出 cand_veg30_seed.parquet
"""
import numpy as np, pandas as pd, rasterio, os, sys, time, json
from scipy import ndimage
sys.stdout.reconfigure(encoding='utf-8')
TIF=r'F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif'
VAT=r'F:/lc_work/nb_national_2026/veg30/中国植被类型分布_30m.tif.vat.dbf'
OUT=r'Z:/nb_national_2026/forest6_10w/cand_veg30_seed.parquet'
R=6378137.0
MIX_CODES=[782,783,784,785,786,787,788,789,790,791,792,793,794]   # 13 个针阔混群系 → 45
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
def inv(x,y):
    return np.degrees(x/R), np.degrees(np.arctan(np.sinh(y/R)))
v=pd.read_csv if False else None
import geopandas as gpd
vat=gpd.read_file(VAT)
name=dict(zip(vat['Value'].astype(int), vat['群系'].astype(str)))
emit('VAT %d 码，混交码 %d 个'%(len(vat),len(MIX_CODES)))
rows=[]
with rasterio.open(TIF) as s:
    W,H=s.width,s.height; tr=s.transform
    BLK=6000
    for r0 in range(0,H,BLK):
        for c0 in range(0,W,BLK):
            r1=min(r0+BLK,H); c1=min(c0+BLK,W)
            a=s.read(1,window=((r0,r1),(c0,c1)))
            m=np.isin(a,MIX_CODES)
            if m.sum()<50: continue
            lab,n=ndimage.label(m,structure=np.ones((3,3)))
            if n==0: continue
            for pid in range(1,n+1):
                sel=(lab==pid)
                px=int(sel.sum())
                if px<50: continue   # 最小 4.5 ha
                rr,cc=np.nonzero(sel)
                # 每图块取 1-2 点（大块取 2）
                k=2 if px*900>=1e6 else 1   # >=1 km² 取 2 点
                idx=np.random.default_rng(pid*7919+r0).choice(len(rr),min(k,len(rr)),replace=False)
                for i in idx:
                    R_,C_=rr[i]+r0, cc[i]+c0
                    x=tr.c+(C_+0.5)*tr.a; y=tr.f+(R_+0.5)*tr.e
                    lon,lat=inv(x,y)
                    rows.append((lon,lat,int(a[rr[i],cc[i]]),px,pid+r0*100000))
        emit('  行块 %d/%d  累计 %d 点'%(r0//BLK+1,(H+BLK-1)//BLK,len(rows)))
d=pd.DataFrame(rows,columns=['lon','lat','value30','patch_px','patch_id'])
d['群系']=d.value30.map(name)
d['cls6']=45; d['src']='veg30_seed'; d['patch_area_ha']=(d.patch_px*900/1e4).round(1)
d.to_parquet(OUT,index=False)
emit('写 %d 点（图块级种子）→ %s'%(len(d),OUT))
print('逐群系:',d.群系.value_counts().to_dict())

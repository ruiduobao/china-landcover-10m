# -*- coding: utf-8 -*-
"""p1c_scan.py — FCS10 撒点（向量化，每瓦 60-70s，只跑需要的瓦数）"""
import os,sys,time,argparse
import numpy as np, pandas as pd, rasterio
from rasterio.windows import Window
sys.stdout.reconfigure(encoding='utf-8')
TILES=r'Z:/nb_national_2026/fcs10_dl/tiles'
PLAN=r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产/plan/tiles.csv'
OUT=r'Z:/nb_national_2026/forest6_10w'
BAND=4096
NEEDLE=np.array([71,72,81,82]); BROAD=np.array([51,52,61,62]); DC=np.array([81,82])
def emit(m): print(time.strftime('[%H:%M:%S] ')+str(m),flush=True)
def ws(m,k=5):
    mi=m.astype(np.int32); hh,ww=mi.shape; h=k//2
    ii=np.zeros((hh+1,ww+1),dtype=np.int32)
    np.cumsum(np.cumsum(mi,axis=0),axis=1,out=ii[1:,1:])
    if hh<=2*h or ww<=2*h: return None
    return (ii[2*h+1:,2*h+1:]-ii[:hh-2*h,2*h+1:]-ii[2*h+1:,:ww-2*h]+ii[:hh-2*h,:ww-2*h])
def scan(path,rng,cap42,capmix):
    P=[];M=[]
    with rasterio.open(path) as s:
        H,W=s.height,s.width; tr=s.transform
        for r0 in range(0,H,BAND):
            r1=min(r0+BAND,H)
            a=s.read(1,window=Window(0,r0,W,r1-r0))
            isn=np.isin(a,NEEDLE); isb=np.isin(a,BROAD)
            if not isn.any() or not isb.any(): del a,isn,isb; continue
            nn=ws(isn); nb=ws(isb); ndc=ws(np.isin(a,DC))
            if nn is None: del a,isn,isb; continue
            cen=a[2:-2,2:-2]
            m42=np.isin(cen,DC)&(ndc>=23)
            rr,cc=np.nonzero(m42)
            if len(rr):
                k=min(cap42,len(rr)); idx=rng.choice(len(rr),k,replace=False)
                for j in idx:
                    P.append((int(cc[j])+2, int(rr[j])+r0+2))
            tot=nn+nb
            with np.errstate(divide='ignore',invalid='ignore'):
                fr=np.where(tot>0,nn/np.maximum(tot,1),np.nan)
            mm=(nn>=3)&(nb>=3)&(fr>=0.15)&(fr<=0.85)
            rr,cc=np.nonzero(mm)
            if len(rr):
                k=min(capmix,len(rr)); idx=rng.choice(len(rr),k,replace=False)
                for j in idx:
                    M.append((int(cc[j])+2, int(rr[j])+r0+2, float(fr[rr[j],cc[j]]),
                              int(nn[rr[j],cc[j]]), int(nb[rr[j],cc[j]]), int(ndc[rr[j],cc[j]])))
            del a,isn,isb,nn,nb,ndc
        def conv(pts):
            if not pts: return np.zeros((0,2))
            arr=np.array([(tr.c+(c+0.5)*tr.a, tr.f+(r+0.5)*tr.e) for c,r,*_ in pts])
            extra=np.array([list(p[2:]) for p in pts]) if len(pts[0])>2 else np.zeros((len(pts),0))
            return np.c_[arr,extra]
    return conv(P), conv(M)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--end',type=int,default=40); ap.add_argument('--cap42',type=int,default=400)
    ap.add_argument('--capmix',type=int,default=1200); ap.add_argument('--seed',type=int,default=20261027)
    a=ap.parse_args()
    plan=pd.read_csv(PLAN,encoding='utf-8-sig')
    have={f[:-len('_fcs10.tif')] for f in os.listdir(TILES) if f.endswith('_fcs10.tif')}
    tiles=[t for t in plan.tile if t in have][a.start:a.end]
    emit('瓦片 %d（%d-%d）'%(len(tiles),a.start,a.end))
    rng=np.random.default_rng(a.seed); P=[];M=[];t0=time.time()
    for i,t in enumerate(tiles):
        try:
            A,B=scan(os.path.join(TILES,t+'_fcs10.tif'),rng,a.cap42,a.capmix)
            if len(A):
                dA=pd.DataFrame(A,columns=['lon','lat']); dA['center_code']=0; dA.insert(0,'tile',t); P.append(dA)
            if len(B):
                dB=pd.DataFrame(B,columns=['lon','lat','frac_needle','n_needle_win','n_broad_win','n_dc_win'])
                dB.insert(0,'tile',t); M.append(dB)
        except Exception as e: emit('  FAIL %s %s'%(t,str(e)[:60]))
        emit('  %d/%d %.1fmin 42=%d mix=%d'%(i+1,len(tiles),(time.time()-t0)/60,
             sum(len(x) for x in P),sum(len(x) for x in M)))
    tag='%d_%d'%(a.start,a.end)
    if P:
        d=pd.concat(P,ignore_index=True)
        d['center_code']=d.center_code.astype(int); d['cls6']=42; d['src']='fcs10_scan'
        d.to_parquet(os.path.join(OUT,'cand_42_fcs10_%s.parquet'%tag),index=False)
    if M:
        d=pd.concat(M,ignore_index=True)
        for c in ['n_needle_win','n_broad_win','n_dc_win']: d[c]=d[c].astype(int)
        d['src']='fcs10_scan'
        d.to_parquet(os.path.join(OUT,'cand_mix_fcs10_%s.parquet'%tag),index=False)
    emit('DONE %s  42=%d mix=%d  %.1fmin'%(tag,sum(len(x) for x in P),sum(len(x) for x in M),(time.time()-t0)/60))
if __name__=='__main__': main()

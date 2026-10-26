# -*- coding: utf-8 -*-
"""b9_watch_apply.py — 后台守候：QGIS 释放 2017 GPKG 后自动应用暂存文件

* 每 60 s 尝试 os.replace(staging → target)；成功后重算 逐年池成员数.csv / 补样来源明细.csv
* 最多守候 --minutes 分钟（默认 120）
* 用法：python b9_watch_apply.py [--minutes 120]
"""
import argparse, os, shutil, sys, time
import pandas as pd, pyogrio
sys.stdout.reconfigure(encoding='utf-8')
CDIR = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/裸地'
STAGE = os.path.join(CDIR, '_staging_2026-10-26')
BK = os.path.join(CDIR, '旧样本_2026-10-26_补样前')

def rebuild_meta():
    rows=[]
    for y in range(2017,2025):
        d=pyogrio.read_dataframe(os.path.join(CDIR,str(y),'裸地_%d_WGS84.gpkg'%y), columns=['src','A00'])
        vc=d.src.value_counts().to_dict(); n=len(d)
        rows.append({'year':y,'n_total':n,
                     'n_base':n-vc.get('fcs10_2023_ne_east_topup',0)-vc.get('multisrc_s2_bare_south',0),
                     'n_topup_east':vc.get('fcs10_2023_ne_east_topup',0),
                     'n_south':vc.get('multisrc_s2_bare_south',0),
                     'aef_missing':int(d.A00.isna().sum())})
    df=pd.DataFrame(rows); df['n_topup']=df.n_topup_east+df.n_south
    df.to_csv(os.path.join(CDIR,'补样来源明细.csv'), index=False, encoding='utf-8-sig')
    df[['year','n_total','n_base','n_topup']].to_csv(os.path.join(CDIR,'逐年池成员数.csv'), index=False, encoding='utf-8-sig')
    print(df.to_string(index=False), flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--minutes',type=int,default=120); a=ap.parse_args()
    t0=time.time()
    while time.time()-t0 < a.minutes*60:
        pending=[]
        for y in range(2017,2025):
            stg=os.path.join(STAGE,'裸地_%d_WGS84.gpkg'%y); fp=os.path.join(CDIR,str(y),'裸地_%d_WGS84.gpkg'%y)
            if not os.path.isfile(stg): continue
            bk=os.path.join(BK,str(y),'裸地_%d_WGS84.gpkg'%y)
            if not os.path.isfile(bk):
                os.makedirs(os.path.dirname(bk),exist_ok=True); shutil.copy2(fp,bk)
            try:
                os.replace(stg,fp); print('applied',y,flush=True)
            except PermissionError:
                pending.append(y)
        if not pending:
            rebuild_meta(); print('全部应用完成',flush=True); return
        time.sleep(60)
    print('超时未释放：',pending,flush=True)

if __name__=='__main__': main()

# -*- coding: utf-8 -*-
"""o2_feats.py — 乔灌园地：候选点 2023 年 AEF64 特征抓取（GEE sampleRegions，分源组/块级可续跑）

* 输入：data/orchard/{cand_fcs10.parquet, cand_vec.parquet, mk_orchard_points.csv}
* 分组：--group mkv = 母库 + 矢量框；--group fcs = FCS10 框（分组落盘，互不干扰）
* 规则源：GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL 2023 mosaic，64 波段（与母库同源 scale=10）；
          逐 band unmask(-9999) 防丢点
* 门槛：每块 5000 点；块文件 feat_{group}_p{ci}.parquet 存在即跳过；UID=frame_lon_lat（确定性）
* 输出：data/orchard/feats/feat_{group}_p{ci}.parquet（point_uid,lon,lat,frame,tile,src_id,A00-A63,aef_ok）
* 用法：python o2_feats.py --group mkv [--acct x] [--part 0 --nparts 4]
"""
import argparse
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
OUTD = os.path.join(WORK, 'data', 'orchard', 'feats')
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
YEAR = 2023
CHUNK = 5000


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def load_group(group):
    ds = []
    if group == 'mkv':
        p = os.path.join(WORK, 'data', 'orchard', 'cand_vec.parquet')
        if os.path.exists(p):
            d = pd.read_parquet(p)
            d['tile'] = ''
            ds.append(d[['lon', 'lat', 'frame', 'tile', 'src_id']])
        p = os.path.join(WORK, 'data', 'orchard', 'mk_orchard_points.csv')
        if os.path.exists(p):
            m = pd.read_csv(p, encoding='utf-8-sig')
            m['frame'] = 'mk_orchard'
            m['tile'] = ''
            m['src_id'] = m['row_id'].astype(str)
            ds.append(m[['lon', 'lat', 'frame', 'tile', 'src_id']])
    elif group == 'fcs':
        p = os.path.join(WORK, 'data', 'orchard', 'cand_fcs10.parquet')
        if os.path.exists(p):
            d = pd.read_parquet(p)
            if 'src_id' not in d.columns:
                d['src_id'] = d['tile'] if 'tile' in d.columns else ''
            ds.append(d[['lon', 'lat', 'frame', 'tile', 'src_id']])
    d = pd.concat(ds, ignore_index=True)
    d = d.drop_duplicates(subset=['lon', 'lat', 'frame']).reset_index(drop=True)
    d['point_uid'] = ['%s_%.6f_%.6f' % (fr, lo, la) for fr, lo, la in zip(d.frame, d.lon, d.lat)]
    return d


def fetch_chunk(ee, pts):
    aef = (ee.ImageCollection(AEF).filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
           .mosaic().select(VC.FEATS))
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                        {'point_uid': r.point_uid}) for r in pts.itertuples()]
    got = aef.sampleRegions(collection=ee.FeatureCollection(feats), scale=10, geometries=False,
                            tileScale=4).getInfo()['features']
    recs = [f['properties'] for f in got]
    df = pd.DataFrame([{'point_uid': r['point_uid'], **{k: r.get(k) for k in VC.FEATS}} for r in recs])
    df['aef_ok'] = df[VC.FEATS].notna().all(axis=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True, choices=['mkv', 'fcs'])
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    allp = load_group(a.group)
    chunks = [allp.iloc[i:i + CHUNK] for i in range(0, len(allp), CHUNK)]
    mine = [(i, c) for i, c in enumerate(chunks) if i % a.nparts == a.part]
    emit('组 %s：%d 点 / %d 块；本进程 %d 块' % (a.group, len(allp), len(chunks), len(mine)))
    if not mine:
        return
    import ee
    VC.ensure_ctx(a.acct)
    for ci, c in mine:
        fp = os.path.join(OUTD, 'feat_%s_p%02d.parquet' % (a.group, ci))
        if os.path.exists(fp):
            emit('  块 %02d 已存在' % ci)
            continue
        for attempt in range(4):
            try:
                df = fetch_chunk(ee, c)
                out = c[['point_uid', 'lon', 'lat', 'frame', 'tile', 'src_id']].merge(df, on='point_uid', how='left')
                out.to_parquet(fp, index=False)
                emit('  块 %02d: %d/%d 命中' % (ci, int(out.aef_ok.sum()), len(c)))
                break
            except Exception as e:
                emit('  块 %02d 重试%d: %s' % (ci, attempt, str(e)[:140]))
                time.sleep(10)
        else:
            emit('  块 %02d 连续失败' % ci)


if __name__ == '__main__':
    main()

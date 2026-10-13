# -*- coding: utf-8 -*-
"""o9_gem_sample.py — GEM-Forest 2020 树作物类（b1==2）中国区候选采样（GEE，5° 格循环，块级可续跑）

* 输入：GEE 资产 projects/danielp-cuni/assets/GEM-Forest/GEM-Forest_2020（945 瓦片，b1：0=非森林/1=森林/2=树作物）
* 规则源：仅取 b1==2 像元；按 5° 格循环，每格对 mask(b1==2) 做 sample（scale=20，numPixels 自适应），
          格落盘 data/orchard/gem/part_<lon>_<lat>.parquet（存在即跳过）
* 门槛：每格目标 ≥200 点（末段按格内容量）；全国后处理去重 + ≥300 m 间距稀疏
* 输出：data/orchard/gem/part_*.parquet（lon,lat,cell）
        data/orchard/gem_points.parquet（全国候选，去重后）
* 用法：python o9_gem_sample.py [--acct zixen8v8] [--combine]
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
OUTD = os.path.join(WORK, 'data', 'orchard', 'gem')
GEM = 'projects/danielp-cuni/assets/GEM-Forest/GEM-Forest_2020'
SCALE = 20
PER_CELL = 4000        # 每格请求采样数（mask 采样，dropNulls 后按 class2 占比得点）


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', default='zixen8v8')
    ap.add_argument('--combine', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    cand = [(lo, la) for lo in range(70, 135, 5) for la in range(15, 55, 5)]
    if a.combine:
        fs = sorted(f for f in os.listdir(OUTD) if f.startswith('part_') and f.endswith('.parquet'))
        d = pd.concat([pd.read_parquet(os.path.join(OUTD, f)) for f in fs], ignore_index=True)
        d = d.drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
        # ≥300 m 间距稀疏（贪心）
        rng = np.random.default_rng(20261012)
        idx = rng.permutation(len(d))
        keep, pts = [], []
        for i in idx:
            lo, la = d.lon.iloc[i], d.lat.iloc[i]
            if pts:
                arr = np.asarray(pts)
                dist = np.hypot((arr[:, 0] - lo) * np.cos(np.radians(la)), arr[:, 1] - la)
                if dist.min() < 300 / 111000.0 * 1.0:
                    continue
            keep.append(i); pts.append((lo, la))
        out = d.iloc[keep].reset_index(drop=True)
        out.to_parquet(os.path.join(WORK, 'data', 'orchard', 'gem_points.parquet'), index=False)
        emit('合并 %d → 去重 %d → 间距后 %d → gem_points.parquet' % (sum(1 for _ in fs), len(d), len(out)))
        print(out.groupby('cell').size().describe())
        return
    import ee
    VC.ensure_ctx(a.acct)
    gem = ee.ImageCollection(GEM).mosaic().select('b1')
    m2 = gem.eq(2)
    done = 0
    for lo, la in cand:
        fp = os.path.join(OUTD, 'part_%d_%d.parquet' % (lo, la))
        if os.path.exists(fp):
            continue
        rect = ee.Geometry.Rectangle([lo, la, lo + 5, la + 5])
        try:
            s = m2.selfMask().sample(region=rect, scale=SCALE, numPixels=PER_CELL,
                                     geometries=True, seed=42, tileScale=8).getInfo()['features']
            pts = [{'lon': f['geometry']['coordinates'][0], 'lat': f['geometry']['coordinates'][1],
                    'cell': '%d_%d' % (lo, la)} for f in s]
            d = pd.DataFrame(pts)
            d.to_parquet(fp, index=False)
            done += 1
            if len(d) > 0:
                emit('  格 %d,%d: %d 点' % (lo, la, len(d)))
            if done % 20 == 0:
                emit('进度 %d 格完成' % done)
        except Exception as e:
            emit('  格 %d,%d 失败: %s' % (lo, la, str(e)[:120]))
            time.sleep(5)
    emit('全格完成')


if __name__ == '__main__':
    main()

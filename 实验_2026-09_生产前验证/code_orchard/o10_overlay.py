# -*- coding: utf-8 -*-
"""o10_overlay.py — GEM-Forest 树作物 × 我方 2023 交付成品的空间叠加（本地只读）

* 输入：data/orchard/gem_points.parquet（o9 候选点）
        我方成品：D:/生产_2023五省/成品/分省/{四川,宁夏,广东,福建,黑龙江}_2023_10m.tif
                  F:/lc_work/prod5p_2023/delivery_r4_甘肃/T*_10m.tif
* 规则源：逐点在成品图上读 1 px（nearest，无重采样）；类码用 24 类体系（VC.V31_NAMES）
* 门槛：仅统计落在成品范围内的点；逐省报告交叉表
* 输出：结果打印 + results/orchard/gem_overlay.json + data/orchard/gem_overlay.csv（逐点）
* 用法：python o10_overlay.py
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows as rw
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
MAPS = {'四川': [r'D:/生产_2023五省/成品/分省/四川_2023_10m.tif'],
        '宁夏': [r'D:/生产_2023五省/成品/分省/宁夏_2023_10m.tif'],
        '广东': [r'D:/生产_2023五省/成品/分省/广东_2023_10m.tif'],
        '福建': [r'D:/生产_2023五省/成品/分省/福建_2023_10m.tif'],
        '黑龙江': [r'D:/生产_2023五省/成品/分省/黑龙江_2023_10m.tif'],
        '甘肃': sorted(glob.glob(r'F:/lc_work/prod5p_2023/delivery_r4_甘肃/T*_10m.tif'))}


def main():
    pts = pd.read_parquet(os.path.join(ORD, 'gem_points.parquet'))
    pts['prov_gem'] = ''
    pts['our_class'] = np.nan
    names = VC.V31_NAMES()
    print('候选 %d 点' % len(pts))
    for prov, fps in MAPS.items():
        hit_total = 0
        for fp in fps:
            if not os.path.exists(fp):
                continue
            with rasterio.open(fp) as ds:
                b = ds.bounds
                m = (pts.lon >= b.left) & (pts.lon <= b.right) & (pts.lat >= b.bottom) & (pts.lat <= b.top) & (pts.prov_gem == '')
                idx = np.where(m.to_numpy())[0]
                if not len(idx):
                    continue
                for i in idx:
                    r, c = ds.index(float(pts.lon.iloc[i]), float(pts.lat.iloc[i]))
                    if 0 <= r < ds.height and 0 <= c < ds.width:
                        v = int(ds.read(1, window=rw.Window(c, r, 1, 1))[0, 0])
                        if v > 0:
                            pts.iloc[i, pts.columns.get_loc('prov_gem')] = prov
                            pts.iloc[i, pts.columns.get_loc('our_class')] = v
                            hit_total += 1
        if hit_total:
            print('  %s: 落在成品内 %d 点' % (prov, hit_total))
    got = pts[pts.our_class.notna()].copy()
    got['our_class_name'] = got.our_class.astype(int).map(lambda k: names.get(str(k), str(k)))
    out = {}
    for prov, g in got.groupby('prov_gem'):
        out[prov] = dict(n=int(len(g)), top=g.our_class_name.value_counts().head(10).to_dict())
        print('\n== %s（n=%d）我方类分布：' % (prov, len(g)))
        print(g.our_class_name.value_counts().head(10).to_string())
    VC.jsave(out, os.path.join(WORK, 'results', 'orchard', 'gem_overlay.json'))
    got.to_csv(os.path.join(ORD, 'gem_overlay.csv'), index=False, encoding='utf-8-sig')
    print('\n合计覆盖 %d 点 → gem_overlay.csv' % len(got))


if __name__ == '__main__':
    main()

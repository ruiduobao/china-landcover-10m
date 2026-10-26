# -*- coding: utf-8 -*-
"""h14b_clcd_check.py — 逐年补样 × CLCD 年度产品一致性（独立旁证，非真值）

* 输入：data/yearly/topup_<year>.parquet；E:/2.1_CLCD土地覆盖数据/30米/CLCD_v01_<year>_albert.tif
* CLCD 码：1 耕地 / 2 林地 / 3 草地 / 4 灌丛 / 5 水体 / 6 冰雪 / 7 裸地 / 8 不透水 / 9 湿地
* 判读：裸地 ← CLCD 7；稀疏 ← CLCD {3,4,7}（无对应类，取草/灌/裸组合）
* 输出：data/yearly/clcd_check.json + reports/clcd_check.md
* 用法：python h14b_clcd_check.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
CLCD = r'E:/2.1_CLCD土地覆盖数据/30米/CLCD_v01_%d_albert.tif'
YEARS = list(range(2017, 2025))
NAMES = {1: '耕地', 2: '林地', 3: '草地', 4: '灌丛', 5: '水体', 6: '冰雪', 7: '裸地', 8: '不透水', 9: '湿地'}


def main():
    out = {}
    for y in YEARS:
        fp = os.path.join(BASE, 'data', 'yearly', 'topup_%d.parquet' % y)
        cf = CLCD % y
        if not os.path.isfile(fp) or not os.path.isfile(cf):
            continue
        d = pd.read_parquet(fp, columns=['lon', 'lat', 'cl'])
        with rasterio.open(cf) as ds:
            tr = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
            x, yy = tr.transform(d.lon.to_numpy(), d.lat.to_numpy())
            vals = np.array([v[0] for v in ds.sample(zip(x, yy), indexes=1)], dtype=np.int16)
        d['clcd'] = vals
        rec = {}
        for fam, cl in [('sparse', 13), ('bare', 22)]:
            g = d[d.cl == cl]
            vc = g.clcd.value_counts().to_dict()
            top = {NAMES.get(int(k), str(k)): int(v) for k, v in
                   sorted(vc.items(), key=lambda kv: -kv[1])[:5]}
            rec[fam] = {'n': int(len(g)), 'top': top,
                        'share_bare7': round(float((g.clcd == 7).mean() * 100), 1),
                        'share_grass3': round(float((g.clcd == 3).mean() * 100), 1),
                        'share_shrub4': round(float((g.clcd == 4).mean() * 100), 1),
                        'share_ok': round(float(g.clcd.isin([3, 4, 7]).mean() * 100), 1)}
        out[y] = rec
        print('%d: 稀疏 CLCD-ok %.1f%%（裸7 %.1f/草3 %.1f/灌4 %.1f）｜裸地 裸7 %.1f%%' % (
            y, rec['sparse']['share_ok'], rec['sparse']['share_bare7'], rec['sparse']['share_grass3'],
            rec['sparse']['share_shrub4'], rec['bare']['share_bare7']), flush=True)
    os.makedirs(os.path.join(BASE, 'reports'), exist_ok=True)
    json.dump(out, open(os.path.join(BASE, 'data', 'yearly', 'clcd_check.json'), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    lines = ['# 逐年补样 × CLCD 年度产品一致性（独立旁证）\n', '```json',
             json.dumps(out, ensure_ascii=False, indent=1), '```']
    open(os.path.join(BASE, 'reports', 'clcd_check.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    print('→ data/yearly/clcd_check.json')


if __name__ == '__main__':
    main()

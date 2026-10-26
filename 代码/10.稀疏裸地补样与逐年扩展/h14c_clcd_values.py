# -*- coding: utf-8 -*-
"""h14c_clcd_values.py — 逐年逐点 CLCD 类别（供 settlement guard 使用）

* 输入：data/yearly/topup_<year>.parquet（或 data/final/final_2023_v2.parquet + yearly_panel）
        这里直接用 final_2023_v2 的 7,098 点，逐年采样 CLCD
* 输出：data/yearly/clcd_values.parquet（pid, year, clcd）
* 用法：python h14c_clcd_values.py
"""
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


def main():
    d = pd.read_parquet(os.path.join(BASE, 'data', 'final', 'final_2023_v2.parquet'),
                        columns=['pid', 'lon', 'lat'])
    rows = []
    for y in YEARS:
        cf = CLCD % y
        if not os.path.isfile(cf):
            continue
        with rasterio.open(cf) as ds:
            tr = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
            x, yy = tr.transform(d.lon.to_numpy(), d.lat.to_numpy())
            vals = np.array([v[0] for v in ds.sample(zip(x, yy), indexes=1)], dtype=np.int16)
        rows.append(pd.DataFrame({'pid': d.pid, 'year': y, 'clcd': vals}))
        print('%d: clcd==8 %d / %d' % (y, int((vals == 8).sum()), len(vals)), flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(os.path.join(BASE, 'data', 'yearly', 'clcd_values.parquet'), index=False)
    print('→ data/yearly/clcd_values.parquet (%d 行)' % len(out))


if __name__ == '__main__':
    main()

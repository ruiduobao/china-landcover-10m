# -*- coding: utf-8 -*-
"""h1d_purity_recheck.py — 交付前逐点栅格纯度复核（消除块边界取舍误差）

* 背景：h1 的块内纯度用 5px 边距的 padded window 计算，坐标换算在极少数块边界会差 1 px，
  自检发现约 1% 的候选 3×3 不纯。本脚本对全部候选按**交付坐标**重读 3×3 与 9×9：
    keep ⇔ 3×3 全为目标类族 且 9×9 同类占比 ≥ 0.70
* 输入/输出：data/candidates/cand_{north,east}.parquet（就地重写；新增 share9_r）
* 用法：python h1d_purity_recheck.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
TILE_ROOT = r'E:/data/全球土地覆盖数据/数据/GLC_FCS10'
FAM = {'sparse': (150, 152, 153), 'bare': (200, 201, 202)}
SHARE = 0.70


def main():
    tiles = []
    for t in sorted(glob.glob(os.path.join(TILE_ROOT, '*', 'GLC_FCS10_2023_*.tif'))):
        with rasterio.open(t) as ds:
            tiles.append((t, ds.bounds))
    out = {}
    for reg in ('north', 'east', 'ne130'):
        fp = os.path.join(BASE, 'data', 'candidates', 'cand_%s.parquet' % reg)
        if not os.path.isfile(fp):
            continue
        d = pd.read_parquet(fp)
        ok3 = np.zeros(len(d), dtype=bool)
        ok9 = np.zeros(len(d), dtype=bool)
        sh9 = np.full(len(d), np.nan, dtype=np.float32)
        for i, r in enumerate(d.itertuples()):
            cand = [x for x in tiles if x[1].left <= r.lon < x[1].right
                    and x[1].bottom <= r.lat < x[1].top]
            if not cand:
                continue
            t = cand[0][0]
            with rasterio.open(t) as ds:
                tr = ds.transform
                col = int(round((r.lon - tr.c) / tr.a - 0.5))
                row = int(round((r.lat - tr.f) / tr.e - 0.5))
                a3 = ds.read(1, window=Window(col - 1, row - 1, 3, 3))
                a9 = ds.read(1, window=Window(col - 4, row - 4, 9, 9))
            codes = FAM[r.fam]
            ok3[i] = bool(np.all(np.isin(a3, codes)))
            if a9.size == 81:
                s = float(np.isin(a9, codes).mean())
                sh9[i] = s
                ok9[i] = s >= SHARE
        keep = ok3 & ok9
        st = {'n_in': int(len(d)), 'drop_pure3': int((~ok3).sum()),
              'drop_share9': int((ok3 & ~ok9).sum()), 'n_out': int(keep.sum())}
        dd = d[keep].copy()
        dd['share9_r'] = sh9[keep]
        dd.to_parquet(fp, index=False)
        json.dump(st, open(fp.replace('.parquet', '_purity.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        out[reg] = st
        print('%s: %s' % (reg, st), flush=True)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()

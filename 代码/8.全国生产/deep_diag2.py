# -*- coding: utf-8 -*-
"""深度诊断 2：全域交叉表 + 游程形态学"""
import sys, json
sys.path.insert(0, '.'); sys.path.insert(0, '../0.本地流水线')
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from lc_conf import CLASSES
import pandas as pd

OURS = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/生产输出/成品/东北三省_2023_100m.tif'
wc = np.load(r'F:/lc_work/wc_ne.npy')
meta = json.load(open(r'F:/lc_work/wc_ne_meta.json'))
x0, res, y0, resy = meta['transform']
Hw, Ww = meta['shape']

# 把 100m 成品 mode 重采样到 WC 网格
with rasterio.open(OURS) as s:
    src = s.read(1)
    dst = np.zeros((Hw, Ww), np.uint8)
    reproject(src, dst, src_transform=s.transform, src_crs=s.crs,
              dst_transform=rasterio.transform.from_origin(x0, y0, res, -resy),
              dst_crs='EPSG:4326', resampling=Resampling.mode)
np.save(r'F:/lc_work/ours_500m.npy', dst)
valid = dst > 0
print('对齐网格 %d×%d，我们有效像元占比 %.1f%%' % (Hw, Ww, 100*valid.mean()))

WCNAME = {10:'WC乔木', 20:'WC灌丛', 30:'WC草地', 40:'WC耕地', 50:'WC建成',
          60:'WC裸/稀', 70:'WC冰雪', 80:'WC水体', 90:'WC湿地', 95:'WC红树', 100:'WC苔藓'}
# --- 交叉表 ---
ct = pd.crosstab(pd.Series(dst[valid], name='本'), pd.Series(wc[valid], name='WC'))
ct = ct.reindex(sorted(ct.index))
ct.columns = [WCNAME.get(int(c), str(c)) for c in ct.columns]
tot_wc = ct.sum(axis=0)
print('\n=== 我们的每个类由 WC 的什么组成（行归一化 %，只列 ≥3% 的 WC 类）===')
for c in ct.index:
    row = ct.loc[c]
    if row.sum() < 3000: continue
    r = (100*row/row.sum()).sort_values(ascending=False)
    top = '  '.join('%s %.0f' % (k, v) for k, v in r.items() if v >= 3)[:110]
    print('  %3d %-12s n=%7d  %s' % (c, CLASSES[int(c)][1], row.sum(), top))
print('\n=== WC 的每个类被我们判成了什么（列归一化 %，只列 ≥5%）===')
for col in ct.columns:
    colv = ct[col]
    if colv.sum() < 3000: continue
    r = (100*colv/colv.sum()).sort_values(ascending=False)
    top = '  '.join('%d:%s %.0f' % (k, CLASSES[int(k)][1], v) for k, v in r.items() if v >= 5)[:120]
    print('  %-8s n=%7d  %s' % (col, colv.sum(), top))
ct.to_csv(r'F:/lc_work/crosstab_ne.csv', encoding='utf-8-sig')
print('\n交叉表 → F:/lc_work/crosstab_ne.csv')

# -*- coding: utf-8 -*-
"""
p1_check.py — G1 门禁：嵌入数据核查 + PIF 漂移探测（交互式，zsi8emo）
* 2017-2024 年份/波段/分辨率（北京 bbox）
* PIF（伪不变地物）跨年漂移：密云/官厅水库深水区 + 高山裸岩点，逐年两两余弦
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

ee, pid = PC.load_account('zsi8emo')

BJ = ee.Geometry(PC.beijing_geojson())
col = ee.ImageCollection(PC.EMB_COL).filterBounds(BJ)

print('=== 1) 年份覆盖 ===')
years = col.aggregate_array('system:time_start').map(
    lambda t: ee.Date(t).get('year')).distinct().sort().getInfo()
print('可用年份:', years)

img = col.first()
print('=== 2) 波段/分辨率 ===')
info = img.bandNames().getInfo()
print('波段数:', len(info), '前5:', info[:5])
print('nominalScale:', img.projection().nominalScale().getInfo())

print('=== 3) PIF 漂移探测 ===')
pifs = {
    '密云水库': (116.85, 40.48),
    '官厅水库': (115.62, 40.36),
}
def emb_at(lon, lat, year):
    ic = (ee.ImageCollection(PC.EMB_COL)
          .filterBounds(ee.Geometry.Point([lon, lat]))
          .filter(ee.Filter.calendarRange(year, year, 'year')))
    n = ic.size().getInfo()
    if n == 0:
        return None
    return ic.first()

# 提取各年 64 维向量
vectors = {}
for name, (lon, lat) in pifs.items():
    vectors[name] = {}
    for y in PC.YEARS:
        im = emb_at(lon, lat, y)
        if im is None:
            continue
        v = im.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=ee.Geometry.Point([lon, lat]),
            scale=10).getInfo()
        vec = np.array([v.get(f'A{i:02d}', np.nan) for i in range(64)], dtype=float)
        vectors[name][y] = vec
        print(f'  {name} {y}: norm={np.linalg.norm(vec):.4f} (取到 {np.isfinite(vec).sum()}/64 维)')

print('\n=== 4) PIF 逐年 vs 2017 的余弦相似度（应≈1，<0.999 视为漂移）===')
worst = 1.0
for name, ys in vectors.items():
    if 2017 not in ys:
        continue
    v0 = ys[2017]
    for y in sorted(ys):
        v = ys[y]
        cos = float(np.dot(v0, v) / (np.linalg.norm(v0) * np.linalg.norm(v) + 1e-12))
        worst = min(worst, cos)
        print(f'  {name} {y} vs 2017: cos={cos:.6f}')
print(f'\n最差 PIF 相似度: {worst:.6f}')
print('G1 判定:', 'PASS（无显著漂移，无需旋转校正）' if worst > 0.999 else 'NEED-CORRECTION（漂移>1e-3，年更新需先做 PIF 校正）')

# 落盘核查结果
json.dump({'years': years, 'bands': len(info), 'scale_m': img.projection().nominalScale().getInfo(),
           'pif_worst_cos': worst},
          open('数据/本地处理/北京试点/p1_check.json', 'w'), ensure_ascii=False, indent=2)
print('已写 数据/本地处理/北京试点/p1_check.json')

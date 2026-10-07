# -*- coding: utf-8 -*-
"""
lc11_中国样本全量统计_v2.py — 用 shapely prepared geometry 精确统计中国陆域样本类别分布
* 修正 lc10b 跨界瓦片 in-china 太慢的问题（shapely.strtree/prepared 加速 ~100x）
* 输出：真实中国陆域 34 类(GLC原始码) 分布 + 稀有类/缺类真实清单
运行: python lc11_中国样本全量统计_v2.py
"""
import os, re, csv, json
from collections import Counter
from multiprocessing import Pool
from shapely.geometry import shape, Point
from shapely.prepared import prep
from lc_conf import CODE_MAP

DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
BOUNDARY = r'数据/边界/china_100000_full.json'
OUT_DIR  = r'数据/本地处理/统计'
os.makedirs(OUT_DIR, exist_ok=True)

# 全局：中国合并多边形 + prepared（worker 各自载入一次）
_CN_PREP = None; _CN = None; _CMAP = None

def load_china_prep(path):
    d = json.load(open(path, encoding='utf-8'))
    polys = []
    for f in d['features']:
        g = f['geometry']
        if g['type'] == 'Polygon':
            polys.append(shape(g))
        elif g['type'] == 'MultiPolygon':
            polys += list(shape(g))
    merged = polys[0]
    for p in polys[1:]:
        merged = merged.union(p)
    return merged, prep(merged)

def _init(bp, cm):
    global _CN, _CN_PREP, _CMAP
    _CN, _CN_PREP = load_china_prep(bp)
    _CMAP = cm

def _tile_relation(lon0, lat0):
    """用中国合并多边形判断瓦片：in / out / cross（shapely 快）"""
    box = None
    from shapely.geometry import box
    b = box(lon0, lat0, lon0+1, lat0+1)
    if _CN.contains(b):
        return 'in'
    if not _CN.intersects(b):
        return 'out'
    return 'cross'

def _process(tile):
    lon0, lat0 = tile
    rel = _tile_relation(lon0, lat0)
    fn = os.path.join(DATA_DIR, f'Sample_Lon{lon0}_Lat{lat0}.csv')
    cnt = Counter(); n_in = 0
    if rel == 'out':
        return {'rel': 'out', 'in': 0, 'total': 0, 'cnt': {}}
    with open(fn, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    total = len(rows)
    if rel == 'in':
        for r in rows:
            c = _CMAP.get(int(r['class']), -1); cnt[c] += 1
        return {'rel': 'in', 'in': total, 'total': total, 'cnt': dict(cnt)}
    # cross：逐点 prepared.contains
    for r in rows:
        if _CN_PREP.contains(Point(float(r['lon']), float(r['lat']))):
            c = _CMAP.get(int(r['class']), -1); cnt[c] += 1
    return {'rel': 'cross', 'in': sum(cnt.values()), 'total': total, 'cnt': dict(cnt)}

def main():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from lc_conf import CODE_MAP
    tiles = []
    for f in os.listdir(DATA_DIR):
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv', f)
        if m: tiles.append((int(m.group(1)), int(m.group(2))))
    print(f'瓦片总数: {len(tiles)}')
    # 先快筛：只在 lon 60-140 / lat 10-60 的瓦片才需要（其余全 out，跳过读文件）
    tiles = [t for t in tiles if 60 <= t[0] < 140 and 10 <= t[1] < 60]
    print(f'候选瓦片(经纬粗筛后): {len(tiles)}')
    agg = Counter(); total_in = 0; total_all = 0; relc = Counter()
    with Pool(6, initializer=_init, initargs=(BOUNDARY, CODE_MAP)) as p:
        for r in p.imap_unordered(_process, tiles, chunksize=20):
            relc[r['rel']] += 1
            total_in += r['in']; total_all += r['total']
            agg.update(r['cnt'])
    print(f'瓦片关系: {dict(relc)}')
    print(f'总点: {total_all:,}  中国境内: {total_in:,} ({100*total_in/total_all:.1f}%)')
    print(f'\n=== 中国境内类别分布（GLC 原始码，经 CODE_MAP 到产品码）===')
    for c, n in agg.most_common():
        print(f'  {c:>4}: {n:,}')
    json.dump({'tiles_relation': dict(relc), 'total_all': total_all,
               'inside': total_in,
               'class_inside': {str(k): v for k, v in agg.most_common()}},
              open(os.path.join(OUT_DIR, 'samples_cn_precise_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n[输出] samples_cn_precise_stats.json')

if __name__ == '__main__':
    main()

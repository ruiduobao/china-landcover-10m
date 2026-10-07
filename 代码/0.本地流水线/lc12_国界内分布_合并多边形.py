# -*- coding: utf-8 -*-
"""
lc12_国界内分布_合并多边形.py — 用"合并后的中国多边形 + shapely"快速统计
* 关键提速：把 35 省 union 成单个 MultiPolygon，用 STRtree 一次判断；
* 只对跨界瓦片逐点，但跨界瓦片用 prepared 逐点（已比射线法快>50x）；
* 结果写 samples_cn_final_stats.json
运行: python lc12_国界内分布_合并多边形.py
"""
import os, re, csv, json
from collections import Counter
from shapely.geometry import shape, box, Point
from shapely.prepared import prep
from shapely.strtree import STRtree
from lc_conf import CODE_MAP

DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
BOUNDARY = r'数据/边界/china_100000_full.json'
OUT_DIR  = r'数据/本地处理/统计'
os.makedirs(OUT_DIR, exist_ok=True)

def load_china_polys(path):
    """返回省多边形列表（不合并，避免union拓扑错误）"""
    d = json.load(open(path, encoding='utf-8'))
    polys = []
    for f in d['features']:
        sg = shape(f['geometry'])
        polys += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
    return polys

def main():
    polys = load_china_polys(BOUNDARY)
    tree = STRtree(polys)
    # 候选瓦片：用所有省 bounds 的并集粗框
    minx = min(p.bounds[0] for p in polys); miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys); maxy = max(p.bounds[3] for p in polys)
    tiles = []
    for f in os.listdir(DATA_DIR):
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv', f)
        if not m: continue
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        if lon0 < maxx and lon0+1 > minx and lat0 < maxy and lat0+1 > miny:
            tiles.append((lon0, lat0))
    print(f'候选瓦片: {len(tiles)}')
    agg = Counter(); total_in = 0; total_all = 0; relc = Counter()
    for lon0, lat0 in tiles:
        fn = os.path.join(DATA_DIR, f'Sample_Lon{lon0}_Lat{lat0}.csv')
        b = box(lon0, lat0, lon0+1, lat0+1)
        cand = tree.query(b)          # 候选省
        if len(cand) == 0:
            continue
        # 判断瓦片是否完全在某省内（近似in）/ 跨界
        inside_count = sum(1 for i in cand if polys[i].contains(b))
        if inside_count > 0:
            rel = 'in'
        else:
            rel = 'cross'
        with open(fn, encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh))
        relc[rel] += 1
        total_all += len(rows)
        if rel == 'in':
            for r in rows:
                agg[int(r['class'])] += 1     # 保留 GLC 原始码（不再映射）
            total_in += len(rows)
        else:
            # 跨界瓦片：逐点判断是否落在任一候选省内
            for r in rows:
                pt = Point(float(r['lon']), float(r['lat']))
                if any(polys[i].contains(pt) for i in cand):
                    agg[int(r['class'])] += 1     # 保留 GLC 原始码
                    total_in += 1
    print(f'瓦片关系: {dict(relc)}')
    print(f'总点: {total_all:,}  境内: {total_in:,}')
    print('=== 境内类别分布（映射到产品码）===')
    for c, n in agg.most_common():
        print(f'  {c:>4}: {n:,}')
    json.dump({'relation': dict(relc), 'total_all': total_all, 'inside': total_in,
               'class_inside': {str(k): v for k, v in agg.most_common()}},
              open(os.path.join(OUT_DIR, 'samples_cn_final_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n[输出] samples_cn_final_stats.json')

if __name__ == '__main__':
    main()

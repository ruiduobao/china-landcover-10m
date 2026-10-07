# -*- coding: utf-8 -*-
"""
lc10_全量边界内统计.py — 用中国边界剔除境外样本，全量重统计（修正 lc01 高估）
* 输出：真实中国陆域样本的 类别分布 + 每类纬度带分布（地理先验验证用）+ 境外剔除统计
运行: python lc10_全量边界内统计.py
"""
import os, re, csv, json, logging
from collections import Counter, defaultdict
from multiprocessing import Pool
from lc09_地理实证 import load_boundary, is_in_china

DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
BOUNDARY = r'数据/边界/china_100000_full.json'
OUT_DIR  = r'数据/本地处理/统计'
LOG_DIR  = r'数据/本地处理/日志'
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, 'lc10.log'), level=logging.INFO,
                    format='%(asctime)s %(message)s', encoding='utf-8')

_BPOLYS = None; _CMAP = None
def _init(bp, cm):
    global _BPOLYS, _CMAP
    _BPOLYS = bp; _CMAP = cm

def _process(fn):
    inside = Counter(); outside = 0; total = 0
    latband = Counter(); mang_lat = Counter()
    with open(fn, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            total += 1
            lon = float(row['lon']); lat = float(row['lat'])
            if not is_in_china(lon, lat, _BPOLYS):
                outside += 1
                latband[int(lat//5)*5] += 1
                continue
            c = int(row['class']); nc = _CMAP.get(c, -1)
            inside[nc] += 1
            if nc == 184: mang_lat[int(lat//5)*5] += 1   # 红树林纬度带
    return {'inside': inside, 'outside': outside, 'total': total,
            'latband': latband, 'mang_lat': mang_lat}

def main():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from lc_conf import CODE_MAP
    bp = load_boundary(BOUNDARY)
    files = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('.csv'):
            files.append(os.path.join(DATA_DIR, f))
    print(f'全量文件: {len(files)}')
    agg = Counter(); outside_n = 0; tot = 0
    lat_band = Counter(); mang_lat = Counter()
    with Pool(5, initializer=_init, initargs=(bp, CODE_MAP)) as p:
        for r in p.imap_unordered(_process, files, chunksize=8):
            agg.update(r['inside']); outside_n += r['outside']; tot += r['total']
            lat_band.update(r['latband']); mang_lat.update(r['mang_lat'])
    res = {'total_all': tot, 'outside': outside_n,
           'inside_total': tot - outside_n,
           'class_inside': {str(k): v for k, v in agg.most_common()},
           'outside_by_lat_band': {str(k): v for k, v in sorted(lat_band.items())},
           'mangrove_by_lat_band': {str(k): v for k, v in sorted(mang_lat.items())}}
    json.dump(res, open(os.path.join(OUT_DIR, 'samples_cn_boundary_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'总文件点: {tot:,}  境外剔除: {outside_n:,} ({100*outside_n/tot:.1f}%)  边界内: {tot-outside_n:,}')
    print(f'\n境外点纬度带分布(证明北纬>49为蒙/俄):')
    for k, v in sorted(lat_band.items()): print(f'  {k}-{k+4}N: {v:,}')
    print(f'\n边界内类别分布:')
    for c, n in agg.most_common():
        print(f'  {c:>4}: {n:,}')
    print(f'\n红树林(184)纬度带:')
    for k, v in sorted(mang_lat.items()): print(f'  {k}-{k+4}N: {v:,}')
    print('\n[输出] samples_cn_boundary_stats.json')

if __name__ == '__main__':
    main()

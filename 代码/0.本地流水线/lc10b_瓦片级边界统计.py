# -*- coding: utf-8 -*-
"""
lc10b_瓦片级边界统计.py — 按瓦片快速剔除境外，重估中国陆域样本（含跨界瓦片精算）
* 策略：每 1°瓦片先判断与中国的包含关系（质心 in-china / 角点 in-china 组合）。
*   完全境外 → 整片跳过；完全境内 → 整片计入；跨界 → 对文件抽样点 in-china 精估。
* 比逐点全量 in-china 快得多（跨界瓦片仅少数，抽样精算即可）。
运行: python lc10b_瓦片级边界统计.py
"""
import os, re, csv, json
from collections import Counter
from multiprocessing import Pool
from lc09_地理实证 import load_boundary, is_in_china

DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
BOUNDARY = r'数据/边界/china_100000_full.json'
OUT_DIR  = r'数据/本地处理/统计'
os.makedirs(OUT_DIR, exist_ok=True)

_BPOLYS = None; _CMAP = None
def _init(bp, cm):
    global _BPOLYS, _CMAP
    _BPOLYS = bp; _CMAP = cm

def tile_relation(lon0, lat0, polys):
    """瓦片(lon0..+1, lat0..+1) 与中国的包含关系：'in','out','cross'"""
    corners = [(lon0, lat0), (lon0+1, lat0), (lon0, lat0+1), (lon0+1, lat0+1)]
    in_c = sum(1 for c in corners if is_in_china(c[0], c[1], polys))
    if in_c == 4: return 'in'
    if in_c == 0: return 'out'
    return 'cross'

def _process(tile):
    lon0, lat0 = tile
    rel = tile_relation(lon0, lat0, _BPOLYS)
    fn = os.path.join(DATA_DIR, f'Sample_Lon{lon0}_Lat{lat0}.csv')
    cnt = Counter(); total = 0; sample_in = None
    if rel == 'out':
        return {'rel': 'out', 'total': 0, 'cnt': {}, 'est_in': 0}
    if rel == 'in':
        with open(fn, encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                total += 1
                c = _CMAP.get(int(row['class']), -1); cnt[c] += 1
        return {'rel': 'in', 'total': total, 'cnt': dict(cnt), 'est_in': total}
    # cross：逐点全量 in_china（跨界瓦片在统计主循环中被单独串行处理，这里直接返回全量精算结果）
    with open(fn, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    total = len(rows)
    for r in rows:
        if is_in_china(float(r['lon']), float(r['lat']), _BPOLYS):
            c = _CMAP.get(int(r['class']), -1); cnt[c] += 1
    return {'rel': 'cross', 'total': total, 'cnt': dict(cnt),
            'est_in': sum(cnt.values())}

def main():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from lc_conf import CODE_MAP
    bp = load_boundary(BOUNDARY)
    files = os.listdir(DATA_DIR)
    tiles = []
    for f in files:
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv', f)
        if m: tiles.append((int(m.group(1)), int(m.group(2))))
    print(f'瓦片数: {len(tiles)}')
    agg = Counter(); total_in = 0; total_all = 0
    relc = Counter()
    # 先并行处理 in/out 瓦片（整片计，快），跨界瓦片攒起来单进程串行
    cross_tiles = []
    with Pool(5, initializer=_init, initargs=(bp, CODE_MAP)) as p:
        for r in p.imap_unordered(_process, tiles, chunksize=40):
            relc[r['rel']] += 1
            if r['rel'] == 'cross':
                cross_tiles.append(r)
                continue
            total_in += r['est_in']; total_all += r['total']
            agg.update(r['cnt'])
    # 跨界瓦片串行逐点精算（量小）
    print(f'跨界瓦片 {len(cross_tiles)} 个，串行精算…')
    for r in cross_tiles:
        total_in += r['est_in']; total_all += r['total']
        agg.update(r['cnt'])
    res = {'tiles': len(tiles), 'rel': dict(relc), 'total_all': total_all,
           'inside_est': total_in,
           'class_inside_est': {str(k): v for k, v in agg.most_common()}}
    json.dump(res, open(os.path.join(OUT_DIR, 'samples_cn_tile_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'瓦片关系: {dict(relc)}')
    print(f'总点(全部文件): {total_all:,}  边界内估算: {total_in:,} ({100*total_in/total_all:.0f}%)')
    print(f'\n边界内类别估算分布:')
    for c, n in agg.most_common():
        print(f'  {c:>4}: {n:,}')
    print('\n[输出] samples_cn_tile_stats.json')

if __name__ == '__main__':
    main()

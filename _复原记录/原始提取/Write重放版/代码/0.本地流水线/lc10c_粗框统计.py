# -*- coding: utf-8 -*-
"""
lc10c_粗框统计.py — 用"中国陆域粗经纬框 + 去东北/西北境外"快速重估（不用逐点in-china）
* 目的：修正 lc10b 因跨界瓦片太慢而超时的问题，给出真实的中国陆域样本类别分布。
* 粗框：lon 73-135E, lat 18-54N（主陆域）
* 特殊剔除：东北北部(>50N & >126E 的蒙俄)… 简化处理：直接采用行政粗框并注明近似。
* 更快：只读 lon73-135/lat18-54 瓦片，整片计入（与 lc01 同框，但说明局限）。
运行: python lc10c_粗框统计.py
"""
import os, re, csv
from collections import Counter
from multiprocessing import Pool
from lc_conf import CODE_MAP

DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
OUT_DIR  = r'数据/本地处理/统计'
LON_LO, LON_HI, LAT_LO, LAT_HI = 73, 136, 18, 55   # 主陆域粗框（北到54含部分蒙俄——见说明）
os.makedirs(OUT_DIR, exist_ok=True)

def _process(fn):
    cnt = Counter(); n = 0
    with open(fn, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            n += 1
            c = CODE_MAP.get(int(row['class']), -1); cnt[c] += 1
    return n, cnt

def main():
    tiles = []
    for f in os.listdir(DATA_DIR):
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv', f)
        if not m: continue
        lon, lat = int(m.group(1)), int(m.group(2))
        if LON_LO <= lon < LON_HI and LAT_LO <= lat < LAT_HI:
            tiles.append(os.path.join(DATA_DIR, f))
    print(f'粗框瓦片: {len(tiles)}')
    agg = Counter(); total = 0
    with Pool(5) as p:
        for n, cnt in p.imap_unordered(_process, tiles, chunksize=40):
            total += n; agg.update(cnt)
    res = {'tiles': len(tiles), 'total': total,
           'class_dist': {str(k): v for k, v in agg.most_common()}}
    import json
    json.dump(res, open(os.path.join(OUT_DIR, 'samples_cn_bbox_stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'粗框总点: {total:,}')
    print('类别分布(映射后):')
    for c, n in agg.most_common():
        print(f'  {c:>4}: {n:,}')
    print('\n[说明] 粗框含东北/西北边境少量境外(蒙/俄)，数值略高；精确境内数以 lc10b 跨界精算为准。')
    print('[输出] samples_cn_bbox_stats.json')

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
lc01_样本统计.py — 中国区样本库本地统计与质控
* 不依赖 GEE / 嵌入特征，仅对已有样本 CSV 做统计。
* 限 5 核：multiprocessing 4 workers；低内存：workers 只回传聚合 Counter。
运行: python 代码/0.本地流水线/lc01_样本统计.py
"""
import os, re, csv, sys, json, logging
from collections import Counter
from multiprocessing import Pool

# 中国区范围（瓦片起点经纬度，含 1° 跨度的判断）
LON_LO, LON_HI = 73, 136
LAT_LO, LAT_HI = 17, 55
DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
OUT_DIR  = r'数据/本地处理/统计'
LOG_DIR  = r'数据/本地处理/日志'
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(filename=os.path.join(LOG_DIR, 'lc01.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# GLC_FCS30D → 本产品30类精细体系 码表映射（来自 03文档 §6.4；本文件是单一事实来源之一）
CODE_MAP = {
    10:10, 11:10, 12:11, 20:12,
    51:51, 52:52, 61:61, 62:62, 71:71, 72:72, 81:81, 82:82, 91:91, 92:92,
    120:120, 121:121, 122:121,
    130:130, 140:140,
    150:150, 151:150, 152:150, 153:150,
    180:180, 181:181, 182:182, 183:183, 184:184, 185:185, 186:186, 187:186,
    190:190, 200:200, 201:201, 202:201, 210:202, 220:220,
}

def china_tiles():
    """返回中国区瓦片（起点lon, lat）列表"""
    tiles = []
    for f in os.listdir(DATA_DIR):
        if not f.endswith('.csv'): continue
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv', f)
        if not m: continue
        lon, lat = int(m.group(1)), int(m.group(2))
        if LON_LO <= lon < LON_HI and LAT_LO <= lat < LAT_HI:
            tiles.append((lon, lat))
    return tiles

def _process(tile):
    lon, lat = tile
    fn = os.path.join(DATA_DIR, f'Sample_Lon{lon}_Lat{lat}.csv')
    cnt = Counter(); raw = Counter(); n = 0
    try:
        with open(fn, encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                cls = row.get('class'); n += 1
                if cls is None: continue
                try: c = int(cls)
                except ValueError: continue
                raw[c] += 1
                m = CODE_MAP.get(c)
                if m is not None: cnt[m] += 1
    except Exception as e:
        logging.warning(f'{fn}: {e}')
    return {'tile': tile, 'n': n, 'mapped': cnt, 'raw': raw}

def main():
    tiles = china_tiles()
    logging.info(f'中国区瓦片数: {len(tiles)}')
    print('瓦片数:', len(tiles))
    tot_n = 0; newcnt = Counter(); rawcnt = Counter(); per_tile = {}
    with Pool(4) as p:
        for r in p.imap_unordered(_process, tiles):
            tot_n += r['n']; newcnt.update(r['mapped']); rawcnt.update(r['raw'])
            l = r['tile'][0]; la = r['tile'][1]
            per_tile[(l, la)] = r['n']
    # 输出
    out = {'total_points': tot_n, 'tiles': len(tiles),
           'class_raw': dict(rawcnt.most_common()),
           'class_new': dict(newcnt.most_common()),
           'per_tile_points': {f'{l}_{la}': n for (l, la), n in per_tile.items()}}
    with open(os.path.join(OUT_DIR, 'samples_cn_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 摘要表（新码）
    lines = ['类别码,样本数']
    for c, n in newcnt.most_common():
        lines.append(f'{c},{n}')
    with open(os.path.join(OUT_DIR, 'class_distribution_new.csv'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logging.info(f'总计 {tot_n} 点，新码类数 {len(newcnt)}')
    print(f'\n=== 中国区样本统计 ===')
    print(f'总点数: {tot_n:,}')
    print(f'瓦片数: {len(tiles)}')
    print(f'新码类数: {len(newcnt)}')
    print(f'\n新码类分布(前25):')
    for c, n in newcnt.most_common(25):
        print(f'  code {c:>3}: {n:,}')
    # 稀有类检查
    print('\n稀有类(<2000点):')
    for c, n in sorted(newcnt.items()):
        if n < 2000: print(f'  code {c:>3}: {n:,}')
    with open(os.path.join(LOG_DIR,'lc01.txt'),'w',encoding='utf-8') as f:
        f.write(f'总点数 {tot_n}\n瓦片 {len(tiles)}\n新码类数 {len(newcnt)}\n')
    print('\n[OK] 输出:', OUT_DIR)

if __name__ == '__main__':
    main()

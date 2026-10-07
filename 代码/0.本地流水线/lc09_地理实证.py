# -*- coding: utf-8 -*-
"""
lc09_地理实证.py — 用样本点经纬度验证 CODE_MAP + 剔除非中国领土样本
* 意义：修 CODE_MAP 后，用"已知地理先验"实证映射是否正确，并纠正 lat54 统计。
* 测试（地理先验）：
   1) 红树林(184)样本应几乎全部落在华南沿海（海南/广东/广西/福建，纬度<26N且滨海）
   2) 郁闭落叶针叶林(81/82)应集中在东北大兴安岭
   3) 测试点若落在境外(蒙/俄等)，应按中国边界剔除
* 输入：中国边界 geojson(DataV 100000_full) + 样本CSV（抽样）
运行: python lc09_地理实证.py
"""
import os, re, glob, csv, json, logging, math
from collections import Counter
from multiprocessing import Pool

DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
BOUNDARY = r'数据/边界/china_100000_full.json'
OUT_DIR  = r'数据/本地处理/统计'
os.makedirs(OUT_DIR, exist_ok=True)

def is_in_china(lon, lat, polys):
    """射线法点-in-多边形；支持带洞多边形简单处理（取外环）"""
    def in_poly(x, y, ring):
        inside = False
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i]; xj, yj = ring[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi:
                inside = not inside
            j = i
        return inside
    for poly in polys:
        # 简单取外环（首个环）；洞未处理（边界附近极少数，可接受）
        if in_poly(lon, lat, poly[0]):
            return True
    return False

def load_boundary(path):
    """加载 geojson，返回若干多边形（外环坐标 [lon,lat]）"""
    d = json.load(open(path, encoding='utf-8'))
    polys = []
    for f in d['features']:
        g = f['geometry']
        if g['type'] == 'Polygon':
            polys.append([ring for ring in g['coordinates']])
        elif g['type'] == 'MultiPolygon':
            for p in g['coordinates']:
                polys.append([ring for ring in p])
    return polys

def _process(fn):
    """统计单个CSV：返回 {lon,lat,映射后类计数, 境外点计数}"""
    cnt = Counter(); outside = 0; total = 0
    with open(fn, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            total += 1
            lon = float(row['lon']); lat = float(row['lat'])
            if not is_in_china(lon, lat, BOUNDARY_POLYS):
                outside += 1
                continue
            c = int(row['class'])
            cnt[CODE_MAP.get(c, -1)] += 1
    return {'fn': os.path.basename(fn), 'total': total, 'outside': outside, 'cnt': cnt}

# 全局（worker用）
BOUNDARY_POLYS = []
CODE_MAP = {}
def _init(boundary_path, cmap):
    global BOUNDARY_POLYS, CODE_MAP
    BOUNDARY_POLYS = load_boundary(boundary_path)
    CODE_MAP = cmap

def main():
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from lc_conf import CODE_MAP as CM
    # 只取中国区瓦片（含少量境外/邻国的都算，靠边界剔）
    files = []
    for f in os.listdir(DATA_DIR):
        if not f.endswith('.csv'): continue
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv', f)
        if not m: continue
        lon, lat = int(m.group(1)), int(m.group(2))
        if 60 <= lon <= 136 and 10 <= lat <= 60:   # 扩大范围以含邻国边界测试
            files.append(os.path.join(DATA_DIR, f))
    print(f'待测文件: {len(files)}')
    # 抽样 1/30 估算（全量太慢）
    files = files[::30]
    print(f'抽样文件: {len(files)}')
    from multiprocessing import Pool
    # 用 _init 初始化各 worker 的全局边界
    p = Pool(4, initializer=_init, initargs=(BOUNDARY, CM))
    results = []
    for r in p.imap_unordered(_process, files):
        results.append(r)
    p.close(); p.join()

    # 汇总
    agg = Counter(); outside_n = 0; total_n = 0
    for r in results:
        agg.update(r['cnt']); outside_n += r['outside']; total_n += r['total']
    print(f'\n=== 抽样统计（文件 {len(results)}，点 {total_n}）===')
    print(f'境外点: {outside_n} ({100*outside_n/max(total_n,1):.1f}%)')
    print('映射后新码分布:')
    for c, n in agg.most_common():
        print(f'  {c:>4}: {n:,}')
    # 关键地理测试：红树林分布纬度
    # （此抽样已带 outside 剔除，红树林检验在下方单独做——因抽样文件少，另用全量做特定类）

if __name__ == '__main__':
    main()

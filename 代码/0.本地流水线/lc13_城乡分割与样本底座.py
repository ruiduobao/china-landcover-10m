# -*- coding: utf-8 -*-
"""
lc13_城乡分割与样本底座.py — 生成本项目"中国区样本底座 v1"
* 步骤：
  1) 读取 lc12 的精确境内统计（samples_cn_final_stats.json 为汇总；这里重新逐瓦片读原始CSV）
  2) 用 STRtree+prepared 加速逐点判断（复用 lc12 逻辑，输出为点级 Parquet，不再只是汇总）
  3) GLC 190 点用 GUB 2020 城市边界分割：in→产品190(城镇)、out→产品200(乡村)
  4) GLC 码 → 产品30类码（lc_conf.CODE_MAP，已修正）
  5) 输出 Parquet：lon/lat/class_glc/class_new/urban_flag/year/tile_id
* 资源：5 进程、分批写、低内存（每批 flush）
运行: python lc13_城乡分割与样本底座.py
"""
import os, re, csv, json, logging
from collections import Counter
from multiprocessing import Pool
import pandas as pd
from shapely.geometry import shape, box, Point
from shapely.strtree import STRtree
import geopandas as gpd

from lc_conf import CODE_MAP

DATA_DIR = r'数据/样本数据/GLC_Samples_Grid_2x2'
BOUNDARY = r'数据/边界/china_100000_full.json'
GUB_SHP  = r'E:\data\非洲城市发展和驱动力分析\数据\GUB_全球城市边界数据\GUB_Global_2020\GUB_Global_2020.shp'
OUT_PARQUET = r'数据/本地处理/样本底座/cn_samples_v1.parquet'
LOG = r'数据/本地处理/日志/lc13.log'
os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
logging.basicConfig(filename=LOG, level=logging.INFO, format='%(asctime)s %(message)s', encoding='utf-8')

# ===== 全局 worker 变量 =====
_g_polys=None; _g_tree=None; _g_gub=None; _g_gub_tree=None; _cmap=None

def _init(bpath, gpath, cmap):
    global _g_polys, _g_tree, _g_gub, _g_gub_tree, _cmap
    d = json.load(open(bpath, encoding='utf-8'))
    polys = []
    for f in d['features']:
        sg = shape(f['geometry'])
        polys += list(sg.geoms) if sg.geom_type=='MultiPolygon' else [sg]
    _g_polys = polys; _g_tree = STRtree(polys)
    gub = gpd.read_file(gpath)
    _g_gub = list(gub.geometry)          # 12万城市多边形
    _g_gub_tree = STRtree(_g_gub)
    _cmap = cmap

def _in_china(lon, lat):
    pt = Point(lon, lat)
    for i in _g_tree.query(pt):
        if _g_polys[i].contains(pt):
            return True
    return False

def _in_gub(lon, lat):
    pt = Point(lon, lat)
    for i in _g_gub_tree.query(pt):
        if _g_gub[i].contains(pt):
            return True
    return False

def _process(tile):
    """处理一个瓦片：返回点级记录列表（境内）"""
    lon0, lat0 = tile
    fn = os.path.join(DATA_DIR, f'Sample_Lon{lon0}_Lat{lat0}.csv')
    b = box(lon0, lat0, lon0+1, lat0+1)
    cand = _g_tree.query(b)
    if len(cand)==0:
        return []
    rows_out = []
    with open(fn, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            lon = float(r['lon']); lat = float(r['lat'])
            pt = Point(lon, lat)
            inside = any(_g_polys[i].contains(pt) for i in cand)
            if not inside: continue
            glc = int(r['class'])
            new = _cmap.get(glc, -1)
            urban = None
            if glc == 190:                       # 不透水→GUB分割
                urban = 1 if _in_gub(lon, lat) else 0
                new = 190 if urban==1 else 200
            rows_out.append({'lon':lon,'lat':lat,'class_glc':glc,
                             'class_new':new,'urban':urban,
                             'year':int(r.get('year',2020)),
                             'tile_id':r.get('tile_id','')})
    return rows_out

def main():
    tiles = []
    for f in os.listdir(DATA_DIR):
        m = re.match(r'Sample_Lon(-?\d+)_Lat(-?\d+)\.csv', f)
        if not m: continue
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        tiles.append((lon0, lat0))
    # 只跑与中国相交的（bounds 粗筛，之后 worker 里再精确判断）
    import json as _j
    bb = _j.load(open(BOUNDARY, encoding='utf-8'))
    xs=[];ys=[]
    for f in bb['features']:
        g=f['geometry']
        if g['type']=='Polygon': c=g['coordinates'][0]
        else: c=g['coordinates'][0][0]
        for x,y in c: xs.append(x); ys.append(y)
    minx,maxx,miny,maxy = min(xs),max(xs),min(ys),max(ys)
    tiles = [t for t in tiles if t[0]<maxx and t[0]+1>minx and t[1]<maxy and t[1]+1>miny]
    logging.info(f'候选瓦片 {len(tiles)}')
    print(f'候选瓦片: {len(tiles)}')

    writer = None
    agg = Counter(); n_total=0
    BATCH = 200
    with Pool(5, initializer=_init, initargs=(BOUNDARY, GUB_SHP, CODE_MAP)) as p:
        buf = []
        for rows in p.imap_unordered(_process, tiles, chunksize=8):
            if rows:
                buf.extend(rows); agg.update(r['class_new'] for r in rows)
                n_total += len(rows)
            if len(buf) >= 50000:
                df = pd.DataFrame(buf)
                if writer is None:
                    df.to_parquet(OUT_PARQUET, index=False)
                    writer = True
                else:
                    df.to_parquet(OUT_PARQUET + f'.part{n_total}.parquet', index=False)
                buf = []
                print(f'  已写 {n_total:,} 点')
        if buf:
            df = pd.DataFrame(buf)
            if writer is None:
                df.to_parquet(OUT_PARQUET, index=False)
            else:
                df.to_parquet(OUT_PARQUET + f'.part{n_total}.parquet', index=False)
    print(f'\n境内总点: {n_total:,}')
    print('产品码分布:')
    for c, n in agg.most_common():
        print(f'  {c:>4}: {n:,}')
    logging.info(f'done {n_total}')
    json.dump({'total':n_total, 'class_new':{str(k):v for k,v in agg.most_common()}},
              open(os.path.join(os.path.dirname(OUT_PARQUET),'cn_samples_v1_summary.json'),'w',encoding='utf-8'),
              ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()

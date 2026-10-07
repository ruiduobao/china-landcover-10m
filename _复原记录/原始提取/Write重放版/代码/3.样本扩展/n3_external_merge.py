# -*- coding: utf-8 -*-
"""
n3_external_merge.py — 外部点级样本整合（筛中国 → 统一列）
* WorldCereal: 递归解嵌套 zip → *.geoparquet → 中国 bbox 过滤 → EWOC/label 映射
* GPW 草地:   harm_point gpkg（bbox 裁中国）→ 类别映射
* 输出: worldcereal_china.parquet / grass_gpw_china.parquet
"""
import sys, os, io, glob, zipfile, tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = '数据/外部样本源'
BJ_ALL = (73, 17, 136, 55)

# EWOC / label → 我们的 level2（部分映射；未列出的丢弃）
EWOC_MAP = {
    'rice': 12, 'paddy rice': 12,
    'rapeseed_rape': 10, 'rapeseed': 10, 'rape': 10,
    'maize': 10, 'wheat': 10, 'soybean': 10, 'barley': 10,
    'cotton': 10, 'sunflower': 10, 'sorghum': 10, 'millet': 10,
    'peanut': 11, 'groundnut': 11, 'tea': 11, 'tea plant': 11,
    'sugarcane': 10, 'sugarbeet': 10, 'potato': 10, 'cassava': 10,
    'vegetables': 10, 'vegetable': 10, 'fruits': 11, 'fruit': 11,
    'orchard': 11, 'citrus': 11, 'apple': 11, 'banana': 11,
}
def map_label(lbl):
    if not isinstance(lbl, str):
        return -1
    k = lbl.strip().lower()
    for key, v in EWOC_MAP.items():
        if key in k:
            return v
    return -1

def load_china_polys():
    import json
    d = json.load(open(r'数据/边界/china_100000_full.json', encoding='utf-8'))
    polys = []
    for f in d['features']:
        sg = shapely.geometry.shape(f['geometry'])
        polys += list(sg.geoms) if sg.geom_type == 'MultiPolygon' else [sg]
    return polys

def china_mask(xs, ys, polys):
    keep = np.zeros(len(xs), dtype=bool)
    from shapely import contains_xy
    m = (xs >= 73) & (xs <= 136) & (ys >= 17) & (ys <= 54)
    xs2, ys2 = xs[m], ys[m]
    km = np.zeros(len(xs2), dtype=bool)
    for p in polys:
        km |= contains_xy(p, xs2, ys2)
    keep[m] = km
    return keep

def iter_geoparquets():
    """递归展开所有 WorldCereal zip（两层），yield (名, bytes)"""
    for zpath in sorted(glob.glob(os.path.join(OUT, 'WorldCereal_*.zip'))):
        z = zipfile.ZipFile(zpath)
        for n in z.namelist():
            if n.endswith('.zip'):
                inner = zipfile.ZipFile(io.BytesIO(z.read(n)))
                for m in inner.namelist():
                    if m.endswith('.geoparquet'):
                        yield f'{os.path.basename(zpath)}/{m}', inner.read(m)
            elif n.endswith('.geoparquet'):
                yield f'{os.path.basename(zpath)}/{n}', z.read(n)

def worldcereal():
    polys = load_china_polys()
    frames = []
    total = 0
    for name, data in iter_geoparquets():
        tmp = os.path.join(tempfile.gettempdir(), 'wc_tmp.geoparquet')
        open(tmp, 'wb').write(data)
        try:
            g = gpd.read_parquet(tmp)
        except Exception as e:
            print('ERR', name, str(e)[:80])
            continue
        total += len(g)
        xs = g.geometry.x.to_numpy(); ys = g.geometry.y.to_numpy()
        keep = china_mask(xs, ys, polys)
        if not keep.any():
            continue
        gc = g[keep].copy()
        lblcol = 'label_full' if 'label_full' in gc.columns else None
        gc['label_full'] = gc.get('label_full', '')
        gc['our_class'] = gc['label_full'].astype(str).map(map_label) if lblcol else -1
        gc['src'] = name.split('/')[0].replace('WorldCereal_ReferenceData_', '').replace('.zip', '')
        keep2 = gc['our_class'] > 0
        gc = gc[keep2]
        if len(gc):
            frames.append(gc[['geometry', 'label_full', 'our_class', 'src'] +
                             (['irrigation_status'] if 'irrigation_status' in gc.columns else []) +
                             (['ewoc_code'] if 'ewoc_code' in gc.columns else [])])
            print(f'{name}: 中国内 {keep.sum()} → 可映射 {len(gc)}')
    if frames:
        g = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
        g['lon'] = g.geometry.x; g['lat'] = g.geometry.y
        g.drop(columns='geometry').to_parquet(os.path.join(OUT, 'worldcereal_china.parquet'), index=False)
        print('WorldCereal 中国样本:', len(g), dict(g.our_class.value_counts()))
    else:
        print('WorldCereal: 中国内无可映射点')

def gpw_grassland():
    fp = os.path.join(OUT, 'gpw_grassland_harm_point.gpkg')
    import pyogrio
    layers = pyogrio.list_layers(fp)
    print('layers:', layers)
    lname = layers[0][0]
    g = gpd.read_file(fp, layer=lname, bbox=(73, 17, 136, 55))
    print('中国bbox内:', len(g), '列:', list(g.columns)[:16])
    polys = load_china_polys()
    xs = g.geometry.x.to_numpy(); ys = g.geometry.y.to_numpy()
    keep = china_mask(xs, ys, polys)
    g = g[keep].copy()
    g['lon'] = g.geometry.x; g['lat'] = g.geometry.y
    print('国界内:', len(g))
    # 类别列探测
    ccol = [c for c in g.columns if c.lower() in ('class', 'grassland', 'lc', 'sample_class', 'interpretation')]
    print('类别列:', ccol, {c: g[c].value_counts().to_dict() for c in ccol})
    keepcols = ['lon', 'lat'] + ccol
    g[keepcols].assign(src='gpw_grass_2024').to_parquet(
        os.path.join(OUT, 'grass_gpw_china.parquet'), index=False)
    print('输出:', len(g))

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('wc', 'all'):
        worldcereal()
    if which in ('gpw', 'all'):
        gpw_grassland()

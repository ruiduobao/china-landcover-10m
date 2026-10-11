# -*- coding: utf-8 -*-
"""k19_provattrib.py — 省域归属的可靠实现（DataV 省界含 invalid 几何，禁用 make_valid 丢省）

* 背景（本次踩坑）：DataV 省界里 18 个省的几何是 **invalid** MultiPolygon；`shapely.make_valid`
  会把它们变成 GeometryCollection，其中一半**不含任何 Polygon**（原多边形被拆进线/点的混合），
  若按 geom_type 过滤就会**整省丢失**——实测四川、甘肃、河北、山东、广东、广西、湖北、湖南、
  河南、浙江、贵州、宁夏、北京、天津、重庆、香港、安徽、陕西 18 省被丢，导致 40% 的点被判"界外"。
* 做法：**直接用原始几何做 contains**（GEOS 对 invalid 几何仍可按拓扑判定，实测成都/川西/兰州均 True）；
        仅当原始几何本身非面类型时，才用 get_parts 抽多边形兜底。
* 用法（作为模块）：
        import k19_provattrib as PA
        prov = PA.attribute(df)      # df 需含 lon/lat
"""
import json, sys
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from shapely import get_parts
from shapely.strtree import STRtree

BND = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json'
_polys, _names = None, None


def load():
    global _polys, _names
    if _polys is not None:
        return _polys, _names
    gj = json.load(open(BND, encoding='utf-8'))
    polys, names = [], []
    for f in gj['features']:
        g = f.get('geometry')
        if not g:
            continue
        gg = shape(g)
        if gg.geom_type in ('Polygon', 'MultiPolygon'):
            polys.append(gg); names.append(f['properties'].get('name', '?'))
        else:                      # 线段等要素：抽其中的多边形部件
            for part in get_parts(gg):
                if part.geom_type == 'Polygon':
                    polys.append(part); names.append(f['properties'].get('name', '?'))
    _polys, _names = polys, names
    return polys, names


def attribute(df, lon_col='lon', lat_col='lat'):
    """返回与 df 等长的省名数组（未落省置 ''）。"""
    polys, names = load()
    tree = STRtree(polys)
    pts = [Point(float(x), float(y)) for x, y in zip(df[lon_col], df[lat_col])]
    idx = tree.query(pts)
    prov = np.array([''] * len(df), dtype=object)
    if idx.size:
        for col in range(idx.shape[1]):
            r, c = int(idx[0, col]), int(idx[1, col])
            if prov[r] == '' and polys[c].contains(pts[r]):
                prov[r] = names[c]
    return prov


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], encoding='utf-8-sig') if len(sys.argv) > 1 else None
    if df is None:
        raise SystemExit(__doc__)
    out = attribute(df)
    print('归属 %d/%d；省数 %d' % (int((out != '').sum()), len(out), len(set(out) - {''})))

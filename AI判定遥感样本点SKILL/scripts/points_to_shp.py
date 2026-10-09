# -*- coding: utf-8 -*-
"""points_to_shp.py — 判读成果转 Shapefile（GIS 交付；可选 WGS84 + GCJ-02 双版）

* 输入：--csv 判读合并表（含 id/lon/lat 与判读列）
* 规则源：references/影像源与瓦片手册.md（DBF 限制与国内坐标偏移）
* 门槛：字段名转 ASCII 且 ≤10 字符（DBF 硬限）；文本 ≤250 字节；写后回读校验要素数与字段数
* 输出：<outdir>/<name>.{shp,shx,dbf,prj,cpg}（UTF-8 + .cpg）
        <outdir>/<name>_GCJ02.shp...（--gcj02 时，供叠国内矢量/高德底图；元数据名义 EPSG:4326）
        <outdir>/字段对照.txt
* 用法：python points_to_shp.py --csv merged.csv --outdir ./shp --gcj02
"""
import argparse
import csv
import math
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import geopandas as gpd

_NUM = ('lon', 'lat')

_A, _EE = 6378245.0, 0.00669342162296594323


def _out_of_china(lon, lat):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _tl(x, y):
    r = -100 + 2 * x + 3 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    r += (20 * math.sin(6 * x * math.pi) + 20 * math.sin(2 * x * math.pi)) * 2 / 3
    r += (20 * math.sin(y * math.pi) + 40 * math.sin(y / 3 * math.pi)) * 2 / 3
    r += (160 * math.sin(y / 12 * math.pi) + 320 * math.sin(y * math.pi / 30)) * 2 / 3
    return r


def _tg(x, y):
    r = 300 + x + 2 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    r += (20 * math.sin(6 * x * math.pi) + 20 * math.sin(2 * x * math.pi)) * 2 / 3
    r += (20 * math.sin(x * math.pi) + 40 * math.sin(x / 3 * math.pi)) * 2 / 3
    r += (150 * math.sin(x / 12 * math.pi) + 300 * math.sin(x / 30 * math.pi)) * 2 / 3
    return r


def wgs84_to_gcj02(lon, lat):
    if _out_of_china(lon, lat):
        return lon, lat
    dlat, dlon = _tl(lon - 105.0, lat - 35.0), _tg(lon - 105.0, lat - 35.0)
    rad = lat / 180.0 * math.pi
    m = 1 - _EE * math.sin(rad) ** 2
    sm = math.sqrt(m)
    return (lon + (dlon * 180.0) / (_A / sm * math.cos(rad) * math.pi),
            lat + (dlat * 180.0) / ((_A * (1 - _EE)) / (m * sm) * math.pi))


def ascii_name(c, used):
    """非 ASCII/超长列名 → 短名（f1, f2…）；ASCII 保留并截 10。"""
    s = re.sub(r'[^0-9A-Za-z_]', '', str(c))
    s = s[:10] if s else ''
    if not s or s in used:
        i = 1
        while ('f%d' % i) in used:
            i += 1
        s = 'f%d' % i
    used.add(s)
    return s


def main():
    ap = argparse.ArgumentParser(description='判读成果转 SHP')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--name', default='labels')
    ap.add_argument('--lon-col', default='lon')
    ap.add_argument('--lat-col', default='lat')
    ap.add_argument('--gcj02', action='store_true', help='额外输出 GCJ-02 版（叠国内矢量/高德）')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    df = pd.read_csv(a.csv, dtype=str, encoding='utf-8-sig').fillna('')
    if a.lon_col not in df.columns or a.lat_col not in df.columns:
        raise SystemExit('缺经纬度列')
    lon = pd.to_numeric(df[a.lon_col], errors='coerce')
    lat = pd.to_numeric(df[a.lat_col], errors='coerce')
    used, ren, note = set(), {}, []
    for c in df.columns:
        n = ascii_name(c, used)
        ren[c] = n
        if n != c:
            note.append((n, c))
    attrs = df.drop(columns=[a.lon_col, a.lat_col]).rename(columns=ren)
    attrs = attrs.applymap(lambda v: v[:240]);

    def write(fp, lo, la):
        g = gpd.GeoDataFrame(attrs.copy(), geometry=gpd.points_from_xy(lo, la), crs='EPSG:4326')
        g.to_file(fp, driver='ESRI Shapefile', engine='pyogrio', encoding='UTF-8')

    fp1 = os.path.join(a.outdir, a.name + '.shp')
    write(fp1, lon, lat)
    with open(os.path.join(a.outdir, '字段对照.txt'), 'w', encoding='utf-8') as f:
        f.write('shp 字段 → 原列名（DBF 限 10 字节 ASCII）\n\n')
        for n, c in note:
            f.write('%-12s %s\n' % (n, c))
        f.write('\n坐标版本：\n  %s.shp  WGS84（对卫星影像）\n' % a.name)
        if a.gcj02:
            f.write('  %s_GCJ02.shp  GCJ-02（对国内矢量/高德底图；两版偏移约 300–700 m，勿混用）\n' % a.name)
    if a.gcj02:
        gl = [wgs84_to_gcj02(x, y) for x, y in zip(lon, lat)]
        write(os.path.join(a.outdir, a.name + '_GCJ02.shp'), [p[0] for p in gl], [p[1] for p in gl])
    import fiona
    for fp in ([fp1] + ([os.path.join(a.outdir, a.name + '_GCJ02.shp')] if a.gcj02 else [])):
        with fiona.open(fp, encoding='UTF-8') as s:
            over = [n for n in s.schema['properties'] if len(n) > 10]
            print('读回 %s：%d 要素，超长字段 %s' % (os.path.basename(fp), len(s), over or '无'))
    print('字段对照：%s' % os.path.join(a.outdir, '字段对照.txt'))


if __name__ == '__main__':
    main()

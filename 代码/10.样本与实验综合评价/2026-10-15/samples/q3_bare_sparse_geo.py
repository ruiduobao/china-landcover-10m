# -*- coding: utf-8 -*-
"""q3_bare_sparse_geo.py — 全国裸地(22)/稀疏植被(13)空间分布：经纬带 / 省 / 东西分界 / 三窗口

* 输入：delivery2023_slim.parquet（2023 交付全量，含 ct_prov）、train2023_v2.parquet（池）、
        train2023_clean.parquet（首批池，30类码）
* 口径：省＝交付内嵌 ct_prov（CTAmap 2026 最近省赋值；训练/评估仍用 DataV 旧界——本表仅做解释）；
        区域窗口＝裸地/稀疏诊断报告 2026-10-14 同款（东北 119–135E/42–54N；华北 113–120E/34–40.5N；
        华东华南 114–125E/18–34N）
* 输出（同目录）：
  - bare_sparse_region_windows.csv   三窗口 × 层（交付/池/首批）×类 计数
  - bare_sparse_latlon_band.csv      5° 纬度带 × 5° 经度带（>90E 部分）计数
  - bare_sparse_province.csv         逐省 × 类（交付口径）
  - bare_sparse_ew_split.csv         东西分界（105/110/115E）计数
* 用法：python q3_bare_sparse_geo.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
SP = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/samples'
WINDOWS = {
    '东北_119_135E_42_54N': (119, 135, 42, 54),
    '华北_113_120E_34_40p5N': (113, 120, 34, 40.5),
    '华东华南_114_125E_18_34N': (114, 125, 18, 34),
}


def windows_count(df, tag):
    rows = []
    for wname, (x0, x1, y0, y1) in WINDOWS.items():
        m = (df.lon >= x0) & (df.lon < x1) & (df.lat >= y0) & (df.lat < y1)
        sub = df[m]
        for cl, g in sub.groupby('cl'):
            rows.append(dict(layer=tag, window=wname, cl=int(cl), n=len(g)))
        rows.append(dict(layer=tag, window=wname, cl=0, n=len(sub)))
    return rows


def main():
    d = pd.read_parquet(os.path.join(SP, 'delivery2023_slim.parquet'), columns=['cl', 'lon', 'lat', 'ct_prov'])
    rows = []
    for tag, df in [('交付2023', d),
                    ('池v2', d[d.cl.isin([13, 22])]),   # 池v2=交付剔除后，见 q2；此处仅用交付近似会被剔除点极少影响
                    ]:
        pass
    # 用真实池 v2 与首批池分别统计
    v2 = pd.read_parquet(r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet', columns=['cl', 'lon', 'lat'])
    v1 = pd.read_parquet(r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet',
                         columns=['class_new', 'lon', 'lat'])
    M30_24 = {51: 4, 52: 4, 61: 5, 62: 5, 71: 6, 72: 6, 81: 7, 82: 7, 91: 8, 92: 8, 190: 21, 200: 21,
              10: 1, 11: 2, 12: 3, 120: 9, 121: 10, 130: 11, 140: 12, 150: 13, 180: 14, 181: 15,
              182: 16, 183: 17, 184: 18, 185: 19, 186: 20, 201: 22, 202: 23, 220: 24}
    v1['cl'] = v1.class_new.map(M30_24)
    v1 = v1[v1.cl.isin([13, 22])].copy()

    dd = d[d.cl.isin([13, 22])].copy()
    rows += windows_count(dd, '交付2023')
    rows += windows_count(v2[v2.cl.isin([13, 22])], '池v2')
    rows += windows_count(v1, '首批池v1')
    pd.DataFrame(rows).to_csv(os.path.join(SP, 'bare_sparse_region_windows.csv'), index=False, encoding='utf-8-sig')

    # ---- 纬度带 × 经度带（中国域 73–135E, 18–54N）----
    bands = []
    for tag, df in [('交付2023', dd), ('池v2', v2[v2.cl.isin([13, 22])]), ('首批池v1', v1)]:
        for cl in [13, 22]:
            g = df[df.cl == cl]
            glat = pd.cut(g.lat, bins=list(range(15, 60, 5)), right=False)
            glon = pd.cut(g.lon, bins=list(range(70, 140, 5)), right=False)
            t = g.assign(lat5=glat.astype(str), lon5=glon.astype(str)).groupby(['lat5', 'lon5']).size()
            for (la, lo), n in t.items():
                bands.append(dict(layer=tag, cl=cl, lat_band=la, lon_band=lo, n=int(n)))
    pd.DataFrame(bands).to_csv(os.path.join(SP, 'bare_sparse_latlon_band.csv'), index=False, encoding='utf-8-sig')

    # ---- 省级（交付 + 池 v2）----
    prov_rows = []
    for tag, df in [('交付2023', dd), ('池v2', v2[v2.cl.isin([13, 22])])]:
        cp = dd[['cl', 'lon', 'lat']]  # placeholder
        if tag == '交付2023':
            t = df.groupby(['cl', 'ct_prov']).size()
            for (cl, pv), n in t.items():
                prov_rows.append(dict(layer=tag, cl=int(cl), ct_prov=pv, n=int(n)))
    # 池 v2 的省：用交付 ct_prov 回接
    slim = pd.read_parquet(os.path.join(SP, 'delivery2023_slim.parquet'), columns=['cl', 'point_id', 'ct_prov'])
    v2f = pd.read_parquet(r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet', columns=['cl', 'point_id'])
    v2f = v2f[v2f.cl.isin([13, 22])]
    j = v2f.merge(slim, on=['cl', 'point_id'], how='left')
    t = j.groupby(['cl', 'ct_prov']).size()
    for (cl, pv), n in t.items():
        prov_rows.append(dict(layer='池v2', cl=int(cl), ct_prov=pv, n=int(n)))
    pd.DataFrame(prov_rows).to_csv(os.path.join(SP, 'bare_sparse_province.csv'), index=False, encoding='utf-8-sig')

    # ---- 东西分界 ----
    ew = []
    for tag, df in [('交付2023', dd), ('池v2', v2[v2.cl.isin([13, 22])]), ('首批池v1', v1)]:
        for cl in [13, 22]:
            g = df[df.cl == cl]
            row = dict(layer=tag, cl=cl, n=len(g))
            for cut in [100, 105, 110, 115]:
                row['west_of_%dE' % cut] = int((g.lon < cut).sum())
                row['east_of_%dE' % cut] = int((g.lon >= cut).sum())
            ew.append(row)
    pd.DataFrame(ew).to_csv(os.path.join(SP, 'bare_sparse_ew_split.csv'), index=False, encoding='utf-8-sig')

    # 控制台速览
    for tag, df in [('交付2023', dd), ('池v2', v2[v2.cl.isin([13, 22])]), ('首批池v1', v1)]:
        for cl in [13, 22]:
            g = df[df.cl == cl]
            ne = ((g.lon >= 119) & (g.lon < 135) & (g.lat >= 42) & (g.lat < 54)).sum()
            ec = ((g.lon >= 114) & (g.lon < 125) & (g.lat >= 18) & (g.lat < 34)).sum()
            print('%s cl=%d n=%d 东北=%d 华东华南=%d <105E=%d' % (tag, cl, len(g), ne, ec, (g.lon < 105).sum()))
    print('done')


if __name__ == '__main__':
    main()

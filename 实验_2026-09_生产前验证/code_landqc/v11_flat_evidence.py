# -*- coding: utf-8 -*-
"""v11_flat_evidence.py — 湖河滩地海岸甄别：证据合并 + 分布体检

* 输入：lists/flat_delivered.parquet（26,507 点）
        feats/flat_p*.parquet（v10：it_n/it_ever/d_it_m/d_sea_m/etopo_m）
        F:/lc_work/v31_exp/data/land_qc/feats/water/land_p*.parquet（v7 本地证据 28 波段）
        交付 SHP（ct_prov/ct_code/ct_km + tier/src）
* 输出：out/evidence_all.parquet（全部证据）+ 屏显分布体检
* 用法：python v11_flat_evidence.py
"""
import glob, os, sys
import numpy as np
import pandas as pd
import shapefile

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
BASE = os.path.join(WORK, 'data', 'flat_fix')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/湖河滩地'
SENT_DIT = 1_390_228.5     # sqrt(2^31-1)*30 → ">9 km（300 px 外）"
SENT_DSEA = 85_977_512.0   # ">111 km（60 px 外）"
COASTAL = {'辽宁省', '河北省', '天津市', '山东省', '江苏省', '上海市', '浙江省', '福建省',
           '广东省', '广西壮族自治区', '海南省', '台湾省', '香港特别行政区', '澳门特别行政区'}


def main():
    pts = pd.read_parquet(os.path.join(BASE, 'lists', 'flat_delivered.parquet'))
    pts['point_id'] = pts.point_id.astype(str)
    # 交付 SHP 属性
    r = shapefile.Reader(os.path.join(DEST, '2023', '湖河滩地_2023_WGS84.shp'), encoding='utf-8')
    fl = [x[0] for x in r.fields[1:]]
    rows = []
    for rec in r.iterRecords():
        rows.append({k: rec[fl.index(k)] for k in ('pid', 'tier', 'src', 'ct_prov', 'ct_code', 'ct_km')})
    r.close()
    att = pd.DataFrame(rows).rename(columns={'pid': 'point_id'})
    att['point_id'] = att.point_id.astype(str)
    # v10 证据
    f10 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(BASE, 'feats', 'flat_p*.parquet')))],
                    ignore_index=True)
    f10['point_id'] = f10.point_id.astype(str)
    print('v10 证据 %d 行' % len(f10))
    # v7 本地证据
    fs7 = sorted(glob.glob(os.path.join(WORK, 'data', 'land_qc', 'feats', 'water', 'land_p*.parquet')))
    ids = set(pts.point_id)
    parts = []
    for f in fs7:
        d = pd.read_parquet(f)
        d['point_id'] = d.point_id.astype(str)
        s = d[d.point_id.isin(ids)]
        if len(s):
            parts.append(s)
    f7 = pd.concat(parts, ignore_index=True).drop_duplicates('point_id')
    print('v7 本地证据命中 %d 行' % len(f7))
    d = pts.merge(att, on='point_id', how='left').merge(f10, on='point_id', how='left').merge(f7, on='point_id', how='left')
    # 哨兵归一
    d['d_it_km'] = np.where(d.d_it_m >= SENT_DIT - 1, np.nan, d.d_it_m / 1000.0)
    d['d_sea_km'] = np.where(d.d_sea_m >= SENT_DSEA - 1, np.nan, d.d_sea_m / 1000.0)
    d['coastal_prov'] = d.ct_prov.isin(COASTAL)
    os.makedirs(os.path.join(BASE, 'out'), exist_ok=True)
    d.to_parquet(os.path.join(BASE, 'out', 'evidence_all.parquet'), index=False)
    # —— 分布体检 ——
    n = len(d)
    print('\n=== 分布体检（n=%d）===' % n)
    print('it_n 分布:', d.it_n.value_counts().sort_index().to_dict())
    print('潮滩像元内（it_n≥1）: %d (%.2f%%)' % ((d.it_n >= 1).sum(), 100 * (d.it_n >= 1).mean()))
    for k in (1, 3, 5):
        print('  it_n≥%d: %d' % (k, (d.it_n >= k).sum()))
    for t in (0.5, 1, 2):
        inb = (d.d_it_km <= t).fillna(False)
        print('d_it ≤ %.1f km: %d（其中 not in tidal footprint: %d）' % (
            t, inb.sum(), (inb & (d.it_n == 0)).sum()))
    print('d_sea ≤ 3 km: %d ｜ d_sea ≤ 1 km: %d' % (
        (d.d_sea_km <= 3).fillna(False).sum(), (d.d_sea_km <= 1).fillna(False).sum()))
    print('沿海省: %d' % d.coastal_prov.sum())
    # 候选池画像
    sel = d[(d.it_n >= 1)]
    if len(sel):
        print('\n—— 潮滩像元内点（%d）画像 ——' % len(sel))
        for c in ('ct_prov',):
            print('  %s top: %s' % (c, sel[c].value_counts().head(8).to_dict()))
        for c in ('max_extent', 'occurrence', 'seasonality', 'max_300', 'dem', 's2_ndvi_xia', 'dw_bare_150', 'bare'):
            if c in sel:
                s = pd.to_numeric(sel[c], errors='coerce')
                print('  %-14s p10/p50/p90 = %.2f / %.2f / %.2f（非空 %d）' % (
                    c, s.quantile(.1), s.median(), s.quantile(.9), s.notna().sum()))
    print('\n→ out/evidence_all.parquet')


if __name__ == '__main__':
    main()

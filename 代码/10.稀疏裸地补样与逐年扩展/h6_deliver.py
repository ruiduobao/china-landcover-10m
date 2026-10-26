# -*- coding: utf-8 -*-
"""h6_deliver.py — 补样点交付（parquet + 逐类 GPKG/SHP + 分省统计 + README）

* 输入：--final data/final/cand_qc.parquet（status=='keep'）；--aef data/aef（h4 块）
* 省界：Y:/CTAmap/.../T2026_province.shp（EPSG:4326；字段 省/省级码）；界外点按最近省赋值 + ct_km
* 输出（--outdir 默认 ../delivery）：
    bare_sparse_v3_2023.parquet（全列 + AEF64 + prov/ct_code/ct_km）
    稀疏植被_2023_WGS84.gpkg / 裸地_2023_WGS84.gpkg（含 AEF64）
    shp/{稀疏植被,裸地}_2023_WGS84.shp（ASCII 字段名 ≤10B，浮点保留）
    分省统计.csv / README_补样说明.md
* 用法：python h6_deliver.py --final ../data/final/cand_qc.parquet --aef ../data/aef
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
from shapely.geometry import Point

sys.stdout.reconfigure(encoding='utf-8')
PROV_SHP = r'Y:/CTAmap/SCI论文/数据/2026CTAmap 省市县行政区划数据/province/T2026_province.shp'
AEQ = ('+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=104 +datum=WGS84 +units=m +no_defs')
FEATS = ['A%02d' % i for i in range(64)]


def load_aef(aefdir):
    fs = sorted(glob.glob(os.path.join(aefdir, 'aef_p*_b*.parquet')))
    if not fs:
        raise SystemExit('缺 AEF 块：%s' % aefdir)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates('pid')
    return d


def add_prov(g):
    prov = gpd.read_file(PROV_SHP)[['省', '省级码', 'geometry']].rename(
        columns={'省': 'ct_prov', '省级码': 'ct_code'})
    prov['geometry'] = prov.geometry.buffer(0)
    inside = gpd.sjoin(g, prov, predicate='within', how='left')
    inside = inside[~inside.index.duplicated(keep='first')]
    inside = inside.drop(columns=[c for c in ('index_right',) if c in inside.columns])
    miss = inside['ct_prov'].isna()
    inside['ct_km'] = 0.0
    if miss.any():
        # 界外点：在等积投影(EPSG 由 AEQ 定义)里做最近邻，距离单位=米 → km
        sub_a = gpd.GeoDataFrame(inside.loc[miss, ['lon', 'lat']].copy(),
                                 geometry=inside.loc[miss, 'geometry'].values,
                                 crs='EPSG:4326').to_crs(AEQ).reset_index()
        prov_a = prov.to_crs(AEQ)
        near = gpd.sjoin_nearest(sub_a, prov_a, how='left', distance_col='_d')
        near = near.drop_duplicates('index').set_index('index')
        inside.loc[miss, 'ct_prov'] = near['ct_prov']
        inside.loc[miss, 'ct_code'] = near['ct_code']
        inside.loc[miss, 'ct_km'] = near['_d'] / 1000.0
    inside['ct_km'] = inside['ct_km'].fillna(0.0)
    return inside.drop(columns=[c for c in ('index_right',) if c in inside.columns])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', required=True)
    ap.add_argument('--aef', required=True)
    ap.add_argument('--outdir', default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'delivery'))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.outdir, 'shp'), exist_ok=True)

    qc = pd.read_parquet(a.final)
    keep = qc[qc.status == 'keep'].copy().reset_index(drop=True)
    aef = load_aef(a.aef)
    d = keep.merge(aef, on='pid', how='left', validate='one_to_one')
    miss = int(d[FEATS].isna().all(axis=1).sum())
    print('keep %d ｜ AEF 全缺失 %d' % (len(d), miss), flush=True)
    d['year'] = 2023
    d['src'] = 'fcs10_2023_ne_east_topup'

    g = gpd.GeoDataFrame(d, geometry=[Point(x, y) for x, y in zip(d.lon, d.lat)], crs='EPSG:4326')
    g = add_prov(g)
    for c in FEATS:
        g[c] = pd.to_numeric(g[c], errors='coerce').astype('float32')
    g['cls24'] = g['fam'].map({'sparse': 13, 'bare': 22}).astype(int)
    g['cls_zh'] = g['fam'].map({'sparse': '稀疏植被', 'bare': '裸地'})
    g['tier'] = 'strong'
    out_cols = (['pid', 'lon', 'lat', 'cls24', 'cls_zh', 'tier', 'src',
                 'fam', 'raw', 'share9', 'ndvi',
                 'bare', 'water', 'built', 'snow_and_ice', 'crops', 'shrub', 'wc',
                 'occurrence', 'ct_prov', 'ct_code', 'ct_km', 'year'] + FEATS)
    g = g[out_cols + ['geometry']]
    g.to_parquet(os.path.join(a.outdir, 'bare_sparse_v3_2023.parquet'), index=False)

    for fam, nm in [('sparse', '稀疏植被'), ('bare', '裸地')]:
        sub = g[g.fam == fam].copy()
        if len(sub) == 0:
            continue
        # GPKG（全字段，含 AEF64）
        fp_gpkg = os.path.join(a.outdir, '%s_2023_WGS84.gpkg' % nm)
        if os.path.isfile(fp_gpkg):
            os.remove(fp_gpkg)
        sub.to_file(fp_gpkg, layer='samples', driver='GPKG', engine='pyogrio')
        # SHP（ASCII 字段名）
        shp = sub
        fp_shp = os.path.join(a.outdir, 'shp', '%s_2023_WGS84.shp' % nm)
        for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
            if os.path.isfile(fp_shp.replace('.shp', ext)):
                os.remove(fp_shp.replace('.shp', ext))
        shp.to_file(fp_shp, driver='ESRI Shapefile', engine='pyogrio')
        # 回读校验（字段值一致）
        rb = pyogrio.read_dataframe(fp_gpkg)
        assert len(rb) == len(sub), (len(rb), len(sub))
        dmax = float(np.nanmax(np.abs(rb.lon.to_numpy() - sub.lon.to_numpy())))
        print('  %s: gpkg %d 点 回读 |Δlon|max=%.2e  shp %s' % (nm, len(sub), dmax, fp_shp), flush=True)

    # 分省统计
    st = g.groupby(['fam', 'ct_prov']).size().unstack(0).fillna(0).astype(int)
    st['合计'] = st.sum(axis=1)
    st = st.sort_values('合计', ascending=False)
    st.to_csv(os.path.join(a.outdir, '分省统计.csv'), encoding='utf-8-sig')
    print(st.head(20).to_string(), flush=True)

    # README
    n_sp = int((g.fam == 'sparse').sum()); n_ba = int((g.fam == 'bare').sum())
    miss_txt = '' if miss == 0 else '\n- ⚠ AEF 全缺失点 %d 个（保留待核），其余 AEF64 完整。' % miss
    readme = f"""# 裸地 / 稀疏植被 东北·东部补样（池 v3 候选层，2023）

> 2026-10-25 ｜ 依据 `F:/lc_work/bare_sparse_v3/spec/预注册判据.md`（阈值先写死后跑）
> 触发：2026-10-14 诊断 `F:/lc_work/v31_exp/reports/bare_sparse_east_gap_2026-10-14.md`（东北稀疏植被缺口）

## 内容
- `稀疏植被_2023_WGS84.gpkg`：**{n_sp:,}** 点（cls24=13，raw∈150/152/153）
- `裸地_2023_WGS84.gpkg`：**{n_ba:,}** 点（cls24=22，raw∈200/201/202）
- `bare_sparse_v3_2023.parquet`：同上长表（含 AEF64 A00–A63、证据列、省界字段）
- `shp/`：同名 Shapefile（ASCII 字段名，浮点保留）
- `分省统计.csv`

## 来源与规则
- 源：**GLC_FCS10 2023 10 m**（E 盘本地瓦片）3×3 全纯 + 9×9 同类占比 ≥0.70；
  0.025° 格配额（稀疏 ≤2 / 裸地 ≤1）、最小间距（300/200 m）、与既有 2023 同类样本 250 m 去重。
- 分窗配额：北区（科尔沁/松嫩/呼伦贝尔/浑善达克/黑吉辽其余）+ 东区（华北/黄淮山东/华东华南/黄土高原东）。
- 独立证据（GEE 2023）：S2 夏 NDVI/NDWI、冬 NDVI；Dynamic World 概率；WorldCover；GSW；SRTM。
  keep 判据沿用 doc67 v1.1 两类口径 + 窗口加固（详见 spec §3）。
- 独立核验：栅格回读 300 点（raw+3×3 纯）、证据双路 200 点（独立构建，max|Δ| 须 0）、AEF 完备性。

## 使用
- 训练池 v3：`src='fcs10_2023_ne_east_topup'`，可直接按 src 并入/回退；**不要**改动 v2 池。
- 注：E 盘 FCS10 的 130–135°E 带已于 2026-10-25 补齐（Zenodo MD5 校验），本层已含该带裸地补样（稀疏植被该带仅约 9 km²，未采）。{miss_txt}

## 字段
`pid, lon, lat, cls24(13/22), cls_zh, tier, src, fam, raw(FCS10 原始码), share9(9×9 同类占比), ndvi, bare/water/built/snow_and_ice/crops/shrub(DW 概率), wc(WorldCover 码), occurrence(GSW), ct_prov/ct_code/ct_km(CTAmap 省界), year, A00–A63(AEF64)`
"""
    open(os.path.join(a.outdir, 'README_补样说明.md'), 'w', encoding='utf-8').write(readme)
    json.dump({'sparse': n_sp, 'bare': n_ba, 'aef_missing': miss},
              open(os.path.join(a.outdir, 'delivery_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('交付完成 →', a.outdir, flush=True)


if __name__ == '__main__':
    main()

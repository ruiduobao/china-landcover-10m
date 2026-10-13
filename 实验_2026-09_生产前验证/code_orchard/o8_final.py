# -*- coding: utf-8 -*-
"""o8_final.py — 乔灌园地：交付装配（年度面板 + 选点 + 判读真值 + SHP + 质检门）

* 输入：data/orchard/yearly_parts/y*/aef_p*.parquet（母库面板）
        data/orchard/yearly_new/y*/aef_p*.parquet（新增点逐年，若存在）
        data/orchard/orchard_samples_2023.csv（o6 选点）
        data/orchard/judge_merged.csv（判读真值）
* 规则源：
  作物亚型映射（src→作物/形态）：citrus_* → 柑橘/乔木园地；aomc_apple_* → 苹果/乔木园地；
    teamap_* → 茶/灌木园地；rubber_* → 橡胶/乔木园地；glc12/glc_fcs10/glc30d → 未明（待判）
  质检门：G1 AEF 命中率 ≥0.98；G2 (row_id,year) 唯一；G3 坐标域检查；G4 年覆盖 8/8；G5 类型字段非空
* 输出：交付目录 = F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/
        - 乔灌园地_母库面板_2017-2024.parquet（含 AEF64）
        - 乔灌园地_母库面板_2017-2024_精简.csv（无 AEF 列）
        - 乔灌园地_2023选点.csv、乔灌园地_判读真值.csv、乔灌园地_新增样本_面板.parquet（若有）
        - shp/（锚点集双坐标 WGS84+GCJ02）
        - QA_gate.json
* 用法：python o8_final.py
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地'
CROP = [('citrus', '柑橘', '乔木园地'), ('aomc', '苹果', '乔木园地'), ('teamap', '茶', '灌木园地'),
        ('rubber', '橡胶', '乔木园地'), ('glc12', '未明', '待判'), ('glc_fcs10', '未明', '待判'),
        ('glc_fcs30d', '未明', '待判'), ('v2_glc', '未明', '待判')]


def crop_of(src):
    s = str(src).lower()
    for k, c, f in CROP:
        if k in s:
            return c, f
    return '未明', '待判'


def main():
    os.makedirs(DEST, exist_ok=True)
    os.makedirs(os.path.join(DEST, 'shp'), exist_ok=True)
    qa = {}
    # ---- 母库面板 ----
    fs = sorted(glob.glob(os.path.join(ORD, 'yearly_parts', 'y*', 'aef_p*.parquet')))
    parts = []
    for f in fs:
        y = int(os.path.basename(os.path.dirname(f))[1:])
        d = pd.read_parquet(f)
        d['year'] = y
        parts.append(d)
    panel = pd.concat(parts, ignore_index=True)
    panel[['crop', 'form']] = panel['src'].map(lambda s: pd.Series(crop_of(s))).apply(pd.Series)
    panel = panel[['row_id', 'year', 'lon', 'lat', 'src', 'crop', 'form', 'aef_ok'] + VC.FEATS]
    panel.to_parquet(os.path.join(DEST, '乔灌园地_母库面板_2017-2024.parquet'), index=False)
    slim = panel.drop(columns=VC.FEATS)
    slim.to_csv(os.path.join(DEST, '乔灌园地_母库面板_2017-2024_精简.csv'), index=False, encoding='utf-8-sig')
    qa['G1_aef_hit'] = round(float(panel.aef_ok.mean()), 4)
    qa['G2_unique_rowid_year'] = bool(not panel.duplicated(subset=['row_id', 'year']).any())
    qa['G3_coord_ok'] = bool(panel.lon.between(73, 136).all() and panel.lat.between(18, 54).all())
    qa['G4_years'] = sorted(panel.year.unique().tolist())
    qa['G5_form_nonnull'] = bool(panel.form.notna().all())
    print('母库面板 %d 行（%d 点 × %d 年）；AEF 命中 %.4f' % (
        len(panel), panel.row_id.nunique(), panel.year.nunique(), qa['G1_aef_hit']))
    # ---- 新增点面板（若有）----
    nfs = sorted(glob.glob(os.path.join(ORD, 'yearly_new', 'y*', 'aef_p*.parquet')))
    if nfs:
        nparts = []
        for f in nfs:
            y = int(os.path.basename(os.path.dirname(f))[1:])
            d = pd.read_parquet(f)
            d['year'] = y
            nparts.append(d)
        newp = pd.concat(nparts, ignore_index=True)
        newp = newp[['point_id', 'year', 'lon', 'lat', 'prov', 'tier', 'aef_ok'] + VC.FEATS]
        newp.to_parquet(os.path.join(DEST, '乔灌园地_新增样本_面板.parquet'), index=False)
        newp.drop(columns=VC.FEATS).to_csv(os.path.join(DEST, '乔灌园地_新增样本_面板_精简.csv'),
                                           index=False, encoding='utf-8-sig')
        print('新增面板 %d 行' % len(newp))
    # ---- 选点 + 判读真值 ----
    p = os.path.join(ORD, 'orchard_samples_2023.csv')
    if os.path.exists(p):
        pd.read_csv(p, encoding='utf-8-sig').to_csv(os.path.join(DEST, '乔灌园地_2023选点.csv'),
                                                    index=False, encoding='utf-8-sig')
    p = os.path.join(ORD, 'judge_merged.csv')
    if os.path.exists(p):
        pd.read_csv(p, encoding='utf-8-sig').to_csv(os.path.join(DEST, '乔灌园地_判读真值.csv'),
                                                    index=False, encoding='utf-8-sig')
    # ---- SHP（锚点：母库点 + 选点）----
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        pts = slim[['row_id', 'lon', 'lat', 'src']].drop_duplicates('row_id').rename(columns={'row_id': 'pid'})
        g = gpd.GeoDataFrame(pts, geometry=[Point(x, y) for x, y in zip(pts.lon, pts.lat)], crs='EPSG:4326')
        g.to_file(os.path.join(DEST, 'shp', '乔灌园地_母库点_WGS84.shp'), encoding='utf-8')
        # GCJ02
        g2 = g.copy()
        g2['geometry'] = [Point(x + 0.0035, y + 0.0016) for x, y in zip(g2.lon, g2.lat)]
        g2.to_file(os.path.join(DEST, 'shp', '乔灌园地_母库点_GCJ02近似.shp'), encoding='utf-8')
        print('SHP 导出 %d 点' % len(g))
    except Exception as e:
        print('SHP 跳过: %s' % str(e)[:100])
    VC.jsave(qa, os.path.join(DEST, 'QA_gate.json'))
    print('QA:', json.dumps(qa, ensure_ascii=False)[:300])
    print('→', DEST)


if __name__ == '__main__':
    main()

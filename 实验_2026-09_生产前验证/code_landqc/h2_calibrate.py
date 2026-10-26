# -*- coding: utf-8 -*-
"""h2_calibrate.py — 混淆对定向修复：病对两端的特征分布对比（阈值标定用）

* 输入：新抓证据 data/confuse_fix/feats/{swamp,shrub,orchard}（S1+DW+S2）
        旧 2023 证据 data/land_qc/feats/{crop,veg,water,urban}（8 个地类）
        data/forest_qc/feats/*（4 类林地：s2 2019/2023 + env + xs）
        交付 GPKG 的 ct_prov（从各 2023 gpkg 读 id+ct_prov）
* 输出：out/joined_<类>.parquet（供应用阶段；含 point_id/类/经纬度/ct_prov/特征）
        out/校准分布.txt + out/校准分布.json（逐对逐特征分位数）
* 用法：python h2_calibrate.py
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import geopandas as gpd

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
BASE = os.path.join(WORK, 'data', 'confuse_fix')
OUT = os.path.join(BASE, 'out')
CLS24 = {1: '草本旱地', 3: '灌溉耕地', 11: '草地', 13: '稀疏植被', 16: '湖河滩地', 22: '裸地', 23: '水体', 21: '不透水面'}
NEW = {'木本沼泽': 'feats/swamp', '草本沼泽': 'feats/swamp', '灌木_常绿': 'feats/shrub', '灌木_落叶': 'feats/shrub',
       '乔灌园地': 'feats/orchard'}
GIDX = {1: 'crop', 3: 'crop', 11: 'veg', 13: 'veg', 22: 'veg', 16: 'water', 23: 'water', 21: 'urban'}
IDCOL = {'木本沼泽': 'pid', '草本沼泽': 'pid', '灌木_常绿': 'point_id', '灌木_落叶': 'point_id', '乔灌园地': 'uid'}
PAIRS = [('草本沼泽', '湖河滩地'), ('草本沼泽', '木本沼泽'), ('灌木_落叶', '草地'), ('灌木_落叶', '林地_落叶阔叶林'),
         ('灌木_常绿', '草地'), ('乔灌园地', '灌溉耕地'), ('乔灌园地', '草本旱地'), ('乔灌园地', '林地_常绿阔叶林'),
         ('裸地', '稀疏植被'), ('水体', '湖河滩地'), ('灌溉耕地', '草本旱地')]
FEAT = ['s1_vv_jja', 's1_vh_jja', 's1_vv_djf', 's1_vh_djf', 'dw_water', 'dw_trees', 'dw_grass', 'dw_crops',
        'dw_shrub', 'dw_bare', 'dw_trees_150', 's2_ndvi_xia', 's2_ndvi_dong', 's2_ndwi_xia',
        'water', 'trees', 'grass', 'crops', 'built', 'bare', 'shrub', 'max_extent', 'seasonality',
        'max_300', 'dem', 'slope', 'wc', 'dw_trees_x', 'dw_shrub_x', 'dw_crops_x', 'dw_grass_x']


def norm_cols(d):
    # 统一新旧列名（v7 用 water/trees/...；新抓用 dw_*）
    ren = {}
    for c in ('water', 'trees', 'grass', 'crops', 'shrub', 'bare'):
        if c in d.columns and 'dw_' + c not in d.columns:
            ren[c] = 'dw_' + c
    return d.rename(columns=ren)


def load_cls(cls):
    if cls in NEW:
        fs = sorted(glob.glob(os.path.join(BASE, NEW[cls], '*.parquet')))
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        d = d[d.cls == cls] if 'cls' in d.columns else d
        # 补 ct_prov / lon-lat
        g = gpd.read_file(os.path.join(ROOT, {'木本沼泽': '木本沼泽/2023/木本沼泽_2023_WGS84.gpkg',
                                              '草本沼泽': '草本沼泽/2023/草本沼泽_2023_WGS84.gpkg',
                                              '灌木_常绿': '常绿灌木/2023/常绿灌木_2023_WGS84.gpkg',
                                              '灌木_落叶': '落叶灌木/2023/落叶灌木_2023_WGS84.gpkg',
                                              '乔灌园地': '乔灌园地/2023/乔灌园地v7_2023_WGS84.gpkg'}[cls]),
                            columns=[IDCOL[cls], 'ct_prov'], engine='pyogrio')
        g['point_id'] = g[IDCOL[cls]].astype(str)
        d['point_id'] = d.point_id.astype(str)
        d = d.merge(g[['point_id', 'ct_prov']], on='point_id', how='left')
        d['lon'] = d.lon if 'lon' in d.columns else np.nan
        return norm_cols(d)
    if cls.startswith('林地_'):
        C4 = {'常绿阔叶林': 4, '落叶阔叶林': 5, '常绿针叶林': 6, '落叶针叶林': 7}
        d = pd.read_parquet(os.path.join(WORK, 'data', 'forest_qc', 'out', 'forest_samples_v1.parquet'),
                            columns=['point_id', 'lon', 'lat', 'prov', 'class4', 'tier', 'src',
                                     'dem', 'wc', 'modis_lc', 'dw_trees', 'dw_shrub', 'dw_crops', 'dw_grass'])
        d = d[d.class4 == C4[cls.replace('林地_', '')]].copy()
        d['point_id'] = d.point_id.astype(str)
        d['ct_prov'] = d['prov']
        return d
    cl = [k for k, v in CLS24.items() if v == cls][0]
    g = pd.read_parquet(os.path.join(WORK, 'data', 'land_qc', 'out', 'land_qc_%s.parquet' % GIDX[cl]))
    g = g[g.class24 == cl].rename(columns={'pid': 'point_id'})
    g['point_id'] = g.point_id.astype(str)
    return norm_cols(g)


def main():
    os.makedirs(OUT, exist_ok=True)
    cache, rep = {}, {}
    lines = []
    for cls in sorted(set([c for p in PAIRS for c in p])):
        try:
            d = load_cls(cls)
        except Exception as e:
            lines.append('%s 载入失败：%s' % (cls, str(e)[:80])); continue
        d.to_parquet(os.path.join(OUT, 'joined_%s.parquet' % cls), index=False)
        cache[cls] = d
        have = [c for c in FEAT if c in d.columns]
        lines.append('%-14s n=%-7d 特征 %d：%s' % (cls, len(d), len(have), ','.join(have)))
    for a, b in PAIRS:
        if a not in cache or b not in cache:
            continue
        da, db = cache[a], cache[b]
        common = [c for c in FEAT if c in da.columns and c in db.columns]
        rep['%s↔%s' % (a, b)] = {}
        lines.append('\n===== %s（n=%d） ↔ %s（n=%d）=====' % (a, len(da), b, len(db)))
        for c in common:
            qa = pd.to_numeric(da[c], errors='coerce')
            qb = pd.to_numeric(db[c], errors='coerce')
            qs = [5, 10, 25, 50, 75, 90, 95]
            va = [round(float(qa.quantile(q / 100.0)), 3) if qa.notna().sum() else None for q in qs]
            vb = [round(float(qb.quantile(q / 100.0)), 3) if qb.notna().sum() else None for q in qs]
            rep['%s↔%s' % (a, b)][c] = dict(A=va, B=vb, nA=int(qa.notna().sum()), nB=int(qb.notna().sum()))
            lines.append('  %-14s %s: %s ｜ %s: %s' % (c, a[:6], va, b[:6], vb))
    open(os.path.join(OUT, '校准分布.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
    json.dump(rep, open(os.path.join(OUT, '校准分布.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n'.join(lines[:40]))
    print('…（全文 → out/校准分布.txt）')


if __name__ == '__main__':
    main()

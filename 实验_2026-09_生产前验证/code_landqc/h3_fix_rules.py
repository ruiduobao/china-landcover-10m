# -*- coding: utf-8 -*-
"""h3_fix_rules.py — 混淆对定向修复：判据冻结 + 删除量评估 + 删除清单（本步不改任何交付件）

* 输入：out/joined_<类>.parquet（h2 产物，含新抓证据与旧 2023 证据）
* 判据（v1 草案，见规则表；评估后冻结进 out/criteria_confuse_v1.json 与 技术方案文档）
    仅对"交付态"点生效（land_qc 的 verdict != quarantine；沼泽/灌木/园地为全部）
* 输出：out/{criteria_confuse_v1.json, 删除量评估.txt, 删除清单_<类>.csv}
* 用法：python h3_fix_rules.py [--apply-freeze]
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
BASE = os.path.join(WORK, 'data', 'confuse_fix')
OUT = os.path.join(BASE, 'out')
def aridlon(d):
    return ((d.ct_prov != '内蒙古自治区') | (d.lon < 112)) & ((d.ct_prov != '甘肃省') | (d.lon < 104))

ARID = ['新疆维吾尔自治区', '甘肃省', '宁夏回族自治区', '青海省', '内蒙古自治区', '西藏自治区']
RULES = {
    '草本沼泽': [('BARE_LIKE_FLAT', '(s2_ndvi_xia < 200) & (dw_bare >= 0.30)'),
               ('WOODY_LIKE', '(dw_trees_150 >= 0.30) | ((dw_trees_150 >= 0.15) & (s1_vh_djf >= -19.0))')],
    '木本沼泽': [('HERB_LIKE', '(dw_trees_150 < 0.06) & ((s1_vh_djf.isna()) | (s1_vh_djf < -23.5))')],
    '湖河滩地': [('VEGETATED_LIKE', '(s2_ndvi_xia >= 250) & ((dw_grass + dw_trees + dw_shrub) >= 0.15)'),
               ('PERMANENT_WATER', '(max_extent == 1) & (seasonality >= 10) & (dw_water >= 0.60)')],
    '灌木_常绿': [('FOREST_LIKE', '(dw_trees >= 0.65)'),
               ('GRASS_LIKE', '(dw_grass >= 0.25) & (dw_trees < 0.20)')],
    '灌木_落叶': [('FOREST_LIKE', '(dw_trees >= 0.50)'),
               ('GRASS_LIKE', '(dw_grass >= 0.30) & (dw_trees < 0.15)')],
    '草地': [('SHRUB_FOREST_LIKE', '(dw_trees >= 0.40) & (dw_grass < 0.15)'),
            ('SHRUB_LIKE', '(dw_shrub >= 0.45) & (dw_trees >= 0.25)')],
    '乔灌园地': [('CROP_LIKE', '(dw_crops >= 0.40) & (dw_trees < 0.25)'),
              ('NON_TREE_BARE_LIKE', '(dw_trees < 0.15) & (s2_ndvi_xia < 400)')],
    '裸地': [('VEGETATED_LIKE', '(s2_ndvi_xia >= 250) & ((dw_grass + dw_shrub + dw_trees) >= 0.30)')],
    '稀疏植被': [('BARE_LIKE', '(s2_ndvi_xia < 50) & (dw_bare >= 0.55) & (dw_shrub < 0.12)')],
    '水体': [('NO_WATER_EVIDENCE', '(max_extent == 0) & (dw_water < 0.30) & (seasonality <= 1)')],
    '灌溉耕地': [('ARID_BARE_ANOMALY', '(s2_ndvi_xia < 250) & (ct_prov in @ARID) & (@ARIDLON)')],
    '草本旱地': [('ARID_OASIS_ANOMALY', '(s2_ndvi_xia >= 700) & (ct_prov in @ARID) & (@ARIDLON)')],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply-freeze', action='store_true')
    a = ap.parse_args()
    lines, frozen, total_del = [], {}, 0
    for cls, rules in RULES.items():
        f = os.path.join(OUT, 'joined_%s.parquet' % cls)
        if not os.path.exists(f):
            lines.append('✗ 缺 %s' % cls); continue
        d = pd.read_parquet(f)
        d['point_id'] = d.point_id.astype(str)
        if 'ct_prov' not in d.columns:   # land 组：ct_prov 从交付 GPKG 补
            import geopandas as gpd
            ROOT = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12'
            G = {'草本旱地': '草本旱地/2023/草本旱地_2023_WGS84.gpkg', '灌溉耕地': '灌溉耕地/2023/灌溉耕地_2023_WGS84.gpkg',
                 '草地': '草地/2023/草地_2023_WGS84.gpkg', '稀疏植被': '稀疏植被/2023/稀疏植被_2023_WGS84.gpkg',
                 '裸地': '裸地/2023/裸地_2023_WGS84.gpkg', '湖河滩地': '湖河滩地/2023/湖河滩地_2023_WGS84.gpkg',
                 '水体': '水体/2023/水体_2023_WGS84.gpkg'}
            g = gpd.read_file(os.path.join(ROOT, G[cls]), columns=['pid', 'ct_prov'], engine='pyogrio')
            g['point_id'] = g['pid'].astype(str)
            d = d.merge(g[['point_id', 'ct_prov']].drop_duplicates('point_id'), on='point_id', how='left')
        if 'verdict' in d.columns:
            d = d[d.verdict != 'quarantine']
        if 'has_ev' in d.columns:
            d = d[d.has_ev == 1]
        n = len(d)
        dele = pd.Series(False, index=d.index)
        reason = pd.Series('', index=d.index)
        for name, expr in rules:
            hit = d.eval(expr, local_dict={'ARID': ARID, 'ARIDLON': aridlon(d)}).fillna(False)
            reason = reason.mask(hit & ~dele, name)
            dele |= hit
        k = int(dele.sum())
        total_del += k
        lines.append('%-8s 交付 n=%-7d 删 %-6d (%.1f%%) ｜ %s' % (
            cls, n, k, 100 * k / max(1, n),
            '；'.join('%s %d' % (nm, int(d.eval(ex, local_dict={'ARID': ARID, 'ARIDLON': aridlon(d)}).fillna(False).sum())) for nm, ex in rules)))
        frozen[cls] = dict(n_delivered=n, n_del=k, rules=[dict(reason=nm, expr=ex) for nm, ex in rules])
        lst = d[dele][['point_id'] + [c for c in ('cls', 'ct_prov', 's2_ndvi_xia', 'dw_trees', 'dw_grass',
                                                  'dw_shrub', 'dw_bare', 'dw_crops', 'dw_trees_150',
                                                  's1_vh_djf', 'max_extent', 'seasonality') if c in d.columns]].copy()
        lst['reason'] = reason[dele].to_numpy()
        lst.to_csv(os.path.join(OUT, '删除清单_%s.csv' % cls), index=False, encoding='utf-8-sig')
    txt = '\n'.join(lines) + '\n合计删除（跨类，含点重复计入）%d' % total_del
    open(os.path.join(OUT, '删除量评估.txt'), 'w', encoding='utf-8').write(txt)
    print(txt)
    if a.apply_freeze:
        json.dump(dict(version='v1-2026-10-14', arid_provs=ARID, classes=frozen),
                  open(os.path.join(OUT, 'criteria_confuse_v1.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print('→ 判据已冻结 criteria_confuse_v1.json')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""o29_v5_assembly.py — 乔灌园地 v5 组装（橡胶三层 + 茶园/多源/锚点沿用 + 面积校准表）

* 输入：data/orchard/{rubber_gem.parquet, rubber_pheno_scored.parquet, tea_2022_cn_points.parquet (v4 沿用)}
        v4 交付目录（茶园/多源/锚点沿用）
* 规则源（**预注册分层**）：
  橡胶 TierA = 橡胶图 ∧ GEM图=2 ∧ 物候≥0.5（三源）  TierB = 橡胶图 ∧（GEM=2 ∨ 物候≥0.7）减 A  TierC = 其余
  面积校准：按省级"层占比 × 产品面积"与官方对照（点位为 200m 网格抽样，比例代表层内份额）
* 输出：交付目录 v5/（橡胶三层 parquet+csv、茶园沿用、多源沿用、锚点、v5_stats.json、面积校准表）
* 用法：python o29_v5_assembly.py
"""
import json
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import v31_common as VC

WORK = r'F:/lc_work/v31_exp'
ORD = os.path.join(WORK, 'data', 'orchard')
D4 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v4_专题样本'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/乔灌园地/v5_精化样本'


def main():
    os.makedirs(DEST, exist_ok=True)
    # ---- 橡胶三层 ----
    rub = pd.read_parquet(os.path.join(ORD, 'rubber_gem.parquet'))[['lon', 'lat', 'tile', 'gem']]
    ph = pd.read_parquet(os.path.join(ORD, 'rubber_pheno_scored.parquet'))
    sc = pd.read_parquet(os.path.join(ORD, 'scored_rubber.parquet'))
    aef = pd.concat([pd.read_parquet(f) for f in sorted(
        __import__('glob').glob(os.path.join(ORD, 'aef', 'rubber_p*.parquet')))], ignore_index=True)
    for d in (rub, ph, sc, aef):
        d['lon6'] = d.lon.round(6); d['lat6'] = d.lat.round(6)
    r = rub.merge(ph[['lon6', 'lat6', 'pheno_p', 'drop'] + [c for c in ph.columns if c.startswith('nd_') or c.startswith('s1_')]],
                  on=['lon6', 'lat6'], how='left')
    r = r.merge(sc[['lon6', 'lat6', 'p_treecrop']], on=['lon6', 'lat6'], how='left')
    r = r.merge(aef[['lon6', 'lat6'] + VC.FEATS], on=['lon6', 'lat6'], how='left')
    import k19_provattrib as PA
    r['prov'] = PA.attribute(r)
    r['subtype'] = '橡胶'
    # 三层
    A = (r.gem == 2) & (r.pheno_p >= 0.5)
    B = ~A & ((r.gem == 2) | (r.pheno_p >= 0.7))
    r['tier'] = np.where(A, 'A_三源', np.where(B, 'B_双源', 'C_单产品'))
    r['grade'] = r.tier
    r = r.drop(columns=['lon6', 'lat6'])
    r.to_parquet(os.path.join(DEST, '乔灌园地v5_橡胶样本.parquet'), index=False)
    r.drop(columns=VC.FEATS).to_csv(os.path.join(DEST, '乔灌园地v5_橡胶样本_精简.csv'),
                                    index=False, encoding='utf-8-sig')
    print('橡胶 v5 分层:', r.tier.value_counts().to_dict())
    # 面积校准表（按省 × 层：层占比 × 产品省面积）
    OFFICIAL = {'海南省': 5300, '云南省': 5600}   # 官方橡胶面积极略值（km²，取中值）
    PROD_AREA = {'海南省': 6238.2, '云南省': 4187.7}  # Rubber_SEA 中国区实测面积
    calib = []
    for prov in ('海南省', '云南省'):
        g = r[r.prov == prov]
        n_all = len(g)
        for t in ('A_三源', 'B_双源', 'C_单产品'):
            n_t = int((g.tier == t).sum())
            calib.append({'prov': prov, 'tier': t, 'n': n_t,
                          'share': round(n_t / max(1, n_all), 3),
                          'area_km2_est': round(PROD_AREA[prov] * n_t / max(1, n_all), 0)})
    cb = pd.DataFrame(calib)
    cb.to_csv(os.path.join(DEST, 'v5_橡胶_面积校准表.csv'), index=False, encoding='utf-8-sig')
    print(cb.to_string(index=False))
    # ---- 沿用 v4 的茶园/多源/锚点 ----
    for f in ['乔灌园地v4_茶园样本.parquet', '乔灌园地v4_茶园样本_精简.csv',
              '乔灌园地v4_多源果园候选.csv', '乔灌园地v4_目视确认锚点.csv']:
        src = os.path.join(D4, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DEST, f.replace('v4', 'v5')))
    # ---- 推荐训练集 v5 ----
    tea = pd.read_parquet(os.path.join(DEST, '乔灌园地v5_茶园样本.parquet'))
    multi = pd.read_csv(os.path.join(DEST, '乔灌园地v5_多源果园候选.csv'), encoding='utf-8-sig')
    anc = pd.read_csv(os.path.join(DEST, '乔灌园地v5_目视确认锚点.csv'), encoding='utf-8-sig')
    parts = []
    rr = r[r.tier != 'C_单产品'].copy()
    parts.append(rr[['lon', 'lat', 'prov', 'subtype', 'tier']].rename(columns={'tier': 'grade'}).assign(label='乔灌园地'))
    tt = tea.copy(); tt['label'] = '乔灌园地'; tt['grade'] = 'B_专题产品'
    parts.append(tt[['lon', 'lat', 'prov', 'subtype', 'grade', 'label']])
    mm = multi.copy(); mm['label'] = '乔灌园地'; mm['grade'] = mm['conf'].astype(str)
    if 'prov' not in mm.columns:
        mm['prov'] = ''
    parts.append(mm[['lon', 'lat', 'prov', 'subtype', 'grade', 'label']])
    aa = anc.copy(); aa['label'] = '乔灌园地'; aa['subtype'] = aa.Q2.fillna('待判'); aa['grade'] = 'A_目视'
    aa['prov'] = ''
    parts.append(aa[['lon', 'lat', 'prov', 'subtype', 'grade', 'label']])
    allp = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['lon', 'lat'])
    allp.to_csv(os.path.join(DEST, '乔灌园地v5_推荐训练集.csv'), index=False, encoding='utf-8-sig')
    stats = {'rubber_tiers': r.tier.value_counts().to_dict(),
             'tea': int(len(tea)), 'multi': int(len(multi)), 'anchors': int(len(anc)),
             'train_total': int(len(allp)), 'train_grade': allp.grade.value_counts().to_dict()}
    VC.jsave(stats, os.path.join(DEST, 'v5_stats.json'))
    print('推荐训练集 v5:', len(allp), stats['train_grade'])
    print('→', DEST)


if __name__ == '__main__':
    main()

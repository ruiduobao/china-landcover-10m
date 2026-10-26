# -*- coding: utf-8 -*-
"""h18_report_yearly.py — 逐年补样支撑 2017–2024 分类的验收报告

* 读取：data/yearly/yearly_stats.json、data/final/yearly_membership.csv、verify_yearly.json、
        clcd_check.json、accuracy_yearly.json、province_yearly.csv、reports/visual_audit/result.json
* 输出：reports/yearly_support_2017_2024.md
"""
import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'


def jload(p):
    try:
        return json.load(open(p, encoding='utf-8'))
    except Exception:
        return None


def main():
    ys = jload(os.path.join(BASE, 'data', 'yearly', 'yearly_stats.json')) or {}
    ver = jload(os.path.join(BASE, 'data', 'yearly', 'verify_yearly.json')) or {}
    clcd = jload(os.path.join(BASE, 'data', 'yearly', 'clcd_check.json')) or {}
    prov = pd.read_csv(os.path.join(BASE, 'data', 'yearly', 'province_yearly.csv'))
    vis = jload(os.path.join(BASE, 'reports', 'visual_audit', 'result.json'))
    acc = jload(os.path.join(BASE, 'data', 'yearly', 'accuracy_yearly.json'))
    L = []
    A = L.append
    A('# 稀疏植被 / 裸地 补样对全国 2017–2024 分类的支撑验收')
    A('')
    A('> 2026-10-25 ｜ 锚年 2023（FCS10-2023 派生）+ 逐年证据判定（S2 夏/宽窗/冬 NDVI、Dynamic World）')
    A('')
    A('## 1. 逐年可用样本量（东北/东部补样层）')
    A('')
    A('| 年 | 稀疏植被 | 裸地 | 合计 | 合并后年度池 |')
    A('|---|---:|---:|---:|---:|')
    tot_s = tot_b = 0
    for y in sorted(ys, key=int):
        v = ys[y]
        tot_s += v['sparse']; tot_b += v['bare']
        A('| %s | %d | %d | %d | %s |' % (y, v['sparse'], v['bare'], v['topup_keep'], v.get('merged_n')))
    A('| **合计（点年）** | **%d** | **%d** | **%d** | — |' % (tot_s, tot_b, tot_s + tot_b))
    A('')
    A('注：点年是同一批点的不同年份成员；2023 年 = 交付层 4,560 点（含三项护栏）。')
    A('')
    A('## 2. 跨年一致性')
    A('')
    cy = ver.get('cross_year', {})
    n0 = max(cy.get('n_points', 1), 1)
    A('- 全 8 年可用：**%s / %s**（%.1f%%）；≥6 年：%s（%.1f%%）；平均可用年数 **%.2f**' % (
        cy.get('keep_all_8'), cy.get('n_points'), 100 * cy.get('keep_all_8', 0) / n0,
        cy.get('keep_ge6'), 100 * cy.get('keep_ge6', 0) / n0, cy.get('keep_mean', 0)))
    A('- 可用年数分布：%s' % json.dumps(cy.get('flip_counts'), ensure_ascii=False))
    A('')
    A('## 3. 区域分布（点年）')
    A('')
    A('| 区域 | 稀疏 | 裸地 | 年均稀疏 | 年均裸地 |')
    A('|---|---:|---:|---:|---:|')
    for k, v in (ver.get('region_point_years') or {}).items():
        A('| %s | %d | %d | %.0f | %.0f |' % (k, v['sparse_point_years'], v['bare_point_years'],
                                              v['sparse_point_years'] / 8, v['bare_point_years'] / 8))
    A('')
    A('省级（逐年明细见 `data/yearly/province_yearly.csv`）：')
    A('')
    pv = prov.pivot_table(index='ct_prov', columns='class', values='n', aggfunc='sum').fillna(0).astype(int)
    A('| 省 | 稀疏(8年合计) | 裸地(8年合计) |')
    A('|---|---:|---:|')
    for p, r in pv.sort_values('sparse' if 'sparse' in pv.columns else pv.columns[0], ascending=False).head(12).iterrows():
        A('| %s | %d | %d |' % (p, r.get('sparse', 0), r.get('bare', 0)))
    A('')
    A('## 4. 精度证据')
    A('')
    A('### 4.1 独立证据双路复核')
    A('')
    A('| 年 | DW bare | DW water | S2 NDVI |')
    A('|---|---|---|---|')
    for y, v in (ver.get('double_arm') or {}).items():
        A('| %s | %s | %s | %s |' % (y, v.get('bare', {}).get('max_abs_diff'),
                                     v.get('water', {}).get('max_abs_diff'),
                                     v.get('s2_ndvi_xia', {}).get('max_abs_diff')))
    A('')
    A('### 4.2 与 CLCD 年度产品的一致性（独立旁证）')
    A('')
    A('| 年 | 稀疏→CLCD{草,灌,裸} | 其中灌丛 | 裸地→CLCD 裸地 |')
    A('|---|---:|---:|---:|')
    for y in sorted(clcd, key=int):
        v = clcd[y]
        A('| %s | %.1f%% | %.1f%% | %.1f%% |' % (y, v['sparse']['share_ok'],
                                                 v['sparse']['share_shrub4'], v['bare']['share_bare7']))
    A('')
    A('### 4.3 S2 真彩色目视抽查（最终护栏层）')
    A('')
    if vis and 'years' in vis:
        A('每类 12 点/年；1 km 视场 + 中心十字；三档判读（符合/模糊/不符）。接触表见 `reports/visual_audit/`。')
        A('')
        A('| 年 | 稀疏 符合/模糊/不符 | 裸地 符合/模糊/不符 |')
        A('|---|---|---|')
        for y in sorted(vis['years'], key=int):
            v = vis['years'][y]
            sp, ba = v['sparse'], v['bare']
            A('| %s | %d/%d/%d | %d/%d/%d |' % (y, sp['clear'], sp['ambiguous'], sp['unclear'],
                                                ba['clear'], ba['ambiguous'], ba['unclear']))
        ag = vis.get('aggregate') or {}
        if ag:
            A('')
            A('合计（n=48/类）：稀疏 **%.1f%% 明确 + %.1f%% 模糊**；裸地 **%.1f%% 明确 + %.1f%% 模糊**。' % (
                ag['sparse']['clear_pct'], ag['sparse']['clear_plus_amb_pct'] - ag['sparse']['clear_pct'],
                ag['bare']['clear_pct'], ag['bare']['clear_plus_amb_pct'] - ag['bare']['clear_pct']))
            A('')
            A('> 注：%s' % vis.get('caveat', ''))
    else:
        A('（未找到 result.json）')
    A('')
    A('### 4.4 逐年分层随机抽样精度量化（≥60 点/类/年；独立参考 = CLCD 年度）')
    A('')
    if acc:
        A('参考映射：稀疏←CLCD{3 草,4 灌}；裸地←CLCD{7}；其余记 other 并剔除。PA=类内一致率（按类分层抽样，UA 为样本条件值）。')
        A('')
        A('| 年 | 有效参考 | OA (95%CI) | 稀疏 PA | 稀疏 UA | 裸地 PA | 裸地 UA |')
        A('|---|---:|---|---:|---:|---:|---:|')
        for y in sorted(acc, key=int):
            r = acc[y]
            A('| %s | %d | %.1f%% (%s) | %.1f%% | %.1f%% | %.1f%% | %.1f%% |' % (
                y, r['n_defined'], r['OA'], r['OA_ci'], r['sparse_PA'], r['sparse_UA'],
                r['bare_PA'], r['bare_UA']))
    else:
        A('（未找到 accuracy_yearly.json）')
    A('')
    A('## 5. 交付物')
    A('')
    A('- 逐年补样层：`data/yearly/topup_<year>.parquet`（含当年 AEF64、证据列）')
    A('- 逐年训练池：`data/pools/train_<year>.parquet`（= `年度子集_含稀有类/r7_train_<year>` + 补样层，30 类码 150/201）')
    A('- 逐年有效期：`data/final/yearly_valid_window.parquet`（valid_from/valid_to，含 2023 的连续区间）')
    A('- 护栏：CLCD 当年不透水剔；稀疏 DW crops≥0.35 剔；裸地要求 DW bare≥0.25')
    A('- 核验：`data/yearly/verify_yearly.json`、`clcd_check.json`、`accuracy_yearly.json`、`reports/distribution.md`')
    A('')
    A('## 6. 局限')
    A('')
    A('1. 2017/2018 南方夏季 S2 档案稀疏，部分点用 4–11 月宽窗或冬 NDVI 兜底。')
    A('2. 逐年"可用"是证据一致性判定（S2+DW），不等同人工真值；目视抽查与 CLCD 抽样为旁证。')
    A('3. 1 km 视场目视对"大景观中的小区块"偏严；像素级以 3×3 全纯 100% + 9×9≥0.70 为准。')
    A('4. 年度池进生产前仍需按生产协议做逐年来源筛选/预注册。')
    fp = os.path.join(BASE, 'reports', 'yearly_support_2017_2024.md')
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('→', fp)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""s4_leaf_habit.py — 给针叶/阔叶样本添加常绿/落叶属性（不新建分类类）

* 目标（用户 2026-10-17 指令）：针叶林/阔叶林为主类；常绿/落叶作为**属性列**，
  供后期用物候等手段单独判定，**不进入 2 类主分类**。
* 输入（只读）：
    - labels/needle_broad_national_v2.parquet（110,653 点，全国 248 瓦）
    - labels/methods/work_base.parquet（S2 物候 v2019/v2023、amp/djf，93.8% 可匹配）
* 规则源（预注册，写死）：
    R0 基础属性 = FCS10 官方图例原义（GLC_FCS10-2023 码表）：
        51/52=常绿阔叶  61/62=落叶阔叶  71/72=常绿针叶  81/82=落叶针叶
        奇=郁闭(Closed)  偶=稀疏(Open)      —— 与 技术文档/77 §3.2 一致
    R1 等级判定（三档，不覆盖 R0）：
        E2E=一致（FCS10 与 S2 两年自一致结论相同）
        E2C=冲突（FCS10 与 S2 两年自一致结论相反）
        E2U=无物候证据（S2 两年未同时判出，或该点不在 work_base）
    R2 物候辅助列（保留原始量，不下结论）：v2019/v2023/amp2019/amp2023/djf2019/djf2023
    R3 **不下最终裁决**：本轮只落"FCS10 属性 + 冲突标记 + 物候原始量"，
        实际采用哪一方由用户在 doc78 的 A/B/C 三方案中裁决（预注册纪律）。
* 输出：
    - labels/needle_broad_national_v2_habit.parquet（v2 原列 + 属性列）
    - labels/habit_stats.json（属性分布 / 冲突率 / 分省 / 分码）
* 用法：python s4_leaf_habit.py
* 资源：单进程，< 2 GB
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_common as C

V2 = os.path.join(C.LAB, 'needle_broad_national_v2.parquet')
WB = r'F:/lc_work/nb_national_2026/labels/methods/work_base.parquet'
OUT = os.path.join(C.LAB, 'needle_broad_national_v2_habit.parquet')

# FCS10 官方图例：码 → (叶型, 常绿/落叶, 郁闭/稀疏)
CODE_ATTR = {
    51: ('needle', 'evergreen', 'closed'),  # 实际 5x = 阔叶，见下方修正
    52: ('needle', 'evergreen', 'open'),
    61: ('needle', 'deciduous', 'closed'),
    62: ('needle', 'deciduous', 'open'),
    71: ('needle', 'evergreen', 'closed'),
    72: ('needle', 'evergreen', 'open'),
    81: ('needle', 'deciduous', 'closed'),
    82: ('needle', 'deciduous', 'open'),
    91: ('mixed', 'mixed', 'mixed'),
    92: ('mixed', 'mixed', 'mixed'),
}
# 修正：5x=阔叶（Broadleaved）
CODE_ATTR[51] = ('broad', 'evergreen', 'closed')
CODE_ATTR[52] = ('broad', 'evergreen', 'open')
CODE_ATTR[61] = ('broad', 'deciduous', 'closed')
CODE_ATTR[62] = ('broad', 'deciduous', 'open')


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def main():
    t0 = time.time()
    v2 = pd.read_parquet(V2)
    emit('v2 %d 点 / %d 瓦' % (len(v2), v2.tile.nunique()))

    # ---- R0：从 FCS10 码直接派生 ----
    v2['habit_fcs10'] = v2.fcs10_raw.map(lambda c: CODE_ATTR.get(int(c), ('?', '?', '?'))[1])
    v2['canopy_fcs10'] = v2.fcs10_raw.map(lambda c: CODE_ATTR.get(int(c), ('?', '?', '?'))[2])
    v2['leaf_fcs10'] = v2.fcs10_raw.map(lambda c: CODE_ATTR.get(int(c), ('?', '?', '?'))[0])
    emit('habit_fcs10: %s' % v2.habit_fcs10.value_counts().to_dict())
    emit('canopy_fcs10: %s' % v2.canopy_fcs10.value_counts().to_dict())

    # ---- R2：拼 S2 物候原始量 ----
    wb = pq.read_table(WB, columns=['lon', 'lat', 'v2019', 'v2023',
                                    'amp2019', 'amp2023', 'djf2019', 'djf2023']).to_pandas()
    wb['k'] = ((wb.lon.round(6) * 1e6).astype('int64').astype(str) + '_' +
               (wb.lat.round(6) * 1e6).astype('int64').astype(str))
    # work_base 已是唯一键，但保险起见去重
    wb = wb.drop_duplicates('k')
    v2['k'] = ((v2.lon.round(6) * 1e6).astype('int64').astype(str) + '_' +
               (v2.lat.round(6) * 1e6).astype('int64').astype(str))
    m = v2.merge(wb.drop(columns=['lon', 'lat']), on='k', how='left')
    assert len(m) == len(v2), (len(m), len(v2))
    v2 = m.drop(columns=['k'])
    emit('物候匹配 %d / %d = %.1f%%' % (
        int(v2.v2019.notna().sum()), len(v2), 100 * v2.v2019.notna().mean()))

    # ---- R1：冲突等级（只在 FCS10 为 E/D 的纯林点上判） ----
    pure = v2.habit_fcs10.isin(['evergreen', 'deciduous'])
    s2_both = v2.v2019.isin(['E', 'D']) & v2.v2023.isin(['E', 'D'])
    s2_same = s2_both & (v2.v2019 == v2.v2023)
    map_ed = {'evergreen': 'E', 'deciduous': 'D'}
    f_as_ed = v2.habit_fcs10.map(map_ed)

    v2['habit_grade'] = 'E2X'      # 非纯林（混交）或无属性
    v2.loc[pure & ~s2_both, 'habit_grade'] = 'E2U'
    v2.loc[pure & s2_both & ~s2_same, 'habit_grade'] = 'E2U'   # 两年打架 → 证据不足
    v2.loc[pure & s2_same, 'habit_grade'] = np.where(
        v2.loc[pure & s2_same, 'v2019'] == f_as_ed[pure & s2_same], 'E2E', 'E2C')
    v2.loc[v2.leaf == 'mixed', 'habit_grade'] = 'E2M'

    emit('habit_grade: %s' % v2.habit_grade.value_counts().to_dict())

    # ---- 物候侧建议值（只作参考，不覆盖） ----
    v2['habit_pheno'] = np.where(
        s2_same, v2.v2019.map({'E': 'evergreen', 'D': 'deciduous'}), 'unknown')

    # ---- R1b：连续量补充档 E2P（只在 E2U 且 amp2023 有值时启用）----
    # 阈值 amp2023 < 350 由阈值扫描定（300–400 一致率平坦，350 峰值 0.8567，
    # 见 doc78 §2.3）；这是"用连续量替代离散结论"，不是新证据源。
    THR_AMP = 350.0
    need = (v2.habit_grade == 'E2U') & v2.amp2023.notna()
    ph_e = v2.amp2023 < THR_AMP                      # True=常绿
    f_e = (v2.habit_fcs10 == 'evergreen')            # True=常绿
    v2.loc[need, 'habit_pheno'] = np.where(ph_e[need], 'evergreen', 'deciduous')
    v2.loc[need, 'habit_grade'] = np.where(
        (ph_e & f_e)[need], 'E2E', 'E2C')
    v2['habit_pheno_src'] = 'unknown'
    v2.loc[s2_same, 'habit_pheno_src'] = 's2_two_year'
    v2.loc[need, 'habit_pheno_src'] = 's2_amp_thr350'
    emit('补 E2P 后 habit_grade: %s' % v2.habit_grade.value_counts().to_dict())

    v2.to_parquet(OUT, index=False)
    emit('写 %s（%d 行，%.1f MB）' % (OUT, len(v2), os.path.getsize(OUT) / 1e6))

    # ---- 统计 ----
    p = v2[pure]
    st = {
        'n_total': int(len(v2)),
        'n_pure': int(len(p)),
        'habit_fcs10': {k: int(v) for k, v in v2.habit_fcs10.value_counts().items()},
        'canopy_fcs10': {k: int(v) for k, v in v2.canopy_fcs10.value_counts().items()},
        'grade': {k: int(v) for k, v in v2.habit_grade.value_counts().items()},
        'agreement': {
            'n_evaluable': int((p.habit_grade.isin(['E2E', 'E2C'])).sum()),
            'agree': int((p.habit_grade == 'E2E').sum()),
            'rate': round(float((p.habit_grade == 'E2E').sum() /
                                max((p.habit_grade.isin(['E2E', 'E2C'])).sum(), 1)), 4),
        },
        'by_leaf': {},
        'by_code': {},
        'by_prov': {},
        'note': ('常绿/落叶为 FCS10 官方图例派生属性，不进 2 类主分类；'
                 'E2C 为与 S2 两年自一致物候结论冲突，需人工裁决'),
    }
    for lf, g in p.groupby('leaf'):
        ev = g[g.habit_grade.isin(['E2E', 'E2C'])]
        st['by_leaf'][lf] = {
            'n': int(len(g)),
            'fcs10_evergreen_share': round(float((g.habit_fcs10 == 'evergreen').mean()), 4),
            'grade': {k: int(v) for k, v in g.habit_grade.value_counts().items()},
            'agree_rate': round(float((ev.habit_grade == 'E2E').mean()), 4) if len(ev) else None,
        }
    for code, g in p.groupby('fcs10_raw'):
        ev = g[g.habit_grade.isin(['E2E', 'E2C'])]
        st['by_code'][str(int(code))] = {
            'n': int(len(g)),
            'agree_rate': round(float((ev.habit_grade == 'E2E').mean()), 4) if len(ev) else None,
            'n_eval': int(len(ev)),
        }
    for pv, g in p.groupby('prov'):
        if not pv:
            continue
        ev = g[g.habit_grade.isin(['E2E', 'E2C'])]
        if len(g) < 100:
            continue
        st['by_prov'][pv] = {
            'n': int(len(g)),
            'fcs10_E': round(float((g.habit_fcs10 == 'evergreen').mean()), 3),
            'pheno_E': round(float((g.habit_pheno == 'evergreen').mean()), 3),
            'agree_rate': round(float((ev.habit_grade == 'E2E').mean()), 4) if len(ev) else None,
        }
    json.dump(st, open(os.path.join(C.LAB, 'habit_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('统计写入 labels/habit_stats.json（%.1f min）' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()

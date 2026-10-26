# -*- coding: utf-8 -*-
"""q6_gates.py — 历史清洗门核验（r7_pool 抽稀前池）：投票门 agree≥3 与 0.025° 落格对分布的解释力

* 输入：数据/本地处理/样本重建/r7_pool.parquet（9,646,715 点，含五产品 level0 投票列 + agree_n）
* 口径：30类码 稀疏植被=150、裸地=201（v31_map identity）；区域窗口同 bare_sparse 诊断报告；
        "落格后"= 0.025° 网格唯一格数（500m 间距/配额步骤为后续近似，本表仅标注）；
        省=池内 province 列（母库构建时口径，非 CTAmap——仅解释用）
* 输出：
  - gates_region_by_class.csv    区域 × 类：候选 / agree≥3 / 过门率 / 0.025°格数 / 格后率
  - gates_vote_detail.csv        稀疏/裸 东北+华东华南窗口 五产品 level0 投票去向
  - gates_province_150_201.csv   省 × 类：候选 / 过门 / 格数
* 用法：python q6_gates.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
SP = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/samples'
POOL = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/样本重建/r7_pool.parquet'
WINDOWS = {
    '东北_119_135E_42_54N': (119, 135, 42, 54),
    '华北_113_120E_34_40p5N': (113, 120, 34, 40.5),
    '华东华南_114_125E_18_34N': (114, 125, 18, 34),
    '全国': (73, 136, 15, 55),
}


def main():
    cols = ['lon', 'lat', 'class_new', 'agree_n', 'province', 'wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0']
    d = pd.read_parquet(POOL, columns=cols)
    print('r7_pool n=', len(d))
    d = d[d.class_new.isin([150, 201])].copy()
    print('稀疏150/裸201 候选合计:', len(d), d.class_new.value_counts().to_dict())
    d['cell0025'] = list(zip((d.lon / 0.025).astype(int), (d.lat / 0.025).astype(int)))

    rows = []
    for wname, (x0, x1, y0, y1) in WINDOWS.items():
        m = (d.lon >= x0) & (d.lon < x1) & (d.lat >= y0) & (d.lat < y1)
        sub = d[m]
        for cl, g in sub.groupby('class_new'):
            n = len(g); g3 = int((g.agree_n >= 3).sum())
            cells = g.loc[g.agree_n >= 3, 'cell0025'].nunique()
            rows.append(dict(window=wname, class_new=int(cl), n_cand=n, n_agree3=g3,
                             pass_rate=round(g3 / n, 4) if n else 0,
                             n_cell0025_after_gate=int(cells),
                             cell_ratio=round(cells / g3, 4) if g3 else 0,
                             n_after_cell=cells))
    pd.DataFrame(rows).to_csv(os.path.join(SP, 'gates_region_by_class.csv'), index=False, encoding='utf-8-sig')
    print(pd.DataFrame(rows).to_string(index=False))

    # 投票去向（东北、华东华南窗）
    det = []
    PROD = ['wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0']
    for wname in ['东北_119_135E_42_54N', '华东华南_114_125E_18_34N']:
        x0, x1, y0, y1 = WINDOWS[wname]
        sub = d[(d.lon >= x0) & (d.lon < x1) & (d.lat >= y0) & (d.lat < y1)]
        for cl, g in sub.groupby('class_new'):
            for p in PROD:
                vc = g[p].value_counts().head(6)
                for v, n in vc.items():
                    det.append(dict(window=wname, class_new=int(cl), product=p, vote=v, n=int(n), total=len(g)))
    pd.DataFrame(det).to_csv(os.path.join(SP, 'gates_vote_detail.csv'), index=False, encoding='utf-8-sig')

    # 省表
    rows = []
    for (pv, cl), g in d.groupby(['province', 'class_new']):
        n = len(g); g3 = int((g.agree_n >= 3).sum())
        cells = g.loc[g.agree_n >= 3, 'cell0025'].nunique()
        rows.append(dict(province=pv, class_new=int(cl), n_cand=n, n_agree3=g3,
                         pass_rate=round(g3 / n, 4) if n else 0, n_cell0025=int(cells)))
    pd.DataFrame(rows).sort_values(['class_new', 'n_cand'], ascending=[True, False]).to_csv(
        os.path.join(SP, 'gates_province_150_201.csv'), index=False, encoding='utf-8-sig')
    print('done')


if __name__ == '__main__':
    main()

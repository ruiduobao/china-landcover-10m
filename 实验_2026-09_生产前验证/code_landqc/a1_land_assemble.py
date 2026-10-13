# -*- coding: utf-8 -*-
"""a1_land_assemble.py — 其余 14 类质检结果合并与交付（keep/flag → 升级样本；quarantine → 剔除清单）

* 输入：out/land_qc_{crop,veg,water,urban}.parquet（v8 产物）
* 输出：out/clean/land_other14_v1.parquet（14 类，tier=strong/weak，含理由）
        out/clean/land24_<cl>_<名>_v1.parquet（逐类交付件）
        out/land_qc_summary.csv + out/land_qc_summary.md（逐类 n/keep/quar/flag 与来源构成）
* 规则：keep→tier=strong；flag→tier=weak（保留）；quarantine→不进交付（另落 out/qc/剔除）
* 用法：python a1_land_assemble.py
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/v31_exp/data/land_qc'; OUT = os.path.join(BASE, 'out'); CLEAN = os.path.join(OUT, 'clean')
NAME = {1: '草本旱地', 3: '灌溉耕地', 11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 22: '裸地',
        16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩', 23: '水体',
        21: '不透水面', 24: '永久冰雪'}


def main():
    fs = sorted(glob.glob(os.path.join(OUT, 'land_qc_*.parquet')))
    fs = [f for f in fs if os.path.basename(f) not in ('land_qc_all.parquet',)]
    if not fs:
        raise SystemExit('无 v8 产物')
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d['point_id'] = d.point_id.astype(str)
    assert d.point_id.is_unique, 'point_id 重复 %d' % (len(d) - d.point_id.nunique())
    d['tier'] = np.where(d.verdict == 'keep', 'strong', np.where(d.verdict == 'flag', 'weak', 'out'))
    os.makedirs(CLEAN, exist_ok=True)
    keep = d[d.verdict != 'quarantine'].copy()
    keep.drop(columns=['verdict']).to_parquet(os.path.join(CLEAN, 'land_other14_v1.parquet'), index=False)
    rows = []
    for cl in sorted(NAME):
        g = d[d.class24 == cl]
        g[g.verdict != 'quarantine'].drop(columns=['verdict']).to_parquet(
            os.path.join(CLEAN, 'land24_%d_%s_v1.parquet' % (cl, NAME[cl])), index=False)
        src = g[g.verdict == 'quarantine'].src.value_counts().head(3).to_dict()
        rows.append(dict(class24=cl, name=NAME[cl], n=len(g),
                         keep=int((g.verdict == 'keep').sum()), flag=int((g.verdict == 'flag').sum()),
                         quar=int((g.verdict == 'quarantine').sum()),
                         keep_pct=round(100.0 * (g.verdict == 'keep').mean(), 2),
                         quar_pct=round(100.0 * (g.verdict == 'quarantine').mean(), 2),
                         quar_top_src=str(src)[:120]))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, 'land_qc_summary.csv'), index=False, encoding='utf-8-sig')
    with open(os.path.join(OUT, 'land_qc_summary.md'), 'w', encoding='utf-8') as f:
        f.write('# 其余 14 类样本质检汇总（%s）\n\n' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
        f.write('总计 %d 点：keep %d / flag %d / quarantine %d\n\n' % (
            len(d), (d.verdict == 'keep').sum(), (d.verdict == 'flag').sum(), (d.verdict == 'quarantine').sum()))
        f.write(s.to_markdown(index=False) + '\n')
    print(s.to_string(index=False))
    print('\n总计 %d：keep %d / flag %d / quar %d' % (
        len(d), (d.verdict == 'keep').sum(), (d.verdict == 'flag').sum(), (d.verdict == 'quarantine').sum()))
    print('交付 →', os.path.join(CLEAN, 'land_other14_v1.parquet'))


if __name__ == '__main__':
    main()

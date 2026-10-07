# -*- coding: utf-8 -*-
"""
v4_level_report.py — 三级粒度精度报告（30 类 / LEVEL1 10 组 / LEVEL0 9 组）
依据（文献核验 2026-09-14）：GLC_FCS30 自身 9 类 OA 82.5% → 16 类 71.4% → 24 细类 68.7%；
FCS30D 10 类 80.88% → 17 类 73.24%。多拆类要付 8–14pp OA 是**产品级常态**，
因此交付应同时给三级口径，避免用 30 类口径误判制图能力。
同时按验证点数给"统计可用性"标注（n<30 的类只报合并级）。
输入: 评估_vfinal/val_predictions.parquet
输出: 评估_vfinal/level_report.json + 控制台表
"""
import os, sys, json
import numpy as np
import pandas as pd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(PROJ, '代码', '4.全国清洗训练'))
from lc_conf import CLASSES, LEVEL1, LEVEL0
import e6_spatial_eval as E6

D = os.path.join(E6.BASE, '评估_vfinal')
P = os.path.join(D, 'val_predictions.parquet')
MIN_N = 30


def agg(ct, pred):
    labs = sorted(set(ct) | set(pred))
    ix = {c: i for i, c in enumerate(labs)}
    cm = np.zeros((len(labs), len(labs)), np.int64)
    for a, b in zip(ct, pred):
        cm[ix[a], ix[b]] += 1
    met, per = E6.metrics_from_cm(cm, labs)
    return met, per


def main():
    df = pd.read_parquet(P)
    print(f'验证预测 {len(df):,}（{df.eval_year.min()}–{df.eval_year.max()}）')
    rep = {'n': int(len(df)), 'levels': {}}
    for lvl, fn in (('30类', lambda c: c),
                    ('LEVEL1_10组', lambda c: LEVEL1[CLASSES[int(c)][2]]),
                    ('LEVEL0_9组', lambda c: LEVEL0[CLASSES[int(c)][3]])):
        ct = np.array([fn(c) for c in df.class_new])
        pr = np.array([fn(c) for c in df.pred])
        met, per = agg(ct, pr)
        rep['levels'][lvl] = {'OA': round(met['OA'], 4), 'macro_F1': round(met['macro_F1'], 4),
                              'balanced_accuracy': round(met['balanced_accuracy'], 4),
                              'n_group': int(len(per))}
        print(f"{lvl:<14} OA={met['OA']:.4f}  macro-F1={met['macro_F1']:.4f}  "
              f"bal-acc={met['balanced_accuracy']:.4f}  组数={len(per)}")
    # 30 类逐类 + 统计可用性
    ct = df.class_new.to_numpy(int); pr = df.pred.to_numpy(int)
    met, per = agg(ct, pr)
    per['name'] = [CLASSES[int(c)][1] for c in per['class']]
    per['stat_ok'] = per.n_ref >= MIN_N
    per.to_csv(os.path.join(D, 'per_class_overall.csv'), index=False, encoding='utf-8-sig')
    weak = per[~per.stat_ok]
    print(f'\n非零类 {int((per.n_pred > 0).sum())}/{len(per)}；'
          f'验证点 <{MIN_N} 的类（只宜报合并级）: '
          + ', '.join(f"{int(r['class'])}({r['name']},n={int(r.n_ref)})" for _, r in weak.iterrows()))
    rep['per_class'] = per.to_dict('records')
    rep['low_stat_classes'] = [int(c) for c in weak['class']]
    rep['n_nonzero_class'] = int((per.n_pred > 0).sum())
    # 按来源分层
    is_f = df.src.astype(str).str.lower().str.contains('fcs10').to_numpy()
    rep['by_source'] = {}
    print('\n来源分层（30 类口径）：')
    for tag, m in (('fcs10', is_f), ('non_fcs10', ~is_f)):
        if int(m.sum()) == 0:
            continue
        g = df[m]
        mm, _ = agg(g.class_new.to_numpy(int), g.pred.to_numpy(int))
        rep['by_source'][tag] = {'OA': round(mm['OA'], 4), 'macro_F1': round(mm['macro_F1'], 4),
                                 'n': mm['n']}
        print(f"  {tag:<10} OA={mm['OA']:.4f} macro-F1={mm['macro_F1']:.4f} n={mm['n']}")
    json.dump(rep, open(os.path.join(D, 'level_report.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print('\n输出:', os.path.join(D, 'level_report.json'))


if __name__ == '__main__':
    main()

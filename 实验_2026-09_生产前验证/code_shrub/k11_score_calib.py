# -*- coding: utf-8 -*-
"""k11_score_calib.py — 用 1,803 个目视判读点训练 AEF 判别器，量出「分数阈值 → 精度」校准曲线

* 目标：在**全国 FCS10 灌丛层（537,361 点，母库已带 AEF64，零 GEE 成本）**上批量筛出
        precision 高于 0.46（该层实测真灌丛率）的灌木样本。可行性的唯一前提是：
        在判读点上，"分类器分数 ≥θ" 的条件精度显著高于 0.46 且召回可用。
* 输入：judged_feats.parquet（1,852 判读点，AEF64 + 物候 + 物理层）
        母池 r7_train_2023.parquet 的 glc_fcs10_2023_shrub 点（AEF64 已在库）
* 规则源（预注册）：
        训练/校准 = 全部可判读判读点（1,803：243 灌丛 / 1,560 非灌丛）
        交叉验证 = GroupKFold(5)，按 src_exp 分组（防同批实验的空间自相关虚高）
        只用 AEF64（与母池同源同尺度，scale=10 已校验 |ΔA00|=0）
        校准曲线 = P(is_shrub | score ≥ θ)，逐 θ∈{0.3,...,0.9} 报 n/precision/Wilson CI/召回
* 门槛：校准曲线上 precision ≥0.60 且召回 ≥0.10 的最高档，作为放量阈值（记入 verdict）
* 输出：data/shrub_nat/score_calib.json + data/shrub_nat/score_calib.md
* 用法：python k11_score_calib.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

OUTD = os.path.join(WORK, 'data', 'shrub_nat')
POOL = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def main():
    j = pd.read_parquet(os.path.join(OUTD, 'judged_feats.parquet'))
    jd = j[j['is_shrub'].isin(['是', '否'])].copy()
    y = (jd['is_shrub'] == '是').to_numpy(int)
    feats = ['A%02d' % i for i in range(64)]
    X = jd[feats].astype(float).fillna(-999.0).to_numpy()
    groups = jd['src_exp'].astype('category').cat.codes.to_numpy()
    emit('判读点 %d（灌丛 %d，非灌丛 %d）；GroupKFold(5) 按实验分组' % (len(jd), y.sum(), (1 - y).sum()))
    p = np.zeros(len(jd))
    gkf = GroupKFold(n_splits=5)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_leaf_nodes=5000,
                                     max_features='sqrt', n_jobs=15, random_state=7,
                                     class_weight='balanced_subsample').fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
        emit('  折%d：训练 %d / 测试 %d（组 %s）' % (k, len(tr), len(te),
                                              sorted(set(groups[te].tolist()))))
    jd['p'] = p
    auc = roc_auc_score(y, p)
    emit('GroupKFold AUC = %.4f' % auc)

    lines = ['# AEF 判别器分数校准（全国 FCS10 层放量的可行性依据）%s' % time.strftime('%Y-%m-%d %H:%M'), '',
             '- 训练/校准集：%d 个目视判读点（灌丛 %d / 非灌丛 %d），GroupKFold(5) 按实验分组' % (
                 len(jd), y.sum(), (1 - y).sum()),
             '- 特征：AEF64 单源（与母库同源同尺度）；AUC = %.4f' % auc,
             '- FCS10 灌丛层基线 precision = 0.460（doc57 无偏实测）',
             '', '| 阈值θ | 判读点中通过 | 其中灌丛 | precision | 95%CI | 召回(判读集内) |', '|---|---|---|---|---|---|']
    res = dict(auc=round(float(auc), 4), n=int(len(jd)), n_pos=int(y.sum()), theta={})
    best = None
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = p >= th
        n, k = int(m.sum()), int(y[m].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        rec = k / max(1, int(y.sum()))
        res['theta'][str(th)] = dict(n=n, shrub=k, precision=(round(prec, 4) if prec is not None else None),
                                     wilson=[lo, hi], recall=round(rec, 4))
        lines.append('| %.1f | %d | %d | %s | [%.2f,%.2f] | %.3f |' % (
            th, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi, rec))
        emit('θ=%.1f：n=%d 灌=%d precision=%s 召回=%.3f' % (th, n, k, ('%.3f' % prec) if prec is not None else '—', rec))
        # 预注册门槛：precision≥0.60 且召回≥0.10 的最高档
        if prec is not None and prec >= 0.60 and rec >= 0.10 and n >= 15:
            best = th
    res['chosen_theta'] = best
    lines += ['', '## 预注册门槛内选定的放量阈值：**θ = %s**' % best, '',
              '该档在判读点上的 precision 见上表；相对 FCS10 层基线 0.460 的增益即为本法的收益。']
    jd[['point_id', 'lon', 'lat', 'is_shrub', 'shr_type' if 'shr_type' in jd.columns else 'shrub_type',
        'src_exp', 'p']].to_parquet(os.path.join(OUTD, 'judged_scored.parquet'), index=False)
    json.dump(res, open(os.path.join(OUTD, 'score_calib.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(OUTD, 'score_calib.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    emit('→ score_calib.json / score_calib.md（选定 θ=%s）' % best)


if __name__ == '__main__':
    main()

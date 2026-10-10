# -*- coding: utf-8 -*-
"""k5_fcs10_rule.py — 以 FCS10 灌丛层为抽样框，量测「AEF 判别器 + 物理屏幕」规则的精度

* 为什么这么设计：doc57 无偏分层判读显示各层精度差距巨大——
        FCS10 灌丛层 0.46（宁 0.25/川 0.50/黑 0.54/闽 0.57）；OSM 0.31；
        WC/LC only（L3）0.18；**DW 灌丛标签（L1/L4）仅 0.027/0.028**——DW 的 shrub_and_scrub
        大量落在草地/稀疏/耕地上，不能当正样本框。故正样本框＝FCS10 灌丛层，判别器提纯。
* 输入：data/shrub_nat/judged_feats.parquet（1,852 判读点，含 AEF64/S2/GEDI/物理层）
        + data/shrub/judged_all.csv（doc57 判读，带 stratum）+ 2023 母池 AEF（一致性校验）
* 规则源（预注册）：
        评估集 E = doc57 的 **FCS10 层随机抽样点**（n=87 判读，框内随机→其精度可代表该层）
                训练集 T = 其余 1,765 个判读点（含 doc57 其他层与前期实验）
        判别器 = RandomForest（AEF64 单源，balanced_subsample，n=300，n_jobs=15，随机种子 7）
        物理屏幕 S = rh98∈[1.5,8] 且 rh98_mx11<8 且 tc<30
* 门槛：θ∈{0.5,0.6,0.7,0.8,0.9}逐档报 n/命中/precision/Wilson CI/召回；另报 +S 的组合
* 输出：data/shrub_nat/rule_fcs10.json + data/shrub_nat/rule_fcs10.md
* 用法：python k5_fcs10_rule.py
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    d57 = pd.read_csv(os.path.join(WORK, 'data', 'shrub', 'judged_all.csv'), encoding='utf-8-sig')
    k57 = list(zip(np.round(d57.lon.to_numpy(float), 6), np.round(d57.lat.to_numpy(float), 6)))
    key57 = {k: i for i, k in enumerate(k57)}          # 坐标 → doc57 行号
    key = {k: i for i, k in enumerate(zip(np.round(j.lon.to_numpy(float), 6),
                                          np.round(j.lat.to_numpy(float), 6)))}
    k57 = list(zip(np.round(d57.lon.to_numpy(float), 6), np.round(d57.lat.to_numpy(float), 6)))
    st57 = d57['stratum'].to_numpy()
    kj = list(zip(np.round(j.lon.to_numpy(float), 6), np.round(j.lat.to_numpy(float), 6)))
    # is_shrub/shrub_type 已在 k6 从 judged_pool 恢复（doc57 + 前期实验统一口径），此处只补 doc57 分层
    j['stratum57'] = [st57[key57[k]] if k in key57 else '' for k in kj]
    jd = j[j['is_shrub'].isin(['是', '否'])].copy()
    jd['y'] = (jd['is_shrub'] == '是').astype(int)
    emit('带目视标签的点：%d（灌丛 %d）' % (len(jd), jd['y'].sum()))
    # ---- AEF 一致性校验：母池 A00..A63 vs 本次抓取 ----
    try:
        pool = pd.read_parquet(POOL, columns=['lon', 'lat'] + ['A%02d' % i for i in range(64)])
        pk = {}
        for lo, la, a0 in zip(pool['lon'].to_numpy(float), pool['lat'].to_numpy(float),
                              pool['A00'].to_numpy(float)):
            pk[(round(lo, 6), round(la, 6))] = a0
        same = 0; tot = 0; diffs = []
        for _, r in jd.iterrows():
            kk = (round(float(r.lon), 6), round(float(r.lat), 6))
            if kk in pk and pd.notna(r['A00']):
                tot += 1
                diffs.append(abs(float(pk[kk]) - float(r['A00'])))
        same = sum(1 for d in diffs if d < 1e-4)
        emit('AEF 一致性：%d/%d 点 |ΔA00|<1e-4，中位差 %.2e（<1e-3 视为同源）' % (
            same, tot, float(np.median(diffs)) if diffs else -1))
        aef_ok = (float(np.median(diffs)) if diffs else 1) < 1e-3
    except Exception as e:
        emit('AEF 一致性校验跳过：%s' % str(e)[:80])
        aef_ok = None
    # ---- 训练/评估划分：E = doc57 的 FCS10 层；T = 其余 ----
    ev = jd[(jd['stratum57'] == 'FCS10')].copy()
    tr = jd[jd['stratum57'] != 'FCS10'].copy()
    emit('评估集 E（doc57 FCS10 层抽样）n=%d 灌丛=%d；训练集 T n=%d 灌丛=%d' % (
        len(ev), ev['y'].sum(), len(tr), tr['y'].sum()))
    feats = ['A%02d' % i for i in range(64)]
    Xtr, ytr = tr[feats].astype(float).fillna(-999).to_numpy(), tr['y'].to_numpy()
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_leaf_nodes=5000,
                                 max_features='sqrt', n_jobs=15, random_state=7,
                                 class_weight='balanced_subsample').fit(Xtr, ytr)
    Xev = ev[feats].astype(float).fillna(-999).to_numpy()
    ev['p'] = clf.predict_proba(Xev)[:, 1]
    emit('判别器（AEF64）在 E 上 AUC = %.3f' % roc_auc_score(ev['y'], ev['p']))

    def phys(d):
        r98 = d['rh98'].to_numpy(float); mx = d['rh98_mx11'].to_numpy(float); tc = d['tc'].to_numpy(float)
        return ((~np.isfinite(r98)) | ((r98 >= 1.5) & (r98 <= 8.0))) & ((~np.isfinite(mx)) | (mx < 8.0)) \
            & ((~np.isfinite(tc)) | (tc < 30))

    P = phys(ev)
    res = dict(n_eval=int(len(ev)), n_pos_eval=int(ev['y'].sum()),
               auc=round(float(roc_auc_score(ev['y'], ev['p'])), 4), aef_consistent=aef_ok,
               theta={}, theta_phys={})
    lines = ['# FCS10 框 + AEF 判别器规则精度 %s' % time.strftime('%Y-%m-%d %H:%M'), '',
             '- 评估集 E＝doc57 的 FCS10 层随机抽样（n=%d，真灌丛 %d，precision=%.3f）' % (
                 len(ev), ev['y'].sum(), ev['y'].mean()),
             '- 训练集 T＝其余判读点（n=%d），判别器＝RF(AEF64)，AUC(E)=%.3f' % (len(tr), res['auc']),
             '- AEF 与母池同源校验：%s' % ('一致（中位差<1e-3）' if aef_ok else '不一致/未校验'),
             '', '| 阈值θ | E中通过 | 其中真灌丛 | precision | 95%CI | 召回(E内) |', '|---|---|---|---|---|---|']
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = (ev['p'] >= th).to_numpy()
        n, k = int(m.sum()), int(ev['y'][m].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        rec = k / max(1, int(ev['y'].sum()))
        res['theta'][str(th)] = dict(n=n, shrub=k, precision=(round(prec, 4) if prec is not None else None),
                                     wilson=[lo, hi], recall=round(rec, 4))
        lines.append('| %.1f | %d | %d | %s | [%.2f,%.2f] | %.3f |' % (
            th, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi, rec))
        emit('θ=%.1f：n=%d 灌=%d precision=%s 召回=%.3f' % (th, n, k, ('%.3f' % prec) if prec is not None else '—', rec))
    lines += ['', '## 叠加物理屏幕 S（rh98∈[1.5,8] 且 rh98_mx11<8 且 tc<30）', '',
              '| 阈值θ | 通过 | 真灌丛 | precision | 95%CI | 召回(E内) |', '|---|---|---|---|---|---|']
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = (ev['p'] >= th).to_numpy() & P
        n, k = int(m.sum()), int(ev['y'][m].sum())
        prec = k / n if n else None
        lo, hi = wilson(k, n)
        res['theta_phys'][str(th)] = dict(n=n, shrub=k, precision=(round(prec, 4) if prec is not None else None),
                                          wilson=[lo, hi], recall=round(k / max(1, int(ev['y'].sum())), 4))
        lines.append('| %.1f | %d | %d | %s | [%.2f,%.2f] | %.3f |' % (
            th, n, k, ('%.3f' % prec) if prec is not None else '—', lo, hi,
            k / max(1, int(ev['y'].sum()))))
        emit('θ=%.1f+S：n=%d 灌=%d precision=%s' % (th, n, k, ('%.3f' % prec) if prec is not None else '—'))
    ev.to_parquet(os.path.join(OUTD, 'eval_fcs10_scored.parquet'), index=False)
    json.dump(res, open(os.path.join(OUTD, 'rule_fcs10.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    open(os.path.join(OUTD, 'rule_fcs10.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    emit('→ rule_fcs10.json / rule_fcs10.md')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""y2b_d2_policy.py — D2 灌丛策略定向复测（本地，零 GEE 成本）

背景（y2a 侦察）：2023/2024 训练池中 24–59% 的点来自 `glc_fcs10_2023_shrub`，
且五个窗的灌丛(09/10)训练点**全部**来自该源；2017/2020 池为 0 灌丛点。
共同评估集（4 年同点、底座来源）中**没有灌丛标签** → E9 的 4–9pp 年度差不是灌丛类被评估，
而是"灌丛训练点把模型往 FCS10 惯例上拉、在底座惯例的评估点上变成错误"。

预注册判据（先写死，再跑）：
  J1 年度一致性：筛除后 2023/2024 共同集 OA 相对全池回升 ≥3pp（三年窗口内，3 种子同向）
  J2 灌丛存活：若某策略下"预测为灌丛的评估点比例"<0.2%（近似哑类），则该策略记为"灌丛哑"
  J3 口径代价：FCS10 评估点（2023/2024）OA 在全池下应显著高于筛除（证明这是标签惯例之争）
  J4 折中有效性：R2/R3（保留 25%/50%）若同时满足 ①年度差收敛到 ≤2pp ②灌丛预测率 >0.2% → 记为"折中可用"

策略（只改训练池，评估点固定）：
  R0 全池（现状）｜R1 全筛（排除 src 以 glc_fcs10 开头）｜R2 保留 25%｜R3 保留 50%
产物：results/d2/*.json + reports/D2灌丛策略复测_YYYYMMDD.md
"""
import os, sys, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

YEARS = [2017, 2020, 2023, 2024]
WS = ['w1', 'w2', 'w3', 'w4', 'w5']
SEEDS = [7, 11, 23]
POLICIES = ['R0', 'R1', 'R2', 'R3', 'R4']
FRACS = {'R2': 25, 'R3': 50}
OUTD = os.path.join(VC.DATA, 'yearly')
RESD = os.path.join(VC.RES, 'd2')
FOCUS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 21, 22, 23]


def apply_policy(tr, pol):
    src = tr['src'].fillna('').astype(str)
    if pol == 'R4':                     # 只筛"灌丛层"，保留 FCS10 混类抽样（含 02 园地等稀有类）
        m = src == 'glc_fcs10_2023_shrub'
        return tr[~m], 0
    m = src.str.startswith('glc_fcs10')
    if not m.any():
        return tr, 0
    if pol == 'R0':
        return tr, int(m.sum())
    if pol == 'R1':
        return tr[~m], 0
    frac = FRACS[pol]
    rid = tr['row_id'].to_numpy()
    h = (rid * 2654435761) % 100
    keep = (~m) | (h < frac)
    return tr[keep], int(keep.sum() - (~m).sum())


def rf(seed):
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
                                  max_features='sqrt', class_weight=None, n_jobs=15, random_state=seed)


def main():
    os.makedirs(RESD, exist_ok=True)
    plan = VC.jload(os.path.join(OUTD, 'plan.json'))
    common = {w: set(plan['common_eval'][w]['rids']) for w in WS}
    names = VC.V31_NAMES()
    rows = []
    t00 = time.time()
    for w in WS:
        com = list(common[w])
        for year in YEARS:
            tr_all = pd.read_parquet(os.path.join(OUTD, 'train_%d_%s.parquet' % (year, w)))
            ev = pd.read_parquet(os.path.join(OUTD, 'eval_%d_%s.parquet' % (year, w)))
            ev_rid = ev.row_id.to_numpy()
            sel_com = np.isin(ev_rid, com)
            # FCS10 口径评估点（2023/2024 才有；按构造=灌丛）
            is_f10 = ev['src'].fillna('').astype(str).str.startswith('glc_fcs10').to_numpy()
            Xev = ev[VC.FEATS].to_numpy('float32')
            y31_ev = VC.to_v31(ev['class_new'].to_numpy(int))
            for pol in POLICIES:
                tr, n_kept_f10 = apply_policy(tr_all, pol)
                if len(tr) < 1000:
                    VC.emit('%s %d %s 训练点不足，跳过' % (w, year, pol)); continue
                Xtr = tr[VC.FEATS].to_numpy('float32')
                ytr = VC.to_v31(tr['class_new'].to_numpy(int))
                oas, mf1s, shps, oas_f10 = [], [], [], []
                conf = None
                for seed in SEEDS:
                    clf = rf(seed).fit(Xtr, ytr)
                    p_all = clf.predict(Xev)
                    p_com = p_all[sel_com]
                    t_com = y31_ev[sel_com]
                    m = VC.metrics(t_com, p_com)
                    oas.append(m['OA']); mf1s.append(m['macroF1'])
                    shps.append(float((p_com == 10).mean() + (p_com == 9).mean()))
                    if is_f10.sum() >= 20:
                        mf = VC.metrics(y31_ev[is_f10], p_all[is_f10])
                        oas_f10.append(mf['OA'])
                    if seed == SEEDS[0]:
                        # 混淆矩阵（底座共同集）
                        K = 25
                        conf = np.zeros((K, K), dtype=int)
                        for t_, p_ in zip(t_com, p_com):
                            if 0 < t_ < K and 0 < p_ < K:
                                conf[t_, p_] += 1
                        per = m['per']
                rec = dict(w=w, year=year, policy=pol, n_train=len(tr), n_f10_kept=n_kept_f10,
                           n_common=int(sel_com.sum()),
                           n_fcs_eval=int(is_f10.sum()),
                           OA=float(np.mean(oas)), OA_sd=float(np.std(oas)),
                           macroF1=float(np.mean(mf1s)),
                           shrub_pred_rate=float(np.mean(shps)),
                           OA_fcs=float(np.mean(oas_f10)) if oas_f10 else None,
                           per=per,
                           conf=[int(x) for x in conf.flatten()])
                VC.jsave(rec, os.path.join(RESD, 'p_%s_%d_%s.json' % (w, year, pol)))
                rows.append(rec)
                VC.emit('%s %d %s: 训练%d(灌丛源保留%d) 共同集n=%d OA=%.4f±%.4f mF1=%.4f 灌丛预测率=%.3f%% FCS10口径OA=%s' % (
                    w, year, pol, len(tr), n_kept_f10, sel_com.sum(), rec['OA'], rec['OA_sd'],
                    rec['macroF1'], rec['shrub_pred_rate'] * 100,
                    ('%.4f' % rec['OA_fcs']) if rec['OA_fcs'] is not None else '—'))
    # ---------- 汇总报告 ----------
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('per', 'conf')} for r in rows])
    L = ['# D2 灌丛策略定向复测 %s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '> 训练：各年窗内 20k 点（年度子集）＋策略过滤；评估：**固定共同集**（4 年同点、底座来源、无灌丛标签）',
         '> 指标为 3 种子（7/11/23）均值；灌丛预测率 = 共同集中被预测为 09/10 的比例', '',
         '## FCS10 派生点全国来源构成（2023 年度子集扫描，doc42 附）', '',
         '| src | 点数 | 主要类别（原码:点数） |', '|---|---|---|',
         '| `glc_fcs10_2023_shrub` | 537,361 | 121:428,825、120:108,536（纯灌丛层） |',
         '| `glc_fcs10_2023_band130` | 47,268 | 121:9,679、62:5,179、92:4,226、82:4,074、72:3,403、61:3,276、10:2,519、181:2,432 |',
         '| `glc_fcs10_2023` | 29,533 | 130:7,298、201:6,383、62:5,164、52:3,884、11:1,862、10:1,182 |',
         '| **合计** | **614,162** | 87% 是灌丛层；13% 是混类抽样（含 02 园地 1,862 点） |', '']
    for pol in POLICIES:
        L += ['## %s' % pol, '', '| 窗 | 2017 | 2020 | 2023 | 2024 | 极差(pp) | 灌丛预测率 2023/2024 |', '|---|---|---|---|---|---|---|']
        for w in WS:
            d = df[(df.w == w) & (df.policy == pol)]
            if not len(d):
                continue
            oa = {int(r.year): r.OA for r in d.itertuples()}
            sh = {int(r.year): r.shrub_pred_rate for r in d.itertuples()}
            rng = max(oa.values()) - min(oa.values())
            L.append('| %s | %s | %s | %s | %s | %.2f | %.3f%% / %.3f%% |' % (
                w, *['%.4f' % oa[y] for y in YEARS], rng * 100,
                sh.get(2023, 0) * 100, sh.get(2024, 0) * 100))
        L.append('')
    # 配对差值（同窗同年同种子口径的均值差）
    L += ['## 配对差值（R1/R2/R3/R4 − R0，共同集 OA，正值=筛除/减量更好）', '',
          '| 窗 | 年 | R1−R0 | R2−R0 | R3−R0 | R4−R0 |', '|---|---|---|---|---|---|']
    for w in WS:
        for y in YEARS:
            cells = []
            for pol in ['R1', 'R2', 'R3', 'R4']:
                a = df[(df.w == w) & (df.year == y) & (df.policy == pol)]
                b = df[(df.w == w) & (df.year == y) & (df.policy == 'R0')]
                cells.append(('%.4f' % (a.OA.iloc[0] - b.OA.iloc[0])) if len(a) and len(b) else '—')
            L.append('| %s | %d | %s |' % (w, y, ' | '.join(cells)))
    # FCS10 口径
    L += ['', '## FCS10 派生评估点 OA（2023/2024，证明"惯例之争"）', '',
          '| 窗 | 年 | R0 | R1 | R2 | R3 | n |', '|---|---|---|---|---|---|---|']
    for w in WS:
        for y in (2023, 2024):
            d = df[(df.w == w) & (df.year == y)]
            if not len(d):
                continue
            n = int(d.n_fcs_eval.iloc[0])
            cells = []
            for pol in POLICIES:
                v = d[d.policy == pol]
                cells.append(('%.4f' % v.OA_fcs.iloc[0]) if len(v) and v.OA_fcs.iloc[0] is not None else '—')
            L.append('| %s | %d | %s | %d |' % (w, y, ' | '.join(cells), n))
    # 焦点类 F1（2023，共同集）
    L += ['', '## 焦点类 F1（2023 共同集，R0 vs R1）', '',
          '| 窗 | 类 | R0 | R1 | n(R0) |', '|---|---|---|---|---|']
    for w in WS:
        a = df[(df.w == w) & (df.year == 2023) & (df.policy == 'R0')]
        b = df[(df.w == w) & (df.year == 2023) & (df.policy == 'R1')]
        if not len(a) or not len(b):
            continue
        pa = [r for r in rows if r['w'] == w and r['year'] == 2023 and r['policy'] == 'R0'][0]['per']
        pb = [r for r in rows if r['w'] == w and r['year'] == 2023 and r['policy'] == 'R1'][0]['per']
        for c in FOCUS:
            ca, cb = pa.get(str(c)), pb.get(str(c))
            if ca or cb:
                L.append('| %s | %02d %s | %s | %s | %s |' % (
                    w, c, names.get(str(c), '?'),
                    '%.3f' % ca['f1'] if ca else '—', '%.3f' % cb['f1'] if cb else '—',
                    ca['n'] if ca else 0))
    fp = os.path.join(VC.REPT, 'D2灌丛策略复测_%s.md' % time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('汇总 → %s（用时 %.1f 分钟）' % (fp, (time.time() - t00) / 60))


if __name__ == '__main__':
    main()

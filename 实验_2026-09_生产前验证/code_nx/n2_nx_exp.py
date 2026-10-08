# -*- coding: utf-8 -*-
"""n2_nx_exp.py — 宁夏"全筛"小样本实验：训练池策略对比（本地按生产配方复刻，零 GEE 成本）

* 输入：F:/lc_work/prod5p_2023/data/train2023_clean.parquet（2023 池 2,060,577 点，含 src）
        F:/lc_work/v31_exp/data/m3/nx_neutral_points.csv（205 点中性集：A=M3-A 独立点 / S1=成品图灌丛内部 / S2=灌丛—草裸界面）
        F:/lc_work/v31_exp/data/m3/nx_neutral_embed.csv（A00–A63，2023 年 AEF，n0_nx_embed.py 取样）
        F:/lc_work/v31_exp/data/m3/nx_判读.csv（判读真值）
* 规则源：生产配方（p1_pilot）：每瓦 ±2° 框内随机抽 20,000 点 → RF(100树, minLeaf2, maxNodes5000, sqrt)
        策略：y2b_d2_policy.apply_policy（R0 全池 / R1 筛 glc_fcs10* / R2 留 25% / R3 留 50% / R4 只筛灌丛层）
* 门槛：主对比 R0 vs R4 = 6 瓦 × 3 种子（7/11/23）；策略曲线 R1/R2/R3 = T1608/T1609 × seed 7
* 输出：F:/lc_work/v31_exp/results/d2/nx_exp.json + reports/宁夏全筛小样本实验_YYYYMMDD.md
* 用法：python n2_nx_exp.py [--quick]
"""
import csv
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import v31_common as VC
import y2b_d2_policy as YB

M3D = r'F:/lc_work/v31_exp/data/m3'
POOL = r'F:/lc_work/prod5p_2023/data/train2023_clean.parquet'
RESD = os.path.join(VC.RES, 'd2')
REPT = VC.REPT
NX = {'T1509': (103, 36, 105, 38), 'T1608': (105, 34, 107, 36), 'T1609': (105, 36, 107, 38),
      'T1610': (105, 38, 107, 40), 'T1709': (107, 36, 109, 38), 'T1710': (107, 38, 109, 40)}
BIG = ['T1608', 'T1609']
N_TRAIN = 20000
SEEDS = [7, 11, 23]
SHRUBG = (9, 10, 11)
NAMES = VC.V31_NAMES()
RULES = [('水田', 3), ('灌溉', 3), ('乔灌园地', 2), ('果园', 2), ('常绿阔叶林', 4), ('落叶阔叶林', 5),
         ('常绿针叶林', 6), ('针叶', 6), ('灌丛', 10), ('草地', 11), ('稀疏植被', 13), ('草本沼泽', 15),
         ('湖河滩地', 16), ('盐渍湿地', 17), ('海岸盐沼', 19), ('冰雪', 24), ('水体', 23), ('裸地', 22),
         ('沙地', 22), ('耕地', 1), ('梯田', 1), ('大棚', 1), ('园地', 2), ('不透水面', 21), ('建成', 21)]


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def code_of(text):
    for k, c in RULES:
        if k in text:
            return c
    return None


def load_truth():
    pts = {r['point_id']: r for r in csv.DictReader(open(os.path.join(M3D, 'nx_neutral_points.csv'), encoding='utf-8-sig'))}
    lab = {r['point_id']: r for r in csv.DictReader(open(os.path.join(M3D, 'nx_判读.csv'), encoding='utf-8-sig'))}
    emb = {r['point_id']: r for r in csv.DictReader(open(os.path.join(M3D, 'nx_neutral_embed.csv'), encoding='utf-8-sig'))}
    rows, skip = [], 0
    for pid, p in pts.items():
        r = lab.get(pid)
        if not r or not r['Q1']:
            skip += 1
            continue
        c = code_of((r['Q2'] or '') + r['note'])
        if c is None or pid not in emb:
            skip += 1
            continue
        tile = next((t for t, (x0, y0, x1, y1) in NX.items()
                     if x0 <= float(p['lon']) <= x1 and y0 <= float(p['lat']) <= y1), None)
        rec = dict(point_id=pid, grp=p['grp'], tile=tile, lon=float(p['lon']), lat=float(p['lat']),
                   truth=c, q2=r['Q2'], shrub=1 if r['Q1'] == '是' else 0)
        rec.update({f: float(emb[pid][f]) for f in VC.FEATS})
        rows.append(rec)
    emit('判读真值 %d 点（未判/无嵌入跳过 %d）' % (len(rows), skip))
    return pd.DataFrame(rows)


def sample_box(df, box, seed):
    x0, y0, x1, y1 = box
    m = df[(df.lon >= x0 - 2) & (df.lon <= x1 + 2) & (df.lat >= y0 - 2) & (df.lat <= y1 + 2)]
    if len(m) > N_TRAIN:
        m = m.sample(n=N_TRAIN, random_state=seed)
    return m


def oa(t, p):
    return float((t == p).mean()) if len(t) else float('nan')


def macro(a):
    return VC.to_macro(np.asarray(a, dtype=int))


def stats(truth, pred, grp_mask):
    t, p = truth[grp_mask], pred[grp_mask]
    tm, pm = macro(t), macro(p)
    return dict(n=int(len(t)), OA24=round(oa(t, p), 4), OA9=round(oa(tm, pm), 4),
                shrub_true=int((t == 10).sum()), shrub_pred=int((p == 10).sum()),
                shrub_hit=int(((t == 10) & (p == 10)).sum()))


def main():
    t0 = time.time()
    os.makedirs(RESD, exist_ok=True)
    quick = '--quick' in sys.argv
    T = load_truth()
    emit('分组：A %d / S1 %d / S2 %d' % ((T.grp == 'A').sum(), (T.grp == 'S1').sum(), (T.grp == 'S2').sum()))
    emit('载入 2023 池 …')
    pool = pd.read_parquet(POOL, columns=['row_id', 'lon', 'lat', 'class_new', 'src'] + VC.FEATS)
    emit('池 %d 点；灌丛层 %d 点（%.1f%%）' % (len(pool), int((pool.src == 'glc_fcs10_2023_shrub').sum()),
                                          100 * (pool.src == 'glc_fcs10_2023_shrub').mean()))

    plan = [(t, pol, sd) for t in NX for pol in ('R0', 'R4') for sd in SEEDS]
    for t in (BIG if not quick else BIG[:1]):
        for pol in ('R1', 'R2', 'R3'):
            plan.append((t, pol, 7))
    emit('训练计划 %d 个模型' % len(plan))

    preds = {}          # (pol, seed) -> np.array 对齐 T 行序（NaN=该点所在瓦未训）
    removed = {}
    for t, pol, sd in plan:
        tr_pol, _ = YB.apply_policy(pool, pol, 'w4')
        box = NX[t]
        x0, y0, x1, y1 = box
        inb = tr_pol[(tr_pol.lon >= x0 - 2) & (tr_pol.lon <= x1 + 2) &
                     (tr_pol.lat >= y0 - 2) & (tr_pol.lat <= y1 + 2)]
        removed[(t, pol)] = int(len(inb))
        tr = sample_box(tr_pol, box, sd)
        X = tr[VC.FEATS].to_numpy('float32')
        y = VC.to_v31(tr['class_new'].to_numpy(int))
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
                                     max_features='sqrt', n_jobs=15, random_state=sd).fit(X, y)
        m = (T.tile == t).to_numpy()
        key = (pol, sd)
        if key not in preds:
            preds[key] = np.full(len(T), -1, dtype=int)
        if m.sum():
            preds[key][m] = clf.predict(T.loc[m, VC.FEATS].to_numpy('float32'))
        emit('  %s %s s%d：框内池 %d → 训练 %d → 预测 %d 点  [%.1f min]' % (
            t, pol, sd, len(inb), len(tr), int(m.sum()), (time.time() - t0) / 60))

    out = {'n_points': int(len(T)), 'plan': [list(p) for p in plan],
           'pool_rows': int(len(pool)), 'box_pool_after_policy': {'%s|%s' % k: v for k, v in removed.items()},
           'results': {}}
    masks = {'ALL': np.ones(len(T), bool), 'A': (T.grp == 'A').to_numpy(),
             'S1': (T.grp == 'S1').to_numpy(), 'S2': (T.grp == 'S2').to_numpy()}
    truth = T.truth.to_numpy(int)
    for (pol, sd), pr in sorted(preds.items()):
        ok = pr > 0
        if ok.sum() < len(T):
            emit('  注意：%s s%d 仅覆盖 %d/%d 点（策略曲线只在两瓦）' % (pol, sd, ok.sum(), len(T)))
        rec = {}
        for gname, gm in masks.items():
            gm2 = gm & ok
            rec[gname] = stats(truth, pr, gm2)
        out['results']['%s|%d' % (pol, sd)] = rec
    VC.jsave(out, os.path.join(RESD, 'nx_exp.json'))

    # ---- 汇总：主对比 R0 vs R4（3 种子均值）----
    L = ['# 宁夏"全筛"小样本实验（本地生产配方复刻） %s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '- 中性判读集：A=M3-A 独立点、S1=成品图灌丛内部、S2=灌丛—草裸界面；真值＝z18 高影像逐点判读',
         '- 配方：每瓦 ±2° 框抽 20,000 点 → RF(100, leaf2, nodes5000)；R0=全池，R4=筛 glc_fcs10_2023_shrub',
         '- 灌丛层在框内被筛掉的比例见 json；判读真值 %d 点（A %d/S1 %d/S2 %d）' % (
             len(T), masks['A'].sum(), masks['S1'].sum(), masks['S2'].sum()), '',
         '## 主对比（3 种子均值）', '',
         '| 策略 | 组 | n | OA24 | OA9 | 真值灌丛 | 预测灌丛 | 命中 |', '|---|---|---|---|---|---|---|---|']
    for pol in ('R0', 'R4'):
        for gname in ('ALL', 'A', 'S1', 'S2'):
            vals = [out['results'].get('%s|%d' % (pol, sd), {}).get(gname) for sd in SEEDS]
            vals = [v for v in vals if v]
            if not vals:
                continue
            L.append('| %s | %s | %d | %.4f | %.4f | %d | %.1f | %d |' % (
                pol, gname, vals[0]['n'],
                np.mean([v['OA24'] for v in vals]), np.mean([v['OA9'] for v in vals]),
                vals[0]['shrub_true'], np.mean([v['shrub_pred'] for v in vals]),
                np.mean([v['shrub_hit'] for v in vals])))
    L += ['', '## 策略曲线（T1608+T1609，seed 7）', '',
          '| 策略 | 组 | n | OA24 | OA9 | 预测灌丛 |', '|---|---|---|---|---|---|']
    for pol in ('R0', 'R1', 'R2', 'R3', 'R4'):
        v = out['results'].get('%s|7' % pol)
        if not v:
            continue
        for gname in ('ALL', 'S1'):
            if gname in v and v[gname]['n']:
                L.append('| %s | %s | %d | %.4f | %.4f | %d |' % (
                    pol, gname, v[gname]['n'], v[gname]['OA24'], v[gname]['OA9'], v[gname]['shrub_pred']))
    fp = os.path.join(REPT, '宁夏全筛小样本实验_%s.md' % time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    emit('→ %s' % fp)
    emit('总耗时 %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()

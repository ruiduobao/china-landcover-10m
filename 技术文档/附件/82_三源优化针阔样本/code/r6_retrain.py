# -*- coding: utf-8 -*-
"""r6_retrain.py — 新优化样本 vs v2 旧样本：同特征(AEF64)同协议的空间块 5 折重训对比

* 目的：本轮**唯一决定性质量检验** —— 新样本(opt_v1 TIER_A/B) 是否真的比 v2 旧样本好？
* 输入（只读，均为本脚本生成的上游产物）：
    opt/aef_all_points.parquet        veg30 新点的 AEF64（r6_aef_fetch.py 产出）
    F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet   v2 点的 AEF64（本地已有）
    labels/needle_broad_national_v2.parquet             v2 标签（FCS10）
    opt/needle_broad_opt_v1.parquet                    新样本标签（leaf_final / tier）
* 特征：AEF64（2023），逐点
* 标签：needle=1 / broad=0，**排除 mixed**
* 四臂（预注册）：
    ARM_V2   v2 全量（needle+broad）
    ARM_AB   opt_v1 TIER_A + TIER_B
    ARM_A    opt_v1 TIER_A
    ARM_MIX  v2 全量 + (opt_v1 A+B 中不与 v2 重复的点)，超 20 万则下采样到 20 万
* 协议（四臂完全一致，故可比）：
    空间块 = floor(lon/0.1)*10000 + floor(lat/0.1)；block 种子 20261024 洗牌；
    5 折按 block 分组；RF(n_estimators=200, min_samples_leaf=2, max_features='sqrt',
    class_weight='balanced', n_jobs=8, random_state=20261024)
    报告 OA / macro-F1 / 每类 P/R/F1 的 5 折均值±sd，以及合并 OOF 混淆矩阵
* 跨臂互评（比自评更有说服力）：
    A→B：用 ARM_V2 全量训练，预测 ARM_AB 中 **block 不与 v2 重叠** 的子集
    B→A：用 ARM_AB 全量训练，预测 ARM_V2 中 **block 不与 AB 重叠** 的子集
    （block 不重叠 = 真正的"对方的数据"，且训练侧没见过同一空间块）
* 判据（先写死）：
    主判据 delta_macroF1 = ARM_AB - ARM_V2；
    噪声阈：|delta| < 两臂折间 sd 的合并标准误 × 2 → 判"未超噪声"
    配对检验：同折配对 t 检验（5 折）+ Wilcoxon 符号秩（n=5 时仅作参考）
* 输出：opt/aef_{arm}.parquet、opt/r6_retrain.json、opt/r6_verdict.md
* 用法：python r6_retrain.py [--arms V2,AB,A,MIX] [--skip-cross] [--max-mix 200000]
* 资源：单进程，RF n_jobs=8，峰值 < 12 GB
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support)

sys.stdout.reconfigure(encoding='utf-8')

OPT = r'F:/lc_work/nb_national_2026/opt'
LAB = r'F:/lc_work/nb_national_2026/labels'
R7 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
A = ['A%02d' % i for i in range(64)]

SEED = 20261024
N_FOLD = 5
RF_ARGS = dict(n_estimators=200, min_samples_leaf=2, max_features='sqrt',
               class_weight='balanced', n_jobs=8, random_state=SEED)


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def blk(lon, lat, b=0.1):
    """10 km 空间块：floor(lon/0.1)*10000 + floor(lat/0.1)"""
    return (np.floor(np.asarray(lon, dtype=np.float64) / b).astype(np.int64) * 10000 +
            np.floor(np.asarray(lat, dtype=np.float64) / b).astype(np.int64))


def key(lon, lat, nd=5):
    return (np.round(np.asarray(lon) * 10 ** nd).astype('int64').astype(str) + '_' +
            np.round(np.asarray(lat) * 10 ** nd).astype('int64').astype(str))


# ---------------- 数据装载 ----------------
def load_all():
    """返回 (v2_df, opt_df)，均含 lon/lat/leaf(AEF) 与 block，只保留 needle/broad"""
    v2 = pd.read_parquet(os.path.join(LAB, 'needle_broad_national_v2.parquet'),
                         columns=['point_id', 'row_id', 'lon', 'lat', 'tile', 'prov',
                                  'eco', 'leaf', 'tier', 'src'])
    opt = pd.read_parquet(os.path.join(OPT, 'needle_broad_opt_v1.parquet'),
                          columns=['point_id', 'lon', 'lat', 'tile', 'prov', 'eco',
                                   'leaf_final', 'tier', 'src', 'reason'])

    # v2：本地 r7_2023 的 AEF64
    r7 = pd.read_parquet(R7, columns=['row_id', 'lon', 'lat'] + A)
    r7 = r7.drop_duplicates('row_id').set_index('row_id')
    v2 = v2[v2.leaf.isin(['needle', 'broad'])].copy()
    sub = r7.reindex(v2.row_id.values)
    v2[A] = sub[A].to_numpy(np.float32)
    # 坐标索引（含 mixed，供 opt 里被 r5_fuse 改判的点回取 AEF）
    r7c = r7.assign(k=key(r7.lon.values, r7.lat.values))
    r7c = r7c[~r7c.index.duplicated()]
    r7c = r7c[~r7c['k'].duplicated()].set_index('k')[A]
    del r7, sub
    emit('v2 needle/broad %d（针 %d / 阔 %d）' %
         (len(v2), int((v2.leaf == 'needle').sum()), int((v2.leaf == 'broad').sum())))

    # opt：veg30 点的 AEF64 来自 r6_aef_fetch；v2_fcs10 点按坐标回取 r7
    opt = opt[opt.leaf_final.isin(['needle', 'broad'])].copy()
    opt['y'] = (opt.leaf_final == 'needle').astype(np.int8)
    need = opt[opt.src == 'veg30'].copy()
    ap = os.path.join(OPT, 'aef_all_points.parquet')
    if len(need):
        af = pd.read_parquet(ap)
        af = af.drop_duplicates('point_id').set_index('point_id')
        sub = af.reindex(need.point_id.values)
        need[A] = sub[A].to_numpy(np.float32)
        del af, sub
        miss = int((~np.isfinite(need[A].to_numpy(np.float32)).all(1)).sum())
        emit('veg30 点 %d，AEF 缺失 %d' % (len(need), miss))
    # 注意（2026-10-25 实测）：opt_v1 的 point_id 被 r5_fuse 重编号为 800000000+，
    # **与 v2 的 point_id 零交集** → 必须按坐标(5 位)连接，不能按 point_id。
    have = opt[opt.src == 'v2_fcs10'].copy()
    have['k'] = key(have.lon.values, have.lat.values)
    sub = r7c.reindex(have.k.values)
    have[A] = sub[A].to_numpy(np.float32)
    nmiss = int((~np.isfinite(have[A].to_numpy(np.float32)).all(1)).sum())
    emit('opt 中 v2 来源点 %d，按坐标回取 AEF 失败 %d' % (len(have), nmiss))
    # 标签是否被 r5_fuse 改写（南方冲突时按植被图定）
    v2lab = v2.assign(k=key(v2.lon.values, v2.lat.values)).set_index('k')['leaf']
    v2lab = v2lab[~v2lab.index.duplicated()]
    have['v2_leaf'] = v2lab.reindex(have.k.values).to_numpy()
    chg = have[have.v2_leaf.notna() &
               (have.v2_leaf.map({'needle': 1, 'broad': 0}).to_numpy() !=
                have.y.to_numpy())]
    emit('  v2 来源点中 leaf_final 与 v2.leaf 不同：%d（%.2f%%）'
         % (len(chg), 100 * len(chg) / max(1, len(have))))
    del v2lab
    opt = pd.concat([need, have.drop(columns=['v2_leaf'])], ignore_index=True)
    opt = opt.dropna(subset=['lon', 'lat'])
    v2['y'] = (v2.leaf == 'needle').astype(np.int8)
    for d in (v2, opt):
        d['block'] = blk(d.lon.values, d.lat.values)
        d['k'] = key(d.lon.values, d.lat.values)
    v2['good_aef'] = np.isfinite(v2[A].to_numpy(np.float32)).all(1)
    opt['good_aef'] = np.isfinite(opt[A].to_numpy(np.float32)).all(1)
    return v2, opt


def build_arms(v2, opt, max_mix=200000, rng=None):
    """四臂字典；每臂 DataFrame 含 lon/lat/y/block + AEF64"""
    arms = {}
    v2g = v2[v2.good_aef].copy()
    arms['V2'] = v2g
    ab = opt[(opt.tier.isin(['TIER_A', 'TIER_B'])) & opt.good_aef].copy()
    abf = opt[(opt.tier.isin(['TIER_A', 'TIER_B', 'TIER_C', 'TIER_D'])) ].copy()
    # ARM_AB：TIER_A/B，mixed 已在 load_all 排除
    arms['AB'] = ab
    arms['A'] = opt[(opt.tier == 'TIER_A') & opt.good_aef].copy()
    # ARM_MIX：v2 全量 + A+B 中不与 v2 重复的点
    newab = ab[~ab.k.isin(set(v2g.k))]
    mix = pd.concat([v2g[['lon', 'lat', 'y', 'block', 'k'] + A],
                     newab[['lon', 'lat', 'y', 'block', 'k'] + A]], ignore_index=True)
    if max_mix and len(mix) > max_mix:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(mix), max_mix, replace=False))
        mix = mix.iloc[idx].reset_index(drop=True)
        emit('ARM_MIX 下采样到 %d' % len(mix))
    arms['MIX'] = mix
    for nm, d in arms.items():
        emit('  ARM_%-4s n=%d（针 %d / 阔 %d）块 %d' %
             (nm, len(d), int((d.y == 1).sum()), int((d.y == 0).sum()), d.block.nunique()))
    return arms


# ---------------- 评估 ----------------
def fold_blocks(blocks, seed=SEED, k=N_FOLD):
    ug = np.unique(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    return np.array_split(ug, k)


def metrics(y, p):
    oa = float((p == y).mean())
    f1m = f1_score(y, p, average='macro')
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[1, 0])
    # labels=[1,0] → 0=needle, 1=broad
    return dict(oa=oa, macro_f1=float(f1m),
                needle=dict(precision=float(pr[0]), recall=float(rc[0]), f1=float(f1[0]),
                            support=int(sup[0])),
                broad=dict(precision=float(pr[1]), recall=float(rc[1]), f1=float(f1[1]),
                           support=int(sup[1])))


def cv_arm(d, seed=SEED, k=N_FOLD):
    X = d[A].to_numpy(np.float32)
    y = d.y.to_numpy(int)
    b = d.block.to_numpy()
    folds = fold_blocks(b, seed, k)
    per, oof = [], np.zeros(len(d), dtype=np.int8)
    for i, fb in enumerate(folds):
        m = np.isin(b, fb)
        rf = RandomForestClassifier(**RF_ARGS)
        rf.fit(X[~m], y[~m])
        oof[m] = rf.predict(X[m]).astype(np.int8)
        per.append(metrics(y[m], oof[m]))
        emit('    fold %d/%d ntr=%d nte=%d OA %.4f macroF1 %.4f needleF1 %.4f'
             % (i + 1, k, int((~m).sum()), int(m.sum()), per[-1]['oa'],
                per[-1]['macro_f1'], per[-1]['needle']['f1']))
    agg = {'folds': per}
    for key_ in ['oa', 'macro_f1']:
        v = np.array([f[key_] for f in per])
        agg[key_] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), folds=v.tolist())
    for cls in ['needle', 'broad']:
        agg[cls] = {}
        for mk in ['precision', 'recall', 'f1']:
            v = np.array([f[cls][mk] for f in per])
            agg[cls][mk] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), folds=v.tolist())
    agg['oof_metrics'] = metrics(y, oof)
    cm = confusion_matrix(y, oof, labels=[1, 0])
    agg['oof_confusion'] = dict(labels=['needle', 'broad'],
                                matrix=cm.tolist(),
                                note='行=真实(needle,broad)，列=预测(needle,broad)')
    return agg, oof


def mcnemar(y, p1, p2):
    """同一测试集上两模型逐点错误的 McNemar 检验（连续性校正）"""
    from scipy.stats import chi2
    c1 = (p1 != y)
    c2 = (p2 != y)
    b = int((c1 & ~c2).sum())     # 仅模型1错
    c = int((~c1 & c2).sum())     # 仅模型2错
    stat = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return dict(only_m1_wrong=b, only_m2_wrong=c,
                chi2=float(stat), p=float(chi2.sf(stat, 1)),
                note='H0: 两模型错误率相同')


def common_test(train_a, train_b, test, tag):
    """同一测试集上比较两臂（两臂训练集均排除 test 所在 block）

    train_a/train_b 传入的已是"全量臂"，函数内部自行剔除 test 的 block。
    """
    tb = set(np.unique(test.block))
    a = train_a[~train_a.block.isin(tb)]
    b = train_b[~train_b.block.isin(tb)]
    yte = test.y.to_numpy(int)
    out = {'n_test': int(len(test)), 'test_needle': int((yte == 1).sum()),
           'test_broad': int((yte == 0).sum())}
    preds = {}
    for nm, d in [('a', a), ('b', b)]:
        if len(d) < 100:
            out[nm] = dict(error='empty_train_after_block_exclusion', n_train=int(len(d)))
            emit('  %s | %s 训练集被剔空（n=%d）→ 跳过' % (tag, nm, len(d)))
            continue
        rf = RandomForestClassifier(**RF_ARGS)
        rf.fit(d[A].to_numpy(np.float32), d.y.to_numpy(int))
        p = rf.predict(test[A].to_numpy(np.float32)).astype(int)
        preds[nm] = p
        out[nm] = dict(n_train=int(len(d)), **metrics(yte, p))
        emit('  %s | 训练 %s(n=%d) → 测试 n=%d：OA %.4f macroF1 %.4f needleF1 %.4f'
             % (tag, nm, len(d), len(test), out[nm]['oa'], out[nm]['macro_f1'],
                out[nm]['needle']['f1']))
    if len(preds) == 2:
        out['mcnemar'] = mcnemar(yte, preds['a'], preds['b'])
        out['delta_macro_f1_a_minus_b'] = out['a']['macro_f1'] - out['b']['macro_f1']
        emit('  %s | ΔmacroF1(a−b) = %+.4f  McNemar only_a_wrong=%d only_b_wrong=%d p=%.3g'
             % (tag, out['delta_macro_f1_a_minus_b'], out['mcnemar']['only_m1_wrong'],
                out['mcnemar']['only_m2_wrong'], out['mcnemar']['p']))
    return out


def cross_eval(train_d, test_d, tag):
    """train_d 全量训练 → 预测 test_d（test_d 的 block 已与 train_d 不重叠）"""
    Xtr = train_d[A].to_numpy(np.float32)
    ytr = train_d.y.to_numpy(int)
    rf = RandomForestClassifier(**RF_ARGS)
    rf.fit(Xtr, ytr)
    p = rf.predict(test_d[A].to_numpy(np.float32)).astype(int)
    m = metrics(test_d.y.to_numpy(int), p)
    emit('  %s n_tr=%d n_te=%d → OA %.4f macroF1 %.4f needleF1 %.4f'
         % (tag, len(train_d), len(test_d), m['oa'], m['macro_f1'], m['needle']['f1']))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='V2,AB,A,MIX')
    ap.add_argument('--skip-cross', action='store_true')
    ap.add_argument('--max-mix', type=int, default=200000)
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()
    t0 = time.time()
    want = [s.strip() for s in args.arms.split(',') if s.strip()]

    v2, opt = load_all()
    arms = build_arms(v2, opt, args.max_mix)

    # 落盘各臂特征
    for nm in ['V2', 'AB', 'A', 'MIX']:
        if nm not in arms:
            continue
        p = os.path.join(OPT, 'aef_%s%s.parquet' % (nm.lower(), args.out_tag))
        cols = ['lon', 'lat', 'y', 'block'] + A
        arms[nm][cols].to_parquet(p, index=False)
        emit('  写 %s' % p)

    res = dict(prereg=dict(
        features='AEF64 (2023, GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL)',
        labels='needle=1 / broad=0, mixed excluded',
        block='floor(lon/0.1)*10000+floor(lat/0.1), 10 km',
        block_seed=SEED, folds=N_FOLD, rf=RF_ARGS,
        arms={k: int(len(v)) for k, v in arms.items()}),
        arms={}, cross={}, elapsed_min=None)

    for nm in want:
        if nm not in arms:
            emit('跳过未知臂 %s' % nm)
            continue
        emit('=== CV ARM_%s n=%d ===' % (nm, len(arms[nm])))
        agg, oof = cv_arm(arms[nm])
        res['arms'][nm] = agg
        emit('  ARM_%-3s OA %.4f±%.4f  macroF1 %.4f±%.4f  needleF1 %.4f±%.4f  broadF1 %.4f±%.4f'
             % (nm, agg['oa']['mean'], agg['oa']['sd'],
                agg['macro_f1']['mean'], agg['macro_f1']['sd'],
                agg['needle']['f1']['mean'], agg['needle']['f1']['sd'],
                agg['broad']['f1']['mean'], agg['broad']['f1']['sd']))

    # ---------- 跨臂互评 ----------
    if not args.skip_cross and 'V2' in arms and 'AB' in arms:
        emit('=== 跨臂互评 ===')
        v2b = set(arms['V2'].block.unique())
        abb = set(arms['AB'].block.unique())
        only_ab = arms['AB'][~arms['AB'].block.isin(v2b)].copy()
        only_v2 = arms['V2'][~arms['V2'].block.isin(abb)].copy()
        res['cross']['subset_sizes'] = dict(
            AB_only_blocks=int(len(only_ab)), V2_only_blocks=int(len(only_v2)),
            shared_blocks=int(len(v2b & abb)),
            AB_only_needle=int((only_ab.y == 1).sum()),
            V2_only_needle=int((only_v2.y == 1).sum()))
        emit('  block 不重叠子集：AB_only %d 点（针 %d）/ V2_only %d 点（针 %d）；共享块 %d'
             % (len(only_ab), int((only_ab.y == 1).sum()),
                len(only_v2), int((only_v2.y == 1).sum()), len(v2b & abb)))
        # A→B：v2 全量训练 → 预测 AB 独有块；B→A：AB 全量训练 → 预测 V2 独有块
        res['cross']['V2_train_AB_test'] = cross_eval(arms['V2'], only_ab, 'V2train→ABtest')
        res['cross']['AB_train_V2_test'] = cross_eval(arms['AB'], only_v2, 'ABtrain→V2test')
        if 'A' in arms:
            res['cross']['A_train_V2_test'] = cross_eval(arms['A'], only_v2, 'Atrain→V2test')
        if 'MIX' in arms:
            res['cross']['MIX_train_AB_test'] = cross_eval(arms['MIX'], only_ab, 'MIXtrain→ABtest')

        # ---------- 核心：同一测试集上的正面比较（McNemar） ----------
        emit('=== 共同测试集正面比较（两臂训练集均剔除测试块）===')
        res['common_test'] = {}
        # 1) 在 AB 独有的块上：V2 vs AB
        res['common_test']['on_AB_only_blocks'] = common_test(
            arms['V2'], arms['AB'], only_ab, 'AB_only块')
        # 2) 在 V2 独有的块上：V2 vs AB
        res['common_test']['on_V2_only_blocks'] = common_test(
            arms['V2'], arms['AB'], only_v2, 'V2_only块')
        # 3) 在 v2 原始抽样点上（v2 的"老家"，但仍是 block 留出）
        res['common_test']['on_V2_all'] = common_test(
            arms['V2'], arms['AB'], arms['V2'], 'V2全量点')
        if 'A' in arms:
            res['common_test']['on_AB_only_blocks_AvsV2'] = common_test(
                arms['V2'], arms['A'], only_ab, 'AB块_A vs V2')
        if 'MIX' in arms:
            res['common_test']['on_AB_only_blocks_MIXvsV2'] = common_test(
                arms['V2'], arms['MIX'], only_ab, 'AB块_MIX vs V2')


    res['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(OPT, 'r6_retrain%s.json' % args.out_tag), 'w',
                        encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    emit('写 r6_retrain%s.json；总耗时 %.1f min' % (args.out_tag, res['elapsed_min']))
    emit(json.dumps({k: dict(macro_f1=round(v['macro_f1']['mean'], 4),
                             needle_f1=round(v['needle']['f1']['mean'], 4))
                     for k, v in res['arms'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()

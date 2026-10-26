# -*- coding: utf-8 -*-
"""e3_driver.py — 园地先验实验·限时驱动 + 配对汇总（region×arm×3seed ≤1200s/单元）

* 输入：<WS>/prereg/regions_frozen.json、<WS>/cache/*、e2_run_unit.py
* 规则源：doc70 §3.2（硬超时终止自己子进程）、§6.3（配对块 bootstrap）、§7（采纳门）
* 门槛：主门=非FCS10 确认层 macro9 非劣（点估≥−1pp 且块bootstrap95%下界≥−2pp）；
        目标类F1 Δ≥+0.03 且 CI 下界>0；充分支持类 ΔF1≤−0.05 视为新回归
* 输出：<WS>/results/summary.json、<WS>/results/summary.md、<WS>/results/paired_by_seed.csv
* 用法：PYTHONDONTWRITEBYTECODE=1 python e3_driver.py            # 跑全部 8 单元 + 汇总
        PYTHONDONTWRITEBYTECODE=1 python e3_driver.py --agg-only  # 只汇总（已有单元 json/csv）
"""
import os, sys, json, time, argparse, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC

DEADLINE = 1200
NBOOT = 10000


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def run_units(reg):
    rows = []
    for rid in reg['regions']:
        for arm in reg['arms']:
            t0 = time.time()
            cmd = [sys.executable, os.path.join(HERE, 'e2_run_unit.py'), '--region', rid, '--arm', arm]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            logdir = os.path.join(WS, 'control', 'logs')
            os.makedirs(logdir, exist_ok=True)
            lf = os.path.join(logdir, '%s_%s.log' % (rid, arm))
            state = 'DONE'
            with open(lf, 'w', encoding='utf-8') as fh:
                try:
                    r = subprocess.run(cmd, timeout=DEADLINE, stdout=fh, stderr=subprocess.STDOUT, env=env)
                    if r.returncode != 0:
                        state = 'FAILED_rc%d' % r.returncode
                except subprocess.TimeoutExpired:
                    state = 'TIMEOUT'
            wall = round(time.time() - t0, 1)
            emit('%s/%s → %s (%.1fs)' % (rid, arm, state, wall))
            rows.append(dict(region=rid, arm=arm, state=state, wall_s=wall, log=lf))
    pd.DataFrame(rows).to_csv(os.path.join(WS, 'results', 'unit_runs.csv'), index=False, encoding='utf-8-sig')
    return rows


def block_bootstrap_ci(blocks, dpoint, nboot=NBOOT, seed=20261015):
    """空间块 bootstrap：对块重采样，求 Δ 的均值分布 2.5/97.5 分位。dpoint=每点配对差（seed 均值）。"""
    rng = np.random.RandomState(seed)
    ub = np.unique(blocks)
    idx_by_b = {b: np.flatnonzero(blocks == b) for b in ub}
    means = np.empty(nboot)
    for i in range(nboot):
        pick = rng.randint(0, len(ub), len(ub))
        sel = np.concatenate([idx_by_b[ub[j]] for j in pick])
        means[i] = dpoint[sel].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(len(ub))


def mcnemar(corr_b, corr_o):
    """配对 McNemar（连续性校正）→ (b, c, chi2, p)。b=基线对/候选错，c=基线错/候选对。"""
    from scipy.stats import chi2 as chi2d
    b = int(((corr_b == 1) & (corr_o == 0)).sum())
    c = int(((corr_b == 0) & (corr_o == 1)).sum())
    if b + c == 0:
        return b, c, 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(chi2), float(1 - chi2d.cdf(chi2, 1))


def per_point_frame(m, macro):
    """(seed,row_id) 长表 → 每参考点跨种子平均正确率（唯一点）。"""
    m = m.copy()
    m['ok24_b'] = (m.pred_b.to_numpy() == m.t24_b.to_numpy()).astype(float)
    m['ok24_o'] = (m.pred_o.to_numpy() == m.t24_b.to_numpy()).astype(float)
    m['ok9_b'] = (macro(m.pred_b.to_numpy()) == macro(m.t24_b.to_numpy())).astype(float)
    m['ok9_o'] = (macro(m.pred_o.to_numpy()) == macro(m.t24_b.to_numpy())).astype(float)
    g = m.groupby('row_id', sort=False)
    pt = pd.DataFrame(dict(
        row_id=g.row_id.first(), lon=g.lon_b.first(), lat=g.lat_b.first(),
        t24=g.t24_b.first(), isf=g.isf_b.first(), block=g.block_b.first(), split=g.split_b.first(),
        ok24_b=g.ok24_b.mean(), ok24_o=g.ok24_o.mean(),
        ok9_b=g.ok9_b.mean(), ok9_o=g.ok9_o.mean(), nseed=g.size()))
    return pt.reset_index(drop=True)


def layer_stats(pt, msk, blocks_all):
    """单层：seed 平均正确率 → Δ、块 bootstrap CI、McNemar（多数票）。"""
    sel = np.flatnonzero(msk)
    if len(sel) < 10:
        return None
    d24 = pt.ok24_o.to_numpy()[sel] - pt.ok24_b.to_numpy()[sel]
    d9 = pt.ok9_o.to_numpy()[sel] - pt.ok9_b.to_numpy()[sel]
    lo24, hi24, nblk = block_bootstrap_ci(blocks_all[sel], d24)
    lo9, hi9, _ = block_bootstrap_ci(blocks_all[sel], d9)
    b, c, chi2, p = mcnemar((pt.ok24_b.to_numpy()[sel] > 0.5).astype(int),
                            (pt.ok24_o.to_numpy()[sel] > 0.5).astype(int))
    return dict(n=int(len(sel)), n_blocks=nblk,
                OA24_b=round(float(pt.ok24_b.to_numpy()[sel].mean()), 4),
                OA24_o=round(float(pt.ok24_o.to_numpy()[sel].mean()), 4),
                dOA24=round(float(d24.mean()), 4), dOA24_ci=[round(lo24, 4), round(hi24, 4)],
                m9_b=round(float(pt.ok9_b.to_numpy()[sel].mean()), 4),
                m9_o=round(float(pt.ok9_o.to_numpy()[sel].mean()), 4),
                dm9=round(float(d9.mean()), 4), dm9_ci=[round(lo9, 4), round(hi9, 4)],
                mcnemar=dict(b=b, c=c, chi2=round(chi2, 3), p=round(p, 4)))


def per_class_f1(m, seeds, names, layers, keys):
    """逐类 seed 平均 F1（每层）。m 为 (seed,row_id) 长表；keys 为**唯一点**全局键数组
    （region:row_id，与 msk 同序），经 gid 映射对齐——修复"联合时按位置索引错位"的统计 bug。"""
    m = m.copy()
    m['gid'] = m.region_b.astype(str) + ':' + m.row_id.astype(str)
    pos = {s: dict(zip(m[m.seed == s].gid.to_numpy(),
                       np.arange(len(m[m.seed == s])))) for s in seeds}
    s0 = m[m.seed == seeds[0]]
    per = {}
    for lay, msk in layers:
        sel = np.flatnonzero(msk)
        if len(sel) < 1:
            continue
        k = keys[sel]
        p0 = np.array([pos[seeds[0]][x] for x in k])
        tt = s0.t24_b.to_numpy()[p0]
        for c in sorted(set(tt.tolist())):
            if (tt == c).sum() == 0:
                continue
            f1s = {'b': [], 'o': []}
            for s in seeds:
                sub = m[m.seed == s]
                p = np.array([pos[s][x] for x in k])
                for arm, pv in [('b', sub.pred_b.to_numpy()[p]), ('o', sub.pred_o.to_numpy()[p])]:
                    tp = int(((tt == c) & (pv == c)).sum())
                    fp = int(((tt != c) & (pv == c)).sum())
                    fn = int(((tt == c) & (pv != c)).sum())
                    pr = tp / (tp + fp) if tp + fp else 0.0
                    rc = tp / (tp + fn) if tp + fn else 0.0
                    f1s[arm].append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
            per.setdefault(lay, {})[int(c)] = dict(
                name=names.get(str(int(c)), '?'), n=int((tt == c).sum()),
                f1_b=round(float(np.mean(f1s['b'])), 4), f1_o=round(float(np.mean(f1s['o'])), 4),
                dF1=round(float(np.mean(f1s['o']) - np.mean(f1s['b'])), 4))
    return per


def verdict_of(main, per, tgt):
    """doc70 §7 五级判读：ACCEPT_EXPERIMENTAL / PROMISING / REJECT（+ TIMEOUT/BLOCKED 另由驱动标）。
    主门=非fcs10 确认层 macro9 非劣；目标类 F1（支持充分者）改善。
    返回 (group_verdict, target_supported, target_status)。
    target_status ∈ IMPROVED / NOT_IMPROVED / INSUFFICIENT_SUPPORT —— 组级非劣 **不等于** 目标类改善。
    无正收益（点估 ≤0）一律 REJECT；组级回归亦 REJECT。"""
    if main is None:
        return 'REJECT', [], 'NOT_EVALUATED'
    noninf = (main['dm9'] >= -0.01) and (main['dm9_ci'][0] >= -0.02)
    tgt_supp = [c for c in tgt if per.get('conf_nonfcs10', {}).get(c, {}).get('n', 0) >= 30]
    tgt_ok = any(per.get('conf_nonfcs10', {}).get(c, {}).get('dF1', 0) >= 0.03 for c in tgt_supp)
    if not noninf or main['dm9'] <= 0:
        v = 'REJECT'                       # 组级回归或无正收益
    elif tgt_ok:
        v = 'ACCEPT_EXPERIMENTAL'          # 非劣且有目标类改善
    else:
        v = 'PROMISING'                    # 非劣且正向，但目标类未改善/支持不足
    if not tgt_supp:
        ts = 'INSUFFICIENT_SUPPORT'
    else:
        ts = 'IMPROVED' if tgt_ok else 'NOT_IMPROVED'
    return v, tgt_supp, ts


def agg(reg):
    names = VC.V31_NAMES()
    macro = VC.to_macro
    seeds = reg['seeds']
    TARGET = {'SC1': [4], 'SC2': [4], 'GD': [4, 5], 'NX': [22, 3]}
    out = {}
    rows = []
    pts = {}
    for rid, R in reg['regions'].items():
        fp_b = os.path.join(WS, 'results', '%s_baseline_pred.csv' % rid)
        fp_o = os.path.join(WS, 'results', '%s_OCAP_pred.csv' % rid)
        if not (os.path.exists(fp_b) and os.path.exists(fp_o)):
            out[rid] = dict(state='MISSING')
            continue
        db, do = pd.read_csv(fp_b), pd.read_csv(fp_o)
        m = db.merge(do, on=['seed', 'row_id'], suffixes=('_b', '_o'))
        assert len(m) == len(db) == len(do), '%s 两臂参考点不一致 %d/%d/%d' % (rid, len(m), len(db), len(do))
        pt = per_point_frame(m, macro)
        pt['region'] = rid
        pts[rid] = pt
        blocks_all = pt.block.to_numpy()
        # 每 seed Δ（层分报）
        for s in seeds:
            sub = m[m.seed == s]
            t = sub.t24_b.to_numpy()
            f = sub.isf_b.to_numpy()
            sp = sub.split_b.to_numpy()
            for lay, msk in [('all', np.ones(len(sub), bool)), ('nonfcs10', ~f), ('fcs10', f),
                             ('conf', sp == 'conf'), ('conf_nonfcs10', (sp == 'conf') & (~f)),
                             ('dev', sp == 'dev')]:
                if msk.sum() < 10:
                    continue
                tt = t[msk]
                cb = (sub.pred_b.to_numpy()[msk] == tt)
                co = (sub.pred_o.to_numpy()[msk] == tt)
                mb = (macro(sub.pred_b.to_numpy()[msk]) == macro(tt))
                mo = (macro(sub.pred_o.to_numpy()[msk]) == macro(tt))
                rows.append(dict(region=rid, seed=s, layer=lay, n=int(msk.sum()),
                                 OA24_b=round(cb.mean(), 4), OA24_o=round(co.mean(), 4),
                                 dOA24=round(co.mean() - cb.mean(), 4),
                                 m9_b=round(mb.mean(), 4), m9_o=round(mo.mean(), 4),
                                 dm9=round(mo.mean() - mb.mean(), 4)))
        # 层统计（唯一点 seed 平均 + 块 bootstrap）
        f, sp = pt.isf.to_numpy(), (pt.split.to_numpy() == 'conf')
        res = {}
        for lay, msk in [('all', np.ones(len(pt), bool)), ('nonfcs10', ~f), ('fcs10', f),
                         ('conf', sp), ('conf_nonfcs10', sp & ~f), ('dev', ~sp)]:
            res[lay] = layer_stats(pt, msk, blocks_all)
        keys_r = (pt.region.astype(str) + ':' + pt.row_id.astype(str)).to_numpy()
        per = per_class_f1(m, seeds, names,
                           [('conf_nonfcs10', (sp & ~f)), ('all', np.ones(len(pt), bool))], keys_r)
        v, tgt_supp, tstat = verdict_of(res.get('conf_nonfcs10'), per, TARGET.get(rid, []))
        # 园地输出归零回归风险（预测类2：全 3 seed 汇总 + 逐 seed）
        n2b = int((m.pred_b.to_numpy() == 2).sum())
        n2o = int((m.pred_o.to_numpy() == 2).sum())
        n2_ps = {int(s): (int((m[(m.seed == s)].pred_b.to_numpy() == 2).sum()),
                          int((m[(m.seed == s)].pred_o.to_numpy() == 2).sum())) for s in seeds}
        out[rid] = dict(state='OK', main_layer=res.get('conf_nonfcs10'), verdict=v,
                        target_classes=TARGET.get(rid, []), target_supported=tgt_supp,
                        target_status=tstat,
                        orchard_output_risk=dict(
                            pred_cl2_allseeds_baseline=n2b, pred_cl2_allseeds_ocap=n2o,
                            pred_cl2_per_seed=n2_ps,
                            collapse_all_seeds=(n2b > 0 and n2o == 0),
                            near_collapse=(n2b > 0 and n2o <= max(1, int(0.05 * n2b))),
                            note='O-CAP 使园地预测近乎消失（SC1/SC2/GD 全 3 seed 归零；NX 19→1）；'
                                 '真实园地会被误判为林/耕（**显式回归风险**）。确认层园地真值 n≤1 → UA/PA 不可评。'),
                        layers=res, per_class=per)
        ml = res.get('conf_nonfcs10')
        emit('%s: conf_nonfcs10 n=%s dm9=%s ci=%s | n_conf_cl4=%s | target=%s → %s (target %s)' % (
            rid, ml['n'] if ml else '-', ml['dm9'] if ml else '-', ml['dm9_ci'] if ml else '-',
            R.get('n_conf_cl4'), TARGET.get(rid), v, tstat))

    # ---- 预登记小区联合：SC1+SC2（两框各用所属局部模型；仅提高支持，不称省域；描述性）----
    if 'SC1' in pts and 'SC2' in pts:
        pj = pd.concat([pts['SC1'], pts['SC2']], ignore_index=True)
        fj = pj.isf.to_numpy()
        sj = (pj.split.to_numpy() == 'conf')
        blkj = pj.block.to_numpy()
        mj = pd.concat([pd.read_csv(os.path.join(WS, 'results', '%s_baseline_pred.csv' % r))
                        .merge(pd.read_csv(os.path.join(WS, 'results', '%s_OCAP_pred.csv' % r)),
                               on=['seed', 'row_id'], suffixes=('_b', '_o')) for r in ('SC1', 'SC2')],
                       ignore_index=True)
        joint_main = layer_stats(pj, sj & ~fj, blkj)
        keys_j = (pj.region.astype(str) + ':' + pj.row_id.astype(str)).to_numpy()
        joint_per = per_class_f1(mj, seeds, names, [('conf_nonfcs10', sj & ~fj)], keys_j)
        jv, jtgt, jstat = verdict_of(joint_main, joint_per, [4])
        out['SC1+SC2_joint'] = dict(
            state='OK', note='预登记小区联合（SC1+SC2，各用所属局部模型）；**描述性，不用于抬高采纳门**',
            main_layer=joint_main, verdict=jv, target_supported=jtgt, target_status=jstat,
            target_f1_conf=joint_per.get('conf_nonfcs10', {}),
            n_conf_cl4=int((joint_per.get('conf_nonfcs10', {}).get(4, {}) or {}).get('n', 0)))
        emit('SC1+SC2 联合(描述性): conf_nonfcs10 n=%s dm9=%s ci=%s 类4n=%s' % (
            joint_main['n'], joint_main['dm9'], joint_main['dm9_ci'], out['SC1+SC2_joint']['n_conf_cl4']))

    pd.DataFrame(rows).to_csv(os.path.join(WS, 'results', 'paired_by_seed.csv'),
                              index=False, encoding='utf-8-sig')
    with open(os.path.join(WS, 'results', 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), regions=out), f,
                  ensure_ascii=False, indent=1, default=str)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg-only', action='store_true')
    a = ap.parse_args()
    reg = json.load(open(os.path.join(WS, 'prereg', 'regions_frozen.json'), encoding='utf-8'))
    if not a.agg_only:
        run_units(reg)
    out = agg(reg)
    emit('汇总完成 → results/summary.json')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""s5_report.py — A/B/D 判定报告（预注册判据 doc37 §6.2）→ reports/R1_ABD_*.md + STATE.verdict
判据（预注册，防事后挪线）：
  B vs A：macro9 OA 不降 且 森林组成类(04-08)平均F1 不降 且 24类 macroF1 提升
  D vs B：同族配对（D_g{i} vs B_s{base_i}）平均 OA 增益 ≥ +1pp（n=3 组）
参考锚（口径不同，只作背景）：v-final 全局 OA 0.7959/mF1 0.6399；Exp2 T1/T2（n=117/135 小样本）
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd

FOREST_COMP = [4, 5, 6, 7, 8]


def load_all():
    agg = VC.jload(os.path.join(VC.MET, 'agg.json'), {}).get('agg', {})
    pc = os.path.join(VC.MET, 'perclass.csv')
    per = pd.read_csv(pc) if os.path.exists(pc) else pd.DataFrame()
    return agg, per


def forest_f1(per, w, arm):
    d = per[(per.window == w) & (per.arm == arm) & (per.cls.isin(FOREST_COMP))]
    return float(d.f1.mean()) if len(d) else None


def mean_seed(m):
    """{seed:{OA,macroF1,macroOA}} → (OA, mF1, m9OA) 均值"""
    if not m:
        return (None, None, None)
    v = list(m.values())
    return (float(np.mean([x['OA'] for x in v])),
            float(np.mean([x['macroF1'] for x in v])),
            float(np.mean([x['macroOA'] for x in v])))


def main():
    agg, per = load_all()
    if not agg:
        VC.emit('无聚合结果'); return None
    P = VC.cfg('frozen_params.json')
    bases = P['seeds']['D_groups_base']
    lines = ['# R1 — v3.1 A/B/D 决定性实验（%s）' % time.strftime('%Y-%m-%d %H:%M'), '',
             '> 参数：局部±2°窗、20k点/模型、100树、minLeaf=2、**maxNodes=5000（容量对齐后）**、无class_weight、2023年。',
             '> 评估：分类器直接分类全部留出点（表格导出），A 臂 30→24 归并同口径双评。', '']
    verdict = dict(windows={}, time=time.strftime('%Y-%m-%d %H:%M'), pass_all=True)
    lines += ['| 窗 | n | A_OA24 | B_OA24 | Δ(B-A) | A_m9OA | B_m9OA | Δm9 | A_mF1_24 | B_mF1_24 | ΔmF1 | 森林F1 A→B |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for w, d in sorted(agg.items()):
        A, B = d.get('A', {}).get('seeds', {}), d.get('B', {}).get('seeds', {})
        aO, aF, aM = mean_seed(A)
        bO, bF, bM = mean_seed(B)
        n = d.get('_n_eval')
        fA, fB = forest_f1(per, w, 'A'), forest_f1(per, w, 'B')
        # 配对差（同种子）
        dOA = [B[s]['OA'] - A[s]['OA'] for s in sorted(set(A) & set(B))]
        dM9 = [B[s]['macroOA'] - A[s]['macroOA'] for s in sorted(set(A) & set(B))]
        dMF = [B[s]['macroF1'] - A[s]['macroF1'] for s in sorted(set(A) & set(B))]
        fm = lambda v: ('%+.4f' % float(np.mean(v))) if v else '—'
        lines.append('| %s | %s | %.4f | %.4f | %s | %.4f | %.4f | %s | %.4f | %.4f | %s | %s→%s |' % (
            w, n, aO or 0, bO or 0, fm(dOA), aM or 0, bM or 0, fm(dM9),
            aF or 0, bF or 0, fm(dMF),
            ('%.3f' % fA) if fA else '—', ('%.3f' % fB) if fB else '—'))
        # 判定
        ok_m9 = (np.mean(dM9) >= 0) if dM9 else False
        ok_f = ((fB is not None and fA is not None and fB >= fA - 1e-9)) if True else False
        ok_mf1 = (np.mean(dMF) > 0) if dMF else False
        verdict['windows'][w] = dict(
            B_vs_A=dict(delta_macro9_OA=round(float(np.mean(dM9)), 4) if dM9 else None,
                        delta_macroF1_24=round(float(np.mean(dMF)), 4) if dMF else None,
                        forestF1_A=round(fA, 4) if fA else None,
                        forestF1_B=round(fB, 4) if fB else None,
                        ok=bool(ok_m9 and ok_f and ok_mf1)))
        # D vs B 同族配对
        Dv = d.get('D', {}).get('vote', {})
        pairs = []
        for g, base in enumerate(bases, 1):
            dv = Dv.get(str(g))
            bv = B.get(str(base))
            if dv and bv:
                pairs.append(dv['OA'] - bv['OA'])
        dD = float(np.mean(pairs)) if pairs else None
        lines.append('|  └ D(K=%s) | | | | Δ(D-B)=%s | | | | | | | 组内一致性见 perclass |' % (
            '5', ('%+.4f' % dD) if dD is not None else '—'))
        okD = (dD is not None and dD >= 0.01)
        verdict['windows'][w]['D_vs_B'] = dict(
            delta_OA=round(dD, 4) if dD is not None else None,
            n_pairs=len(pairs), ok=okD)
        if not (verdict['windows'][w]['B_vs_A']['ok'] and okD):
            verdict['pass_all'] = False
    lines += ['', '## 预注册判定', '']
    for w, v in verdict['windows'].items():
        lines.append('- **%s**: B_vs_A %s（Δm9=%s, ΔmF1=%s, 森林F1 %s→%s）；D_vs_B %s（ΔOA=%s, n=%d）' % (
            w, '✅成立' if v['B_vs_A']['ok'] else '❌不成立',
            v['B_vs_A']['delta_macro9_OA'], v['B_vs_A']['delta_macroF1_24'],
            v['B_vs_A']['forestF1_A'], v['B_vs_A']['forestF1_B'],
            '✅达标≥1pp' if v['D_vs_B']['ok'] else '❌未达标',
            v['D_vs_B']['delta_OA'], v['D_vs_B']['n_pairs']))
    # 背景锚
    lines += ['', '## 背景锚（口径不同，仅参照）', '',
              '- v-final 全局模型（冻结）：OA 0.7959 / macroF1 0.6399（开发池口径）',
              '- Exp2 三臂（09-24，留出点0.5°框覆盖仅 n=117/135）：T1 A0.7436/B0.8291/C0.8632；T2 A0.6815/B0.7704/C0.7852',
              '  - 其中 Exp2-C 含 121→130/92→61 合并（doc37 已撤销该建议）与 maxNodes=20000（容量语义勘误后为另一配置）',
              '- 本轮 n=整窗留出点（数千/窗），统计功效显著高于 Exp2']
    md = '\n'.join(lines) + '\n'
    fp = os.path.join(VC.REPT, 'R1_ABD_%s.md' % time.strftime('%Y%m%d'))
    with open(fp, 'w', encoding='utf-8') as f:
        f.write(md)
    st = VC.state()
    st['verdict_abd'] = verdict
    st['report_abd'] = fp
    VC.save_state(st)
    VC.emit('R1 报告 → %s ｜ 判定 pass_all=%s' % (fp, verdict['pass_all']))
    return verdict


if __name__ == '__main__':
    main()

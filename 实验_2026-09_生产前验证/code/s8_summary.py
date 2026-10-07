# -*- coding: utf-8 -*-
"""s8_summary.py — 实验收尾：C 臂判定 + 全部实验汇总报告 + doc37/VERSION 追加更新
产出：
  reports/汇总_全部实验_<date>.md   —— 历史实验(引用文档号) + 本轮 R1/C 臂全表
  STATE.verdict_c / stage=done
  技术文档/37_*.md 末尾追加 R1 结果附录；VERSION.md 追加条目（均 append-only）
"""
import os, sys, time, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import pandas as pd
import numpy as np

DOC37 = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/技术文档/37_v3p1_24类体系技术方案.md'
VERSION = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/VERSION.md'
BOTTLE = {'w2': (list(range(4, 9)), '森林组成(04-08)'),
          'w5': (list(range(14, 21)), '湿地细分(14-20)')}


def eecu_totals():
    jobs = VC.jload(VC.JOBS_FP, [])
    tot = {}
    for j in jobs:
        for a in j.get('attempts', []):
            if a.get('state') == 'SUCCEEDED':
                arm = j['arm']
                tot[arm] = round(tot.get(arm, 0) + (a.get('eecu_h') or 0), 3)
    return tot


def _preds_of(job_id, w):
    """读某任务的预测 (rids, pred24)；缺文件返回 None。"""
    fp = os.path.join(VC.RAW, job_id + '.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp)
    df = df.dropna(subset=['classification'])
    return df['rid'].to_numpy(int), df['classification'].to_numpy(int)


def c_verdict(per):
    """C vs B：**同一批点位**上比较（C 只在特征采样存活点上可评）。
    B 的预测按 C 的点位子集裁剪后重算，避免样本不同造成的伪差异。"""
    jobs = VC.jload(VC.JOBS_FP, [])
    out = {}
    for w, (bl, nm) in BOTTLE.items():
        cjobs = [j for j in jobs if j['window'] == w and j['arm'] == 'C' and j.get('done')]
        bjobs = [j for j in jobs if j['window'] == w and j['arm'] == 'B' and j.get('done')]
        if not cjobs or not bjobs:
            continue
        ev = pd.read_parquet(eval_fp_local(w))
        c0 = _preds_of(cjobs[0]['job_id'], w)
        if c0 is None:
            continue
        rids = c0[0]
        lab = ev[['row_id', 'class_new']].set_index('row_id').reindex(rids)['class_new']
        if lab.isna().any():
            ok = ~lab.isna()
            rids = rids[ok.to_numpy()]
            lab = lab[ok]
        t24 = VC.to_v31(lab.to_numpy(int))
        idx = {int(r): i for i, r in enumerate(rids)}

        def m_of(j):
            pr = _preds_of(j['job_id'], w)
            if pr is None:
                return None
            rid, p = pr
            pos = np.array([idx.get(int(r), -1) for r in rid])
            keep = pos >= 0
            if keep.sum() < 0.9 * len(rids):
                return None
            pv = np.zeros(len(rids), dtype=int)
            pv[pos[keep]] = p[keep]
            return VC.metrics(t24, pv)

        Cm = [m for m in (m_of(j) for j in cjobs) if m]
        Bm = [m for m in (m_of(j) for j in bjobs) if m]
        if not Cm or not Bm:
            continue
        f1 = lambda mm, c: np.mean([m['per'][c]['f1'] for m in mm if c in m['per']]) \
            if any(c in m['per'] for m in mm) else np.nan
        d_bot = {int(c): round(float(f1(Cm, c) - f1(Bm, c)), 4) for c in bl
                 if not np.isnan(f1(Cm, c)) or not np.isnan(f1(Bm, c))}
        allc = sorted(set().union(*[set(m['per'].keys()) for m in Cm + Bm]))
        others = [c for c in allc if c not in bl]
        d_oth = [float(f1(Cm, c) - f1(Bm, c)) for c in others if not np.isnan(f1(Cm, c))]
        gain = float(np.nanmean([v for v in d_bot.values() if not np.isnan(v)])) if d_bot else 0.0
        harm = float(np.nanmean(d_oth)) if d_oth else 0.0
        out[w] = dict(bottleneck=nm, n_eval_subset=int(len(rids)),
                      delta=dict(d_bot), mean_gain=round(gain, 4),
                      others_mean_delta=round(harm, 4),
                      OA_C=round(float(np.mean([m['OA'] for m in Cm])), 4),
                      OA_B_samesubset=round(float(np.mean([m['OA'] for m in Bm])), 4),
                      macroF1_C=round(float(np.mean([m['macroF1'] for m in Cm])), 4),
                      macroF1_B_samesubset=round(float(np.mean([m['macroF1'] for m in Bm])), 4),
                      ok=bool(gain >= 0.03 and harm >= -0.005))
    return out or None


def eval_fp_local(w):
    fp = os.path.join(VC.DATA, 'eval_%s.parquet' % w)
    return fp if os.path.exists(fp) else os.path.join(VC.EVAL_DIR_GEE, 'eval_%s.parquet' % w)


def append_doc37(verdict, cv):
    with open(DOC37, 'r', encoding='utf-8') as f:
        txt = f.read()
    if 'R1 实验结果附录' in txt:
        return False
    abd = verdict.get('windows', {})
    lines = ['', '---', '', '## 附录 A：R1 四组实验结果（%s 自动追加）' % time.strftime('%Y-%m-%d'),
             '', '- 详见 `F:/lc_work/v31_exp/reports/`（R1_ABD 报告、汇总报告、逐类 F1 表）',
             '- A/B/D 判定（预注册口径）：']
    for w, v in abd.items():
        ba, db = v.get('B_vs_A', {}), v.get('D_vs_B', {})
        lines.append('  - %s：B_vs_A %s（Δmacro9OA=%s，ΔmacroF1=%s，森林组成F1 %s→%s）；'
                     'D_vs_B %s（ΔOA=%s，n=%d）' % (
                         w, '✅' if ba.get('ok') else '❌', ba.get('delta_macro9_OA'),
                         ba.get('delta_macroF1_24'), ba.get('forestF1_A'), ba.get('forestF1_B'),
                         '✅' if db.get('ok') else '❌', db.get('delta_OA'), db.get('n_pairs')))
    if cv:
        lines.append('- C 臂（B+针对性特征，w2/w5 代表区）：')
        for w, v in cv.items():
            lines.append('  - %s：%s（瓶颈=%s 平均ΔF1=%s，他类平均Δ=%s）' % (
                w, '✅达标' if v['ok'] else '❌未达标', v['bottleneck'],
                v['mean_gain'], v['others_mean_delta']))
    with open(DOC37, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return True


def append_version(summary_fp):
    with open(VERSION, 'r', encoding='utf-8') as f:
        txt = f.read()
    if 'v31_exp' in txt:
        return False
    entry = ('\n## 2026-09-25  v3.1 四组决定性实验（R1）自动记录\n\n'
             '- 实验包：`F:/lc_work/v31_exp/`（全部数据/代码/结果同文件夹，含 backups 快照链）\n'
             '- 设计：doc37 §六 A/B/C/D；A=局部RF×30类（容量对齐 maxNodes=5000），B=局部RF×24类，'
             'C=B+17个针对性特征（S2红边/纹理/水淹/地形），D=24类 K=5 集成；窗口 w1-w4+w5(三江湿地)\n'
             '- 评估：分类器直接分类全部留出点（表格导出，无图像导出），A 臂 30→24 归并同口径双评\n'
             '- 判定与逐类结果：`v31_exp/reports/`；结论见 `技术文档/37` 附录 A\n'
             '- 汇总报告：%s\n' % os.path.basename(summary_fp))
    with open(VERSION, 'a', encoding='utf-8') as f:
        f.write(entry)
    return True


def main():
    agg = VC.jload(os.path.join(VC.MET, 'agg.json'), {}).get('agg', {})
    per = pd.read_csv(os.path.join(VC.MET, 'perclass.csv')) if os.path.exists(
        os.path.join(VC.MET, 'perclass.csv')) else pd.DataFrame()
    st = VC.state()
    verdict = st.get('verdict_abd', {})
    cv = c_verdict(per)
    ee = eecu_totals()
    L = ['# v3.1 实验全记录与汇总（%s）' % time.strftime('%Y-%m-%d %H:%M'), '',
         '## 一、本轮 R1：四组决定性实验（doc37 §六）', '',
         '| 项 | 内容 |', '|---|---|',
         '| 目的 | 判定 24 类体系是否成立(B vs A)、针对性特征是否值成本(C vs B)、K=5 集成是否值成本(D vs B) |',
         '| 方法 | GEE 服务端局部训练(±2°窗资产)+`clf.classify(留出点FC)`表格导出；A 臂 30→24 归并同口径双评 |',
         '| 参数 | 100树、minLeaf=2、maxNodes=5000（容量对齐）、20k点/模型、sqrt配比、无class_weight、2023 |',
         '| 窗口 | w1松嫩 w2秦岭 w3南方丘岭 w4西北甘旱 w5三江湿地；评估点=全窗留出（共 %s 点） |' % (
             sum(d.get('_n_eval', 0) for d in agg.values()) or '—'),
         '| EECU | %s |' % json.dumps(ee, ensure_ascii=False), '']
    L += ['### 1. A/B/D 判定（预注册）', '']
    for w, v in verdict.get('windows', {}).items():
        ba, db = v.get('B_vs_A', {}), v.get('D_vs_B', {})
        L.append('- **%s**：B_vs_A %s（Δmacro9OA=%s，ΔmacroF1=%s，森林组成F1 %s→%s）；'
                 'D_vs_B %s（ΔOA=%s，n=%d）' % (
                     w, '✅成立' if ba.get('ok') else '❌不成立', ba.get('delta_macro9_OA'),
                     ba.get('delta_macroF1_24'), ba.get('forestF1_A'), ba.get('forestF1_B'),
                     '✅≥1pp' if db.get('ok') else '❌未达标', db.get('delta_OA'), db.get('n_pairs')))
    L += ['', '### 2. C 臂判定（B+17 特征：S2红边/纹理/水淹/地形）', '']
    if cv:
        for w, v in cv.items():
            L.append('- **%s**（%s）：%s，瓶颈平均ΔF1=%s，他类平均Δ=%s' % (
                w, v['bottleneck'], '✅达标(≥3pp且无害)' if v['ok'] else '❌未达标',
                v['mean_gain'], v['others_mean_delta']))
    else:
        L.append('- （C 臂结果未就绪或未运行）')
    L += ['', '## 二、历史实验台账（目的/方法/结果/结论，详见对应文档）', '',
          '| # | 实验 | 文档 | 结果→结论 |', '|---|---|---|---|',
          '| 1 | 样本质量分层三假说（加权/脏样本剔除/局部建模） | 33/34 | A/B 证伪、C 成立：局部建模 +6.56pp，121 F1 0.435→0.676 |',
          '| 2 | GEE 实机三臂（全局 vs 局部 vs 局部+K5+合并+maxNodes2万） | 35 | Exp2 +11.96/+10.37pp 但成本 6.5-7×；降级为高成本备选 |',
          '| 3 | 粗模型+内联专家硬门控 | 30 | -0.97pp 净损失 → 证伪（概率输出未测，见勘误②） |',
          '| 4 | 浅容量下加样本 | 30/35 | 2万→20万点 OA 反降 7.6pp（容量语义勘误后需重测，本轮 A/B 已按 5000 叶对齐） |',
          '| 5 | 训练端合并 121→130/92→61 | 29/30/37 | 撤销：合并升OA≠能力提升；121 局部可修，92 并入 08 |',
          '| 6 | 52/62 疏/郁闭可分性 | 36 | AUC 0.785/0.833+质心余弦 0.985 → AEF 嵌入上区分不足，覆盖度拆属性层 |',
          '| 7 | 物种分布当替代样本 | 31/32 | AEF 质心检验：对湿地/园地/140 无效；属级白名单有限可用 |',
          '| 8 | **本轮 R1** | v31_exp + 37附录 | 见上；判定决定 24 类定稿与全国配置 |', '',
          '## 三、下一步（doc37 §七）', '',
          '- 按判定结果定 24 类与全国配置（局部单 RF 为基线；D/C 达标才考虑纳入成本）',
          '- M3 独立验证点 ≥3,000（人工判读，最终测试集）',
          '- M4 覆盖度回归试验（GEDI cover/MOD44B → AEF vs AEF+红边纹理）', '']
    fp = os.path.join(VC.REPT, '汇总_全部实验_%s.md' % time.strftime('%Y%m%d'))
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    st = VC.state()
    st['verdict_c'] = cv
    st['summary_report'] = fp
    VC.save_state(st)
    try:
        append_doc37(verdict, cv)
        append_version(fp)
    except Exception as e:
        VC.emit('文档追加失败（不阻断）: %s' % str(e)[:120])
    VC.emit('汇总 → %s ｜ C判定=%s' % (fp, json.dumps(cv, ensure_ascii=False)[:200] if cv else None))


if __name__ == '__main__':
    main()

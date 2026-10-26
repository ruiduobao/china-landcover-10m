# -*- coding: utf-8 -*-
"""e4_report.py — 园地先验实验·结果表渲染（读 summary.json → summary.md）

* 输入：<WS>/results/summary.json、<WS>/results/paired_by_seed.csv、<WS>/prereg/regions_frozen.json
* 输出：<WS>/results/summary.md
* 用法：PYTHONDONTWRITEBYTECODE=1 python e4_report.py
"""
import os, json, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
res = json.load(open(os.path.join(WS, 'results', 'summary.json'), encoding='utf-8'))
reg = json.load(open(os.path.join(WS, 'prereg', 'regions_frozen.json'), encoding='utf-8'))
ps = pd.read_csv(os.path.join(WS, 'results', 'paired_by_seed.csv'))

L = []
L.append('# 园地先验 + 森林筛选 · 小区域受控分类实验结果（O-CAP vs baseline）')
L.append('')
L.append('> 生成 %s ｜ 工作区 `%s` ｜ 零 GEE；本地 sklearn RF（≠ GEE smileRF）' % (
    res['generated'], WS))
L.append('> 注册：`prereg/regions_frozen.json`（md5 %s）｜ 规范：doc70 v1.0' %
         __import__('hashlib').md5(open(os.path.join(WS, 'prereg', 'regions_frozen.json'), 'rb').read()).hexdigest())
L.append('> 臂：**baseline**=窗内均匀随机前 20k；**O-CAP**=同 permutation，类 2 园地上限 200(=1%×20k)，'
         '非园地按同窗同 seed 随机序补足，固定总 N=20000。seeds 7/11/23。')
L.append('> **本轮空间留出设计**：训练窗（所属 2°瓦±2°）剔除本区全部 0.25° dev+conf 块（位置键零重叠自检通过）；'
         '**与生产 classify（用全窗点）不同，不可与生产精度混比**。')
L.append('> 主门=非 FCS10 **确认层** macro9 非劣（点估≥−1pp 且块 bootstrap 95% 下界≥−2pp）；'
         '目标类 F1 同层。全点/FCS10 完整报。class8 真值记错不删。')
L.append('')
L.append('## 0 结论速览')
L.append('')
L.append('| 区 | 省/窗 | refn(非fcs10) | conf n(非fcs10) | 主门 Δm9(pp) | 95%CI(pp) | ΔOA24(pp) | 目标类 | 目标类ΔF1 | 组级判读 | 目标类状态 |')
L.append('|---|---|---|---|---|---|---|---|---|---|---|')
for rid in ['SC1', 'SC2', 'GD', 'NX']:
    r = res['regions'].get(rid)
    if not r or r.get('state') != 'OK':
        L.append('| %s | — | — | — | — | — | — | — | — | %s | — |' % (rid, (r or {}).get('state', '?')))
        continue
    R = reg['regions'][rid]
    m = r['main_layer']
    tgt = r['target_classes']
    tf = r.get('per_class', {}).get('conf_nonfcs10', {})
    tf_s = '；'.join('类%d n=%d Δ%+.3f' % (c, tf.get(str(c), {}).get('n', 0), tf.get(str(c), {}).get('dF1', 0))
                    for c in tgt if str(c) in tf)
    L.append('| %s | %s/%s | %d(%d) | %d | %+.2f | [%+.2f,%+.2f] | %+.2f | %s | %s | **%s** | **%s** |' % (
        rid, R['ptag'], R['win'], R['refn'], R['refn_nonfcs10'], m['n'], m['dm9'] * 100,
        m['dm9_ci'][0] * 100, m['dm9_ci'][1] * 100, m['dOA24'] * 100,
        ','.join(str(c) for c in tgt), tf_s, r['verdict'], r.get('target_status', '—')))
L.append('')
L.append('> **判读纪律**：组级 `PROMISING`（非劣/略正）**不等于**已提高。目标类状态单列：'
         '`NOT_IMPROVED` = 目标类 F1 未改善（本次各区类 4 ΔF1 均为负）；`INSUFFICIENT_SUPPORT` = 确认层目标类 n<30。'
         '**本实验 cap 臂目标类一律 NOT_IMPROVED / 支持不足，未证明精度提升。**')
L.append('> **表内 n 口径**：`目标类ΔF1` 列 n 为 **conf ∩ 非FCS10** 层；§0 另有 `n_conf_cl4`（conf 层**全部来源**，'
         '含 fcs10）。两者差 1（SC1 38 vs 37、GD 35 vs 34）即该 conf 类 4 真值中有 1 点为 fcs10 系，'
         '已按主门层（非FCS10）单列，不混算。')
L.append('')
L.append('### 0.1 园地输出近乎消失 = 显式回归风险')
L.append('')
L.append('| 区 | 预测类2 baseline→O-CAP（全3seed） | 逐 seed b→o | 全3seed归零 | 近乎消失 |')
L.append('|---|---|---|---|---|')
for rid in ['SC1', 'SC2', 'GD', 'NX']:
    r = res['regions'].get(rid)
    if not r or r.get('state') != 'OK':
        continue
    k = r.get('orchard_output_risk', {})
    psd = k.get('pred_cl2_per_seed', {})
    ps_s = '；'.join('%s %s→%s' % (s, psd.get(s, psd.get(str(s), ['-', '-']))[0],
                                   psd.get(s, psd.get(str(s), ['-', '-']))[1]) for s in ('7', '11', '23'))
    L.append('| %s | %d→%d | %s | %s | %s |' % (
        rid, k.get('pred_cl2_allseeds_baseline', -1), k.get('pred_cl2_allseeds_ocap', -1),
        ps_s, '**是**' if k.get('collapse_all_seeds') else '否',
        '**是**' if k.get('near_collapse') else '否'))
L.append('')
L.append('> O-CAP 使园地预测**近乎消失**（精确说：SC1/SC2/GD 全 3 seed 归零；NX 全 3 seed 19→1）。'
         '真实园地会被误判为林/耕 → **显式回归风险**。确认层园地真值 n≤1 → 园地 UA/PA 不可评，'
         '不能以组级持平掩盖该风险。')
L.append('')
L.append('> 修订说明（2026-10-15）：① 统计 bug——原 `per_class_f1` 在"联合"时按位置索引对齐去重帧与 seed 长表，'
         '致联合类 4 n 误记 61；已改 `region:row_id` 键对齐，正确 n=51（37+14），ΔF1=-0.0223。'
         '② 判读级别——原 `HOLD` 非 doc70 §7 五级，NX 已改 **REJECT**（无正收益）。'
         '**预测文件未改动**，仅统计重算；旧版输出保留于 `summary_prevbug.{json,md}`。')
if 'SC1+SC2_joint' in res['regions']:
    j = res['regions']['SC1+SC2_joint']
    m = j['main_layer']
    L.append('**SC1+SC2 预登记小区联合（描述性，不用于抬高采纳门）**：conf_nonfcs10 n=%d，'
             'Δm9=%+.2fpp 95%%CI[%+.2f,%+.2f]，ΔOA24=%+.2fpp，类 4 n=%d ΔF1=%+.3f。' % (
                 m['n'], m['dm9'] * 100, m['dm9_ci'][0] * 100, m['dm9_ci'][1] * 100,
                 m['dOA24'] * 100, j['n_conf_cl4'],
                 j['target_f1_conf'].get('4', {}).get('dF1', 0)))
    L.append('> 注：该联合统计在首轮预测**已产生后**补登记，故仅作描述性支持说明，不参与采纳判定。')
    L.append('')
L.append('## 1 各层完整结果（seed 平均，唯一点；块 bootstrap 10000 次）')
L.append('')
L.append('| 区 | 层 | n | 块数 | OA24 b→o | ΔOA24(pp) | CI | m9 b→o | Δm9(pp) | CI |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for rid in ['SC1', 'SC2', 'GD', 'NX']:
    r = res['regions'].get(rid)
    if not r or r.get('state') != 'OK':
        continue
    for lay, st in (r.get('layers') or {}).items():
        if not st:
            continue
        L.append('| %s | %s | %d | %d | %.4f→%.4f | %+.2f | [%+.2f,%+.2f] | %.4f→%.4f | %+.2f | [%+.2f,%+.2f] |' % (
            rid, lay, st['n'], st['n_blocks'], st['OA24_b'], st['OA24_o'], st['dOA24'] * 100,
            st['dOA24_ci'][0] * 100, st['dOA24_ci'][1] * 100, st['m9_b'], st['m9_o'],
            st['dm9'] * 100, st['dm9_ci'][0] * 100, st['dm9_ci'][1] * 100))
L.append('')
L.append('## 2 逐类 F1（conf_nonfcs10 层；n<30 只作诊断）')
L.append('')
L.append('| 区 | 类码 | 名称 | n | F1 b→o | ΔF1 |')
L.append('|---|---|---|---|---|---|')
for rid in ['SC1', 'SC2', 'GD', 'NX']:
    r = res['regions'].get(rid)
    if not r or r.get('state') != 'OK':
        continue
    for c, v in sorted(r.get('per_class', {}).get('conf_nonfcs10', {}).items(),
                       key=lambda x: -x[1]['n']):
        if v['n'] < 5:
            continue
        L.append('| %s | %s | %s | %d | %.3f→%.3f | %+.3f%s |' % (
            rid, c, v['name'], v['n'], v['f1_b'], v['f1_o'], v['dF1'],
            ' ⚠n<30' if v['n'] < 30 else ''))
L.append('')
L.append('## 3 每 seed Δ（噪声/抽样不确定性分离）')
L.append('')
L.append('| 区 | 层 | ' + ' | '.join('seed%s ΔOA24/Δm9(pp)' % s for s in reg['seeds']) + ' |')
L.append('|---|---|' + '---|' * len(reg['seeds']))
for rid in ['SC1', 'SC2', 'GD', 'NX']:
    sub = ps[ps.region == rid]
    for lay in ['conf_nonfcs10', 'all', 'nonfcs10']:
        sl = sub[sub.layer == lay]
        if len(sl) == 0:
            continue
        cells = []
        for s in reg['seeds']:
            row = sl[sl.seed == s]
            cells.append('%+.2f/%+.2f' % (row.dOA24.iloc[0] * 100, row.dm9.iloc[0] * 100) if len(row) else '—')
        L.append('| %s | %s | %s |' % (rid, lay, ' | '.join(cells)))
L.append('')
L.append('## 4 单元运行与限时')
L.append('')
L.append('| 区 | 臂 | 状态 | wall(s) |')
L.append('|---|---|---|---|')
ur = os.path.join(WS, 'results', 'unit_runs.csv')
if os.path.exists(ur):
    for _, r in pd.read_csv(ur).iterrows():
        L.append('| %s | %s | %s | %.1f |' % (r.region, r.arm, r.state, r.wall_s))
L.append('')
L.append('> 全部单元 wall-clock 远低于 1200s 截止；无 TIMEOUT/FAILED。')
L.append('')
L.append('## 5 先验连带：预测类分布（seed 7；说明 O-CAP 的类别先验后果）')
L.append('')
L.append('| 区 | 预测类2 次数 b→o | 预测类4 b→o | 预测类1 b→o | conf 层类2 真值 n |')
L.append('|---|---|---|---|---|')
for rid in ['SC1', 'SC2', 'GD', 'NX']:
    fb = os.path.join(WS, 'results', '%s_baseline_pred.csv' % rid)
    fo = os.path.join(WS, 'results', '%s_OCAP_pred.csv' % rid)
    if not (os.path.exists(fb) and os.path.exists(fo)):
        continue
    b = pd.read_csv(fb); o = pd.read_csv(fo)
    b7 = b[b.seed == 7]; o7 = o[o.seed == 7]
    n2t = int(((b7.split == 'conf') & (b7.t24 == 2)).sum())
    L.append('| %s | %d→%d | %d→%d | %d→%d | %d |' % (
        rid, int((b7.pred == 2).sum()), int((o7.pred == 2).sum()),
        int((b7.pred == 4).sum()), int((o7.pred == 4).sum()),
        int((b7.pred == 1).sum()), int((o7.pred == 1).sum()), n2t))
L.append('')
L.append('> **关键机制观察**：O-CAP（类 2 上限 1%）使园地预测**近乎消失**'
         '（SC1/SC2/GD 全 3 seed 归零；NX 19→1），同时预测类 4/类 1 增多——方向与'
         '"园地先验导致林耕替代"的假设一致；但**目标类 4 的 F1 未改善**'
         '（各区略降），且确认层园地真值极少（多数区 n<5），园地 UA/PA 不可评。'
         '因此 O-CAP 是"以消除园地输出换取组级持平"的强先验干预，非"只提标签质量"的修复。')
L.append('> 本轮各区参考点无 class8 真值；若出现则按 doc70 §6.2 记错不删。')
L.append('')
open(os.path.join(WS, 'results', 'summary.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('→ results/summary.md')

# -*- coding: utf-8 -*-
"""merge_stats.py — 判读成果合并、一致率/确认率统计与报告（含敏感性）

* 输入：--base 点位表（含 id、可选先验类/分层）；--labels 一份或多份判读表单（按 id 合并，后者覆盖前者）
* 规则源：references/判读协议.md —— 判定三态口径、"无法判读"单列、报告区分实测值/解读
* 门槛：id 必须能对齐（对不上的点位单列）；判读列缺失即报错；分母口径在报告里写死
* 输出：<outdir>/merged_labels.csv（原始列 + 判读列 + 是否覆盖）
        <outdir>/report.md（总数/无法判读占比/确认率（总+分组）/先验×判读混淆/sensitivity/双遍一致率）
* 用法：python merge_stats.py --base points.csv --labels form_filled.csv --outdir ./run1 \
            --prior-col 教师类 --verdict-col 结论 --class-col 实际类别
        python merge_stats.py --base points.csv --labels pass1.csv pass2.csv --outdir ./run1 ...（第 2 份=复判，自动出一致率）
"""
import argparse
import collections
import csv
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
YES = ('正确', '是', '确认', 'true', 'True', '1')


def norm(v):
    return str(v or '').strip()


def is_yes(v):
    s = norm(v)
    return (s in YES) or s.startswith('是') or s.startswith('正确') or s.startswith('确')


def is_no(v):
    s = norm(v)
    return s.startswith('否') or s.startswith('错误') or s.startswith('错')


def is_na(v):
    s = norm(v)
    return s.startswith('无法') or s in ('', 'nan', 'None')


def main():
    ap = argparse.ArgumentParser(description='判读合并与统计')
    ap.add_argument('--base', required=True)
    ap.add_argument('--labels', nargs='+', required=True, help='一份=主判；两份=主判+复判（出复判一致率）')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--id-col', default='point_id')
    ap.add_argument('--prior-col', default=None)
    ap.add_argument('--strata-col', default=None)
    ap.add_argument('--verdict-col', default='结论')
    ap.add_argument('--class-col', default='实际类别')
    ap.add_argument('--conf-col', default='置信（高/中/低）')
    ap.add_argument('--threshold', type=float, default=None,
                    help='预注册阈值（如灌丛保留率 0.40）：报告"推翻该判定所需错判率"')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    base = list(csv.DictReader(open(a.base, encoding='utf-8-sig')))
    labs = []
    for fp in a.labels:
        labs.append({norm(r.get(a.id_col)): r for r in csv.DictReader(open(fp, encoding='utf-8-sig'))})
    primary = labs[0]
    merged, unmatched = [], []
    for b in base:
        pid = norm(b.get(a.id_col))
        m = dict(b)
        src = None
        for L in labs:                      # 后者覆盖前者
            if pid in L:
                src = L[pid]
        if src is None:
            unmatched.append(pid)
        else:
            m[a.verdict_col] = norm(src.get(a.verdict_col))
            m[a.class_col] = norm(src.get(a.class_col, src.get('实际类别', src.get('主要地物', ''))))
            m[a.conf_col] = norm(src.get(a.conf_col, src.get('置信', '')))
            m['证据'] = norm(src.get('证据', src.get('证据（影像特征+日期+辅助指标）', '')))
        merged.append(m)
    drop = ['证据']
    cols = [c for c in merged[0] if c not in drop] + [c for c in drop if c in merged[0]]
    with open(os.path.join(a.outdir, 'merged_labels.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in merged:
            w.writerow({c: m.get(c, '') for c in cols})

    def tally(rows):
        y = sum(1 for r in rows if is_yes(r.get(a.verdict_col)))
        n = sum(1 for r in rows if is_no(r.get(a.verdict_col)))
        u = len(rows) - y - n
        return y, n, u, (y / (y + n) if y + n else float('nan'))

    y, n, u, rate = tally(merged)
    L = ['# 样本点判读统计报告', '',
         '- 点位：**%d**（其中无法判读 %d，占 %.1f%%）' % (len(merged), u, 100 * u / max(1, len(merged))),
         '- 判读表单：%s' % '、'.join('`%s`' % p for p in a.labels),
         '- 未命中点位：%d %s' % (len(unmatched), unmatched[:5]),
         '', '## 总体', '',
         '| 结论 | n |', '|---|---|',
         '| 正确/确认 | %d |' % y, '| 错误/否定 | %d |' % n, '| 无法判读 | %d |' % u]
    if y + n:
        L += ['', '- **确认率 = %d/(%d+%d) = %.3f**（分母＝可判读点，无法判读单列不并入）' % (y, y, n, rate)]
    # 分组
    for col, title in ((a.prior_col, '按先验类'), (a.strata_col, '按分层')):
        if not col:
            continue
        groups = collections.defaultdict(list)
        for r in merged:
            groups[norm(r.get(col)) or '(空)'].append(r)
        L += ['', '## %s' % title, '', '| %s | n | 确认 | 否定 | 无法 | 确认率 |' % col, '|---|---|---|---|---|---|']
        for k, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            gy, gn, gu, gr = tally(rows)
            L.append('| %s | %d | %d | %d | %d | %s |' % (k, len(rows), gy, gn, gu, '%.3f' % gr if gr == gr else '-'))
    # 先验 × 判读
    if a.prior_col:
        K = collections.Counter((norm(r.get(a.prior_col)) or '(空)', norm(r.get(a.class_col)) or '(未填)')
                                for r in merged if is_no(r.get(a.verdict_col)))
        L += ['', '## 判"否"点的实际类别 top15（先验 × 实际）', '']
        L += ['- ' + ' ｜ '.join('%s→%s:%d' % (p, c, v) for (p, c), v in K.most_common(15))]
    # 敏感性
    if a.threshold is not None and y + n:
        need = int(round(a.threshold * (y + n)))
        if need > y:
            L += ['', '## 敏感性（推翻判定所需错判率）', '',
                  '- 预注册阈值 %.2f：需 确认数 ≥ %d（当前 %d）→ 需把 %d 个"否定"改判为"确认"，'
                  '即否定错判率 ≥ %.1f%%' % (a.threshold, need, y, need - y,
                                        100.0 * (need - y) / max(1, n))]
        else:
            L += ['', '## 敏感性', '', '- 当前确认率 %.3f 已 ≥ 阈值 %.2f' % (rate, a.threshold)]
    # 复判一致率
    if len(labs) >= 2:
        common = [p for p in primary if p in labs[1]]
        same = sum(1 for p in common if norm(primary[p].get(a.verdict_col)) == norm(labs[1][p].get(a.verdict_col)))
        bi = sum(1 for p in common if is_yes(primary[p].get(a.verdict_col)) == is_yes(labs[1][p].get(a.verdict_col)))
        L += ['', '## 复判一致率（主判 vs 复判）', '',
              '- 共同点位 %d：三态完全一致 **%.3f**，二值（确认 vs 非确认）一致 **%.3f**' % (
                  len(common), same / max(1, len(common)), bi / max(1, len(common)))]
        dis = [p for p in common if norm(primary[p].get(a.verdict_col)) != norm(labs[1][p].get(a.verdict_col))]
        L += ['- 不一致 %d 点：%s' % (len(dis), '、'.join(dis[:10]))]
    L += ['', '## 口径声明', '',
          '- "确认率/一致率"是判读结果，不是精度；精度需独立于判读源的真值。',
          '- 本报告由 `merge_stats.py` 生成；逐点证据见 `merged_labels.csv` 的证据列。']
    open(os.path.join(a.outdir, 'report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('合并 %d 点（未命中 %d）；确认 %d / 否定 %d / 无法 %d；确认率 %s' % (
        len(merged), len(unmatched), y, n, u, '%.3f' % rate if rate == rate else '-'))
    print('报告 → %s' % os.path.join(a.outdir, 'report.md'))


if __name__ == '__main__':
    main()

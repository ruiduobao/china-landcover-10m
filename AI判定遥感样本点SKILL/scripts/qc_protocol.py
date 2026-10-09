# -*- coding: utf-8 -*-
"""qc_protocol.py — 判读质控协议：复判抽样 / 随机"否定"审计 / 最小复核清单

* 输入：`sample` 子命令：--labels 首遍判读表单（含结论列）[--strata-col 分层列]；
        `review` 子命令：--labels 主判 --recheck 复判（第二遍）
* 规则源：references/判读协议.md —— 复判 ≥10% 且覆盖全部低置信/无法判读；"否定"按层各抽 N 做审计；
        分歧 + 无法判读 + 低置信"确认" → 最小复核清单（人工只核这些）
* 门槛：随机种子固定（--seed 可复现）；抽样不重复；批次清单只含 point_id（盲判）
* 输出：sample → qc/复判批次.csv、qc/审计批次.csv、qc/复判_blind.csv、qc/审计_blind.csv、qc/质控摘要.md
        review → qc/最小复核清单.csv（含不一致原因）、qc/复核_blind.csv、qc/review_summary.md
* 用法：python qc_protocol.py sample --labels pass1.csv --outdir ./qc --double-pass 0.1 --audit-per-strata 20 --strata-col w
        python qc_protocol.py review --labels pass1.csv --recheck pass2.csv --outdir ./qc
"""
import argparse
import collections
import csv
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')


def is_yes(v):
    s = str(v or '').strip()
    return s.startswith('是') or s.startswith('正确') or s.startswith('确')


def is_no(v):
    s = str(v or '').strip()
    return s.startswith('否') or s.startswith('错误') or s.startswith('错')


def is_na(v):
    s = str(v or '').strip()
    return s.startswith('无法') or s in ('', 'nan')


def write_ids(fp, ids, label='point_id'):
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq', label])
        for i, p in enumerate(ids):
            w.writerow([i + 1, p])


def cmd_sample(a):
    rows = list(csv.DictReader(open(a.labels, encoding='utf-8-sig')))
    rnd = random.Random(a.seed)
    idc = a.id_col
    lowconf = [r for r in rows if str(r.get(a.conf_col, '')).strip() in ('低', 'low')]
    na = [r for r in rows if is_na(r.get(a.verdict_col))]
    conf_yes = [r for r in rows if is_yes(r.get(a.verdict_col))]
    no_rows = [r for r in rows if is_no(r.get(a.verdict_col))]
    # 复判：随机比例 + 全部低置信 + 全部无法判读
    k = max(1, int(round(len(rows) * a.double_pass)))
    rand_ids = [r[idc] for r in rnd.sample(rows, min(k, len(rows)))]
    recheck = list(dict.fromkeys(rand_ids + [r[idc] for r in lowconf] + [r[idc] for r in na]))
    # 审计：各层随机"否定"
    audit = []
    if a.strata_col:
        g = collections.defaultdict(list)
        for r in no_rows:
            g[str(r.get(a.strata_col, '')) or '(空)'].append(r[idc])
        for k2, ids in sorted(g.items()):
            audit += rnd.sample(ids, min(a.audit_per_strata, len(ids)))
    else:
        audit = rnd.sample([r[idc] for r in no_rows], min(a.audit_per_strata, len(no_rows)))
    os.makedirs(a.outdir, exist_ok=True)
    write_ids(os.path.join(a.outdir, '复判批次.csv'), recheck, idc)
    write_ids(os.path.join(a.outdir, '审计批次.csv'), audit, idc)
    write_ids(os.path.join(a.outdir, '复判_blind.csv'), recheck, idc)
    write_ids(os.path.join(a.outdir, '审计_blind.csv'), audit, idc)
    L = ['# 质控抽样摘要', '',
         '- 判读表：`%s`（%d 点）' % (a.labels, len(rows)),
         '- 结论构成：确认 %d ｜ 否定 %d ｜ 无法判读 %d' % (len(conf_yes), len(no_rows), len(na)),
         '', '## 复判批次', '',
         '- 抽样：随机 %.0f%%（%d 点）+ 全部低置信（%d）+ 全部无法判读（%d）→ 合计 **%d 点**（去重后）' % (
             100 * a.double_pass, len(rand_ids), len(lowconf), len(na), len(recheck)),
         '', '## "否定"审计批次', '',
         '- 方式：%s，每层 %d 点 → 合计 **%d 点**' % (
             ('按 `%s` 分层随机' % a.strata_col) if a.strata_col else '整体随机', a.audit_per_strata, len(audit)),
         '', '## 判据（默认，可预注册覆盖）', '',
         '- 复判三态一致率 <0.90 → 复核全部不一致点并抽更多复判；',
         '- "否定"审计准确率 <0.90 → 否定结论不可信，需扩大审计/重判；',
         '- 低置信"确认"点进入最小复核清单（它们决定分母）。',
         '', '## 下一步', '',
         '1. 用 `复判_blind.csv` / `审计_blind.csv`（只含编号）做**盲判**：',
         '   `fetch_chips.py --points 复判_blind.csv …` → `build_sheets.py` → 判读填表；',
         '2. `python qc_protocol.py review --labels 首遍.csv --recheck 复判填完.csv --outdir ' + a.outdir + '`；',
         '3. `python merge_stats.py --base 点位.csv --labels 首遍.csv 复判.csv …` 出统计。']
    open(os.path.join(a.outdir, '质控摘要.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('复判 %d 点、审计 %d 点 → %s（盲判清单已去标签）' % (len(recheck), len(audit), a.outdir))


def cmd_review(a):
    p1 = {str(r.get(a.id_col)).strip(): r for r in csv.DictReader(open(a.labels, encoding='utf-8-sig'))}
    p2 = {str(r.get(a.id_col)).strip(): r for r in csv.DictReader(open(a.recheck, encoding='utf-8-sig'))}
    items = []
    for pid, r2 in p2.items():
        r1 = p1.get(pid)
        if not r1:
            continue
        v1, v2 = str(r1.get(a.verdict_col, '')).strip(), str(r2.get(a.verdict_col, '')).strip()
        if v1 != v2:
            items.append((pid, '复判不一致：主判=%s / 复判=%s' % (v1, v2), r2.get(a.class_col, '')))
    for pid, r in p1.items():
        if is_na(r.get(a.verdict_col)):
            items.append((pid, '主判无法判读：%s' % (r.get(a.conf_col, '') or ''), ''))
        elif is_yes(r.get(a.verdict_col)) and str(r.get(a.conf_col, '')).strip() in ('低', 'low'):
            items.append((pid, '低置信"确认"（决定分母）', r.get(a.class_col, '')))
    seen, uniq = set(), []
    for pid, why, cls in items:
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append((pid, why, cls))
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, '最小复核清单.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([a.id_col, '复核原因', '复判填写类别', '复核结论', '复核人', '备注'])
        for pid, why, cls in sorted(uniq):
            w.writerow([pid, why, cls, '', '', ''])
    write_ids(os.path.join(a.outdir, '复核_blind.csv'), [p for p, _, _ in sorted(uniq)], a.id_col)
    L = ['# 最小复核清单摘要', '', '- 需人工/高等级复核：**%d 点**' % len(uniq),
         '- 来源：复判不一致 + 主判无法判读 + 低置信"确认"（三者去重）',
         '', '## 明细构成', '']
    c = collections.Counter('复判不一致' if w.startswith('复判不一致') else
                            ('无法判读' if w.startswith('主判无法') else '低置信确认') for _, w, _ in uniq)
    for k, v in c.most_common():
        L.append('- %s：%d' % (k, v))
    L += ['', '## 产物', '', '- `最小复核清单.csv`（可交人工逐点裁决）｜ `复核_blind.csv`（盲判清单）',
          '- 复核完成后：`merge_stats.py` 会把复核结果并入（作为最后一份 --labels 传入即覆盖）。']
    open(os.path.join(a.outdir, 'review_summary.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('最小复核清单 %d 点 → %s' % (len(uniq), os.path.join(a.outdir, '最小复核清单.csv')))


def main():
    ap = argparse.ArgumentParser(description='判读质控协议')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--labels', required=True)
    s.add_argument('--outdir', required=True)
    s.add_argument('--id-col', default='point_id')
    s.add_argument('--verdict-col', default='结论')
    s.add_argument('--conf-col', default='置信（高/中/低）')
    s.add_argument('--strata-col', default=None)
    s.add_argument('--double-pass', type=float, default=0.1)
    s.add_argument('--audit-per-strata', type=int, default=20)
    s.add_argument('--seed', type=int, default=20260101)
    s.set_defaults(func=cmd_sample)
    r = sub.add_parser('review')
    r.add_argument('--labels', required=True)
    r.add_argument('--recheck', required=True)
    r.add_argument('--outdir', required=True)
    r.add_argument('--id-col', default='point_id')
    r.add_argument('--verdict-col', default='结论')
    r.add_argument('--class-col', default='实际类别')
    r.add_argument('--conf-col', default='置信（高/中/低）')
    r.set_defaults(func=cmd_review)
    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()

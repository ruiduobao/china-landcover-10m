# -*- coding: utf-8 -*-
"""r8_qc_deblind.py — 轴标签制盲判表的质控抽样、复判一致率与解盲匹配

* 背景：本协议盲判表是**轴标签制**（结构/叶型轴/物候轴/实际类别），**没有"正确/错误"列**，
        skill 的 qc_protocol.py / merge_stats.py 以"结论=正确/错误/无法判读"为前提，直接套会失败；
        故本脚本用同一套质控三率口径重实现，并做解盲匹配（先验×判读）。
* 子命令：
    sample  --labels <首遍填表.csv> [--strata-col ...] → qc/复判_blind.csv、qc/审计_blind.csv、qc/质控摘要.md
    review  --labels <首遍> --recheck <复判> → qc/最小复核清单.csv、qc/review_summary.md（标签一致率）
    deblind --labels <首遍> [--recheck <复判>] → private/deblind_report.md（先验×判读、确认率、按 role/组/轴分层）
* 门槛（skill references/判读协议.md §4）：复判三态/标签一致率 ≥0.90；"否定"审计准确率 ≥0.90。
* 纪律：deblind 输出**只进 private/**（含先验）；复判/审计 blind 清单只给 point_id。
* 用法：python r8_qc_deblind.py sample --labels F:/.../form_filled.csv --outdir F:/lc_work/.../qc
"""
import argparse
import collections
import csv
import os
import random
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_conf import *   # noqa

NA_TOKENS = ('无法', '', 'nan', '不确定')
AXES = ['结构大类（林/灌/草/耕/园/水/建成/裸/湿/冰）', '叶型轴（针/阔/混交/无法判定）',
        '物候轴（常绿/落叶/无法判定）', '实际类别（24类中文名，或"无法判定"）']
CONF = '置信（高/中/低）'
EVID = '证据（影像特征+影像源+日期/季节+辅助指标）'
WHY = '无法判定原因（云/阴影/分辨率/混合像元/无日期）'


def is_na(v):
    s = str(v or '').strip()
    return any(s.startswith(t) for t in NA_TOKENS if t)


def read(fp):
    d = pd.read_csv(fp, encoding='utf-8-sig')
    d['point_id'] = d['point_id'].astype(str)
    return d


def write_ids(fp, ids):
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq', 'point_id'])
        for i, p in enumerate(ids):
            w.writerow([i + 1, p])


def cmd_sample(a):
    rows = read(a.labels)
    rnd = random.Random(a.seed)
    low = [r['point_id'] for _, r in rows.iterrows() if str(r.get(CONF, '')).strip() in ('低', 'low')]
    na = [r['point_id'] for _, r in rows.iterrows() if is_na(r.get('实际类别（24类中文名，或"无法判定"）'))]
    k = max(1, int(round(len(rows) * a.double_pass)))
    rnd_ids = [str(x) for x in rnd.sample(list(rows['point_id']), min(k, len(rows)))]
    recheck = list(dict.fromkeys(rnd_ids + low + na))
    audit = []
    if a.strata_col and a.strata_col in rows.columns:
        g = collections.defaultdict(list)
        for _, r in rows.iterrows():
            g[str(r.get(a.strata_col, '')) or '(空)'].append(r['point_id'])
        for _, ids in sorted(g.items()):
            audit += rnd.sample(ids, min(a.audit_per_strata, len(ids)))
    else:
        audit = rnd.sample(list(rows['point_id']), min(a.audit_per_strata, len(rows)))
    os.makedirs(a.outdir, exist_ok=True)
    write_ids(os.path.join(a.outdir, '复判_blind.csv'), recheck)
    write_ids(os.path.join(a.outdir, '审计_blind.csv'), audit)
    # 附坐标（盲判取像需要），坐标只来自 blind 面（无先验）
    pts = pd.read_csv(os.path.join(BLIND, 'points_blind.csv'), encoding='utf-8-sig')
    pts['point_id'] = pts['point_id'].astype(str)
    pm = pts.set_index('point_id')
    for nm, ids in (('复判_blind', recheck), ('审计_blind', audit)):
        sub = pd.DataFrame(dict(point_id=ids))
        sub['lon'] = [float(pm.loc[p, 'lon']) for p in ids]
        sub['lat'] = [float(pm.loc[p, 'lat']) for p in ids]
        sub.to_csv(os.path.join(a.outdir, nm + '_points.csv'), index=False, encoding='utf-8-sig')
    L = ['# 质控抽样摘要（轴标签制）', '',
         '- 首遍判读表：`%s`（%d 点）' % (a.labels, len(rows)),
         '- 复判批次：随机 %.0f%%（%d）+ 全部低置信（%d）+ 全部无法判定（%d）→ **%d 点**（去重）' % (
             100 * a.double_pass, len(rnd_ids), len(low), len(na), len(recheck)),
         '- 审计批次：%s，每层 %d → **%d 点**' % (('按 `%s` 分层随机' % a.strata_col) if a.strata_col else '整体随机',
                                          a.audit_per_strata, len(audit)),
         '', '## 判据', '',
         '- 复判一致率（轴标签逐列比）≥0.90；"否定"审计准确率 ≥0.90（`references/判读协议.md` §4）；',
         '- 低置信"确认"与"无法判定"进最小复核清单；',
         '- **盲判面不含先验**；一致率与确认率是判读结果，不是精度。']
    open(os.path.join(a.outdir, '质控摘要.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('复判 %d 点 / 审计 %d 点 → %s' % (len(recheck), len(audit), a.outdir))


def cmd_review(a):
    p1, p2 = read(a.labels), read(a.recheck)
    m1 = p1.set_index('point_id')
    m2 = p2.set_index('point_id')
    common = [p for p in m2.index if p in m1.index]
    agree = {c: 0 for c in AXES}
    items = []
    for p in common:
        r1, r2 = m1.loc[p], m2.loc[p]
        for c in AXES:
            v1, v2 = str(r1.get(c, '')).strip(), str(r2.get(c, '')).strip()
            if v1 == v2:
                agree[c] += 1
            elif not (is_na(v1) and is_na(v2)):
                items.append((p, '复判不一致：%s %s→%s' % (c.split('（')[0], v1, v2), v2))
    for p, r in m1.iterrows():
        if is_na(r.get(AXES[3])):
            items.append((p, '主判无法判定：%s' % (r.get(WHY, '') or ''), ''))
        elif str(r.get(CONF, '')).strip() in ('低', 'low'):
            items.append((p, '低置信（决定分母）', r.get(AXES[3], '')))
    seen, uniq = set(), []
    for pid, why, cls in items:
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append((pid, why, cls))
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, '最小复核清单.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['point_id', '复核原因', '复判填写类别', '复核结论', '复核人', '备注'])
        for r in sorted(uniq):
            w.writerow(list(r) + ['', ''])
    write_ids(os.path.join(a.outdir, '复核_blind.csv'), [p for p, _, _ in sorted(uniq)])
    L = ['# 复判复核摘要', '', '- 共同点 %d' % len(common)]
    for c in AXES:
        L.append('- %s 一致率 **%.3f**' % (c.split('（')[0], agree[c] / max(1, len(common))))
    L += ['', '- 最小复核清单 **%d 点**（复判不一致 + 无法判定 + 低置信）' % len(uniq)]
    open(os.path.join(a.outdir, 'review_summary.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('一致率：' + '；'.join('%s=%.3f' % (c.split('（')[0], agree[c] / max(1, len(common))) for c in AXES)
          + '；最小复核 %d 点' % len(uniq))


def cmd_deblind(a):
    lab = read(a.labels)
    side = pd.read_csv(os.path.join(PRIV, 'uid_map.csv'), encoding='utf-8-sig')
    side['point_id'] = side['point_id'].astype(str)
    m = lab.merge(side, on='point_id', how='left')
    m['判读类'] = m[AXES[3]].fillna('').astype(str).str.strip()
    m['无法'] = [is_na(x) for x in m[AXES[3]]]
    L = ['# 解盲匹配报告（仅 private/）', '',
         '- 判读表：`%s`（%d 点，命中 sidecar %d）' % (a.labels, len(m), int(m['role'].notna().sum())),
         '- **口径**：确认率＝判读结果与先验一致 / 可判读点；不是精度；无法判定单列不并入分母。', '']
    for grp in ('forest_pair', 'dispute'):
        g = m[m['group'] == grp]
        if not len(g):
            continue
        L += ['## %s（n=%d）' % (grp, len(g)), '',
              '| 先验类（v2 池/两版） | 判读类 | n |', '|---|---|---|']
        cnt = collections.Counter(zip(g['cl_name'].fillna(''), g['判读类']))
        for (p, c), v in cnt.most_common(15):
            L.append('| %s | %s | %d |' % (p or '(无池标签)', c or '(空)', v))
        same = sum(1 for _, r in g.iterrows()
                   if str(r['cl_name'] or '') == r['判读类'] and not r['无法'])
        L += ['', '- 与先验一致 %d / 可判读 %d = **%.3f**' % (
            same, int((~g['无法']).sum()), same / max(1, int((~g['无法']).sum()))), '']
    # 坡向配对对照（同 pair 两点判读是否同型）
    fp = m[m['group'] == 'forest_pair']
    L += ['## 坡向配对对照（同 pair 两点）', '', '| pair | 阳坡判读 | 阴坡判读 | 同型? |', '|---|---|---|---|']
    n_same = n_pair = 0
    for pid, g in fp.groupby('pair_id'):
        if len(g) != 2:
            continue
        a1 = g[g['slope_cls'] == 'sun']
        b1 = g[g['slope_cls'] == 'shade']
        if len(a1) != 1 or len(b1) != 1:
            continue
        va, vb = str(a1.iloc[0]['判读类']), str(b1.iloc[0]['判读类'])
        n_pair += 1
        n_same += int(va == vb and not is_na(va))
        L.append('| %s | %s | %s | %s |' % (pid, va, vb, '是' if va == vb else '否'))
    L += ['', '- 配对同型率 %d/%d = %.3f（**真实生态差异不应被强行配平**，此表仅描述）' % (
        n_same, n_pair, n_same / max(1, n_pair))]
    open(os.path.join(PRIV, 'deblind_report.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    m.to_csv(os.path.join(PRIV, 'deblind_labels.csv'), index=False, encoding='utf-8-sig')
    print('解盲匹配 → private/deblind_report.md（含先验，勿外发）')


def main():
    ap = argparse.ArgumentParser(description='轴标签制盲判 QC 与解盲')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--labels', required=True)
    s.add_argument('--outdir', default=QC)
    s.add_argument('--strata-col', default=None)
    s.add_argument('--double-pass', type=float, default=0.1)
    s.add_argument('--audit-per-strata', type=int, default=20)
    s.add_argument('--seed', type=int, default=SEED)
    s.set_defaults(func=cmd_sample)
    r = sub.add_parser('review')
    r.add_argument('--labels', required=True)
    r.add_argument('--recheck', required=True)
    r.add_argument('--outdir', default=QC)
    r.set_defaults(func=cmd_review)
    d = sub.add_parser('deblind')
    d.add_argument('--labels', required=True)
    d.set_defaults(func=cmd_deblind)
    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()

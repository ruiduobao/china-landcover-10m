# -*- coding: utf-8 -*-
"""a3_qc_review.py — 复判/否定审计三率与可判覆盖（未知独立成态；分歧不强判）

* 输入：
    --pass1 / --pass2 ：首判与复判填表（csv；blind_id, 实际类别, 置信, 证据, [各轴列]）
    --verdicts-pass1  ：解盲后由 audit 生成的判定表（csv；blind_id, verdict∈{确认,否定,未知}[, prior_code, role, source]）
    --verdicts-pass2  ：可选；缺省时从 pass2 的 verdict 列读取
    --ledger          ：a2 私有底账（区分 recheck / audit_negation 子集）
    --sidecar         ：私有侧表（role/source 分层）
* 规则源：doc72 §5（三态一致率≥0.90、未知独立成态、否定审计≥0.90、分歧全部进入补充核查、
          不能用提高置信度或多数票强制补全）；判读协议 §4（三率与报告口径）
* 门槛：三态一致率（未知独立成态）≥0.90；否定审计准确率 ≥0.90；未达 → 建议扩到 100% 复判。
        实际类别／各轴一致率另列为主门（防"全判未知"把三态一致率做虚高）。
* 输出：<ws>/audit/out/qc_review_<ts>.json + qc_conflicts_<ts>.csv + qc_axis_<ts>.csv
* 用法：python a3_qc_review.py --pass1 qc/pass1.csv --pass2 qc/pass2.csv \
            --verdicts-pass1 qc/verdicts1.csv --ledger qc/qc_sample_ledger_x.json --sidecar private/sidecar.parquet
* 说明：只统计，不重判、不改判读结论；未裁决分歧一律单列，不自动取高置信一方。
"""
import argparse
import collections
import csv
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

WS_DEFAULT = r'F:/lc_work/label_repair_2023_20261015'
UNKNOWN_PREFIX = '无法'
LOWCONF = {'低', 'low', '低置信'}
AXIS_CANDIDATES = ['大类结构', '针阔', '常绿落叶', '园地管理', '耕地类型', '2023适用性']
OK_STATES = {'确认', '是', '正确'}
NO_STATES = {'否定', '否', '错误'}
NA_STATES = {'未知', '无法', '无法判读', ''}


def norm(v):
    return str(v if v is not None else '').strip()


def read_table(path):
    if path.lower().endswith(('.parquet', '.pq')):
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def id_col_of(rows, prefer):
    if not rows:
        return prefer
    cols = list(rows[0].keys())
    for c in (prefer, 'blind_id', 'point_id', 'id'):
        if c in cols:
            return c
    return cols[0]


def state_of(verdict):
    v = norm(verdict)
    if v in OK_STATES:
        return 'OK'
    if v in NO_STATES:
        return 'NO'
    if v in NA_STATES or v.startswith(UNKNOWN_PREFIX):
        return 'NA'
    return 'OTHER'


def main():
    ap = argparse.ArgumentParser(description='复判/否定审计三率与可判覆盖')
    ap.add_argument('--ws', default=WS_DEFAULT)
    ap.add_argument('--pass1', required=True)
    ap.add_argument('--pass2', required=True)
    ap.add_argument('--verdicts-pass1', required=True)
    ap.add_argument('--verdicts-pass2', default=None)
    ap.add_argument('--ledger', default=None)
    ap.add_argument('--sidecar', default=None)
    ap.add_argument('--id-col', default='blind_id')
    ap.add_argument('--class-col', default='实际类别')
    ap.add_argument('--conf-col', default='置信')
    ap.add_argument('--axis-cols', nargs='*', default=None)
    ap.add_argument('--gate', type=float, default=0.90)
    ap.add_argument('--unknown-dominant', type=float, default=0.30)
    ap.add_argument('--min-repair-n', type=int, default=30)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ws = a.ws.replace('\\', '/')
    outd = os.path.join(ws, 'audit', 'out')
    os.makedirs(outd, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    r1, r2 = read_table(a.pass1), read_table(a.pass2)
    a.id_col = a.id_col if (r1 and a.id_col in r1[0]) else id_col_of(r1, a.id_col)
    p1 = {norm(r[a.id_col]): r for r in r1}
    p2 = {norm(r.get(a.id_col, r.get('point_id', r.get('blind_id')))): r for r in r2}
    vr1 = read_table(a.verdicts_pass1)
    v1 = {norm(r.get(a.id_col, r.get('blind_id', r.get('point_id')))): r for r in vr1}
    if a.verdicts_pass2:
        v2 = {norm(r.get(a.id_col, r.get('blind_id', r.get('point_id')))): r
              for r in read_table(a.verdicts_pass2)}
    else:
        v2 = p2
    if not p1 or not p2:
        raise SystemExit('pass1/pass2 为空')
    common = sorted(set(p1) & set(p2))
    if not common:
        raise SystemExit('pass1 与 pass2 无共同 blind_id（检查是否同一盲批命名空间）')

    axis_cols = a.axis_cols or [c for c in AXIS_CANDIDATES if c in p1[common[0]]]
    ledger = json.load(open(a.ledger, encoding='utf-8')) if a.ledger and os.path.exists(a.ledger) else {}
    items = ledger.get('items', {})
    if a.sidecar and os.path.exists(a.sidecar):
        srows = read_table(a.sidecar)
        side = {}
        for r in srows:
            k = ''
            for cand in ('blind_id', 'point_id', 'id', a.id_col):
                if str(r.get(cand, '') or '').strip():
                    k = str(r[cand]).strip()
                    break
            side[k] = r
    else:
        side = {}

    def kinds(i):
        return set(items.get(i, {}).get('kinds', [])) if items else set()

    # ---- 1) 三态一致率（未知独立成态）----
    tri_agree = tri_den = 0
    tri_break = collections.Counter()
    for i in common:
        s1, s2 = state_of((v1.get(i) or {}).get('verdict')), state_of((v2.get(i) or {}).get('verdict'))
        if s1 == 'OTHER' or s2 == 'OTHER':
            tag = '未匹配别名（待裁决，不计分母）' if (
                norm((v1.get(i) or {}).get('verdict')) == '未匹配'
                or norm((v2.get(i) or {}).get('verdict')) == '未匹配') else 'verdict_列无法识别'
            tri_break[tag] += 1
            continue
        tri_den += 1
        if s1 == s2:
            tri_agree += 1
            tri_break['%s=%s' % (s1, s2)] += 1
        else:
            tri_break['分歧 %s/%s' % (s1, s2)] += 1
    tri_rate = tri_agree / tri_den if tri_den else float('nan')

    # ---- 2) 二值一致（旁证，已知会被未知抬升）----
    bi_agree = bi_den = 0
    for i in common:
        s1, s2 = state_of((v1.get(i) or {}).get('verdict')), state_of((v2.get(i) or {}).get('verdict'))
        if s1 == 'OTHER' or s2 == 'OTHER':
            continue
        bi_den += 1
        bi_agree += 1 if ((s1 == 'OK') == (s2 == 'OK')) else 0
    bi_rate = bi_agree / bi_den if bi_den else float('nan')

    # ---- 3) 实际类别与各轴一致率（主门旁证；仅在双方给出具体类别时计入）----
    def agree_on(col):
        n = k = 0
        for i in common:
            c1, c2 = norm(p1[i].get(col)), norm(p2[i].get(col))
            if not c1 or not c2 or c1.startswith(UNKNOWN_PREFIX) or c2.startswith(UNKNOWN_PREFIX):
                continue
            n += 1
            k += 1 if c1 == c2 else 0
        return k, n, (k / n if n else float('nan'))

    cls_k, cls_n, cls_rate = agree_on(a.class_col)
    axis = {}
    for c in axis_cols:
        k, n, r = agree_on(c)
        axis[c] = {'n_judgeable': n, 'agree': k, 'rate': r}

    # ---- 4) 否定审计（首判否定点中被复判推翻的比例）----
    neg_total = sum(1 for i in v1 if state_of((v1.get(i) or {}).get('verdict')) == 'NO')
    aud = [i for i in common if 'audit_negation' in kinds(i)] if items else []
    if not aud:
        aud = [i for i in common if state_of((v1.get(i) or {}).get('verdict')) == 'NO']
    aud_den = aud_still = 0
    for i in aud:
        s2 = state_of((v2.get(i) or {}).get('verdict'))
        if s2 == 'OTHER':
            continue
        aud_den += 1
        aud_still += 1 if s2 == 'NO' else 0
    aud_acc = aud_still / aud_den if aud_den else float('nan')
    aud_overturned = aud_den - aud_still

    # ---- 5) 可判覆盖与 unknown 主导 ----
    unk1 = [i for i in p1 if norm(p1[i].get(a.class_col)).startswith(UNKNOWN_PREFIX)]
    unk_rate1 = len(unk1) / len(p1) if p1 else float('nan')
    judgeable1 = len(p1) - len(unk1)
    by_role = collections.Counter()
    for i in p1:
        by_role[norm((side.get(i) or {}).get('role')) or '(未提供)'] += 1
    unk_by_role = collections.Counter()
    for i in unk1:
        unk_by_role[norm((side.get(i) or {}).get('role')) or '(未提供)'] += 1
    unknown_dominant = unk_rate1 > a.unknown_dominant

    # ---- 6) R 修复可用性 ----
    repair_ok, repair_block = 0, collections.Counter()
    for i, r in p1.items():
        if norm((side.get(i) or {}).get('role')) != 'R':
            continue
        if norm(r.get(a.conf_col)) not in ('中', '高', 'mid', 'high'):
            repair_block['低置信'] += 1
            continue
        if norm(r.get(a.class_col)).startswith(UNKNOWN_PREFIX):
            repair_block['无法判读'] += 1
            continue
        if '2023适用性' in r and norm(r.get('2023适用性')) not in ('是', '可用', '成立'):
            repair_block['年度不适用'] += 1
            continue
        repair_ok += 1
    repair_supported = repair_ok >= a.min_repair_n

    # ---- 7) 未裁决分歧清单（不取高置信一方）----
    conflicts = []
    for i in common:
        s1, s2 = state_of((v1.get(i) or {}).get('verdict')), state_of((v2.get(i) or {}).get('verdict'))
        why = []
        if s1 != s2:
            why.append('三态 %s/%s' % (s1, s2))
        for c in [a.class_col] + axis_cols:
            c1, c2 = norm(p1[i].get(c)), norm(p2[i].get(c))
            if c1 and c2 and c1 != c2:
                why.append('%s %s/%s' % (c, c1, c2))
        if why:
            conflicts.append({'blind_id': i, 'reasons': '; '.join(why),
                              'conf_pass1': norm(p1[i].get(a.conf_col)),
                              'conf_pass2': norm(p2[i].get(a.conf_col)),
                              'resolution': '未裁决（交第三人仲裁；不得取高置信一方）'})
    cpath = os.path.join(outd, 'qc_conflicts_%s.csv' % ts)
    with open(cpath, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['blind_id', 'reasons', 'conf_pass1', 'conf_pass2', 'resolution'])
        w.writeheader()
        w.writerows(conflicts)
    apath = os.path.join(outd, 'qc_axis_%s.csv' % ts)
    with open(apath, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['axis', 'n_judgeable', 'agree', 'rate'])
        w.writerow([a.class_col, cls_n, cls_k, cls_rate])
        for c, v in axis.items():
            w.writerow([c, v['n_judgeable'], v['agree'], v['rate']])

    gates = {
        'three_state_agreement': {'value': tri_rate, 'n': tri_den, 'gate': a.gate,
                                  'pass': bool(tri_den and tri_rate >= a.gate)},
        'negation_audit_accuracy': {'value': aud_acc, 'n': aud_den, 'gate': a.gate,
                                    'pass': bool(aud_den and aud_acc >= a.gate),
                                    'overturned': aud_overturned},
        'actual_class_agreement': {'value': cls_rate, 'n': cls_n, 'note': '主门旁证，不与三态混报'},
    }
    all_pass = gates['three_state_agreement']['pass'] and gates['negation_audit_accuracy']['pass']
    claim_limits = [
        '一致率是判读结果，不是精度；正式精度需独立人工双盲核验',
        '本批全部为 AI 预判读，未完成正式人工复核',
    ]
    if unknown_dominant:
        claim_limits.append('未知占比 %.1f%% > %.0f%%：即使一致率高，也不得声明"标签精度已验证"；'
                            '可判覆盖不足时只能给合并级/结构级结论'
                            % (100 * unk_rate1, 100 * a.unknown_dominant))
    if neg_total and len(aud) < neg_total:
        claim_limits.append('否定审计覆盖 %d/%d 点（受混排包总上限约束）：accuracy 只在已审层内成立，'
                            '未审层的否定结论不得据本次审计作整体断言'
                            % (len(aud), neg_total))
    if not repair_supported:
        claim_limits.append('R 修复可用点 %d < %d：输出 NO_REPAIR_SUPPORT，不臆造提升'
                            % (repair_ok, a.min_repair_n))

    rec = {
        'schema': 'qc_review/v1', 'generated_at': datetime.now().isoformat(timespec='seconds'),
        'n_pass1': len(p1), 'n_pass2': len(p2), 'n_common': len(common),
        'three_state': {'agree': tri_agree, 'denominator': tri_den, 'rate': tri_rate,
                        'breakdown': dict(tri_break)},
        'binary_secondary': {'agree': bi_agree, 'denominator': bi_den, 'rate': bi_rate,
                             'warning': '二值把"未知 vs 错误"算一致，会被未知抬升；仅旁证'},
        'actual_class_agreement': {'agree': cls_k, 'n_judgeable': cls_n, 'rate': cls_rate},
        'axis_agreement': axis,
        'negation_audit': {'audited': len(aud), 'negation_total': neg_total,
                           'coverage': (len(aud) / neg_total) if neg_total else None,
                           'effective_denominator': aud_den, 'still_negation': aud_still,
                           'overturned': aud_overturned, 'accuracy': aud_acc,
                           'note': '口径(c)：层内 N=20、混排包总上限 80；未覆盖层记未审计，'
                                   'accuracy 只在已审层内解释，不外推全部否定点'},
        'unknown': {'n': len(unk1), 'rate': unk_rate1, 'judgeable': judgeable1,
                    'dominant': unknown_dominant, 'by_role': dict(unk_by_role)},
        'repair_eligibility': {'role_R_eligible': repair_ok, 'min_required': a.min_repair_n,
                               'supported': repair_supported, 'blocked_reasons': dict(repair_block)},
        'conflicts': {'n': len(conflicts), 'file': cpath, 'policy': '全部分歧进补充核查，不强判'},
        'gates': gates,
        'gate': 'PASS' if all_pass else 'BELOW_GATE',
        'next_action': ('扩大复判至 100% 并对全部分歧仲裁' if not all_pass else '按 doc72 §5 记录三率并进入 R 修复'),
        'claim_limits': claim_limits,
        'inputs': {k: v for k, v in [('pass1', a.pass1), ('pass2', a.pass2),
                                     ('verdicts_pass1', a.verdicts_pass1),
                                     ('ledger', a.ledger)] if v},
    }
    fp = a.out or os.path.join(outd, 'qc_review_%s.json' % ts)

    def clean(o):
        """NaN/Inf → None，保证输出是严格 JSON（分母为 0 时不写 NaN）。"""
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float) and (o != o or o in (float('inf'), float('-inf'))):
            return None
        return o

    json.dump(clean(rec), open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('共同点 %d：三态一致率 %.3f（n=%d，门 %.2f）｜实际类别一致率 %s（n=%d）｜否定审计 %.3f（n=%d）'
          % (len(common), tri_rate if tri_rate == tri_rate else -1, tri_den, a.gate,
             ('%.3f' % cls_rate) if cls_rate == cls_rate else '-', cls_n,
             aud_acc if aud_acc == aud_acc else -1, aud_den))
    print('未知 %d（%.1f%%）｜R 可用修复点 %d（门 %d）｜未裁决分歧 %d'
          % (len(unk1), 100 * (unk_rate1 if unk_rate1 == unk_rate1 else 0), repair_ok, a.min_repair_n, len(conflicts)))
    print('gate=%s  next=%s' % (rec['gate'], rec['next_action']))
    print('→ %s ｜ 分歧 %s ｜ 轴一致 %s' % (fp, cpath, apath))


if __name__ == '__main__':
    main()

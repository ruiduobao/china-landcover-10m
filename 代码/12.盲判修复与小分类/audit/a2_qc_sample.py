# -*- coding: utf-8 -*-
"""a2_qc_sample.py — 盲判 QC 抽样（随机≥10% + 全部低置信/无法 + 否定审计）与混排盲包

* 输入：
    --labels  ：首判填表（csv；列：blind_id, 实际类别, 置信, 证据, …）；stage=post 另需 --verdicts
    --verdicts：解盲后由 audit 生成的判定表（csv；列：blind_id, verdict∈{确认,否定,未知}, source）
    --sidecar ：私有侧表（取 role/source 分层；不进入盲包）
* 规则源：doc72 §5（抽样口径：≥随机10%、全部低置信/无法、解盲后各组随机否定点；
          否定≤40 全查，更多时按私有来源层各抽 20 或全查不足 20 的层；seed 固定 20261015）
* 门槛：盲包只含中性 blind_id；不得出现 kind（复判/审计）、原因、先验、来源、分层、role
* 输出：qc/qc_blind_mixed_pre.csv（stage=pre 初版）或 qc/qc_blind_mixed.csv（stage=post 正式混排盲包）
        qc/qc_sample_ledger_<ts>.json（私有底账：每点的 kind/来源层/role/初判，供 audit 复核）
* 用法：python a2_qc_sample.py --stage pre  --labels qc/pass1.csv --sidecar private/sidecar.parquet
        python a2_qc_sample.py --stage post --labels qc/pass1.csv --verdicts qc/verdicts.csv --sidecar ...
* 说明：不用技能 qc_protocol.py 抽样——它按"结论"列（先验相对）分层且输出原因分离的两个包；
        本轮改为 glue：单包混排、只给 blind_id、原因与分层只留在私有底账。
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

WS_DEFAULT = r'F:/lc_work/label_repair_2023_20261015'
SEED_DEFAULT = 20261015
UNKNOWN_PREFIX = '无法'
LOWCONF = {'低', 'low', '低置信'}


def norm(v):
    return str(v if v is not None else '').strip()


def read_table(path):
    if path.lower().endswith(('.parquet', '.pq')):
        import pandas as pd
        return pd.read_parquet(path).to_dict('records')
    return list(csv.DictReader(open(path, encoding='utf-8-sig')))


def is_unknown(r, class_col):
    return norm(r.get(class_col)).startswith(UNKNOWN_PREFIX)


def is_low(r, conf_col):
    return norm(r.get(conf_col)) in LOWCONF


def id_col_of(rows, prefer):
    """id 列自适应（本批盲包用 point_id，值为 B####）。"""
    if not rows:
        return prefer
    cols = list(rows[0].keys())
    for c in (prefer, 'blind_id', 'point_id', 'id'):
        if c in cols:
            return c
    return cols[0]


def sid(r):
    """侧表 id 取值：映射后标准名为 blind_id，兼容 point_id/id。"""
    for k in ('blind_id', 'point_id', 'id'):
        v = r.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ''


def apply_map(rows, map_path):
    """map 为 {标准列名: 本侧表列名}；反向把侧表列名改成标准名。"""
    if not map_path:
        return rows
    m = json.load(open(map_path, encoding='utf-8'))
    inv = {v: k for k, v in m.items() if v and not str(v).startswith('__')}
    return [{inv.get(k, k): v for k, v in r.items()} for r in rows]


def fhash(path):
    h = hashlib.sha256()
    h.update(open(path, 'rb').read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description='盲判 QC 抽样与混排盲包')
    ap.add_argument('--ws', default=WS_DEFAULT)
    ap.add_argument('--labels', required=True, help='首判填表（盲批表单）')
    ap.add_argument('--verdicts', default=None, help='stage=post：解盲后判定表（blind_id,verdict,source）')
    ap.add_argument('--sidecar', required=True, help='私有侧表（role/source）')
    ap.add_argument('--stage', choices=['pre', 'post'], default='pre')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--seed', type=int, default=SEED_DEFAULT)
    ap.add_argument('--double-pass', type=float, default=0.10)
    ap.add_argument('--neg-all-max', type=int, default=40, help='否定总数≤此值则全查')
    ap.add_argument('--neg-per-source', type=int, default=20)
    ap.add_argument('--neg-total-cap', type=int, default=80,
                    help='否定审计混排包总上限（口径c：层内 N=20、总上限 80，超限按层降序取前若干层）')
    ap.add_argument('--sidecar-map', default=None, help='JSON：{标准列名: 本侧表列名}')
    ap.add_argument('--id-col', default='blind_id')
    ap.add_argument('--class-col', default='实际类别')
    ap.add_argument('--conf-col', default='置信')
    a = ap.parse_args()
    ws = a.ws.replace('\\', '/')
    outd = a.outdir or os.path.join(ws, 'qc')
    os.makedirs(outd, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    rows = read_table(a.labels)
    if not rows:
        raise SystemExit('首判填表为空：%s' % a.labels)
    a.id_col = a.id_col if a.id_col in rows[0] else id_col_of(rows, a.id_col)
    rnd = random.Random(a.seed)
    byid = {norm(r[a.id_col]): r for r in rows}
    ids_sorted = sorted(byid)                      # 排序后抽样：可复现，与文件行序无关

    low = [i for i in ids_sorted if is_low(byid[i], a.conf_col)]
    unk = [i for i in ids_sorted if is_unknown(byid[i], a.class_col)]
    blank = [i for i in ids_sorted if norm(byid[i].get(a.class_col)) == '']
    k = max(1, int(round(len(ids_sorted) * a.double_pass)))
    rand = rnd.sample(ids_sorted, min(k, len(ids_sorted)))

    kinds = collections.defaultdict(set)
    for i in rand:
        kinds[i].add('recheck_random')
    for i in low:
        kinds[i].add('recheck_lowconf')
    for i in unk:
        kinds[i].add('recheck_unknown')

    audit_ids, audit_alloc, audit_unaudited = [], {}, {}
    neg_budget = 0
    if a.stage == 'post':
        if not a.verdicts or not os.path.exists(a.verdicts):
            raise SystemExit('stage=post 需要 --verdicts（解盲后判定表）')
        side = {sid(r): r for r in apply_map(read_table(a.sidecar), a.sidecar_map)}
        neg = [sid(r) for r in read_table(a.verdicts) if norm(r.get('verdict')) == '否定']
        neg = [i for i in sorted(set(neg)) if i in byid]
        # 混排包总上限口径(c)：先留出"随机10%+全部低/未知"的必抽集合，余量才是否定审计预算
        recheck_n = len(kinds)
        budget = max(0, a.neg_total_cap - recheck_n)
        neg_budget = budget
        if len(neg) <= a.neg_all_max:
            audit_ids = neg
            audit_alloc = {'(all)': len(neg)}
        else:
            g = collections.defaultdict(list)
            for i in neg:
                s = norm((side.get(i) or {}).get('source')) or '(未知来源)'
                g[s].append(i)
            used = 0
            # 口径(c)：层内 N=neg_per_source；按层否定点数降序取整层，超总上限即停，其余层记未审计
            for s, v in sorted(g.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                want = min(a.neg_per_source, len(v))
                if used + want > budget:
                    audit_unaudited[s] = len(v)
                    continue
                take = v if len(v) <= a.neg_per_source else rnd.sample(v, a.neg_per_source)
                audit_ids += take
                audit_alloc[s] = len(take)
                used += len(take)
            audit_ids = sorted(set(audit_ids))
        for i in audit_ids:
            kinds[i].add('audit_negation')

    final = sorted(kinds)
    mixed = list(final)
    rnd.shuffle(mixed)                              # 混排：判读者无法从顺序推断 kind
    out_name = 'qc_blind_mixed_pre.csv' if a.stage == 'pre' else 'qc_blind_mixed.csv'
    fp = os.path.join(outd, out_name)
    with open(fp, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([a.id_col])
        for i in mixed:
            w.writerow([i])

    side = {sid(r): r for r in apply_map(read_table(a.sidecar), a.sidecar_map)}
    ledger = {
        'schema': 'qc_sample_ledger/v1',
        'stage': a.stage,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'seed': a.seed,
        'double_pass': a.double_pass,
        'n_labels': len(ids_sorted),
        'counts': {'random_10pct': len(rand), 'lowconf': len(low), 'unknown': len(unk),
                   'blank_class': len(blank), 'negation_audit': len(audit_ids),
                   'packet_total': len(final)},
        'negation_audit_allocation_by_source': audit_alloc,
        'negation_audit_unaudited_by_source': audit_unaudited,
        'negation_total_cap': a.neg_total_cap,
        'recheck_mandatory_n': len(kinds),
        'negation_budget': neg_budget,
        'negation_cap_used': sum(audit_alloc.values()),
        'negation_cap_leftover': max(0, neg_budget - sum(audit_alloc.values())),
        'packet_over_cap': len(final) > a.neg_total_cap,
        'packet_over_cap_reason': ('必抽复判集（随机10%+全部低置信/无法）本身 %d 点已超总上限 %d，'
                                   '按协议不可缩减，否定审计预算为 0' % (len(kinds), a.neg_total_cap))
                                  if len(final) > a.neg_total_cap else None,
        'negation_audit_rule': '口径(c)：层内 N=%d；按层否定点数降序取整层，累计超总上限 %d 即停，'
                               '其余层记未审计并留底账；否定总数≤%d 时全查；'
                               '混排包总上限 %d 为"必抽复判集 + 否定审计"之和'
                               % (a.neg_per_source, a.neg_total_cap, a.neg_all_max, a.neg_total_cap),
        'blank_class_ids': blank,
        'policy': [
            '判读者只能被告知"复核盲包：判实际类别/各轴/置信/证据"，不得告知初判结论、否定/未知原因或分组',
            '正确/错误为解盲后由 audit 计算，盲包与 judge prompt 均不得出现',
            '本底账为私有：kind/来源层/role/初判不进入盲包'],
        'items': {i: {'kinds': sorted(kinds[i]),
                      'role': norm((side.get(i) or {}).get('role')),
                      'source': norm((side.get(i) or {}).get('source')),
                      'block_025': norm((side.get(i) or {}).get('block_025')),
                      'conf_pass1': norm(byid[i].get(a.conf_col)),
                      'class_pass1': norm(byid[i].get(a.class_col))} for i in final},
        'inputs': {p: fhash(p) for p in [a.labels, a.sidecar] + ([a.verdicts] if a.verdicts else [])},
        'packet': {'file': fp, 'sha256': fhash(fp), 'n': len(mixed)},
    }
    lp = os.path.join(outd, 'qc_sample_ledger_%s.json' % ts)
    json.dump(ledger, open(lp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('QC 包(%s)：随机 %d + 低置信 %d + 无法 %d + 否定审计 %d → 去重 %d 点（混排）'
          % (a.stage, len(rand), len(low), len(unk), len(audit_ids), len(final)))
    if blank:
        print('警告：%d 点实际类别为空白——空白≠无法判读，应退回补全，不参与统计' % len(blank))
    print('盲包 → %s（只含 %s）' % (fp, a.id_col))
    print('私有底账 → %s' % lp)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""a1_timing_audit.py — 20 分钟硬截止 / prepare 单列 / 中止任务终止性 独立核验

* 输入：runner 记录（JSON 或 JSONL），每单元至少含：
    experiment_id, status, deadline_s,
    t_train_start, t_end            # ISO 或 epoch 秒；train->eval->persist 的 wall-clock
    可选 prepare_seconds, phases{}, exit_code, orphan_pids[], orphan_check, output_files[]
* 规则源：doc70 §3.2（单元=区域×方案×种子集；train->eval->persist ≤ deadline；prepare 单列）
* 门槛：wall <= deadline_s（默认 1200）；status==TIMEOUT 时须有终止证据；不得有孤儿进程
* 输出：out/timing_audit_<ts>.json + .csv
* 用法：python a1_timing_audit.py <records.json|.jsonl> [--deadline 1200] [--out-prefix ...]
* 说明：只读核验，不改实验记录；schema 缺失时显式报错，不猜测。
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
OUT = os.path.join(AUDIT, 'out')

REQUIRED = ['experiment_id', 'status', 'deadline_s', 't_train_start', 't_end']


def load_records(path):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        sys.exit('[ERR] not found: %s' % path)
    text = open(path, encoding='utf-8').read().strip()
    if path.lower().endswith('.jsonl'):
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and 'records' in obj:
        return obj['records']
    return [obj]


def to_epoch(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return datetime.fromisoformat(str(v).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def audit_one(rec, default_deadline):
    out = {'experiment_id': rec.get('experiment_id')}
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        out['verdict'] = 'SCHEMA_MISSING'
        out['missing_fields'] = missing
        return out

    dl = rec.get('deadline_s') or default_deadline
    t0, t1 = to_epoch(rec.get('t_train_start')), to_epoch(rec.get('t_end'))
    wall = None if (t0 is None or t1 is None) else round(t1 - t0, 3)
    out.update({
        'status': rec.get('status'),
        'deadline_s': dl,
        'wall_seconds': wall,
        'prepare_seconds': rec.get('prepare_seconds'),   # 单列，不计入 deadline
        'within_deadline': (wall is not None and wall <= dl),
    })

    problems = []
    if wall is None:
        problems.append('WALLTIME_UNPARSEABLE')
    elif wall > dl:
        problems.append('OVER_DEADLINE')

    status = str(rec.get('status', '')).upper()
    if status == 'TIMEOUT':
        # 中止任务验证：须有子进程终止证据、无孤儿、退出码记录
        if rec.get('orphan_pids'):
            problems.append('ORPHAN_PIDS:%s' % rec['orphan_pids'])
        if not rec.get('orphan_check'):
            problems.append('NO_ORPHAN_CHECK_EVIDENCE')
        if rec.get('exit_code') is None:
            problems.append('NO_EXIT_CODE')
        out['timeout_evidence'] = {
            'orphan_pids': rec.get('orphan_pids'),
            'orphan_check': rec.get('orphan_check'),
            'exit_code': rec.get('exit_code'),
            'phases': rec.get('phases'),
        }

    # 不得以“准备”隐藏训练/调参
    if rec.get('prepare_seconds') is not None and rec.get('phases'):
        ph = rec['phases'] or {}
        if isinstance(ph, dict) and ph.get('train_inside_prepare'):
            problems.append('TRAIN_HIDDEN_IN_PREPARE')

    out['problems'] = problems
    if not problems:
        out['verdict'] = 'PASS'
    elif 'SCHEMA_MISSING' in problems:
        out['verdict'] = 'SCHEMA_MISSING'
    else:
        out['verdict'] = 'FAIL'
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('records')
    ap.add_argument('--deadline', type=float, default=1200.0)
    ap.add_argument('--out-prefix', default=None)
    args = ap.parse_args()

    recs = load_records(args.records)
    rows = [audit_one(r, args.deadline) for r in recs]

    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = args.out_prefix or os.path.join(OUT, 'timing_audit_%s' % ts)
    with open(prefix + '.json', 'w', encoding='utf-8') as f:
        json.dump({'deadline_default': args.deadline, 'n': len(rows), 'records': rows}, f,
                  ensure_ascii=False, indent=2)

    cols = ['experiment_id', 'status', 'deadline_s', 'wall_seconds', 'prepare_seconds',
            'within_deadline', 'verdict', 'problems']
    with open(prefix + '.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            r = dict(r)
            r['problems'] = ';'.join(r.get('problems') or [])
            w.writerow(r)

    npass = sum(1 for r in rows if r.get('verdict') == 'PASS')
    print('[OK] %d/%d PASS -> %s.{json,csv}' % (npass, len(rows), prefix))
    for r in rows:
        if r.get('verdict') != 'PASS':
            print('  [%s] %s %s' % (r.get('verdict'), r.get('experiment_id'), r.get('problems')))


if __name__ == '__main__':
    main()

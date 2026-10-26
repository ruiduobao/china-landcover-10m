# -*- coding: utf-8 -*-
"""h15_run_yearly_fetch.py — 用多账号并行补齐 8 年 AEF64 与逐年证据

* 来源：config/pids_mined.json（从各账号 _任务登记.md 挖出的项目ID）
* 规则：每年一个账号（子进程隔离 HOME/USERPROFILE），AEF 与证据各用 8 个账号
* 幂等：h4/h11 块级跳过已存在文件 → 可反复重跑续跑
* 输出：data/aef_yearly/<year>/、data/evidence_yearly/<year>/、logs/yz/*.log、logs/yz/run_summary.json
* 用法：python h15_run_yearly_fetch.py [--workers 16] [--years 2018 2019 ...]
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
CODE = os.path.join(BASE, 'code')
LOGS = os.path.join(BASE, 'logs', 'yz')
PIDS = json.load(open(os.path.join(BASE, 'config', 'pids_mined.json'), encoding='utf-8'))
PY = sys.executable


def pick_accounts(exclude, n):
    out = []
    for a in sorted(PIDS):
        if not PIDS[a] or a in exclude:
            continue
        out.append((a, PIDS[a][0]))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=list(range(2017, 2025)))
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    used = set()
    aef_accts = pick_accounts(used, len(a.years)); used |= {x[0] for x in aef_accts}
    ev_accts = pick_accounts(used, len(a.years))
    procs = []
    for i, y in enumerate(a.years):
        acct, pid = aef_accts[i % len(aef_accts)]
        log = open(os.path.join(LOGS, 'aef_%d.log' % y), 'a')
        cmd = [PY, '-u', os.path.join(CODE, 'h4_aef.py'), '--list',
               os.path.join(BASE, 'data', 'final', 'final_2023_v2.parquet'),
               '--year', str(y), '--acct', acct, '--pid', pid,
               '--outdir', os.path.join(BASE, 'data', 'aef_yearly', str(y))]
        procs.append(('aef', y, acct, pid, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    for i, y in enumerate(a.years):
        acct, pid = ev_accts[i % len(ev_accts)]
        log = open(os.path.join(LOGS, 'ev_%d.log' % y), 'a')
        cmd = [PY, '-u', os.path.join(CODE, 'h11_yearly_evidence.py'), '--year', str(y),
               '--acct', acct, '--pid', pid,
               '--outdir', os.path.join(BASE, 'data', 'evidence_yearly', str(y))]
        procs.append(('ev', y, acct, pid, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    print('启动 %d 个子进程（AEF %d / 证据 %d）' % (len(procs), len(a.years), len(a.years)), flush=True)
    t0 = time.time()
    summary = []
    for kind, y, acct, pid, p, log in procs:
        rc = p.wait()
        log.close()
        summary.append({'kind': kind, 'year': y, 'acct': acct, 'pid': pid, 'rc': rc})
        print('%s %d %s rc=%d (%.0fs)' % (kind, y, acct, rc, time.time() - t0), flush=True)
    json.dump(summary, open(os.path.join(LOGS, 'run_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    bad = [s for s in summary if s['rc'] != 0]
    print('完成：%d 个子进程，失败 %d' % (len(summary), len(bad)), flush=True)
    for b in bad:
        print('  FAIL', b, flush=True)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""b4_run_yearly.py — 为南方裸地候选并行抓 8 年 AEF 与逐年证据

* 输入：data/final/bare_south_cand.parquet
* 输出：data/aef_yearly_bare/<year>/、data/evidence_yearly_bare/<year>/
* 用法：python b4_run_yearly.py [--workers 8]
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
CODE = os.path.join(BASE, 'code')
LOGS = os.path.join(BASE, 'logs', 'bare_yearly')
GOOD = json.load(open(os.path.join(BASE, 'config', 'accounts_good.json'), encoding='utf-8'))
PY = sys.executable
LIST = os.path.join(BASE, 'data', 'final', 'bare_south_cand.parquet')


def run(kind, y, idx):
    for k in range(len(GOOD)):
        a = GOOD[(idx + k) % len(GOOD)]
        log = open(os.path.join(LOGS, '%s_%d_try%d.log' % (kind, y, k)), 'a')
        if kind == 'aef':
            cmd = [PY, '-u', os.path.join(CODE, 'h4_aef.py'), '--list', LIST, '--year', str(y),
                   '--acct', a['acct'], '--pid', a['pid'],
                   '--outdir', os.path.join(BASE, 'data', 'aef_yearly_bare', str(y))]
        else:
            cmd = [PY, '-u', os.path.join(CODE, 'h11_yearly_evidence.py'), '--year', str(y),
                   '--list', LIST, '--acct', a['acct'], '--pid', a['pid'],
                   '--outdir', os.path.join(BASE, 'data', 'evidence_yearly_bare', str(y))]
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT); log.close()
        print('%s %d ← %s rc=%d' % (kind, y, a['acct'], rc), flush=True)
        if rc == 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    tasks = [(k, y) for k in ('ev', 'aef') for y in range(2017, 2025)]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, k, y, i) for i, (k, y) in enumerate(tasks)]
        for (k, y), f in zip(tasks, futs):
            print('%s %d → %s' % (k, y, 'OK' if f.result() else 'FAIL'), flush=True)
if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""h15b_dispatch.py — 缺块补齐调度器（按 shard 轮换账号，直到成功）

* 依据现有文件判断缺块：AEF 每年 8 块、证据每年 5 块（h4/h11 的块数上限，缺失即补）
* 账号：config/accounts_good.json（h0c 实测可用）；每个 shard 失败则换下一个账号重试
* 用法：python h15b_dispatch.py [--years 2017 2018 ...] [--workers 4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
BASE = r'F:/lc_work/bare_sparse_v3'
CODE = os.path.join(BASE, 'code')
LOGS = os.path.join(BASE, 'logs', 'yz')
GOOD = json.load(open(os.path.join(BASE, 'config', 'accounts_good.json'), encoding='utf-8'))
PY = sys.executable
N_AEF, N_EV = 8, 5


def missing(kind, y):
    d = os.path.join(BASE, 'data', 'aef_yearly' if kind == 'aef' else 'evidence_yearly', str(y))
    n = len([f for f in os.listdir(d) if f.endswith('.parquet')]) if os.path.isdir(d) else 0
    return max(0, (N_AEF if kind == 'aef' else N_EV) - n)


def cmd_for(kind, y, acct, pid):
    if kind == 'aef':
        return [PY, '-u', os.path.join(CODE, 'h4_aef.py'), '--list',
                os.path.join(BASE, 'data', 'final', 'final_2023_v2.parquet'), '--year', str(y),
                '--acct', acct, '--pid', pid,
                '--outdir', os.path.join(BASE, 'data', 'aef_yearly', str(y))]
    return [PY, '-u', os.path.join(CODE, 'h11_yearly_evidence.py'), '--year', str(y),
            '--acct', acct, '--pid', pid,
            '--outdir', os.path.join(BASE, 'data', 'evidence_yearly', str(y))]


def run_shard(kind, y, idx):
    for k in range(len(GOOD)):
        a = GOOD[(idx + k) % len(GOOD)]
        log = open(os.path.join(LOGS, '%s_%d_retry%d.log' % (kind, y, k)), 'a')
        try:
            rc = subprocess.call(cmd_for(kind, y, a['acct'], a['pid']),
                                 stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        print('%s %d ← %s rc=%d' % (kind, y, a['acct'], rc), flush=True)
        if rc == 0 and missing(kind, y) == 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', nargs='*', type=int, default=list(range(2017, 2025)))
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    shards = []
    for y in a.years:
        for kind in ('aef', 'ev'):
            m = missing(kind, y)
            if m:
                shards.append((kind, y, m))
    if not shards:
        print('无缺块'); return
    print('缺块 shard：', shards, flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_shard, kind, y, i) for i, (kind, y, _) in enumerate(shards)]
        for (kind, y, m), f in zip(shards, futs):
            print('%s %d 缺 %d 块 → %s' % (kind, y, m, 'OK' if f.result() else '仍失败'), flush=True)
    left = [(k, y, missing(k, y)) for k, y, _ in shards if missing(k, y)]
    print('结束，剩余缺块：', left, flush=True)


if __name__ == '__main__':
    main()

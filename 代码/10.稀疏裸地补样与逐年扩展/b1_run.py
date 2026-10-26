# -*- coding: utf-8 -*-
"""b1_run.py — 多账号并行运行 b1 候选生成（按 shard 轮换账号重试）

* shards：(tag, part, nparts) 组合；账号来自 config/accounts_good.json
* 用法：python b1_run.py [--workers 12]
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
LOGS = os.path.join(BASE, 'logs', 'bare_cand')
GOOD = json.load(open(os.path.join(BASE, 'config', 'accounts_good.json'), encoding='utf-8'))
PY = sys.executable
SHARDS = [('south', i, 10) for i in range(10)] + [('east', i, 10) for i in range(10)]


def run_shard(shard, idx):
    tag, part, nparts = shard
    for k in range(len(GOOD)):
        a = GOOD[(idx + k) % len(GOOD)]
        log = open(os.path.join(LOGS, '%s_p%02d_try%d.log' % (tag, part, k)), 'a')
        cmd = [PY, '-u', os.path.join(CODE, 'b1_candidates_gee.py'), '--tag', tag,
               '--acct', a['acct'], '--pid', a['pid'], '--part', str(part), '--nparts', str(nparts)]
        try:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        print('%s p%d ← %s rc=%d' % (tag, part, a['acct'], rc), flush=True)
        if rc == 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_shard, s, i) for i, s in enumerate(SHARDS)]
        for s, f in zip(SHARDS, futs):
            print('%s p%d → %s' % (s[0], s[1], 'OK' if f.result() else 'FAIL'), flush=True)
    # 汇总
    tot = 0
    for tag in ('south', 'east', 'north_plain', 'loess'):
        fs = [os.path.join(BASE, 'data', 'bare_cand', tag, f)
              for f in os.listdir(os.path.join(BASE, 'data', 'bare_cand', tag))]
        fs = [f for f in fs if f.endswith('.parquet')]
        import pandas as pd
        n = sum(len(pd.read_parquet(f)) for f in fs) if fs else 0
        tot += n
        print('%s: %d 块文件，%d 候选' % (tag, len(fs), n), flush=True)
    print('合计候选 %d（%.0fs）' % (tot, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()

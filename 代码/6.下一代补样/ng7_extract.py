# -*- coding: utf-8 -*-
"""
ng7_extract.py — 下一代样本的年度 AlphaEarth 嵌入提取（舰队驱动）

对 `ng_chunks_index.parquet` 的每个块，用账号专属提取器 `fleet/nge_<acct>.py` 拉 64 维嵌入；
块按 `chunk_id % NACC` 静态分配，落盘即断点续跑。

用法:
  python ng7_extract.py --accts iu5f4z,nhqz5uj,hte4021n,e5h08k --workers 4
  python ng7_extract.py --status
"""
import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ng0_paths as P
import ng0_spec as S

PY = sys.executable


def status():
    idx = os.path.join(P.NG, 'ng_chunks_index.parquet')
    if not os.path.isfile(idx):
        print('尚无年度索引（先跑 ng6_build_samples.py）'); return
    import pandas as pd
    d = pd.read_parquet(idx)
    n_chunk = int(d.chunk_id.max()) + 1
    out_dir = os.path.join(P.NG, 'ng_emb')
    done = 0
    if os.path.isdir(out_dir):
        done = len([f for f in os.listdir(out_dir) if f.endswith('.parquet')])
    print(f'年度索引 {len(d):,} point-years / {n_chunk} 块；已完成 {done} 块')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--accts', default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    P.ensure_all()
    if a.status or not a.accts:
        status(); return
    accts = [x.strip() for x in a.accts.split(',') if x.strip()]
    fleet = json.load(open(P.fleet_json_path(), encoding='utf-8'))
    rows = [(x, fleet[x]['anchor']) for x in accts if x in fleet]
    if not rows:
        raise SystemExit('账号不在指纹台账里: %s' % accts)

    def one(t):
        acct, anchor = t
        wf = os.path.join(P.FLEETD, f'nge_{acct}.py')
        lf = P.log_path(acct, 'emb')
        t0 = time.time()
        with open(lf, 'w', encoding='utf-8') as fh:
            pr = subprocess.run([PY, '-u', wf], stdout=fh, stderr=subprocess.STDOUT,
                                timeout=21600)
        tail = open(lf, encoding='utf-8', errors='replace').read().strip().splitlines()
        return acct, pr.returncode, time.time() - t0, (tail[-1][:100] if tail else '')

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for acct, rc, dt, last in ex.map(one, rows):
            print(f'{"OK " if rc == 0 else "NG "} {acct:16s} {dt/60:6.1f}min  {last}')
    status()


if __name__ == '__main__':
    main()

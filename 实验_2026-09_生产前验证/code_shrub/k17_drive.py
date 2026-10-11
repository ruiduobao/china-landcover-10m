# -*- coding: utf-8 -*-
"""k17_drive.py — S2 物候抓取的多进程驱动（12 账号分片，块级落盘可续跑）

* 输入：data/shrub_nat/nat_points.csv；config/accounts_pool.json 的 workers
* 输出：data/shrub_nat/s2_parts/s2_p{c,c}.parquet（已存在的块跳过 → 幂等续跑）
* 用法：python k17_drive.py [--procs 12]
"""
import argparse, json, os, subprocess, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
CODE = os.path.join(WORK, 'code')
PY = r'F:/anaconda/envs/geeMAP_env/python.exe'
OUTD = os.path.join(WORK, 'data', 'shrub_nat')
LOGD = os.path.join(OUTD, 's2_log')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(LOGD, exist_ok=True)
    pool = json.load(open(os.path.join(WORK, 'config', 'accounts_pool.json'), encoding='utf-8'))
    accts = (pool['workers'] * 2)[:a.procs]
    procs = []
    for i in range(a.procs):
        log = open(os.path.join(LOGD, 'p%02d.log' % i), 'w', encoding='utf-8')
        cmd = [PY, os.path.join(CODE, 'k17_s2_fetch.py'), '--part', str(i),
               '--nparts', str(a.procs), '--acct', accts[i]]
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        emit('分片 %d → %s' % (i, accts[i]))
        time.sleep(3)
    t0 = time.time()
    while True:
        alive = [1 for _, pr, _ in procs if pr.poll() is None]
        parts = os.path.join(OUTD, 's2_parts')
        n = len([f for f in os.listdir(parts) if f.endswith('.parquet')]) if os.path.isdir(parts) else 0
        emit('存活 %d/%d ｜ 块 %d/29 ｜ %.0f 分钟' % (len(alive), len(procs), n, (time.time() - t0) / 60))
        if not alive:
            break
        time.sleep(300)
    emit('结束：%d 块' % len([f for f in os.listdir(os.path.join(OUTD, 's2_parts')) if f.endswith('.parquet')]))


if __name__ == '__main__':
    main()

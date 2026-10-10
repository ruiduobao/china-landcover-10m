# -*- coding: utf-8 -*-
"""h2b_drive.py — h2b_feats 的多进程驱动（贵层特征并行抓取，账号轮换）

* 输入：点表 csv/parquet；config/accounts_pool.json 的 workers
* 门槛：每进程写 <outstem>_p%02d.parquet；已存在的跳过（幂等）
* 输出：同 --outstem 的多个分片 parquet
* 用法：python h2b_drive.py --in 点表.csv --out 结果.parquet --procs 8 [--nbr 1]
"""
import argparse, json, os, subprocess, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
CODE = os.path.join(WORK, 'code')
PY = r'F:/anaconda/envs/geeMAP_env/python.exe'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--nbr', type=int, default=1)
    a = ap.parse_args()
    pool = json.load(open(os.path.join(WORK, 'config', 'accounts_pool.json'), encoding='utf-8'))
    accts = (pool['workers'] * 3)[:a.procs]
    base = os.path.splitext(a.out)[0]
    procs = []
    for i in range(a.procs):
        fp = '%s_p%02d.parquet' % (base, i)
        if os.path.exists(fp):
            emit('分片 %d 已存在，跳过' % i)
            continue
        log = open(fp + '.log', 'w', encoding='utf-8')
        cmd = [PY, os.path.join(CODE, 'h2b_feats.py'), '--in', a.inp, '--out', a.out,
               '--part', str(i), '--nparts', str(a.procs), '--nbr', str(a.nbr),
               '--acct', accts[i]]
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        emit('分片 %d → %s' % (i, accts[i]))
        time.sleep(3)
    while True:
        alive = [1 for _, pr, _ in procs if pr.poll() is None]
        done = len([f for f in os.listdir(os.path.dirname(a.out) or '.')
                    if f.startswith(os.path.basename(base) + '_p') and f.endswith('.parquet')])
        emit('存活 %d/%d ｜ 已落盘 %d 分片' % (len(alive), len(procs), done))
        if not alive:
            break
        time.sleep(90)
    emit('结束')


if __name__ == '__main__':
    main()

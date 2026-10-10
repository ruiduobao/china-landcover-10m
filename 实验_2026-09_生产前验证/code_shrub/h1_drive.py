# -*- coding: utf-8 -*-
"""h1_drive.py — 全国灌丛候选生成的多进程驱动（账号轮换 + 进度看板）

* 输入：config/accounts_pool.json 的 workers 账号；h1_nat_cand.py 的盒表
* 规则源：单账号单进程（ee.Initialize 非线程安全）；进程数 = 账号数
* 门槛：每进程落 data/shrub_nat/parts/cand_<part>.parquet；已存在的 part 跳过（幂等续跑）
* 输出：data/shrub_nat/parts/*.parquet + data/shrub_nat/drive_log/ 子进程日志
* 用法：python h1_drive.py [--procs 11] [--cap 1500] [--only 326]
"""
import argparse, json, os, subprocess, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
CODE = os.path.join(WORK, 'code')
PY = r'F:/anaconda/envs/geeMAP_env/python.exe'
OUTD = os.path.join(WORK, 'data', 'shrub_nat')
PARTD = os.path.join(OUTD, 'parts')
LOGD = os.path.join(OUTD, 'drive_log')


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=11)
    ap.add_argument('--cap', type=int, default=1500)
    ap.add_argument('--rand', type=int, default=25000)
    ap.add_argument('--tag', default='', help='输出 part 名后缀')
    ap.add_argument('--only', type=str, default='', help='逗号分隔的 part 索引，仅跑这些')
    a = ap.parse_args()
    os.makedirs(PARTD, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)
    pool = json.load(open(os.path.join(WORK, 'config', 'accounts_pool.json'), encoding='utf-8'))
    accts = pool['workers'][:a.procs]
    parts = list(range(a.procs))
    if a.only:
        parts = [int(x) for x in a.only.split(',') if x != '']
        accts = (pool['workers'] * len(parts))[:len(parts)]
    emit('驱动启动：%d 进程 ｜ 账号 %s' % (len(parts), ' '.join(accts)))
    procs = []
    for i, p in enumerate(parts):
        fp = os.path.join(PARTD, 'cand_%03d.parquet' % p)
        if os.path.exists(fp):
            emit('  part %d 已存在，跳过' % p)
            continue
        log = open(os.path.join(LOGD, 'part_%03d.log' % p), 'w', encoding='utf-8')
        cmd = [PY, os.path.join(CODE, 'h1_nat_cand.py'), '--part', str(p),
               '--nparts', str(a.procs), '--cap', str(a.cap), '--rand', str(a.rand),
               '--acct', accts[i]]
        if a.tag:
            cmd += ['--tag', a.tag]
        procs.append((p, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        emit('  part %d → %s' % (p, accts[i]))
        time.sleep(3)
    # 看板
    t0 = time.time()
    while True:
        alive = [(p, pr) for p, pr, _ in procs if pr.poll() is None]
        done = [f for f in os.listdir(PARTD) if f.endswith('.parquet')]
        emit('存活 %d/%d ｜ 已落盘 %d 个 part ｜ 运行 %.0f 分钟' % (
            len(alive), len(procs), len(done), (time.time() - t0) / 60))
        if not alive:
            break
        time.sleep(120)
    emit('全部进程结束，落盘 %d 个 part' % len([f for f in os.listdir(PARTD) if f.endswith('.parquet')]))


if __name__ == '__main__':
    main()

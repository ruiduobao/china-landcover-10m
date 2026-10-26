# -*- coding: utf-8 -*-
"""run_g1.py — G-30N 几何实验 harness：**父进程硬截止先启动**，再跑子实验

* 职责（doc70 §3.2 / §4.1）
    1) 单实例锁 + 资源守卫；
    2) **deadline 时钟在启动子进程（首个训练/推理请求）之前起算**；
    3) 子进程 stdout/stderr 落 `logs/<exp_id>/run.log`，原子写 `run_results/<exp_id>.json`；
    4) 到点（或超 `--limit`）**终止自己的子进程树**（taskkill /T），记 `TIMEOUT`、退出码、已完成阶段；
    5) 结果登记进 `registry/experiments.json`（read-modify-write + 锁文件，保留他人条目）。
* 纪律：只终止自己启动的子进程；不取消任何 GEE 任务（本实验不发 batch）；不改冻结 c3。
* 用法
    python run_g1.py --exp G30N_seed7 --limit 1200 -- --seed 7
    python run_g1.py --exp G30N_tiny   --limit 300  -- --tiny
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
RUND = os.path.join(HERE, 'run_results')
LOGD = os.path.join(HERE, 'logs')
LOCKD = os.path.join(HERE, 'locks')
REG = os.path.join(WS, 'registry', 'experiments.json')
SCRIPT = os.path.join(HERE, 'g1_30n.py')


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def merge_registry(exp_id, rec):
    """把本实验结果并入共享 registry（保留其他 agent 条目）。"""
    os.makedirs(os.path.dirname(REG), exist_ok=True)
    lock = REG + '.lock'
    for _ in range(40):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.25)
    else:
        log('⚠️ registry 锁未取得，只写本地 reg/')
        return False
    try:
        d = {}
        if os.path.exists(REG):
            try:
                d = json.load(open(REG, encoding='utf-8'))
            except Exception:
                d = {}
        d.setdefault('schema', 'improve_pilot_registry/v1')
        d.setdefault('experiments', {})
        d['experiments'][exp_id] = rec
        d['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        tmp = REG + '.tmp%d' % os.getpid()
        json.dump(d, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        os.replace(tmp, REG)
        return True
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--limit', type=float, default=1200.0)
    ap.add_argument('--script', default=SCRIPT)
    ap.add_argument('rest', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = [x for x in a.rest if x != '--']
    os.makedirs(RUND, exist_ok=True)
    os.makedirs(os.path.join(LOGD, a.exp), exist_ok=True)
    os.makedirs(LOCKD, exist_ok=True)

    lk = os.path.join(LOCKD, 'geometry.lock')
    if os.path.exists(lk):
        try:
            old = int(open(lk).read().strip() or 0)
        except Exception:
            old = 0
        if old and old != os.getpid():
            out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % old],
                                 capture_output=True, text=True).stdout
            if 'python.exe' in out:
                log('已有几何实验在跑（PID %d），退出' % old)
                sys.exit(2)
    open(lk, 'w').write(str(os.getpid()))

    # ---- deadline 在首个请求之前起算 ----
    t_start = time.time()
    deadline = t_start + a.limit
    log('exp=%s limit=%.0fs script=%s args=%s' % (a.exp, a.limit, os.path.basename(a.script), rest))
    logf = open(os.path.join(LOGD, a.exp, 'run.log'), 'w', encoding='utf-8')
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    cmd = [sys.executable, '-u', a.script] + rest
    logf.write('CMD %s\n' % ' '.join(cmd))
    logf.flush()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding='utf-8', errors='ignore', env=env)
    status, rc, timed_out = 'done', None, False
    try:
        while True:
            line = p.stdout.readline()
            if line:
                logf.write(line)
                logf.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            if p.poll() is not None:
                for line in p.stdout:
                    logf.write(line)
                rc = p.returncode
                break
            if time.time() > deadline:
                timed_out = True
                log('⏱ 到点 %.0fs，终止子进程树 PID %d' % (a.limit, p.pid))
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)],
                               capture_output=True, text=True)
                time.sleep(1.0)
                if p.poll() is None:
                    p.kill()
                rc = p.returncode
                status = 'timeout'
                break
    finally:
        logf.close()
        try:
            os.remove(lk)
        except OSError:
            pass

    wall = round(time.time() - t_start, 1)
    if timed_out:
        status = 'timeout'
    elif rc != 0:
        status = 'failed'
    rec = dict(exp_id=a.exp, owner='geometry', stage='S4', limit_s=a.limit,
               status=status, rc=rc, timed_out=timed_out, wall_s=wall,
               started_at=time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t_start)),
               finished_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
               log=os.path.join(LOGD, a.exp, 'run.log'),
               budget_ok=bool(wall <= a.limit and not timed_out))
    json.dump(rec, open(os.path.join(RUND, '%s.json' % a.exp), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    merge_registry(a.exp, rec)
    log('%s → %s（wall %.1fs / limit %.0fs）' % (status.upper(), os.path.join(RUND, a.exp + '.json'),
                                                 wall, a.limit))
    # 汇总实验自身 eval（若已产出）
    ev = os.path.join(HERE, 'out', '%s_eval.json' % a.exp)
    if os.path.exists(ev):
        d = json.load(open(ev, encoding='utf-8'))
        log('eval 关键量：%s' % json.dumps({k: d.get(k) for k in
            ['evm_in_tile_valid_px', 'legacy_gap_px', 'planar_gap_px', 'restored_px',
             'outside_wedge_agree_rate', 'req_s', 'elapsed_s']}, ensure_ascii=False))
    sys.exit(0 if status == 'done' else (3 if status == 'timeout' else 4))


if __name__ == '__main__':
    main()

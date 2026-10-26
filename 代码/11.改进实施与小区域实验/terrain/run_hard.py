# -*- coding: utf-8 -*-
"""run_hard.py — 父进程硬截止包装：子进程运行 t2_rf_eval.py，父进程超时强杀（默认 1200s）

* 为什么：t2 内置的循环 deadline 检查无法中断 sklearn 的单次 fit；真正的硬截止必须由父进程
  对子进程施加（subprocess timeout → kill）。doc70 §3.2 要求"监督进程设置硬超时，达到截止
  终止自己启动的子进程"。
* 输出：stdout 追加 [HARD] 计时/退出码；out/t2_runlog_<frame>.json 记录 wall/rc/deadline/命令。
* 用法：python run_hard.py --frame F1 --arms main --seeds 7,11,23 --n-jobs 10 [--deadline 1200]
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'out')


def main():
    argv = sys.argv[1:]
    deadline = 1200
    frame = 'F?'
    for i, a in enumerate(argv):
        if a == '--deadline' and i + 1 < len(argv):
            deadline = int(argv[i + 1])
        if a == '--frame' and i + 1 < len(argv):
            frame = argv[i + 1]
    cmd = [sys.executable, os.path.join(HERE, 't2_rf_eval.py')] + argv
    print('[HARD] deadline=%ds cmd=%s' % (deadline, ' '.join(cmd)), flush=True)
    t0 = time.time()
    rc, timed_out = None, False
    try:
        p = subprocess.run(cmd, timeout=deadline)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = -9
        print('[HARD] TIMEOUT：%ds 到，父进程强杀子进程' % deadline, flush=True)
    wall = round(time.time() - t0, 1)
    print('[HARD] wall=%.1fs rc=%s timed_out=%s' % (wall, rc, timed_out), flush=True)
    os.makedirs(OUTD, exist_ok=True)
    with open(os.path.join(OUTD, 't2_runlog_%s.json' % frame), 'w', encoding='utf-8') as f:
        json.dump(dict(frame=frame, cmd=cmd, deadline_s=deadline, wall_seconds=wall,
                       returncode=rc, timed_out=timed_out,
                       generated=time.strftime('%Y-%m-%d %H:%M:%S')), f,
                  ensure_ascii=False, indent=1)
    sys.exit(0 if rc == 0 else 1)


if __name__ == '__main__':
    main()

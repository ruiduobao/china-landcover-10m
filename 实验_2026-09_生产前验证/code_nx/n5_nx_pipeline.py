# -*- coding: utf-8 -*-
"""n5_nx_pipeline.py — 等 6 瓦终态 → 下载+保真 → 与交付 R0 构成对比（无人值守）

* 输入：F:/lc_work/prod5p_2023/nx_r4_state.json
* 规则源：n3_nx_r4.py 的 fetch/compare；每完成一瓦即下载，全部终态后出对比
* 门槛：保真 3000 点须 100%；失败瓦片打印原因（不中断）
* 输出：rasters_r4/<tile>_10m.tif、rasters_r4/构成对比.csv、日志 stdout
* 用法：python n5_nx_pipeline.py
"""
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
CODE = r'F:/lc_work/v31_exp/code'
STATE = r'F:/lc_work/prod5p_2023/nx_r4_state.json'
TERMINAL = ('COMPLETED', 'FAILED', 'CANCELLED')
DEADLINE = time.time() + 8 * 3600
PY = sys.executable


def log(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def run(args):
    r = subprocess.run([PY, CODE + '/n3_nx_r4.py'] + args, capture_output=True, text=True,
                       encoding='utf-8', errors='ignore', cwd=CODE)
    for line in (r.stdout or '').splitlines():
        if 'retry' in line or 'Warning' in line or 'warn' in line:
            continue
        log('  ' + line)
    if r.returncode != 0:
        log('  命令失败 rc=%d：%s' % (r.returncode, (r.stderr or '')[-200:]))


def main():
    while time.time() < DEADLINE:
        import json
        st = json.load(open(STATE, encoding='utf-8'))
        states = {t: r.get('state') for t, r in st['tiles'].items()}
        done = [t for t, s in states.items() if s == 'COMPLETED']
        todo_dl = [t for t in done if not st['tiles'][t].get('fidelity', {}).get('rate') == 1.0]
        if todo_dl:
            log('下载 %d 瓦：%s' % (len(todo_dl), ','.join(sorted(todo_dl))))
            run(['fetch', '--tiles', ','.join(sorted(todo_dl))])
        if all(s in TERMINAL for s in states.values()):
            log('全部终态：%s' % states)
            log('构成对比 …')
            run(['compare'])
            break
        time.sleep(300)


if __name__ == '__main__':
    main()

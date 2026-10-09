# -*- coding: utf-8 -*-
"""f13_sc_v2_pipeline.py — 四川 v2 流水线：等终态 → 逐瓦下载+保真 → 逐瓦面积表

* 输入：F:/lc_work/prod5p_2023/sc_v2_state.json
* 门槛：保真 3000 点须 100%；失败瓦片打印原因不中断
* 输出：rasters_v2_sc/<tile>_10m.tif + qa_v2_sc/<tile>_面积.csv
* 用法：python f13_sc_v2_pipeline.py
"""
import subprocess, sys, time
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
STATE = r'F:/lc_work/prod5p_2023/sc_v2_state.json'
TERM = ('COMPLETED', 'FAILED', 'CANCELLED')
PY_EXE = r'F:/anaconda/envs/geeMAP_env/python.exe'
DEADLINE = time.time() + 10 * 3600

def log(m): print(time.strftime('[%m-%d %H:%M:%S] ')+str(m), flush=True)

def run(args):
    r = subprocess.run([PY_EXE, r'F:/lc_work/v31_exp/code/f11_sichuan_v2.py'] + args,
                       capture_output=True, text=True, encoding='utf-8', errors='ignore',
                       cwd=r'F:/lc_work/v31_exp/code')
    for ln in (r.stdout or '').splitlines():
        if any(k in ln for k in ('retry', 'Warning', 'warn')): continue
        log('  ' + ln)

def main():
    while time.time() < DEADLINE:
        st = VC.jload(STATE, {'tiles': {}})
        done = [t for t, v in st['tiles'].items() if v.get('state') == 'COMPLETED']
        todo = [t for t in done if not (st['tiles'][t].get('fidelity', {}) or {}).get('rate') == 1.0]
        if todo:
            log('下载 %d 瓦：%s' % (len(todo), ','.join(sorted(todo))))
            run(['fetch', '--tiles', ','.join(sorted(todo))])
        if all(v.get('state') in TERM for v in st['tiles'].values()):
            log('全部终态'); break
        time.sleep(300)

if __name__ == '__main__': main()

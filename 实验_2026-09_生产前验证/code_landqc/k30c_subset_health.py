# -*- coding: utf-8 -*-
"""k30c_subset_health.py — 指定账号子集的项目级健康检查（零EECU；子进程隔离 HOME）

* 输入：命令行账号名列表（默认用下方 DEF 列表）
* 方法：复用 s0b_health.check（每账号一子进程，并行 --workers）
* 输出：config/fleet_health_subset.json
* 用法：python k30c_subset_health.py acct1 acct2 ... [--workers 6] [--tag landqc]
"""
import argparse, json, os, subprocess, sys, time
sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'; CODE = os.path.join(WORK, 'code'); PY = sys.executable
OUT = os.path.join(WORK, 'config', 'fleet_health_subset.json')
DEF = ['ccstqqb','ch2fg2t9','chengruiduobao','hrbm3pdw','ief3nj','n08zt4',
       'n51j08y','nama8lg','neg69g','nhqz5uj','nkhn6s9','s4ezbd','sgqv7bf','udn4q4',
       'v054r8u','vh481j','w2qe4hiu','x8diho2n','xysongf','zixen8v8','zsi8emo',
       'vidalmvtpronet18','vlgvl6u','x2zdn8','zvhvax2j','ughwvm7968.med']
CHILD = r'''
import json, os, sys
sys.path.insert(0, r"%s")
import s0b_health
print(json.dumps(s0b_health.check(r"%%s"), ensure_ascii=False))
''' % CODE


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('accts', nargs='*')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    accts = a.accts or DEF
    rows = []
    pend = list(accts)
    running = []
    while pend or running:
        while pend and len(running) < a.workers:
            ac = pend.pop(0)
            p = subprocess.Popen([PY, '-c', CHILD % ac], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, encoding='utf-8')
            running.append((ac, p, time.time()))
        ac, p, t0 = running.pop(0)
        try:
            out, _ = p.communicate(timeout=240)
            r = json.loads(out.strip().splitlines()[-1]); r['acct'] = ac
        except Exception as e:
            r = dict(acct=ac, ok=False, reason='TIMEOUT/ERR %s' % str(e)[:60])
        rows.append(r)
        emit('%s %s %s' % (ac, 'OK' if r.get('ok') else 'BAD', str(r.get('reason'))[:60]))
    ok = [r['acct'] for r in rows if r.get('ok')]
    json.dump(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), n=len(rows), healthy=ok, rows=rows),
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('健康 %d/%d：%s' % (len(ok), len(rows), ok))


if __name__ == '__main__':
    main()

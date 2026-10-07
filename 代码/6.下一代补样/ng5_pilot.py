# -*- coding: utf-8 -*-
"""
ng5_pilot.py — 小区域试点驱动（跑通链路后才允许批量）

顺序（严格遵守"先试点、后批量"）：
  1. ng1_plan.py        生成分片清单（本地，零 EECU）
  2. ng2_preflight.py   每账号：凭证 + 数据集可读 + 门槛构图（只读）
  3. ngw_<acct>.py      各账号跑自己的分片（GEE 交互式，不占 batch 额度）
  4. ng9_qa.py          合并 + 配额 + 硬门槛质检
  5. 出试点报告 技术文档/24_下一代补样试点报告.md

用法:
  python ng5_pilot.py --plan                # 只生成分片清单
  python ng5_pilot.py --preflight           # 只做前置体检（全部编队账号）
  python ng5_pilot.py --run --workers 4     # 跑试点（并发=账号数，本机 CPU 负载可忽略）
  python ng5_pilot.py --qa                  # 只做质检
  python ng5_pilot.py --all --workers 4     # 一条龙
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


def load_accounts():
    fp = os.path.join(P.FLEET, 'ng_accounts.json')
    if not os.path.isfile(fp):
        raise SystemExit(f'缺少 {fp}')
    return json.load(open(fp, encoding='utf-8'))


def sh(cmd, log=None, timeout=7200):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       encoding='utf-8', errors='replace')
    dt = time.time() - t0
    tail = (r.stdout or '')[-1500:] + (r.stderr or '')[-1500:]
    if log:
        open(log, 'a', encoding='utf-8').write(tail)
    return r.returncode, dt, tail


def do_plan(a):
    return sh([PY, os.path.join(P.CODED, 'ng1_plan.py')])


def do_preflight(rows):
    def one(r):
        return (r['acct'],) + sh([PY, os.path.join(P.CODED, 'ng2_preflight.py'),
                                  '--acct', r['acct'], '--anchor', r['anchor'],
                                  '--spec', r['spec']], timeout=1800)
    ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for acct, rc, dt, tail in ex.map(one, rows):
            flag = '✅' if rc == 0 else '❌'
            ok += (rc == 0)
            print(f'{flag} {acct:16s} {dt:6.0f}s  {tail.strip().splitlines()[-1][:90] if tail.strip() else ""}')
    print(f'前置体检: {ok}/{len(rows)} 通过')
    return ok


def do_run(rows, workers):
    def one(r):
        wf = os.path.join(P.FLEETD, f'ngw_{r["acct"]}.py')
        lf = P.log_path(r['acct'], 'pilot')
        t0 = time.time()
        # 实时落盘：工作器一提交就能 tail 日志，避免"看不到进度＝以为卡死"
        with open(lf, 'w', encoding='utf-8') as fh:
            pr = subprocess.run([PY, '-u', wf, '--spec', r['spec']],
                                stdout=fh, stderr=subprocess.STDOUT, timeout=10800)
        dt = time.time() - t0
        tail = open(lf, encoding='utf-8', errors='replace').read()[-1500:]
        return r['acct'], r['spec'], pr.returncode, dt, tail
    res = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for acct, spec, rc, dt, tail in ex.map(one, rows):
            last = [l for l in tail.strip().splitlines() if l.strip()]
            print(f'{"✅" if rc == 0 else "❌"} {acct:16s} {spec:14s} {dt/60:5.1f}min  '
                  f'{last[-1][:95] if last else ""}')
            res.append({'acct': acct, 'spec': spec, 'rc': rc, 'min': round(dt / 60, 1)})
    return res


def do_qa(specs):
    out = []
    for s in specs:
        rc, dt, tail = sh([PY, os.path.join(P.CODED, 'ng9_qa.py'), '--spec', s])
        print(tail.strip().splitlines()[-1] if tail.strip() else f'{s}: 无输出')
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--qa', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only-spec')
    ap.add_argument('--phase', type=int, default=1,
                    help='1=已实现门槛的家族（试点）；0=全部编队账号')
    a = ap.parse_args()
    P.ensure_all()
    rows = load_accounts()
    if a.phase:
        rows = [r for r in rows if int(r.get('phase', 1)) == a.phase]
    if a.only_spec:
        rows = [r for r in rows if r['spec'] == a.only_spec]
    specs = sorted({r['spec'] for r in rows})

    print(f'试点编队: {len(rows)} 账号 / {len(specs)} spec')
    t0 = time.time()
    if a.plan or a.all:
        rc, dt, tail = do_plan(a)
        print(tail.strip())
    if a.preflight or a.all:
        do_preflight(rows)
    if a.run or a.all:
        do_run(rows, a.workers)
    if a.qa or a.all:
        do_qa(specs)
    print(f'总耗时 {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()

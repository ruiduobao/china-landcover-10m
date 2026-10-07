# -*- coding: utf-8 -*-
"""
p4_watch.py — 生产监控与额度台账（只读；随时可跑）
做三件事：
  1) 逐账号读 operations（GET .../v1/projects/{pid}/operations，**路径不带 :list**）
  2) 汇总各账号 活动任务数 / 累计 EECU·h / 最近状态变化，写 dashboard
  3) 从各账号台账 jsonl 统计产出进度（按 tile×year），给出剩余与预估
输出：Z:/生产输出/元数据/dashboard.json + dashboard.csv，控制台表
用法：python p4_watch.py [--acct a b] [--no-net]
"""
import os, sys, io, json, time, glob, argparse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def read_ops(acct):
    import requests, google.oauth2.credentials
    from google.auth.transport.requests import AuthorizedSession
    from ee import oauth as ee_oauth
    pid = C.ACCOUNTS[acct]['proj']
    row = {'acct': acct, 'pid': pid, 'run': 0, 'pend': 0, 'succ': 0, 'fail': 0,
           'cancel': 0, 'eecu_h': 0.0, 'live': []}
    cred = os.path.join(C.cred_home(acct), '.config', 'earthengine', 'credentials')
    if not (pid and os.path.isfile(cred)):
        row['err'] = 'no cred/proj'
        return row
    with io.open(cred, encoding='utf-8') as f:
        cj = json.load(f)
    creds = google.oauth2.credentials.Credentials(
        token=None, refresh_token=cj['refresh_token'],
        token_uri='https://oauth2.googleapis.com/token',
        client_id=ee_oauth.CLIENT_ID, client_secret=ee_oauth.CLIENT_SECRET,
        scopes=cj.get('scopes', ['https://www.googleapis.com/auth/earthengine']))
    sess = AuthorizedSession(creds)
    sess.proxies = C.PROXY
    try:
        r = sess.get('https://earthengine.googleapis.com/v1/projects/%s/operations' % pid,
                     params={'pageSize': 250},
                     headers={'X-Goog-User-Project': pid}, timeout=60)
        if r.status_code != 200:
            row['err'] = 'HTTP %d' % r.status_code
            return row
        rows = []
        for op in r.json().get('operations', []):
            md = op.get('metadata', {})
            st = md.get('state') or ('DONE' if op.get('done') else 'UNKNOWN')
            ec = 0.0
            for k, v in md.items():
                if 'ecu' in k.lower():
                    try:
                        ec = float(v)
                    except (TypeError, ValueError):
                        pass
            row['eecu_h'] += ec / 3600.0
            key = {'RUNNING': 'run', 'PENDING': 'pend', 'SUCCEEDED': 'succ',
                   'FAILED': 'fail', 'CANCELLED': 'cancel'}.get(st)
            if key:
                row[key] += 1
            if key in ('run', 'pend'):
                row['live'].append({'desc': md.get('description', ''),
                                    'state': st, 'task': op.get('name', '').split('/')[-1]})
            rows.append({'acct': acct, 'task': op.get('name', '').split('/')[-1],
                         'desc': md.get('description', ''), 'state': st, 'eecu_s': ec,
                         'begin': md.get('startTime'), 'end': md.get('endTime'),
                         'ts': time.strftime('%Y-%m-%d %H:%M:%S')})
        # 回写本账号台账（追加）
        if rows:
            os.makedirs(C.LEDGER, exist_ok=True)
            with io.open(os.path.join(C.LEDGER, 'ledger_%s.jsonl' % acct), 'a',
                         encoding='utf-8') as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + '\n')
        row['eecu_h'] = round(row['eecu_h'], 1)
    except Exception as e:
        row['err'] = str(e)[:80]
    return row


def read_progress():
    """从台账统计每个 (tile,year) 的最终状态"""
    st, task_state = {}, {}
    rows = []
    for fp in glob.glob(os.path.join(C.LEDGER, 'ledger_*.jsonl')):
        for line in io.open(fp, encoding='utf-8', errors='replace'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append(r)
            if r.get('task'):
                task_state[r['task']] = r.get('state', '')
    for r in rows:      # 两遍扫描：先收全 task→state，再匹配瓦片（行是按时间追加的）
        t, y = r.get('tile'), r.get('year')
        if not t or y is None:
            continue
        # **以最后一次提交为准**（同一瓦片失败重提后，旧 FAILED 不能盖掉新状态）
        s = task_state.get(r.get('task'), r.get('state', ''))
        if s == 'DONE_FETCHED':      # 已下载 ⇒ 必然已 SUCCEEDED
            s = 'SUCCEEDED'
        st[(t, int(y))] = s
    # 只统计区域计划内的瓦片（排除 walkthrough 的 X001/2022 之类）
    valid = set()
    for fp in glob.glob(os.path.join(C.PLAN, 'tiles_ne*_*.json')):
        try:
            d_ = json.load(open(fp, encoding='utf-8'))
        except Exception:
            continue
        yl = d_.get('years', [2023])
        for tt in d_.get('tiles', []):
            for yy in yl:
                valid.add((tt['tile'], int(yy)))
    return {k: v for k, v in st.items() if k in valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', nargs='*', default=None)
    ap.add_argument('--no-net', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    C.ensure_dirs()
    accts = a.acct or [k for k, v in C.ACCOUNTS.items() if v['role'] in ('free', 'busy')]
    rows = []
    if not a.no_net:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(read_ops, accts))
    else:
        rows = [{'acct': x, 'eecu_h': None, 'run': None, 'pend': None} for x in accts]
    print(f"\n{'账号':<16}{'项目':<26}{'RUN':>5}{'PEND':>6}{'SUCC':>6}{'FAIL':>6}{'EECU·h':>9}")
    for r in sorted(rows, key=lambda x: (x.get('run') or 0, x.get('eecu_h') or 0),
                    reverse=True):
        print(f"{r['acct']:<16}{r.get('pid',''):<26}{r.get('run',-1):>5}{r.get('pend',-1):>6}"
              f"{r.get('succ',-1):>6}{r.get('fail',-1):>6}{r.get('eecu_h',-1):>9}"
              f"{'  ' + r.get('err','') if r.get('err') else ''}")
    tot_eecu = sum(r.get('eecu_h') or 0 for r in rows)
    tot_live = sum((r.get('run') or 0) + (r.get('pend') or 0) for r in rows)
    print(f"\n合计 EECU·h ≈ {tot_eecu:,.0f} ｜ 活动任务 {tot_live} 个 ｜ 可用账号 {len(C.usable_accounts())}")

    st = read_progress()
    # 目标任务数：有区域计划文件就用区域口径，否则用全国口径
    import glob as _glob
    _all = os.path.join(C.PLAN, 'tiles_ne_all.json')   # 全集清单（36），避免计划文件被拆分后目标数缩水
    reg = ([_all] if os.path.isfile(_all) else []) or sorted(_glob.glob(os.path.join(C.PLAN, 'tiles_ne*_*.json')))
    if reg:
        uniq = set()      # 支援计划会与省计划重叠 → 必须按 (tile, year) 去重
        for fp in reg:
            d_ = json.load(open(fp, encoding='utf-8'))
            for tt in d_.get('tiles', []):
                for yy in d_.get('years', [2023]):
                    uniq.add((tt['tile'], int(yy)))
        n_total = len(uniq)
    else:
        n_total = 313 * len(C.YEARS)
    n_ok = sum(1 for v in st.values() if v == 'SUCCEEDED')
    n_fail = sum(1 for v in st.values() if v == 'FAILED')
    print(f"产出进度：SUCCEEDED {n_ok} / 目标 {n_total}（FAILED {n_fail}）"
          f"  ≈ {100.0*n_ok/n_total:.2f}%")
    rep = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'accounts': rows,
           'total_eecu_h': round(tot_eecu, 1), 'live_tasks': tot_live,
           'progress': {'succeeded': n_ok, 'failed': n_fail, 'target': n_total}}
    os.makedirs(C.LEDGER, exist_ok=True)
    json.dump(rep, open(os.path.join(C.LEDGER, 'dashboard.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(C.LEDGER, 'dashboard.csv'),
                                  index=False, encoding='utf-8-sig')
    except Exception:
        pass
    print('台账目录:', C.LEDGER)


if __name__ == '__main__':
    main()

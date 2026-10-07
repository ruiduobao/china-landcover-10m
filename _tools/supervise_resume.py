# -*- coding: utf-8 -*-
"""supervise_resume.py — 主嵌入续跑监督器（10 账号编队，2026-09-12 用户确认版）
* 编队: gee_accounts\\fleet_new_13accounts.json（10 账号，禁动 19 个之外；3 个备用未入队）
* 每 5 分钟: ①进程死亡自愈拉起 ②新增嵌入块增量备份到 K 盘 ③刷新资源台账
  每 30 分钟: batch EECU 扫描（GET operations，只读，证明不占 150h/仓额度）
* 启动时: 向每账号 _任务登记.md 追加任务行；完成时回写。
* 4,758 块齐 → 回写登记 + 退出。
"""
import os, sys, json, time, glob, shutil, subprocess, datetime

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
ACC_BASE = r'F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts'
FLEET_JSON = os.path.join(ACC_BASE, 'fleet_new_13accounts.json')
OUT_DIR = r'Z:\地理所\论文\中国土地覆盖数据_2017-2024\数据\本地处理\全国清洗训练\emb_parts_r7'
BK_DIR = r'Z:\地理所\论文\中国土地覆盖数据_2017-2024\数据\本地处理\全国清洗训练\emb_parts_r7'  # 迁移后与 OUT_DIR 相同，backup 自动跳过
IDX = r'F:\r7_prod\chunks_index_r7.parquet'
LOGD = r'F:\r7_prod\logs'
LEDGER_JSON = os.path.join(LOGD, '资源台账.json')
LEDGER_MD = os.path.join(LOGD, '资源台账.md')
TOTAL_TARGET = 4758
PROXY = {"http": "socks5h://127.0.0.1:7890", "https": "socks5h://127.0.0.1:7890"}
os.makedirs(LOGD, exist_ok=True)
os.makedirs(BK_DIR, exist_ok=True)

_START = time.time()
_SESS = {}          # acct -> AuthorizedSession（EECU 扫描用）


def now():
    return datetime.datetime.now().strftime('%m-%d %H:%M')


def done_count():
    return len(glob.glob(os.path.join(OUT_DIR, 'chunkr7_*.parquet')))


def register_task(accts, done=False):
    """向每账号 _任务登记.md 追加/回写任务行"""
    for a in accts:
        p = os.path.join(ACC_BASE, a, '_任务登记.md')
        if not os.path.isfile(p):
            continue
        s = open(p, encoding='utf-8').read()
        mark = 'e1b_resume_20260912'
        if done:
            line = (f"| 2026-09-12 | (多仓交互) | {mark} | AEF年度嵌入交互提取(断点续跑) "
                    f"| {now()} | {now()} | 完成 | 交互(不占batch) | — | 全部块齐，监督器退出 |")
        else:
            line = (f"| 2026-09-12 | (多仓交互) | {mark} | AEF年度嵌入交互提取(断点续跑) "
                    f"| {now()} | — | 进行中 | 交互(不占batch) | — "
                    f"| 监督器自愈续跑；台账 F:\\\\r7_prod\\\\logs\\\\资源台账.md |")
        if mark not in s:
            s = s.rstrip() + '\n' + line + '\n'
            open(p, 'w', encoding='utf-8', newline='').write(s)


def backup_new():
    """把新块增量拷到备份目录（迁移后 OUT=BK，自动跳过）"""
    if os.path.abspath(OUT_DIR) == os.path.abspath(BK_DIR):
        return 0
    n = 0
    for f in glob.glob(os.path.join(OUT_DIR, 'chunkr7_*.parquet')):
        dst = os.path.join(BK_DIR, os.path.basename(f))
        if not os.path.exists(dst):
            try:
                shutil.copyfile(f, dst)
                n += 1
            except Exception as e:
                print(f'[backup] {os.path.basename(f)} ERR {str(e)[:80]}', flush=True)
    return n


def alive(acct):
    try:
        import psutil
        for p in psutil.process_iter(['cmdline']):
            s = ' '.join(p.info.get('cmdline') or [])
            if 'run_e1b_resume.py' in s and acct in s:
                return True
        return False
    except ImportError:
        r = subprocess.run(['wmic', 'process', 'get', 'commandline'],
                           capture_output=True, text=True, timeout=60)
        return f'run_e1b_resume.py {acct}' in (r.stdout or '')


def launch(acct, todo_n):
    log = open(os.path.join(LOGD, f'resume_{acct}.log'), 'a', encoding='utf-8')
    subprocess.Popen([sys.executable, os.path.join(ROOT, '_tools', 'run_e1b_resume.py'), acct],
                     stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f'[{now()}] 拉起 {acct}（待 {todo_n} 块）', flush=True)
    time.sleep(3)


def acct_blocks(acct):
    """从 worker 日志统计该账号累计成功块数与 FINAL-FAIL 数"""
    ok = fail = 0
    fp = os.path.join(LOGD, f'resume_{acct}.log')
    if os.path.isfile(fp):
        for ln in open(fp, encoding='utf-8', errors='ignore'):
            if ' 行 (' in ln:
                ok += 1
            elif 'FINAL-FAIL' in ln:
                fail += 1
    return ok, fail


def eecu_scan(fleet):
    """GET operations 批量额度扫描（只读）；返回 {acct: eecu_h}"""
    global _SESS
    out = {}
    try:
        import requests as _rq
        import google.oauth2.credentials
        from google.auth.transport.requests import AuthorizedSession
        from ee import oauth as ee_oauth
        for acct, (anchor, _fp) in fleet.items():
            s = _SESS.get(acct)
            if s is None:
                with open(os.path.join(ACC_BASE, acct, '.config', 'earthengine',
                                       'credentials')) as f:
                    c = json.load(f)
                creds = google.oauth2.credentials.Credentials(
                    token=None, refresh_token=c['refresh_token'],
                    token_uri='https://oauth2.googleapis.com/token',
                    client_id=ee_oauth.CLIENT_ID, client_secret=ee_oauth.CLIENT_SECRET,
                    scopes=c.get('scopes', ['https://www.googleapis.com/auth/earthengine',
                                            'https://www.googleapis.com/auth/cloud-platform']))
                s = AuthorizedSession(creds)
                s.proxies = PROXY
                _SESS[acct] = s
            tot = 0.0
            for pid in [anchor]:
                try:
                    r = s.get(f'https://earthengine.googleapis.com/v1/projects/{pid}/operations?pageSize=100',
                              headers={'X-Goog-User-Project': pid}, timeout=45)
                    for op in (r.json().get('operations', []) if r.status_code == 200 else []):
                        md = op.get('metadata', {})
                        for k, v in md.items():
                            if 'ecu' in k.lower():
                                try:
                                    tot += float(v)
                                except (TypeError, ValueError):
                                    pass
                except Exception:
                    pass
            out[acct] = round(tot / 3600.0, 2)
    except Exception as e:
        print(f'[eecu] 扫描失败 {str(e)[:80]}', flush=True)
    return out


def write_ledger(fleet, acct_done):
    el = (time.time() - _START) / 3600
    js = {'updated': now(), 'elapsed_h': round(el, 2),
          'total_done': done_count(), 'target': TOTAL_TARGET, 'accounts': {}}
    md = ['# 资源台账（主嵌入续跑 10 账号编队）', '',
          f'更新: {now()} ｜ 已完成 {done_count()}/{TOTAL_TARGET} 块 ｜ 本监督器运行 {el:.1f}h', '',
          '| 账号 | 本次会话出块 | FINAL-FAIL | batchEECU(anchor仓,h) |', '|---|---|---|---|']
    ee = eecu_scan(fleet)
    for a in fleet:
        ok, fail = acct_blocks(a)
        js['accounts'][a] = {'blocks_session': ok, 'final_fail': fail,
                             'batch_eecu_anchor_h': ee.get(a)}
        md.append(f"| {a} | {ok} | {fail} | {ee.get(a, '?')} |")
    json.dump(js, open(LEDGER_JSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    open(LEDGER_MD, 'w', encoding='utf-8', newline='').write('\n'.join(md) + '\n')


def account_cids(fleet):
    import pandas as pd
    idx = pd.read_parquet(IDX, columns=['chunk_id'])
    cids = sorted(idx.chunk_id.unique().tolist())
    per = {}
    for ai, a in enumerate(fleet):
        per[a] = {c for c in cids if c % len(fleet) == ai}
    return per


def _lock_check():
    pf = os.path.join(LOGD, 'supervisor.pid')
    if os.path.isfile(pf):
        try:
            old = int(open(pf).read().strip())
            import psutil
            p = psutil.Process(old)
            if 'supervise_resume' in ' '.join(p.cmdline() or []):
                print(f'[supervisor] 已有实例在跑 pid={old}，退出', flush=True)
                sys.exit(0)
        except Exception:
            pass
    os.makedirs(LOGD, exist_ok=True)
    open(pf, 'w').write(str(os.getpid()))


def main():
    _lock_check()
    fleet = json.load(open(FLEET_JSON, encoding='utf-8'))
    accts = list(fleet)
    nacc = len(accts)
    per = account_cids(fleet)
    print(f'[supervisor] {now()} 编队 {nacc} 账号，目标 {TOTAL_TARGET}，当前 {done_count()}', flush=True)
    register_task(accts)
    last_scan = 0.0
    while True:
        done_files = {int(os.path.basename(f).split('_')[1].split('.')[0])
                      for f in glob.glob(os.path.join(OUT_DIR, 'chunkr7_*.parquet'))}
        if len(done_files) >= TOTAL_TARGET:
            print(f'[supervisor] {now()} 全部 {TOTAL_TARGET} 块齐，回写登记并退出', flush=True)
            register_task(accts, done=True)
            write_ledger(fleet, done_files)
            break
        for a in accts:
            todo = per[a] - done_files
            if todo and not alive(a):
                launch(a, len(todo))
        nb = backup_new()
        if nb:
            print(f'[supervisor] {now()} 备份 {nb} 块到 K 盘（累计 {len(glob.glob(os.path.join(BK_DIR, "chunkr7_*.parquet")))}）', flush=True)
        if time.time() - last_scan > 1800:
            write_ledger(fleet, done_files)
            print(f'[supervisor] {now()} 进度 {len(done_files)}/{TOTAL_TARGET}，台账已刷新', flush=True)
            last_scan = time.time()
        time.sleep(300)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
p0_doctor.py — 生产前环境自检（只读，不提交任何任务）
检查项：
  1) 依赖与版本指纹
  2) 代理 7890 是否监听（socks5h 必通）
  3) 磁盘余量（F 工作区 / Z 成品 / K 镜像）
  4) 母库与关键产物的存在性与行数
  5) 账号凭证文件 + anchor 仓 + 当前 operations 状态（GET，路径不带 :list）
  6) 资产根是否已建（缺失则给出建根命令，不自动建）
输出：控制台表 + Z:/生产输出/日志/doctor_<ts>.json
用法：python p0_doctor.py [--acct a b c] [--no-net] [--fix-roots]
"""
import os, sys, io, json, time, socket, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prod_conf as C

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


def check_deps():
    out = {}
    for m in ['numpy', 'pandas', 'pyarrow', 'sklearn', 'scipy', 'rasterio',
              'requests', 'ee', 'joblib', 'matplotlib']:
        try:
            mod = __import__(m)
            out[m] = getattr(mod, '__version__', '?')
        except Exception:
            out[m] = None
    out['python'] = sys.version.split()[0]
    return out


def check_proxy():
    try:
        s = socket.create_connection(('127.0.0.1', 7890), timeout=3)
        s.close()
        return True
    except Exception:
        return False


def check_disk():
    out = {}
    for tag, p in [('F_workspace', 'F:/lc_work'), ('Z_output', C.KB), ('K_mirror', 'K:/')]:
        try:
            import shutil
            u = shutil.disk_usage(p)
            out[tag] = {'free_gb': round(u.free / 1e9, 1), 'total_gb': round(u.total / 1e9, 1)}
        except Exception as e:
            out[tag] = {'error': str(e)[:60]}
    return out


def check_data():
    import pandas as pd
    out = {}
    for tag, p in [('母库', C.MOTHER), ('validity', C.VALIDITY),
                   ('holdout块', C.HOLDOUT), ('验证池', C.POOL), ('边界', C.BOUNDARY)]:
        out[tag] = {'exists': os.path.isfile(p),
                    'mb': round(os.path.getsize(p) / 1e6, 1) if os.path.isfile(p) else None}
    try:
        import pyarrow.parquet as pq
        out['母库']['rows'] = pq.ParquetFile(C.MOTHER).metadata.num_rows
    except Exception as e:
        out['母库']['rows'] = 'ERR ' + str(e)[:40]
    for tag, p in [('emb_parts_r7', os.path.join(C.KB, '数据/本地处理/全国清洗训练/emb_parts_r7')),
                   ('年度子集_含稀有类', os.path.join(C.KB, '数据/本地处理/全国清洗训练/年度子集_含稀有类'))]:
        out[tag] = {'n_items': len(os.listdir(p)) if os.path.isdir(p) else 0}
    return out


def acct_probe(acct, fix_roots=False):
    """只读探活：凭证 → CRM/项目 → operations（GET 不带 :list）→ 资产根"""
    row = {'acct': acct, 'proj': C.ACCOUNTS[acct]['proj'], 'folder': C.ACCOUNTS[acct]['folder'],
           'cred': False, 'ee': None, 'run': 0, 'pend': 0, 'eecu_h': 0.0, 'root': None}
    home = os.path.join(C.CRED, acct)
    cred = os.path.join(home, '.config', 'earthengine', 'credentials')
    row['cred'] = os.path.isfile(cred)
    if not row['cred']:
        row['err'] = '无凭证文件'
        return row
    if fix_roots:
        # 建根要写操作，只在显式 --fix-roots 时做
        code = ("import os;os.environ['HOME']=r'%s';os.environ['USERPROFILE']=r'%s';"
                "import ee;ee.Initialize(project='%s');"
                "import ee.data as d;"
                "p='projects/%s/assets/%s';"
                "\ntry:\n d.createAsset({'type':'Folder'},p,None);print('CREATED')\nexcept Exception as e:\n print('EXISTS' if 'already exists' in str(e) else str(e)[:80])"
                % (home, home, row['proj'], row['proj'], row['folder']))
        try:
            r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                               timeout=180, env={**os.environ, 'HTTP_PROXY': C.PROXY_URL,
                                                 'HTTPS_PROXY': C.PROXY_URL})
            row['fix'] = (r.stdout or r.stderr).strip()[-120:]
        except Exception as e:
            row['fix'] = 'fix err ' + str(e)[:60]
    try:
        import requests, google.oauth2.credentials
        from google.auth.transport.requests import AuthorizedSession
        from ee import oauth as ee_oauth
        with io.open(cred, encoding='utf-8') as f:
            cj = json.load(f)
        creds = google.oauth2.credentials.Credentials(
            token=None, refresh_token=cj['refresh_token'],
            token_uri='https://oauth2.googleapis.com/token',
            client_id=ee_oauth.CLIENT_ID, client_secret=ee_oauth.CLIENT_SECRET,
            scopes=cj.get('scopes', ['https://www.googleapis.com/auth/earthengine',
                                     'https://www.googleapis.com/auth/cloud-platform']))
        sess = AuthorizedSession(creds)
        sess.proxies = C.PROXY
        # operations：GET，路径**不带** :list（写成 :list 会 404+HTML 且每项目都 404）
        url = ('https://earthengine.googleapis.com/v1/projects/%s/operations' % row['proj'])
        r = sess.get(url, params={'pageSize': 200},
                     headers={'X-Goog-User-Project': row['proj']}, timeout=45)
        row['ee'] = (r.status_code == 200)
        if r.status_code == 200:
            for op in r.json().get('operations', []):
                md = op.get('metadata', {})
                st = md.get('state') or ('DONE' if op.get('done') else 'UNKNOWN')
                if st in ('RUNNING', 'PENDING'):
                    row['run' if st == 'RUNNING' else 'pend'] += 1
                if st == 'SUCCEEDED':
                    for k, v in md.items():
                        if 'ecu' in k.lower():
                            try:
                                row['eecu_h'] += float(v) / 3600.0
                            except (TypeError, ValueError):
                                pass
            row['eecu_h'] = round(row['eecu_h'], 1)
        else:
            row['err'] = 'HTTP %d' % r.status_code
        # 资产根
        try:
            au = ('https://earthengine.googleapis.com/v1/projects/%s/assets/%s' %
                  (row['proj'], row['folder']))
            ra = sess.get(au, headers={'X-Goog-User-Project': row['proj']}, timeout=30)
            row['root'] = (ra.status_code == 200)
        except Exception:
            row['root'] = None
    except Exception as e:
        row['err'] = str(e)[:90]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--acct', nargs='*', default=None)
    ap.add_argument('--no-net', action='store_true')
    ap.add_argument('--fix-roots', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    C.ensure_dirs()
    rep = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'spec': 'v-final.2026-09-14'}
    emit('== 1) 依赖 ==')
    rep['deps'] = check_deps()
    bad = [k for k, v in rep['deps'].items() if v is None]
    for k, v in rep['deps'].items():
        emit('   %-12s %s' % (k, v or '缺失!'))
    emit('== 2) 代理 7890 ==')
    rep['proxy'] = check_proxy()
    emit('   %s' % ('OK' if rep['proxy'] else '未监听 —— 所有 GEE 调用会失败'))
    emit('== 3) 磁盘 ==')
    rep['disk'] = check_disk()
    for k, v in rep['disk'].items():
        emit('   %-12s %s' % (k, v))
    emit('== 4) 数据 ==')
    rep['data'] = check_data()
    for k, v in rep['data'].items():
        emit('   %-16s %s' % (k, v))
    if not a.no_net:
        emit('== 5) 账号（只读探活）==')
        accts = a.acct or C.usable_accounts(include_busy=True)
        res = []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:   # 代理对高并发 token 刷新敏感
            for r in ex.map(lambda x: acct_probe(x, a.fix_roots), accts):
                res.append(r)
                emit('   %-16s cred=%s ee=%s RUN=%d PEND=%d EECU=%sh root=%s %s'
                     % (r['acct'], r['cred'], r['ee'], r['run'], r['pend'],
                        r['eecu_h'], r['root'], r.get('err', '')))
        rep['accounts'] = res
    os.makedirs(C.PLOG, exist_ok=True)
    fp = os.path.join(C.PLOG, 'doctor_%s.json' % time.strftime('%Y%m%d_%H%M'))
    json.dump(rep, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    emit('报告: ' + fp)
    emit('结论: %s' % ('可开跑' if not bad and rep['proxy'] else '有阻塞项，见上'))
    return 0 if (not bad and rep['proxy']) else 1


if __name__ == '__main__':
    raise SystemExit(main())

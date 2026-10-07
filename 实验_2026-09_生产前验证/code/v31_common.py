# -*- coding: utf-8 -*-
"""v31_common.py — v3.1 四组决定性实验公共库（doc37 §六）
职责：路径/配置加载、GEE 账号上下文（沿 exp_common 验证过的套路）、v31 映射、精度指标、
资产存在性检查、单实例锁。约束：CPU≤15核/内存≤10G（本包纯 IO+轻计算，无本地RF）。
"""
import os, sys, io, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = os.path.join(ROOT, 'config')
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
RAW = os.path.join(RES, 'raw')
MET = os.path.join(RES, 'metrics')
REPT = os.path.join(ROOT, 'reports')
LOGS = os.path.join(ROOT, 'logs')
BAKS = os.path.join(ROOT, 'backups')
STATE_FP = os.path.join(ROOT, 'STATE.json')
JOBS_FP = os.path.join(ROOT, 'jobs.json')

PROD = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产'
sys.path.insert(0, PROD)
import prod_conf as C  # noqa


def _pick_proxy():
    """代理端口会漂移（7890 ↔ 7897），启动时探测可用端口并覆盖 prod_conf。"""
    import socket
    cands = []
    for p in (7890, 7897, 7891):
        s = socket.socket()
        s.settimeout(0.6)
        try:
            if s.connect_ex(('127.0.0.1', p)) == 0:
                cands.append(p)
        finally:
            s.close()
    if not cands:
        return None
    port = cands[0]
    url = 'socks5h://127.0.0.1:%d' % port
    C.PROXY = {'http': url, 'https': url}
    C.PROXY_URL = url
    return port


_PROXY_PORT = _pick_proxy()

FEATS = ['A%02d' % i for i in range(64)]
EVAL_DIR_GEE = 'F:/lc_work/样本修正/geetest'   # 已验证的 w1-w4 留出切分所在
SRC2023 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
SQRT_CSV = r'F:/lc_work/prod_sample_sqrt.csv'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def jload(fp, default=None):
    try:
        with io.open(fp, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def jsave(obj, fp):
    tmp = fp + '.tmp'
    with io.open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, fp)


def cfg(name):
    return jload(os.path.join(CFG, name), {})


# ---------- 资产注册表（uploader 可能轮换，跨账号资产 ID 以注册表为准） ----------
def reg_set(key, full_id):
    fp = os.path.join(DATA, 'asset_registry.json')
    d = jload(fp, {})
    d[key] = full_id
    jsave(d, fp)


def reg_get(key):
    return jload(os.path.join(DATA, 'asset_registry.json'), {}).get(key)


STATE2_FP = os.path.join(ROOT, 'STATE2.json')


def state2():
    return jload(STATE2_FP, {})


def save_state2(s):
    s['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    jsave(s, STATE2_FP)


def state():
    return jload(STATE_FP, {})


def save_state(s):
    s['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    jsave(s, STATE_FP)


# ---------- 单实例锁（防 cron 与手工并发跑；Windows 无 fcntl，用 O_EXCL 原子建锁） ----------
def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return True  # 判断不了就当作活着（保守）


def acquire_lock():
    fp = os.path.join(LOGS, '.advance.lock')
    os.makedirs(LOGS, exist_ok=True)
    try:
        if os.path.exists(fp):
            try:
                old = int(open(fp).read().strip() or 0)
            except Exception:
                old = 0
            stale = (time.time() - os.path.getmtime(fp) > 2 * 3600) or \
                    (old and not _pid_alive(old))
            if stale:
                emit('清理陈旧锁（PID %s 已不存在）' % old)
                os.remove(fp)
        fd = os.open(fp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError:
        return None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return fp


def release_lock(lk):
    try:
        if lk and os.path.exists(lk):
            os.remove(lk)
    except Exception:
        pass


# ---------- GEE 账号（套路沿 exp_common：prod_conf + extra_accounts） ----------
def pid_of(acct):
    try:
        return C.ACCOUNTS[acct]['proj']
    except KeyError:
        d = jload(os.path.join(os.path.dirname(os.path.abspath(EVAL_DIR_GEE)),
                               'geetest', 'extra_accounts.json'), {})
        if acct in d:
            return d[acct]
        raise KeyError('账号 %s 无法解析项目ID' % acct)


def ctx(acct):
    C.apply_account_env(acct)
    import ee
    pid = pid_of(acct)
    for i in range(5):
        try:
            ee.Initialize(project=pid)
            _CUR_ACCT['acct'] = acct
            return ee, pid
        except Exception as e:
            if i == 4:
                raise
            emit('init retry %d %s' % (i + 1, str(e)[:80]))
            time.sleep(15 + 5 * i)


def sess(acct):
    import google.oauth2.credentials
    from google.auth.transport.requests import AuthorizedSession
    from ee import oauth as ee_oauth
    pid = pid_of(acct)
    cj = json.load(io.open(os.path.join(C.cred_home(acct), '.config', 'earthengine', 'credentials'),
                           encoding='utf-8'))
    creds = google.oauth2.credentials.Credentials(
        token=None, refresh_token=cj['refresh_token'],
        token_uri='https://oauth2.googleapis.com/token',
        client_id=ee_oauth.CLIENT_ID, client_secret=ee_oauth.CLIENT_SECRET,
        scopes=cj.get('scopes', ['https://www.googleapis.com/auth/earthengine']))
    s = AuthorizedSession(creds)
    s.proxies = C.PROXY
    return s, pid


def qdepth(acct, tries=4):
    last = ''
    for i in range(tries):
        try:
            s, pid = sess(acct)
            r = s.get('https://earthengine.googleapis.com/v1/projects/%s/operations' % pid,
                      params={'pageSize': 50}, headers={'X-Goog-User-Project': pid}, timeout=60)
            r.raise_for_status()
            n = 0
            for o in r.json().get('operations', []):
                if (o.get('metadata', {}).get('state')) in ('RUNNING', 'PENDING'):
                    n += 1
            return n, ''
        except Exception as e:
            last = str(e)[:90]
            time.sleep(6 * (i + 1))
    return None, last


_CUR_ACCT = {'acct': None}


def ensure_ctx(acct):
    """按需切换已初始化的账号（避免频繁 ee.Initialize 与列表资产串号）。"""
    if _CUR_ACCT['acct'] == acct:
        return
    C.apply_account_env(acct)
    import ee
    pid = pid_of(acct)
    for i in range(3):
        try:
            ee.Initialize(project=pid)
            _CUR_ACCT['acct'] = acct
            return
        except Exception as e:
            if i == 2:
                raise
            emit('ensure_ctx retry %d %s' % (i + 1, str(e)[:70]))
            time.sleep(10 + 5 * i)


def list_assets(acct, parent, tries=3):
    """列资产名集合；失败返回 None。**会先把 EE 会话切到该账号**。"""
    import ee
    last = ''
    for i in range(tries):
        try:
            ensure_ctx(acct)
            r = ee.data.listAssets({'parent': parent})
            return set(a['name'].split('/')[-1] for a in r.get('assets', []))
        except Exception as e:
            last = str(e)[:100]
            if 'client library not initialized' in last or 'Permission' in last:
                _CUR_ACCT['acct'] = None
            time.sleep(6 * (i + 1))
    emit('listAssets %s 失败: %s' % (parent, last))
    return None


def asset_exists(acct, full_id):
    name = full_id.split('/')[-1]
    parent = full_id.rsplit('/', 1)[0]
    have = list_assets(acct, parent)
    return (have is not None) and (name in have)


# ---------- v31 映射 ----------
def v31_lut():
    """旧码→v31码 的扁平 dict（str→int）。"""
    m = cfg('v31_map.json')
    lut = {}
    for new, olds in m['merges'].items():
        for o in olds:
            lut[str(o)] = int(new)
    for o, new in m['identity'].items():
        lut[str(o)] = int(new)
    return lut


def v31_lut_ee():
    import ee
    return ee.Dictionary(v31_lut())


def remap_fc_v31(fc):
    """训练端 30→24 服务端重标（doc37 §2.2；ee.Dictionary.get 查表，缺键保原值）。"""
    import ee
    lut = v31_lut_ee()
    return fc.map(lambda f: f.set(
        'cl', ee.Number(lut.get(ee.Number(f.get('cl')).format('%d'), f.get('cl')))))


def to_v31(arr):
    """ndarray 旧码→v31码（缺键保持原值）。"""
    lut = {int(k): v for k, v in v31_lut().items()}
    import numpy as np
    out = arr.copy()
    for o, n in lut.items():
        out[arr == o] = n
    return out


V31_NAMES = lambda: cfg('v31_map.json')['names']
MACRO = lambda: cfg('macro9.json')['macros']
MACRO_NAMES = lambda: cfg('macro9.json')['macro_names']


def to_macro(arr24):
    """v31 24码 → 9 大类码（1-9）。"""
    import numpy as np
    out = np.zeros(len(arr24), dtype=int)
    for i, (m, lst) in enumerate(sorted(MACRO().items()), 1):
        for c in lst:
            out[arr24 == c] = i
    return out


# ---------- 精度指标 ----------
def _f1(tp, fp, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc


def metrics(t, p):
    """t/p: 等长 ndarray。返回 OA、macroF1、逐类 {c:{f1,pa,ua,n}}（对出现在 t∪p 的类）。"""
    import numpy as np
    t = np.asarray(t, dtype=int)
    p = np.asarray(p, dtype=int)
    ok = int((t == p).sum())
    n = len(t)
    oa = ok / n if n else 0.0
    per = {}
    f1s = []
    for c in sorted(set(t.tolist()) | set(p.tolist())):
        tp = int(((t == c) & (p == c)).sum())
        fp = int(((t != c) & (p == c)).sum())
        fn = int(((t == c) & (p != c)).sum())
        if tp + fp + fn == 0:
            continue
        f1, pa, ua = _f1(tp, fp, fn)
        per[int(c)] = dict(f1=round(f1, 4), ua=round(ua, 4), pa=round(pa, 4), n=tp + fn)
        f1s.append(f1)
    mf1 = float(np.mean(f1s)) if f1s else 0.0
    return dict(OA=round(oa, 4), macroF1=round(mf1, 4), n=n, per=per)

# -*- coding: utf-8 -*-
"""b1_batch.py — 五省 2023 批 2/3（74 瓦片）无人值守批量生产驱动
* 输入：plan/tiles_frozen.csv（76 瓦片冻结清单，跳过试产 T2713/T2102）
    批 2 = prov∈{宁,粤,闽} 减 T2102（22 瓦片，先行）
    批 3 = prov∈{川,黑} 减 T2713（52 瓦片，批 2 前 4 瓦片 QA 完成后放量）
* 规则源：冻结配置与 p1_pilot 完全一致（smp2023_merged 20k 点 / RF100,minLeaf2,maxNodes5000,seed7 /
  AEF 2023 mosaic / uint8 clip / Export toAsset scale=10 EPSG:4326 shardSize=16）
  + 批 1 修复：geefast 下载按资产 affine 精确对齐（--affine + --grid-w/h）、保真 unmask(255) 处理掩膜
* 门槛：批预算 ≤1390 EECU·h（超即停新提交）；每轮(5min)新提交 ≤2；每账号 pending ≤2（qdepth 探）；
  每瓦片最多 2 路；赛跑只救真停滞（READY>40min 或 RUNNING>2.5h 零 EECU，原任务不取消）；
  保真 3000 点须 100%（复跑一次含重下载仍 <100% → 熔断）；连续 3 瓦片全 FAILED → 熔断；
  非沿海瓦片 EECU>19.4 超过 2 个 → 熔断；总截止 24 小时
* 输出：batch/state.json；rasters_batch/{tile}_10m.tif（Z 盘未挂载的 fallback，留痕）；
  qa/{tile}_qa.json + {tile}_面积.csv；ledger/eecu_ledger_batch.csv + batch_board.csv
* 用法：python F:/lc_work/v31_exp/code/launch_bg.py b1_batch -- python F:/lc_work/prod5p_2023/pilot/b1_batch.py auto
"""
import os, sys, time, json, math, csv, subprocess
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
import v31_common as VC
import numpy as np
import rasterio

ROOT = r'F:/lc_work/prod2nd_2023'
PD = os.path.join(ROOT, 'run')
BD = os.path.join(PD, 'batch')
STATE_FP = os.path.join(BD, 'state.json')
LEDGER = os.path.join(ROOT, 'ledger', 'eecu_ledger_2nd.csv')
BOARD = os.path.join(ROOT, 'ledger', 'board_2nd.csv')
RDIR = os.path.join(ROOT, 'rasters_batch')
CFG = VC.jload(os.path.join(ROOT, 'config.json'), {})
SRC = CFG.get('asset') or '(未配置)'
REMAP = bool(CFG.get('remap', True))   # 旧池=30类原始码须 remap；v2 池已是 24 类码 → config remap=false

YEAR = 2023
DEADLINE = time.time() + 24 * 3600
BUDGET = 1390.0                 # （已废弃，保留字段兼容；实际用 PROJ_CAP 项目口径）
PROJ_CAP = VC.jload(os.path.join(ROOT, 'config.json'), {}).get('budget_eecu', 950.0)               # 项目总 EECU·h 硬上限（含批 1 的 87.9；裁决 2026-10-04，无弹性）
B1_SPENT = 0.0                 # 批 1 试产已花（含赛跑烧掉的重复路）
B2_GATE = 4                     # 批 2 前 4 瓦片 QA 完成后才放批 3
MAX_ROUTES = 2                  # 每瓦片最多路数
NEW_PER_ROUND = 6               # 每轮新提交上限（2026-10-14 提速：2→6，编队 21 账号，用户要求尽快完成）
OVER_LIMIT = 24.0               # 内陆单瓦片 EECU 上限（G4 校准：n=9 实测 17.7-23.7，原 19.4 偏紧）
GEEFAST = r'C:/Users/Administrator/.zcode/skills/geefast-download/scripts/gee_rest_compute_pixels.py'
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
from d_tiles import COLORS

POOL = VC.jload(os.path.join(VC.CFG, 'accounts_pool.json'), {})
CAND = VC.jload(os.path.join(ROOT, 'config.json'), {}).get('fleet') or ['e5h08k', 'hj74bml', 'hqzub6', 'ccstqqb', 'ch2fg2t9', 'hrbm3pdw', 'n51j08y', 'nama8lg']
OVR = [2, 4, 8, 16, 32]
N_FID = 3000
ACTIVE = ('SUBMITTED', 'PENDING', 'RUNNING', 'READY')


def log(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def load_tiles():
    """读第二批次冻结清单（川24/粤10/宁6=40 瓦）→ (顺序表, [], 元数据表)"""
    order, meta = [], {}
    with open(os.path.join(ROOT, 'plan', 'tiles_secondrun.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = row['tile']
            meta[t] = dict(box=[float(row['x0']), float(row['y0']),
                                float(row['x1']), float(row['y1'])],
                           prov=row['prov'], land_frac=float(row['land_frac']),
                           zone='沿海' if float(row['land_frac']) < 1.0 else '内陆')
            order.append(t)
    return order, [], meta


B2, B3, TILES = load_tiles()
ORDER = B2 + B3

LOCK_FP = os.path.join(BD, 'b1_batch.lock')


def acquire_lock():
    """单实例锁：防双驱动并发写 state / 重复提交。"""
    if os.path.exists(LOCK_FP):
        try:
            old = int(open(LOCK_FP).read().strip() or 0)
        except Exception:
            old = 0
        if old:
            out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % old],
                                 capture_output=True, text=True).stdout
            if ('python.exe' in out) and (str(old) in out):
                log('已有实例在跑（PID %d），退出' % old)
                sys.exit(2)
            log('锁文件残留（PID %d 已死），接管' % old)
    with open(LOCK_FP, 'w') as f:
        f.write(str(os.getpid()))


def S():
    st = VC.jload(STATE_FP, {})
    st.setdefault('tiles', {})
    st.setdefault('dead_accts', {})
    st.setdefault('over_tiles', [])
    st.setdefault('fail_streak', 0)
    return st


def save(st):
    VC.jsave(st, STATE_FP)


def ledger(acct, tile, task, desc, state, eecu, note=''):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['time', 'acct', 'tile', 'task_id', 'desc', 'state', 'eecu_h', 'note'])
        w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), acct, tile, task, desc,
                    state, '%.4f' % eecu, note])


def email(acct):
    return 'user:' + POOL['emails'][acct]


def pick_acct(st, exclude):
    for a in CAND:
        if a in exclude or a in st['dead_accts']:
            continue
        q, _ = VC.qdepth(a)
        if q is not None and q <= 2:
            return a
    return None


def total_eecu(st):
    return round(sum(r['eecu'] for t in st['tiles'].values()
                     for r in t.get('runs', {}).values()), 3)


def acct_err(st, acct, msg):
    """账号连续异常计数，≥5 次摘除并留痕（restricted / 失联防挂死）。"""
    if not acct:
        return
    errs = st.setdefault('acct_err', {})
    errs[acct] = errs.get(acct, 0) + 1
    if errs[acct] >= 5 and acct not in st['dead_accts']:
        st['dead_accts'][acct] = 'consecutive errors: %s' % msg
        log('⚠️ 账号 %s 连续异常 %d 次，摘除（%s）' % (acct, errs[acct], msg))
    errs[acct + '_msg'] = msg


def acct_ok(st, acct):
    errs = st.setdefault('acct_err', {})
    if acct in errs:
        errs[acct] = 0


def winner(t, st):
    runs = st['tiles'].get(t, {}).get('runs', {})
    for suf in sorted(runs):
        if runs[suf].get('state') in ('COMPLETED', 'SUCCEEDED'):
            return suf, runs[suf]
    return None, None


def b2_done_n(st):
    return sum(1 for t in B2 if st['tiles'].get(t, {}).get('fidelity', {}).get('rate') == 1.0
               and os.path.exists(os.path.join(ROOT, 'qa', '%s_qa.json' % t)))


def qa_done(t, st):
    return st['tiles'].get(t, {}).get('fidelity', {}).get('rate') == 1.0 and \
        os.path.exists(os.path.join(ROOT, 'qa', '%s_qa.json' % t))


# ---------- 阶段：训练池就绪 + ACL ----------
def wait_ready(st, ee):
    try:
        ee.data.getAsset(SRC)
    except Exception as e:
        log('等 SRC 异常 %s' % str(e)[:80])
        return False
    want = [email(a) for a in CAND if a in POOL.get('emails', {})]
    try:
        cur = list(ee.data.getAssetAcl(SRC).get('readers', []))
        add = [x for x in want if x not in cur]
        if add:
            ee.data.setAssetAcl(SRC, {'readers': cur + add})
            log('SRC ACL 增 %d readers（覆盖全部 11 账号）' % len(add))
        st['src_ok'] = True
    except Exception as e:
        log('ACL 失败 %s' % str(e)[:80])
    return True


# ---------- 阶段：提交 ----------
def submit_tile(t, st, force=False):
    tinfo = st['tiles'].setdefault(t, {})
    runs = tinfo.setdefault('runs', {})
    # force=True 仅由赛跑救援调用：跳过"原路 ACTIVE"守卫，为真停滞瓦片补第二路
    # （2026-10-14 实测坑：run-1 沿用至今的旧守卫 any(ACTIVE) 使赛跑永不落地——
    #   33 次"触发赛跑"0 次提交成功，而 16 账号空转、尾部瓦卡 5 账号排队）
    if winner(t, st)[0] or (not force and any(r['state'] in ACTIVE for r in runs.values())):
        return False
    if len(runs) >= MAX_ROUTES:
        return False
    acct = pick_acct(st, set(r['acct'] for r in runs.values()))
    if not acct:
        return False
    ee, pid = VC.ctx(acct)
    meta = TILES[t]
    boxp = ee.Geometry.Rectangle([meta['box'][0] - 2, meta['box'][1] - 2,
                                  meta['box'][2] + 2, meta['box'][3] + 2])
    fc = (ee.FeatureCollection(SRC).filterBounds(boxp)
          .randomColumn('rc', 7).sort('rc').limit(20000))
    tr = VC.remap_fc_v31(fc) if REMAP else fc
    # ⚠️ remap_fc_v31 的 LUT 输入键是 30 类码：对已是 24 类码的 v2 池会把 10/11/12 错映成 1/2/3
    #    （落叶灌木→草本旱地、草地→灌溉耕地、地衣苔藓→乔灌园地）——v2 池必须 REMAP=False（2026-10-14 实测）
    clf = (ee.Classifier.smileRandomForest(numberOfTrees=100, minLeafPopulation=2,
           maxNodes=5000, seed=7).train(tr, 'cl', VC.FEATS))
    box = ee.Geometry.Rectangle(meta['box'])
    aef = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
           .filterBounds(box).mosaic().select(VC.FEATS))
    cls = aef.classify(clf).rename('class').uint8().clip(box)
    aid = 'projects/%s/assets/prod2nd/tile_%s_%d' % (pid, t, YEAR)
    suf = chr(ord('a') + len(runs))
    desc = 'p5t_%s_%s_%s' % (t, suf, time.strftime('%m%d%H%M%S'))
    task = ee.batch.Export.image.toAsset(image=cls, description=desc, assetId=aid,
                                         scale=10, crs='EPSG:4326', region=meta['box'],
                                         maxPixels=10 ** 12, shardSize=16)
    task.start()
    runs[suf] = dict(acct=acct, task=task.id, asset=aid, desc=desc,
                     submitted=time.time(), eecu=0.0, state='SUBMITTED')
    tinfo['batch'] = 'B2' if t in B2 else 'B3'
    ledger(acct, t, task.id, desc, 'SUBMITTED', 0.0, 'submit')
    log('%s 提交路 %s @%s → %s' % (t, suf, acct, task.id))
    return True


# ---------- 阶段：轮询 + 赛跑（只救真停滞） ----------
def poll_tile(t, st):
    tinfo = st['tiles'].setdefault(t, {})
    for suf, r in tinfo.get('runs', {}).items():
        if r['state'] in ('COMPLETED', 'SUCCEEDED', 'FAILED', 'CANCELLED'):
            continue
        try:
            e, _ = VC.ctx(r['acct'])
            ts = e.data.getTaskStatus(r['task'])[0]
            acct_ok(st, r['acct'])
            old = r['state']
            r['state'] = ts.get('state', '?')
            r['eecu'] = round(float(ts.get('batch_eecu_usage_seconds') or 0) / 3600, 3)
            if r['state'] != old:
                log('%s 路%s %s→%s EECU=%.2f' % (t, suf, old, r['state'], r['eecu']))
                ledger(r['acct'], t, r['task'], r['desc'], r['state'], r['eecu'], 'poll')
        except Exception as e:
            log('%s 路%s 轮询异常 %s' % (t, suf, str(e)[:60]))
            acct_err(st, r['acct'], str(e)[:60])
    runs = tinfo.get('runs', {})
    if runs and not any(r['state'] == 'COMPLETED' for r in runs.values()) \
            and len(runs) < MAX_ROUTES:
        r0 = min(runs.values(), key=lambda r: r['submitted'])
        age = time.time() - r0['submitted']
        stall = (r0['state'] == 'READY' and age > 40 * 60) or \
                (age > 2.5 * 3600 and r0['eecu'] == 0)
        if stall and not any(r['state'] in ACTIVE for r in runs.values() if r is not r0):
            log('%s 触发赛跑（路状态 %s，已 %.1f h 零进展）' % (t, r0['state'], age / 3600.0))
            ledger(r0['acct'], t, 'RACE', r0['desc'], 'RACE', 0.0, 'stall-reroute')
            try:
                if submit_tile(t, st, force=True):
                    log('%s 赛跑第二路已提交' % t)
                else:
                    log('%s 赛跑无可用账号（全部 q>2 或被占用），原地等队列' % t)
            except Exception as e:
                log('%s 赛跑重投失败 %s' % (t, str(e)[:80]))
    return any(r['state'] == 'COMPLETED' for r in runs.values())


# ---------- 阶段：下载（按资产 affine 精确对齐） ----------
def asset_grid(acct, asset, box):
    ee, _ = VC.ctx(acct)
    tr = ee.Image(asset).projection().getInfo()['transform']
    sx, sy, tx, ty = tr[0], tr[4], tr[2], tr[5]
    w = int(math.ceil((box[2] - tx) / sx))
    h = int(math.ceil((ty - box[1]) / abs(sy)))
    aff = '%r,0,%r,0,%r,%r' % (sx, tx, sy, ty)
    return aff, w, h


def download(t, meta, r):
    raw = os.path.join(RDIR, '_raw_%s.tif' % t)
    os.makedirs(RDIR, exist_ok=True)
    env = dict(os.environ)
    env['HOME'] = VC.C.cred_home(r['acct'])
    env['USERPROFILE'] = env['HOME']
    env['EE_PROJECT'] = VC.pid_of(r['acct'])
    aff, gw, gh = asset_grid(r['acct'], r['asset'], meta['box'])
    cmd = [sys.executable, GEEFAST, '--image', r['asset'],
           '--bbox', ','.join(str(x) for x in meta['box']), '--bands', 'class',
           '--scale', '10', '--affine', aff,
           '--grid-width', str(gw), '--grid-height', str(gh),
           '--auto-tile', '--workers', '20',
           '--dtype', 'uint16', '--output', raw, '--overwrite']
    log('%s 开始下载（资产网格 %dx%d）' % (t, gw, gh))
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         encoding='utf-8', errors='ignore')
    dt = (time.time() - t0) / 60
    # 体积门槛只防"空文件"：全海瓦片 deflate 后仅数 MB（T2001 实测 5 MB），正确性由保真兜底
    ok = res.returncode == 0 and os.path.exists(raw) and os.path.getsize(raw) > 1e6
    log('%s 下载%s（%.1f 分钟，%.0f MB）' % (t, '完成' if ok else '失败', dt,
        os.path.getsize(raw) / 1e6 if os.path.exists(raw) else 0))
    if not ok:
        log('  tail: %s' % ((res.stdout or res.stderr or '')[-300:].replace('\n', ' | ')))
    return ok


# ---------- 阶段：上色 ----------
def palette():
    pal = {i: (230, 230, 230, 0) for i in range(256)}
    for k, c in COLORS.items():
        pal[k] = (int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255)), 255)
    for k in range(25, 256):
        pal[k] = (200, 200, 200, 255)
    return pal


def colorize(t):
    """概览配方（2026-10-04 审查二次裁决）：GDAL build_overviews 产出的概览页内容确定性损坏
    （与 'w'/'r+' 模式无关），且 rasterio out_shape 读概览级会现算、不能作自检。
    流程：底图干净重写（uint8/deflate/tiled512/nodata=0+colormap）→ close →
    'r+' build_overviews 仅作 IFD 脚手架 → close → ov_build.py 手工重写存储页 +
    PIL 全平面验收（隶属/TL 双 100%，5 级），验收 returncode 即每瓦片必过门槛。"""
    src_fp = os.path.join(RDIR, '_raw_%s.tif' % t)
    dst_fp = os.path.join(RDIR, '%s_10m.tif' % t)
    tmp_fp = dst_fp + '.tmp.tif'
    with rasterio.open(src_fp) as ds:
        prof = ds.profile.copy()
        prof.update(dtype='uint8', compress='deflate', predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, count=1, nodata=0)
        with rasterio.open(tmp_fp, 'w', **prof) as dst:
            for r0 in range(0, ds.height, 1024):
                hh = min(1024, ds.height - r0)
                win = rasterio.windows.Window(0, r0, ds.width, hh)
                blk = ds.read(1, window=win)
                dst.write(np.where(blk > 250, 0, blk).astype('uint8'), 1, window=win)
            dst.write_colormap(1, palette())
    os.replace(tmp_fp, dst_fp)
    with rasterio.open(dst_fp, 'r+') as dst:
        dst.build_overviews(OVR, rasterio.enums.Resampling.nearest)
        dst.update_tags(ns='rio_overview', resampling='nearest')
    with open(os.path.join(RDIR, '%s.clr' % t), 'w', encoding='utf-8') as f:
        f.write('# QGIS/ArcGIS color file: value R G B (0-255)\n')
        for k, c in sorted(COLORS.items()):
            f.write('%d %d %d %d\n' % (k, round(c[0] * 255), round(c[1] * 255), round(c[2] * 255)))
    ov_rewrite(t, dst_fp)
    log('%s 上色完成 %s' % (t, dst_fp))


def ov_rewrite(t, fp):
    """调 ov_build.py 手工重建概览存储页并 PIL 全平面验收（隶属/TL 双 100%，5 级）。
    判定以 returncode==0 为权威（工具仅 verify 失败时 SystemExit 非零）；
    stdout 尾部常被 DeprecationWarning 噪音占据，不以末行内容判定（T2103 误判教训）。
    验收失败 → RuntimeError → postprocess 重试路径（重下载重上色）。"""
    res = subprocess.run([sys.executable, os.path.join(PD, 'ov_build.py'), fp],
                         capture_output=True, text=True, encoding='utf-8', errors='ignore')
    out = res.stdout or ''
    good = res.returncode == 0 and ('✅ 验收通过' in out or '验收通过' in out or '跳过重建' in out)
    for ln in (out + (res.stderr or '')).strip().splitlines()[-3:]:
        log('  ov_build| %s' % ln)
    if not good:
        raise RuntimeError('%s ov_build 验收未通过（rc=%d）' % (t, res.returncode))


# ---------- 阶段：保真校验（3000 点逐点 = 100%，unmask255 含掩膜） ----------
def get_proxy():
    import socket
    for p in (7890, 7897, 7891):
        s = socket.socket()
        s.settimeout(0.6)
        try:
            if s.connect_ex(('127.0.0.1', p)) == 0:
                return {'http': 'socks5h://127.0.0.1:%d' % p,
                        'https': 'socks5h://127.0.0.1:%d' % p}
        finally:
            s.close()
    return {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}


def fidelity(t, meta, r):
    rng = np.random.RandomState(7)
    xs = rng.uniform(meta['box'][0] + 0.02, meta['box'][2] - 0.02, N_FID)
    ys = rng.uniform(meta['box'][1] + 0.02, meta['box'][3] - 0.02, N_FID)
    ee, _ = VC.ctx(r['acct'])
    feats = [ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'i': i})
             for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist()))]
    out = (ee.Image(r['asset']).select('class').unmask(255)
           .sampleRegions(collection=ee.FeatureCollection(feats),
                          properties=['i'], scale=10, tileScale=4))
    url = out.getDownloadURL(filetype='csv', selectors=['i', 'class'])
    proxy = get_proxy()
    import requests, io
    txt = requests.get(url, proxies=proxy, timeout=300).text
    import pandas as pd
    df = pd.read_csv(io.StringIO(txt)).set_index('i')['class']
    delp = os.path.join(RDIR, '%s_10m.tif' % t)
    bad = 0
    miss = 0
    with rasterio.open(delp) as ds:
        vals = list(ds.sample([(float(x), float(y)) for x, y in zip(xs, ys)]))
    for i, v in enumerate(vals):
        if int(i) not in df.index:
            miss += 1
            continue
        g = int(df.loc[int(i)])
        loc = int(v[0])
        if (g > 250 and loc != 0) or (g <= 250 and loc != g):
            bad += 1
    rate = 1.0 - (bad + miss) / N_FID
    log('%s 保真：一致率 %.4f%%（bad=%d miss=%d）' % (t, 100 * rate, bad, miss))
    return dict(n=N_FID, bad=bad, miss=miss, rate=round(rate, 6))


# ---------- 阶段：QA（10m 原生网格本地统计，纬度加权） ----------
def qa(t, meta):
    """MF2 配方：像元口径与面积口径分开报（旧公式 cov=zero_km2/tot_px 混单位已废）；
    附加：sea_classified_pct（有 AEF 数据的海面被分类 = 像元覆盖 - 100*land_frac，政策默认保留待用户定夺）
    + rare_classes（孤类 <0.01% 清单），两行写入 reports/batch23_delivery_notes.md。"""
    fp = os.path.join(RDIR, '%s_10m.tif' % t)
    names = VC.V31_NAMES()
    cnt = {}
    tot = 0
    with rasterio.open(fp) as ds:
        res = ds.res
        for r0 in range(0, ds.height, 2048):
            hh = min(2048, ds.height - r0)
            blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
            lat0 = ds.xy(r0, 0, offset='ul')[1]
            lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
            cell = (res[0] * 111.32 * math.cos(math.radians((lat0 + lat1) / 2))) * (abs(res[1]) * 110.57)
            v, c = np.unique(blk, return_counts=True)
            for a, b in zip(v.tolist(), c.tolist()):
                p, s = cnt.get(a, (0, 0.0))
                cnt[a] = (p + int(b), s + int(b) * cell)
            tot += hh * ds.width
    zero_px, zero_km2 = cnt.get(0, (0, 0.0))
    areas = {k: v for k, v in cnt.items() if k > 0}
    a_px = sum(v[0] for v in areas.values())
    a_km2 = sum(v[1] for v in areas.values())
    cov_px = 100 * (1 - zero_px / tot) if tot else 0.0
    sea_pct = round(max(0.0, cov_px - 100 * meta['land_frac']), 3) \
        if meta['land_frac'] < 1 else 0.0
    rare = [dict(code=k, name=names.get(str(k), '?'), km2=round(v[1], 4),
                 pct=round(100 * v[1] / a_km2, 4))
            for k, v in sorted(areas.items(), key=lambda x: x[1][1])
            if 100 * v[1] / a_km2 < 0.01]
    qd = os.path.join(ROOT, 'qa')
    os.makedirs(qd, exist_ok=True)
    with open(os.path.join(qd, '%s_面积.csv' % t), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['code', 'name', 'area_km2', 'share_pct'])
        for k, v in sorted(areas.items(), key=lambda x: -x[1][1]):
            w.writerow([k, names.get(str(k), '?'), round(v[1], 3), round(100 * v[1] / a_km2, 3)])
    doc = dict(tile=t, prov=meta['prov'], zone=meta['zone'],
               caliber='coverage_px_pct=像元口径；coverage_km2_pct=面积口径；'
                       'sea_classified_pct=有AEF数据海面被分类(像元覆盖-100*land_frac，政策默认保留待用户定夺)',
               coverage_px_pct=round(cov_px, 3),
               coverage_km2_pct=round(100 * a_km2 / (a_km2 + zero_km2), 3) if (a_km2 + zero_km2) else None,
               land_frac=meta['land_frac'],
               land_note=('海面掩膜使总覆盖率低于内陆瓦片；land_adj=coverage_px/land_frac 仅为粗折算'
                          if meta['land_frac'] < 1 else '内陆瓦片，无海面'),
               sea_classified_pct=sea_pct,
               rare_classes=rare,
               n_classes=len(areas), total_km2=round(a_km2, 1),
               top5=[dict(code=k, name=names.get(str(k), '?'), pct=round(100 * v[1] / a_km2, 2))
                     for k, v in sorted(areas.items(), key=lambda x: -x[1][1])[:5]])
    json.dump(doc, open(os.path.join(qd, '%s_qa.json' % t), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    rp = os.path.join(ROOT, 'reports')
    os.makedirs(rp, exist_ok=True)
    with open(os.path.join(rp, 'batch23_delivery_notes.md'), 'a', encoding='utf-8') as f:
        f.write('- %s（%s/%s）：海面被分类占比 %.3f%%；孤类(<0.01%%) %d 个%s\n' % (
            t, meta['prov'], meta['zone'], sea_pct, len(rare),
            ('：' + '、'.join('%s %.4f%%' % (r['name'], r['pct']) for r in rare[:8]) if rare else '')))
    log('%s QA：像元覆盖 %.2f%%（面积口径 %.2f%%）海面被分类 %.2f%% 类数 %d 面积 %.0f km²' % (
        t, cov_px, doc['coverage_km2_pct'] or 0, sea_pct, len(areas), a_km2))
    return doc


# ---------- 看板 ----------
def board(st):
    rows = []
    for t in ORDER:
        tinfo = st['tiles'].get(t, {})
        runs = tinfo.get('runs', {})
        suf, r = winner(t, st)
        act = next((x for x in runs.values() if x['state'] in ACTIVE), None)
        rows.append([t, tinfo.get('batch', 'B2' if t in B2 else 'B3'),
                     'QA_OK' if qa_done(t, st) else ('BLOCKED' if tinfo.get('blocked') else
                      ('DONE' if r else (act['state'] if act else
                       ('FAILED' if runs else 'WAIT')))),
                     (r or act or {}).get('acct', ''),
                     (r or {}).get('asset', ''),
                     round(sum(x['eecu'] for x in runs.values()), 2),
                     1 if qa_done(t, st) else 0])
    with open(BOARD, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['tile', 'batch', 'state', 'acct', 'asset', 'eecu', 'qa_done'])
        w.writerows(rows)


def postprocess(t, st):
    """串行后处理：下载 → 上色 → 保真（复跑一次含重下载）→ QA。返回最终保真率或 None。"""
    meta = TILES[t]
    suf, r = winner(t, st)
    delp = os.path.join(RDIR, '%s_10m.tif' % t)
    tinfo = st['tiles'][t]
    try:
        for attempt in (1, 2):
            if not os.path.exists(delp):
                rawp = os.path.join(RDIR, '_raw_%s.tif' % t)
                if not (os.path.exists(rawp) and os.path.getsize(rawp) > 1e6):
                    if not download(t, meta, r):
                        return None
                colorize(t)
            fid = fidelity(t, meta, r)
            tinfo['fidelity'] = fid
            if fid['rate'] == 1.0:
                qa(t, meta)
                return 1.0
            log('%s 保真未达 100%%（第 %d 次），重下载复跑' % (t, attempt))
            for p in (delp, os.path.join(RDIR, '_raw_%s.tif' % t)):
                if os.path.exists(p):
                    os.remove(p)
        tinfo['blocked'] = True
        ledger(r['acct'], t, r['task'], r['desc'], 'FID_FAIL', r['eecu'], 'fidelity<100% twice')
        return fid['rate']
    except Exception as e:
        log('%s 后处理异常 %s' % (t, str(e)[:100]))
        return None


def summary(st, why):
    st['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    st['why'] = why
    st['total_eecu'] = total_eecu(st)
    st['done_n'] = sum(1 for t in ORDER if qa_done(t, st))
    for t in ORDER:
        suf, r = winner(t, st)
        st['tiles'][t]['winner'] = dict(suf=suf, asset=r['asset'] if r else None,
                                        acct=r['acct'] if r else None,
                                        eecu=r['eecu'] if r else None) if r else None
    save(st)
    board(st)
    log('=== 总结（%s）：EECU=%.2f/%.0f，完成 %d/%d ===' % (
        why, st['total_eecu'], PROJ_CAP, st['done_n'], len(ORDER)))


# ---------- 主循环 ----------
def auto():
    os.makedirs(BD, exist_ok=True)
    os.makedirs(RDIR, exist_ok=True)
    acquire_lock()
    st = S()
    log('b1_batch 启动：批2 %d + 批3 %d = %d 瓦片（预算 %.0f EECU·h，截止 %s）' % (
        len(B2), len(B3), len(ORDER), PROJ_CAP - B1_SPENT,
        time.strftime('%m-%d %H:%M', time.localtime(DEADLINE))))
    HOST = VC.jload(os.path.join(ROOT, 'config.json'), {}).get('host', 'zitwwufh')
    ee0, _ = VC.ctx(HOST)
    while time.time() < DEADLINE:
        if not st.get('src_ok'):
            try:
                if not wait_ready(st, ee0):
                    save(st)
                    time.sleep(300)
                    continue
            except Exception as e:
                log('等 SRC 异常 %s' % str(e)[:80])
                save(st)
                time.sleep(300)
                continue
        fuse = st.get('fuse')
        tot = total_eecu(st)
        if not fuse and B1_SPENT + tot > PROJ_CAP:
            fuse = st['fuse'] = 'BUDGET'
            log('⛔ 项目总预算超限（批1 %.1f + 批 %.1f > %.0f），停止新提交并等指令'
                % (B1_SPENT, tot, PROJ_CAP))
        if not fuse:
            # 预测熔断（裁决口径）：批1已花 + 批已花 + 在跑瓦×实均价 > 1,600 → 停新提交上报
            settled = [t for t in ORDER if st['tiles'].get(t, {}).get('runs')]
            avg = tot / len(settled) if settled else 19.0
            act_tiles = sum(1 for t in ORDER
                            if any(x['state'] in ACTIVE
                                   for x in st['tiles'].get(t, {}).get('runs', {}).values()))
            projected = B1_SPENT + tot + act_tiles * avg
            if projected > PROJ_CAP:
                fuse = st['fuse'] = 'PROJECTED_BUDGET'
                log('⛔ 预计超项目总上限（批1 %.1f + 批已花 %.1f + 在跑 %d 瓦×均价 %.1f ≈ %.0f > %.0f），停止新提交并等指令'
                    % (B1_SPENT, tot, act_tiles, avg, projected, PROJ_CAP))
        # ---- 提交（≤2/轮；批3 受批2闸门）----
        if not fuse:
            gate3 = b2_done_n(st) >= B2_GATE
            new_n = 0
            for t in ORDER:
                if new_n >= NEW_PER_ROUND:
                    break
                if t in B3 and not gate3:
                    continue
                try:
                    if submit_tile(t, st):
                        new_n += 1
                except Exception as e:
                    log('%s 提交异常 %s' % (t, str(e)[:80]))
                    acct_err(st, pick_acct(st, set()), str(e)[:60])
        # ---- 轮询 + 赛跑 ----
        fail_now = 0
        for t in ORDER:
            try:
                poll_tile(t, st)
            except Exception as e:
                log('%s 轮询异常 %s' % (t, str(e)[:80]))
            suf, r = winner(t, st)
            tinfo = st['tiles'].get(t, {})
            runs = tinfo.get('runs', {})
            if r and not qa_done(t, st) and not tinfo.get('blocked'):
                rate = postprocess(t, st)
                if rate == 1.0:
                    st['fail_streak'] = 0
                    meta = TILES[t]
                    if meta['land_frac'] == 1.0 and r['eecu'] > OVER_LIMIT:
                        if t not in st['over_tiles']:
                            st['over_tiles'].append(t)
                            log('⚠️ %s 非沿海 EECU=%.1f > %.1f' % (t, r['eecu'], OVER_LIMIT))
                elif rate is None:
                    pass  # 网络等瞬态异常，下轮重试
                else:  # 保真两次 <100% → 熔断
                    st['fuse'] = 'FIDELITY:%s' % t
                    log('⛔ 熔断：%s 保真两次复跑仍 <100%%' % t)
            elif not r and runs and len(runs) >= MAX_ROUTES and \
                    not any(x['state'] in ACTIVE for x in runs.values()) and \
                    not tinfo.get('failed_marked'):
                tinfo['failed_marked'] = True
                fail_now += 1
                ledger(runs[sorted(runs)[0]]['acct'], t,
                       runs[sorted(runs)[0]]['task'], 'ALL_FAILED', 'FAILED',
                       sum(x['eecu'] for x in runs.values()), 'no winner after %d routes' % len(runs))
        if fail_now:
            st['fail_streak'] = st.get('fail_streak', 0) + fail_now
            if st['fail_streak'] >= 3 and not st.get('fuse'):
                st['fuse'] = 'GEE_FAILED_X3'
                log('⛔ 熔断：连续 %d 瓦片 GEE 全 FAILED' % st['fail_streak'])
        if len(st['over_tiles']) > 2 and not st.get('fuse'):
            st['fuse'] = 'EECU_OVER_X3'
            log('⛔ 熔断：非沿海 EECU>%.1f 瓦片达 %d 个' % (OVER_LIMIT, len(st['over_tiles'])))
        # ---- 收尾判定 ----
        done = sum(1 for t in ORDER if qa_done(t, st))
        blk = sum(1 for t in ORDER if st['tiles'].get(t, {}).get('blocked'))
        tot = total_eecu(st)
        board(st)
        save(st)
        if done + blk >= len(ORDER):
            summary(st, 'DONE' if blk == 0 else 'DONE_WITH_BLOCKED')
            return
        act_n = sum(1 for t in ORDER for x in st['tiles'].get(t, {}).get('runs', {}).values()
                    if x['state'] in ACTIVE)
        if act_n == 0 and not st.get('fuse') and B1_SPENT + tot <= PROJ_CAP:
            # 无在跑、无熔断、未完成 → 可能全卡死，防呆记数
            st['idle_rounds'] = st.get('idle_rounds', 0) + 1
            if st['idle_rounds'] >= 6:
                summary(st, 'IDLE_STALL')
                return
        else:
            st['idle_rounds'] = 0
        if st.get('fuse') and act_n == 0:
            summary(st, 'FUSED:' + st['fuse'])
            return
        log('本轮汇总：EECU=%.2f/%.0f，完成=%d 剩=%d 在跑=%d 熔断=%s' % (
            tot, PROJ_CAP - B1_SPENT, done, len(ORDER) - done - blk, act_n, st.get('fuse')))
        time.sleep(300)
    summary(st, 'DEADLINE')


if __name__ == '__main__':
    auto()

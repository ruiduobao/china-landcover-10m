# -*- coding: utf-8 -*-
"""r4_audit.py — 30 m《100 万植被图》针/阔标签的 2019–2024 独立盲判审计
* 输入：opt/needle_broad_veg30_v1.parquet（138,175 行；字段 leaf/gen/prov/dist_v2_m）
* 抽样：G1 南方马尾松 70 / G2 南方杉木 60 / G3 南方其他针叶 40 /
        G4 南方阔叶对照 30 / G5 北方针叶对照 40（合计 240）
        约束：组内与全局空间间隔 ≥0.15°；优先 dist_v2_m ≥1000；随机种子 20261024
* 指标：S2_SR_HARMONIZED 2019-06-01~2024-09-30，QA60+SCL 去云，
        CLOUDY_PIXEL_PERCENTAGE<40；ndvi_gs(6–9 月均值) / ndvi_djf(12–2 月均值)
        amp = ndvi_gs − ndvi_djf（报告同时给 NDVI 单位与 ×1000 毫单位，判据文字用 250/400）
        S1_GRD（IW，VH）2019–2024 冬季（12–2 月）均值 dB
* 判据（预注册、判完不改）：TH；敏感性 TH_VARIANTS
* 输出：_r4_pts.csv（抽样）、_r4_raw.json（指标缓存）、r4_blind.csv（盲判：仅 lon/lat/指标）、
        r4_result.csv（merge 标签）、r4_audit.json（统计/敏感性/典型案例）
* 用法：python r4_audit.py --stage all
        python r4_audit.py --stage sample|extract|blind|report
* 盲判纪律：blind 阶段**只读** _r4_raw.json 的指标，不读 leaf/gen；report 阶段才 merge。

【设计上限，跑前声明】NDVI 季节振幅无法区分"落叶针叶（落叶松 Larix）"与"落叶阔叶"——
两者都冬季落叶、振幅大。故 G5 北方针叶对照**必须先按 落叶松 / 常绿针叶 分层**再解读，
否则会把判据的固有不可分误读成"北方针叶标签失效"。南方马尾松/杉木均为常绿针叶，不受此限。
"""
import os, sys, json, time, argparse, warnings
sys.stdout.reconfigure(encoding='utf-8')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '8.全国生产'))
import prod_conf as C
for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
    os.environ.pop(k, None)
os.environ['HTTP_PROXY'] = C.PROXY_URL
os.environ['HTTPS_PROXY'] = C.PROXY_URL
os.environ['ALL_PROXY'] = C.PROXY_URL
os.environ['NO_PROXY'] = ''

import numpy as np
import pandas as pd
import ee   # 惰性对象图；凭证在 init_ee() 里按账号切换

OUT = r'F:/lc_work/nb_national_2026/opt'
SRC = os.path.join(OUT, 'needle_broad_veg30_v1.parquet')
PTS = os.path.join(OUT, '_r4_pts.csv')
RAW = os.path.join(OUT, '_r4_raw.json')
BLIND = os.path.join(OUT, 'r4_blind.csv')
RES = os.path.join(OUT, 'r4_result.csv')
JSN = os.path.join(OUT, 'r4_audit.json')

# 健康账号（2026-10-25 01:10 串行 s0_sweep --force + s0b_health：仅此两个 OK，其余 21 个 restricted）
ACCTS = ['ybzljyma', 'nnjxsn']
PIDS = {'ybzljyma': 'modern-tangent-509301-q8', 'nnjxsn': 'melodic-gamma-509323-u5'}
SEED = 20261024
YEARS = list(range(2019, 2025))
SOUTH = ['广东省', '广西壮族自治区', '福建省', '浙江省', '江西省', '湖南省']
NORTH = ['黑龙江省', '吉林省', '内蒙古自治区', '辽宁省']
MIN_GAP = 0.15          # 度
DIST_MIN = 1000.0       # m，优先"新样本"
GROUPS = [('G1', 70), ('G2', 60), ('G3', 40), ('G4', 30), ('G5', 40)]

# ---- 预注册阈值（amp 用 ×1000 毫 NDVI 单位，与任务表字面一致）----
TH = dict(ndvi_djf_hi=0.45, ndvi_djf_lo=0.35, ndvi_evg=0.55,
          amp_hi=250.0, amp_lo=400.0, vh_hi=-16.0, vh_lo=-17.0,
          min_gs=0.30, min_obs=3)
TH_VARIANTS = {
    'primary':      dict(amp_hi=250.0, amp_lo=400.0, vh_hi=-16.0, vh_lo=-17.0, use_vh=True),
    'amp300':       dict(amp_hi=300.0, amp_lo=300.0, vh_hi=-16.0, vh_lo=-17.0, use_vh=True),
    'vh18':         dict(amp_hi=250.0, amp_lo=400.0, vh_hi=-18.0, vh_lo=-18.0, use_vh=True),
    'amp300_vh18':  dict(amp_hi=300.0, amp_lo=300.0, vh_hi=-18.0, vh_lo=-18.0, use_vh=True),
    'no_vh':        dict(amp_hi=250.0, amp_lo=400.0, vh_hi=-16.0, vh_lo=-17.0, use_vh=False),
}


# ------------------------------ GEE ------------------------------
def init_ee(prefer=None):
    import ee
    order = ([prefer] if prefer else []) + [a for a in ACCTS if a != prefer]
    for a in order:
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                C.apply_account_env(a)
                ee.Initialize(project=PIDS[a])
                msgs = ' '.join(str(w.message) for w in ws)
                if 'restricted' in msgs or 'quota' in msgs:
                    print('  acct', a, 'RESTRICTED(warn)'); continue
            print('  using', a, PIDS[a], flush=True)
            return ee, a
        except Exception as e:
            print('  acct', a, 'FAIL', repr(e)[:140], flush=True)
    raise RuntimeError('no usable GEE account')


def s2_masked(ee, region=None):
    """S2 SR harmonized 2019-06-01~2024-09-30；QA60 + SCL 去云；CLOUDY_PIXEL_PERCENTAGE<40
    region: 必给的 ee.Geometry —— 不给 bounds 会扫全球集合而超时（实测）"""
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate('2019-06-01', '2024-09-30')
          .filterBounds(region)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))

    def m(img):
        scl = img.select('SCL')
        qa = img.select('QA60')
        bad = (scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
               .Or(qa.bitwiseAnd(1 << 10).neq(0))
               .Or(qa.bitwiseAnd(1 << 11).neq(0)))
        nd = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
        return nd.updateMask(bad.Not()).copyProperties(img, ['system:time_start'])
    return s2.map(m)


def _gs(ic):
    return ic.filter(ee.Filter.calendarRange(6, 9, 'month'))


def _djf(ic):
    return ic.filter(ee.Filter.Or(ee.Filter.calendarRange(12, 12, 'month'),
                                  ee.Filter.calendarRange(1, 2, 'month')))


def build_s2(ee, ic):
    gs = _gs(ic).select('ndvi')
    dj = _djf(ic).select('ndvi')
    b = [gs.mean().rename('ndvi_gs'),
         dj.mean().rename('ndvi_djf'),
         gs.count().rename('n_gs'),
         dj.count().rename('n_djf'),
         ic.select('ndvi').count().rename('n_all')]
    # 逐年（证据用）：生长季 6–9 月、冬季 12–次年 2 月
    for y in YEARS:
        b.append(ic.filterDate(f'{y}-06-01', f'{y}-10-01').select('ndvi')
                 .mean().rename(f'ndvi_gs_{y}'))
    for y in range(2019, 2025):
        b.append(ic.filterDate(f'{y}-12-01', f'{y+1}-03-01').select('ndvi')
                 .mean().rename(f'ndvi_djf_{y}'))
    return ee.Image.cat(b)


def build_s1(ee, region=None):
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filterDate('2019-01-01', '2025-01-01')
          .filterBounds(region)
          .select(['VH', 'VV', 'angle']))
    dj = _djf(s1)
    b = [dj.select('VH').mean().rename('vh_djf'),
         dj.select('VH').count().rename('n_vh_djf'),
         dj.select('VV').mean().rename('vv_djf'),
         dj.select('angle').mean().rename('ang_djf')]
    for y in range(2019, 2025):
        b.append(s1.filterDate(f'{y}-12-01', f'{y+1}-03-01').select('VH')
                 .mean().rename(f'vh_djf_{y}'))
    return ee.Image.cat(b)


def reduce_img(ee, img, fc, scale, tag):
    for attempt in range(5):
        try:
            t0 = time.time()
            r = img.reduceRegions(collection=fc, reducer=ee.Reducer.median(),
                                  scale=scale, tileScale=8).getInfo()
            print(f'   {tag} ok n={len(r["features"])} {time.time()-t0:.0f}s', flush=True)
            return {str(f['properties']['pid']): f['properties'] for f in r['features']}
        except Exception as e:
            print(f'   {tag} retry{attempt} {repr(e)[:170]}', flush=True)
            time.sleep(6 * (attempt + 1))
    print(f'   {tag} FAILED', flush=True)
    return {}


# ------------------------------ 抽样 ------------------------------
def sample_points():
    df = pd.read_parquet(SRC)
    d = df.copy()
    g = pd.Series(index=d.index, dtype=object)
    g[d.prov.isin(SOUTH) & d.gen.str.contains('马尾松') & (d.leaf == 'needle')] = 'G1'
    g[d.prov.isin(SOUTH) & d.gen.str.contains('杉木') & (d.leaf == 'needle')] = 'G2'
    g[d.prov.isin(SOUTH) & (d.leaf == 'needle') & g.isna()] = 'G3'
    g[d.prov.isin(SOUTH) & (d.leaf == 'broad')] = 'G4'
    g[d.prov.isin(NORTH) & (d.leaf == 'needle')] = 'G5'
    d['group'] = g
    d = d[d.group.notna()].copy()

    rng = np.random.default_rng(SEED)
    picked = []                      # 已选（全局间隔约束）
    log = []
    for gname, n in GROUPS:
        c = d[(d.group == gname) & (d.dist_v2_m >= DIST_MIN)].copy()
        n_pool = len(c)
        if n_pool == 0:
            log.append(dict(group=gname, n_pool=0, n_got=0)); continue
        # 距离分带（500 m）内先随机，再按带从高到低取 → 大 dist 优先但不取极端尾部
        c['band'] = (c.dist_v2_m // 500).astype(int)
        parts = []
        for b, sub in c.groupby('band'):
            parts.append(sub.sample(frac=1.0, random_state=int(rng.integers(1 << 30))))
        c = pd.concat(parts).sort_values('band', ascending=False, kind='stable')
        acc, relaxed = [], 0
        for r in c.itertuples():
            ok_g = all(((r.lon - x.lon) ** 2 + (r.lat - x.lat) ** 2) ** 0.5 >= MIN_GAP
                       for x in acc)
            if not ok_g:
                continue
            ok_all = all(((r.lon - x.lon) ** 2 + (r.lat - x.lat) ** 2) ** 0.5 >= MIN_GAP
                         for x in picked)
            if not ok_all:
                relaxed += 1
                # 先记备选；仅当组内不足时才启用跨组放宽
                continue
            acc.append(r)
            if len(acc) >= n:
                break
        if len(acc) < n:            # 跨组间隔放宽补齐
            for r in c.itertuples():
                if len(acc) >= n:
                    break
                if any(r.point_id == a.point_id for a in acc):
                    continue
                if all(((r.lon - a.lon) ** 2 + (r.lat - a.lat) ** 2) ** 0.5 >= MIN_GAP
                       for a in acc):
                    acc.append(r)
        picked.extend(acc)
        log.append(dict(group=gname, n_pool=int(n_pool), n_got=len(acc),
                        n_need=int(n), n_relaxed_cross_group=relaxed))
        print(f'  {gname}: pool {n_pool} -> picked {len(acc)}/{n}', flush=True)

    out = pd.DataFrame([dict(group=r.group, point_id=int(r.point_id),
                             lon=float(r.lon), lat=float(r.lat),
                             prov=r.prov, gen=r.gen, leaf=r.leaf,
                             code30=int(r.code30), ctype=r.ctype, vtype=r.vtype,
                             dist_v2_m=float(r.dist_v2_m),
                             ambig_level=int(r.ambig_level))
                        for r in picked])
    out.to_csv(PTS, index=False, encoding='utf-8-sig')
    # 实现质量：各组最小间距
    qual = {}
    for gname, sub in out.groupby('group'):
        a = sub[['lon', 'lat']].values
        if len(a) > 1:
            dm = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
            np.fill_diagonal(dm, 9e9)
            qual[gname] = dict(n=len(a), min_gap_deg=round(float(dm.min()), 3),
                               dist_median_m=round(float(sub.dist_v2_m.median()), 0),
                               dist_min_m=round(float(sub.dist_v2_m.min()), 0),
                               prov={k: int(v) for k, v in sub.prov.value_counts().items()})
    json.dump(dict(seed=SEED, min_gap=MIN_GAP, dist_min=DIST_MIN,
                   groups=log, realized=qual, n_total=len(out)),
              open(os.path.join(OUT, '_r4_sample_log.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('sample ->', PTS, len(out))
    return out


# ------------------------------ 提取 ------------------------------
def stage_extract(cell=2.0, chunk=60, resume=True):
    """按 2° 空间格分批：每格单独 filterBounds，避免扫全球集合（实测无 bounds 必超时）"""
    ee, acct = init_ee()
    pts = pd.read_csv(PTS)
    pts['gx'] = (pts.lon // cell).astype(int)
    pts['gy'] = (pts.lat // cell).astype(int)
    pts['cell'] = pts.gx.astype(str) + '_' + pts.gy.astype(str)
    raw = {}
    if resume and os.path.exists(RAW):
        try:
            old = json.load(open(RAW, encoding='utf-8'))
            raw = old.get('raw', {})
            print('resume: cached', len(raw), flush=True)
        except Exception:
            raw = {}
    cells = list(pts.groupby('cell'))
    t00 = time.time()
    for ci, (cname, sub) in enumerate(cells):
        todo = sub[~sub.point_id.astype(str).isin(raw.keys())]
        if len(todo) == 0:
            print(f'  cell {cname} cached ({len(sub)})', flush=True)
            continue
        lo1, lo2 = int(todo.gx.iloc[0]) * cell, int(todo.gy.iloc[0]) * cell
        reg = ee.Geometry.Rectangle([lo1 - 0.05, lo2 - 0.05,
                                     lo1 + cell + 0.05, lo2 + cell + 0.05])
        ic = s2_masked(ee, reg)
        s2img = build_s2(ee, ic)
        s1img = build_s1(ee, reg)
        for i in range(0, len(todo), chunk):
            c = todo.iloc[i:i + chunk]
            fc = ee.FeatureCollection([ee.Feature(
                ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(15),
                {'pid': str(r.point_id)}) for r in c.itertuples()])
            a = reduce_img(ee, s2img, fc, 10, f's2 {cname}[{i}:{i+len(c)}]')
            b = reduce_img(ee, s1img, fc, 10, f's1 {cname}[{i}:{i+len(c)}]')
            for pid in c.point_id.astype(str):
                raw[pid] = dict(s2=a.get(pid, {}), s1=b.get(pid, {}))
        # 每格落盘一次 → 断点续跑
        json.dump(dict(acct=acct, th=TH, s2_window='2019-06-01/2024-09-30',
                       s1_window='2019-01-01/2025-01-01', raw=raw),
                  open(RAW, 'w', encoding='utf-8'), ensure_ascii=False, default=str)
        print(f'  cell {cname} done ({len(todo)}) {ci+1}/{len(cells)} '
              f'elapsed {time.time()-t00:.0f}s', flush=True)
    got = sum(1 for v in raw.values() if v['s2'].get('ndvi_djf') is not None)
    gotv = sum(1 for v in raw.values() if v['s1'].get('vh_djf') is not None)
    print('extract done; s2 nonnull', got, '/', len(raw), '; s1 nonnull', gotv, flush=True)


# ------------------------------ 判读 ------------------------------
def _f(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def flatten(v):
    f = {}
    f.update(v.get('s2') or {})
    f.update(v.get('s1') or {})
    return f


def _cond(vals, ops, ths):
    """返回 (n_true, n_missing, detail)"""
    t = m = 0
    det = []
    for v, op, t_ in zip(vals, ops, ths):
        if not np.isfinite(v):
            m += 1; det.append(None); continue
        ok = (v > t_) if op == '>' else (v < t_)
        t += int(ok); det.append(bool(ok))
    return t, m, det


def judge(f, th):
    nd_djf = _f(f.get('ndvi_djf')); nd_gs = _f(f.get('ndvi_gs'))
    vh = _f(f.get('vh_djf'))
    amp = (nd_gs - nd_djf) if (np.isfinite(nd_gs) and np.isfinite(nd_djf)) else np.nan
    amp_m = 1000.0 * amp if np.isfinite(amp) else np.nan
    n_djf = _f(f.get('n_djf'))
    base = dict(ndvi_gs=nd_gs, ndvi_djf=nd_djf, amp=amp, amp_m=amp_m,
                vh_djf=vh, n_gs=_f(f.get('n_gs')), n_djf=n_djf,
                n_vh_djf=_f(f.get('n_vh_djf')))
    # 指标投票（诊断用，-3..+3；+ 指针叶）
    v_nd = 1 if (np.isfinite(nd_djf) and nd_djf > TH['ndvi_djf_hi']) else (
        -1 if (np.isfinite(nd_djf) and nd_djf < TH['ndvi_djf_lo']) else 0)
    v_amp = 1 if (np.isfinite(amp_m) and amp_m < th['amp_hi']) else (
        -1 if (np.isfinite(amp_m) and amp_m > th['amp_lo']) else 0)
    v_vh = 1 if (np.isfinite(vh) and vh > th['vh_hi']) else (
        -1 if (np.isfinite(vh) and vh < th['vh_lo']) else 0)
    base['score'] = v_nd + v_amp + v_vh
    base['votes'] = f'ndvi{v_nd:+d}/amp{v_amp:+d}/vh{v_vh:+d}'

    if not (np.isfinite(nd_djf) and np.isfinite(nd_gs)):
        return dict(label='无法判读', conf='低', note='S2 无有效季节观测', **base)
    if nd_gs < TH['min_gs']:
        return dict(label='无法判读', conf='低', note=f'生长季 NDVI {nd_gs:.2f}<0.30（非植被/裸地/水体）', **base)
    if np.isfinite(n_djf) and n_djf < TH['min_obs']:
        return dict(label='无法判读', conf='低', note=f'冬季有效观测仅 {int(n_djf)} 景', **base)

    use_vh = th.get('use_vh', True)
    vh_vals = [vh] if use_vh else []
    # 针叶：冬季仍绿(NDVI高) + 振幅小 + VH 高(生物量/结构持续)
    t_n, m_n, d_n = _cond([nd_djf, amp_m] + vh_vals,
                          ['>', '<'] + (['>'] if use_vh else []),
                          [TH['ndvi_djf_hi'], th['amp_hi']] + ([th['vh_hi']] if use_vh else []))
    # 落叶阔叶：冬季裸(NDVI低) + 振幅大
    t_d, m_d, d_d = _cond([nd_djf, amp_m], ['<', '>'], [TH['ndvi_djf_lo'], th['amp_lo']])
    # 常绿阔叶：冬季也高 NDVI + 振幅小 + VH 低
    t_e, m_e, d_e = _cond([nd_djf, amp_m] + vh_vals,
                          ['>', '<'] + (['<'] if use_vh else []),
                          [TH['ndvi_evg'], th['amp_hi']] + ([th['vh_lo']] if use_vh else []))

    def full(t, m, tot):
        return (t == tot and m == 0)

    cands = []
    if full(t_n, m_n, 3 if use_vh else 2):
        cands.append(('针叶', '高'))
    if full(t_d, m_d, 2):
        cands.append(('阔叶(落叶)', '高'))
    if full(t_e, m_e, 3 if use_vh else 2):
        cands.append(('阔叶(常绿)', '高'))
    if len(cands) > 1:
        return dict(label='无法判读', conf='低',
                    note='规则冲突：' + '+'.join(c[0] for c in cands), **base)
    if len(cands) == 1:
        return dict(label=cands[0][0], conf=cands[0][1], note='', **base)

    # 中置信：2 项同向 + 1 项缺失
    if use_vh:
        if t_n == 2 and m_n == 1:
            return dict(label='针叶', conf='中', note='VH 缺失，NDVI/振幅两项支持针叶', **base)
        if t_e == 2 and m_e == 1:
            return dict(label='阔叶(常绿)', conf='中', note='VH 缺失，NDVI/振幅两项支持常绿阔叶', **base)
    if t_d == 1 and m_d == 1:
        return dict(label='阔叶(落叶)', conf='低', note='仅 1 项可判', **base)
    note = ('VH 不可用：常绿针叶/常绿阔叶不可分' if (not use_vh and t_n == 2 and t_e == 2)
            else f'条件不全（针{t_n}/3 落{t_d}/2 常{t_e}/3）')
    return dict(label='无法判读', conf='低', note=note, **base)


def evidence(f, j):
    p = []
    if np.isfinite(j['ndvi_djf']):
        p.append(f"冬NDVI {j['ndvi_djf']:.2f}")
    if np.isfinite(j['ndvi_gs']):
        p.append(f"夏NDVI {j['ndvi_gs']:.2f}")
    if np.isfinite(j['amp_m']):
        p.append(f"振幅 {j['amp']:.2f}({j['amp_m']:.0f})")
    if np.isfinite(j['vh_djf']):
        p.append(f"冬VH {j['vh_djf']:.1f}dB")
    yb = [(y, _f(f.get(f'ndvi_djf_{y}'))) for y in YEARS]
    yb = [(y, x) for y, x in yb if np.isfinite(x)]
    if yb:
        p.append('冬NDVI逐年 ' + '/'.join(f'{y%100:02d}:{x:.2f}' for y, x in yb))
    yv = [(y, _f(f.get(f'vh_djf_{y}'))) for y in YEARS]
    yv = [(y, x) for y, x in yv if np.isfinite(x)]
    if yv:
        p.append('冬VH逐年 ' + '/'.join(f'{y%100:02d}:{x:.1f}' for y, x in yv))
    if np.isfinite(j['n_djf']):
        p.append(f"冬{j['n_djf']:.0f}景")
    if np.isfinite(j['n_vh_djf']):
        p.append(f"VH{j['n_vh_djf']:.0f}景")
    return '; '.join(p)


def stage_blind():
    """盲判：只读指标缓存，绝不读 leaf/gen/prov"""
    pts = pd.read_csv(PTS)[['group', 'point_id', 'lon', 'lat']]   # 只带抽样键，不带标签
    rw = json.load(open(RAW, encoding='utf-8'))['raw']
    rows = []
    for r in pts.itertuples():
        f = flatten(rw.get(str(r.point_id), {}))
        j = judge(f, TH)
        rows.append(dict(group=r.group, point_id=int(r.point_id),
                         lon=round(float(r.lon), 6), lat=round(float(r.lat), 6),
                         ndvi_gs=None if not np.isfinite(j['ndvi_gs']) else round(j['ndvi_gs'], 4),
                         ndvi_djf=None if not np.isfinite(j['ndvi_djf']) else round(j['ndvi_djf'], 4),
                         amp=None if not np.isfinite(j['amp']) else round(j['amp'], 4),
                         amp_m=None if not np.isfinite(j['amp_m']) else round(j['amp_m'], 1),
                         vh_djf=None if not np.isfinite(j['vh_djf']) else round(j['vh_djf'], 2),
                         n_djf=None if not np.isfinite(j['n_djf']) else int(j['n_djf']),
                         n_vh_djf=None if not np.isfinite(j['n_vh_djf']) else int(j['n_vh_djf']),
                         score=j['score'], votes=j['votes'],
                         verdict=j['label'], conf=j['conf'], note=j['note'],
                         evidence=evidence(f, j)))
    df = pd.DataFrame(rows)
    df.to_csv(BLIND, index=False, encoding='utf-8-sig')
    print('blind ->', BLIND, df.verdict.value_counts().to_dict(), flush=True)
    return df


# ------------------------------ 统计 ------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _r(x, n=4):
    return None if (x is None or (isinstance(x, float) and not np.isfinite(x))) else round(float(x), n)


def blk(labels):
    s = pd.Series(list(labels))
    n = len(s)
    dec = int(s.isin(['针叶', '阔叶(落叶)', '阔叶(常绿)']).sum())
    k = int((s == '针叶').sum())
    nb = int(s.isin(['阔叶(落叶)', '阔叶(常绿)']).sum())
    lo, hi = wilson(k, dec)
    return dict(n=n, n_decided=dec, n_needle=k, n_broad=nb,
                n_broad_decid=int((s == '阔叶(落叶)').sum()),
                n_broad_everg=int((s == '阔叶(常绿)').sum()),
                n_undecided=int((s == '无法判读').sum()),
                decided_pct=round(100 * dec / n, 1) if n else None,
                needle_rate=_r(k / dec) if dec else None,
                needle_rate_pct=round(100 * k / dec, 1) if dec else None,
                wilson95_pct=[round(100 * lo, 1), round(100 * hi, 1)] if dec else None)


def stage_report():
    pts = pd.read_csv(PTS)
    b = pd.read_csv(BLIND)
    df = b.merge(pts[['point_id', 'prov', 'gen', 'leaf', 'code30', 'vtype', 'ctype',
                      'dist_v2_m', 'ambig_level']], on='point_id', how='left')
    df['says'] = np.where(df.leaf == 'needle', '针叶', '阔叶')
    df['agree'] = np.where(df.verdict.isin(['针叶', '阔叶(落叶)', '阔叶(常绿)']),
                           np.where((df.verdict == '针叶') == (df.says == '针叶'), 1, 0),
                           np.nan)
    # 仅针叶 / 仅阔叶 二值化（用于与 30m 二分类对比）
    df['verdict_leaf'] = df.verdict.map({'针叶': '针叶', '阔叶(落叶)': '阔叶',
                                         '阔叶(常绿)': '阔叶', '无法判读': '无法判读'})
    df.to_csv(RES, index=False, encoding='utf-8-sig')

    GNAME = {'G1': '南方马尾松（30m 说针叶）', 'G2': '南方杉木（30m 说针叶）',
             'G3': '南方其他针叶（30m 说针叶）', 'G4': '南方阔叶对照（30m 说阔叶）',
             'G5': '北方针叶对照（30m 说针叶）'}
    res = dict(generated=pd.Timestamp.now().isoformat(timespec='seconds'),
               accounts_used=json.load(open(RAW, encoding='utf-8')).get('acct'),
               thresholds=TH, amp_unit='amp_m = 1000*(ndvi_gs-ndvi_djf)；判据 250/400 即 0.250/0.400 NDVI',
               windows=dict(s2='2019-06-01/2024-09-30 (CLOUDY_PIXEL_PERCENTAGE<40, QA60+SCL)',
                            s1='2019-01-01/2025-01-01, IW VH, DJF mean'),
               n_total=int(len(df)), by_group={}, overall={}, by_prov={}, sensitivity={},
               tristate={str(k): int(v) for k, v in df.verdict.value_counts().items()},
               conf_dist={str(k): int(v) for k, v in df.conf.value_counts().items()})

    for g, sub in df.groupby('group'):
        x = blk(sub.verdict)
        x['group_name'] = GNAME.get(g, g)
        x['says'] = sub.says.iloc[0]
        x['agreement'] = (x['needle_rate'] if x['says'] == '针叶'
                          else (1 - x['needle_rate'] if x['needle_rate'] is not None else None))
        lo, hi = wilson(int((sub.agree == 1).sum()), int(sub.agree.notna().sum()))
        x['agreement_wilson95_pct'] = [round(100 * lo, 1), round(100 * hi, 1)]
        x['conf'] = {str(k): int(v) for k, v in sub.conf.value_counts().items()}
        res['by_group'][g] = x

    south_needle = df[df.group.isin(['G1', 'G2', 'G3'])]
    ov = blk(south_needle.verdict)
    ov['scope'] = '南方 6 省 30m 说针叶的点（G1+G2+G3）'
    res['overall']['south_30m_needle'] = ov
    all_needle = df[df.says == '针叶']
    res['overall']['all_30m_needle'] = blk(all_needle.verdict)
    res['overall']['all_30m_needle']['scope'] = '全部 30m 说针叶的点（G1+G2+G3+G5）'
    res['overall']['all_30m_broad_G4'] = blk(df[df.group == 'G4'].verdict)
    res['overall']['all_30m_broad_G4']['scope'] = '30m 说阔叶对照（G4）'

    for p, sub in south_needle.groupby('prov'):
        res['by_prov'][p] = blk(sub.verdict)
    res['by_prov']['_all_south'] = blk(south_needle.verdict)
    res['by_prov_group'] = {f'{p}|{g}': blk(s.verdict)
                            for (p, g), s in south_needle.groupby(['prov', 'group'])}
    # 北方对照分层：落叶松（判据结构性不可分）vs 常绿针叶
    g5 = df[df.group == 'G5'].copy()
    g5['sub'] = np.where(g5.gen.str.contains('落叶松'), '落叶松(落叶针叶)',
                         np.where(g5.gen.str.contains('云杉|冷杉'), '云杉/冷杉(常绿针叶)',
                                  '其他常绿针叶(松类)'))
    res['G5_north_by_sub'] = {k: blk(s.verdict) for k, s in g5.groupby('sub')}
    res['G5_north_by_prov'] = {k: blk(s.verdict) for k, s in g5.groupby('prov')}
    res['G5_north_detail'] = {k: dict(n=len(s), gen={a: int(b) for a, b in
                                                     s.gen.value_counts().items()})
                             for k, s in g5.groupby('sub')}

    # 假阳性参照：30m 说阔叶的对照里被判针叶的比例
    g4 = res['by_group']['G4']
    res['false_positive_ref'] = dict(
        G4_needle_pct=g4['needle_rate_pct'],
        note='常绿阔叶与常绿针叶在 NDVI 季节曲线/振幅上高度相似，主要靠 VH 分离；'
             'G4 被判"针叶"的比例即本判据的假阳性量级。')

    # 阈值敏感性
    rw = json.load(open(RAW, encoding='utf-8'))['raw']
    prim = df.set_index('point_id').verdict
    for vn, vth in TH_VARIANTS.items():
        th2 = dict(TH); th2.update(vth)
        labs = {int(pid): judge(flatten(rw.get(str(pid), {})), th2)['label']
                for pid in df.point_id}
        ser = pd.Series(labs)
        rb = blk(ser[df.set_index('point_id').index.map(labs).notna()].tolist()) \
            if False else blk([labs[i] for i in df.point_id])
        both = [(prim[i], labs[i]) for i in df.point_id
                if prim[i] in ('针叶', '阔叶(落叶)', '阔叶(常绿)')
                and labs[i] in ('针叶', '阔叶(落叶)', '阔叶(常绿)')]
        rb['flip_vs_primary_pct'] = (round(100 * sum(1 for x, y in both if x != y) / len(both), 1)
                                     if both else None)
        rb['common_decided'] = len(both)
        rb['params'] = vth
        rb['south_needle_only'] = blk([labs[i] for i in south_needle.point_id])
        res['sensitivity'][vn] = rb
    # 敏感性：分组针叶率
    res['sensitivity_by_group'] = {}
    for vn, vth in TH_VARIANTS.items():
        th2 = dict(TH); th2.update(vth)
        res['sensitivity_by_group'][vn] = {
            g: blk([judge(flatten(rw.get(str(pid), {})), th2)['label'] for pid in s.point_id])
            for g, s in df.groupby('group')}

    # 辅助诊断
    def med(s):
        return round(float(s.median()), 3) if s.notna().any() else None
    res['aux'] = dict(
        ndvi_djf_median_by_group={g: med(s.ndvi_djf) for g, s in df.groupby('group')},
        amp_median_by_group={g: med(s.amp) for g, s in df.groupby('group')},
        vh_djf_median_by_group={g: med(s.vh_djf) for g, s in df.groupby('group')},
        vh_coverage_pct=round(100 * df.vh_djf.notna().mean(), 1),
        s2_djf_obs_median=med(df.n_djf),
        s1_djf_obs_median=med(df.n_vh_djf),
        score_dist={str(k): int(v) for k, v in df.score.value_counts().sort_index().items()},
        undecided_pct=round(100 * (df.verdict == '无法判读').mean(), 1))

    o = res['overall']['south_30m_needle']
    o2 = res['overall']['all_30m_needle']
    res['verdict'] = dict(
        headline_south=f"30m 说针叶的南方 6 省点，2019–2024 实测针叶率 {o['needle_rate_pct']}%"
                       f"（n={o['n_decided']}/{o['n']}，95%CI {o['wilson95_pct']}）",
        headline_all=f"全部 30m 针叶样本点实测针叶率 {o2['needle_rate_pct']}%"
                     f"（n={o2['n_decided']}/{o2['n']}，95%CI {o2['wilson95_pct']}）",
        masson_pct=res['by_group']['G1']['needle_rate_pct'],
        fir_pct=res['by_group']['G2']['needle_rate_pct'],
        other_south_needle_pct=res['by_group']['G3']['needle_rate_pct'],
        guangdong_pct=(res['by_prov'].get('广东省') or {}).get('needle_rate_pct'),
        G4_broad_control_needle_pct=res['by_group']['G4']['needle_rate_pct'],
        G5_north_pct=res['by_group']['G5']['needle_rate_pct'],
        G5_north_evergreen_pct=(res['G5_north_by_sub'].get('其他常绿针叶(松类)') or {}).get('needle_rate_pct'),
        G5_north_spruce_pct=(res['G5_north_by_sub'].get('云杉/冷杉(常绿针叶)') or {}).get('needle_rate_pct'),
        G5_north_larch_pct=(res['G5_north_by_sub'].get('落叶松(落叶针叶)') or {}).get('needle_rate_pct'),
    )

    # 典型案例
    res['typical_cases'] = pick_cases(df)
    json.dump(res, open(JSN, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res['verdict'], ensure_ascii=False, indent=1))
    return res


def pick_cases(df):
    cases = []

    def add(r, why):
        cases.append(dict(point_id=int(r.point_id), group=r.group, prov=r.prov, gen=r.gen,
                          lon=round(float(r.lon), 4), lat=round(float(r.lat), 4),
                          ndvi_gs=_r(r.ndvi_gs, 3), ndvi_djf=_r(r.ndvi_djf, 3),
                          amp=_r(r.amp, 3), vh_djf=_r(r.vh_djf, 1),
                          verdict=r.verdict, conf=r.conf, says=r.says, why=why,
                          evidence=r.evidence))
    # 1) 马尾松被判阔叶且三项一致（最高说服力）
    c = df[(df.group == 'G1') & (df.verdict != '针叶') & (df.conf == '高')]
    c = c[c.vh_djf.notna()].sort_values('amp', ascending=False)
    if len(c):
        add(c.iloc[0], 'G1 马尾松：冬季 NDVI 低、振幅大、VH 弱 → 落叶阔叶特征，30m 标签说针叶')
    # 2) 马尾松被判针叶且三项一致（判据不是一味偏阔叶）
    c = df[(df.group == 'G1') & (df.verdict == '针叶') & (df.conf == '高')]
    c = c[c.vh_djf.notna()].sort_values('vh_djf', ascending=False)
    if len(c):
        add(c.iloc[0], 'G1 马尾松：冬季仍绿+VH 强+振幅小 → 常绿针叶成立，30m 标签正确')
    # 3) 杉木被判阔叶
    c = df[(df.group == 'G2') & (df.verdict != '针叶') & (df.conf == '高')]
    c = c[c.vh_djf.notna()]
    if len(c):
        add(c.iloc[0], 'G2 杉木：被判非针叶，同一图斑的 30m 标签已不代表当下')
    # 4) 北方常绿针叶被判针叶（判据有效性检验）
    c = df[(df.group == 'G5') & (df.verdict == '针叶') & (df.conf == '高')]
    c = c[c.gen.str.contains('松|云杉|冷杉')]
    if len(c):
        add(c.iloc[0], 'G5 北方常绿针叶：判据在同一套规则下正确识别针叶 → 判据本身有效')
    # 5) 落叶松被判阔叶（判据固有不可分，必须写明）
    c = df[(df.group == 'G5') & df.gen.str.contains('落叶松') & (df.verdict != '针叶')]
    if len(c):
        add(c.iloc[0], 'G5 落叶松：冬季落叶 → 判据无法与落叶阔叶区分（判据局限，非标签错误）')
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'sample', 'extract', 'blind', 'report'])
    ap.add_argument('--chunk', type=int, default=60)
    ap.add_argument('--cell', type=float, default=2.0)
    a = ap.parse_args()
    if a.stage in ('all', 'sample'):
        sample_points()
    if a.stage in ('all', 'extract'):
        stage_extract(cell=a.cell, chunk=min(a.chunk, 200))
    if a.stage in ('all', 'blind'):
        stage_blind()
    if a.stage in ('all', 'report'):
        stage_report()


if __name__ == '__main__':
    main()

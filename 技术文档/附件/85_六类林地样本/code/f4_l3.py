# -*- coding: utf-8 -*-
"""f4_l3.py — 六类林地样本的 L3 独立影像验证（doc84 §4.3 验收口径）

* 输入（只读）:
    F:/lc_work/nb_national_2026/forest6/forest6_v1.parquet  （472,307 × 113）
    —— 判定阶段**只读 uid/lon/lat 与标记列，不读 cls6_train**（盲判纪律）
* 规则源:
    F:/地理所/论文/中国土地覆盖数据_2017-2024/技术文档/84_全国六类林地样本汇总.md §4.3
    判据先写死在本文件 §判据（PREREG），判读完不改
* 数据源:
    S2 = COPERNICUS/S2_SR_HARMONIZED  2019-2024 合并（DJF / GS / AUG / OCT 四窗）
         SCL 掩膜 = 剔 3(阴影)/8(云中概率)/9(云高概率)      ← 与 w1_forest_fetch.py 生产口径一致
         合成 = median(B8,B4) → NDVI → ×1000（与 amp2023 同量纲，可直接对照）
         旁证 djf_ns = 额外剔 SCL 11(雪)，用于量化北方积雪污染
    S1 = COPERNICUS/S1_GRD  VH 极化，DJF 窗口（dB）——仅作旁证（doc82 §4：北方 VH 不可靠）
* 门槛:
    每类 700 点（共 4,200）；按生态区分层、每类每生态区 ≥60（不足则全取）
    空间间隔 ≥0.25°（度平面欧氏）；随机种子 20261025
    难点子集：41 含 ≥100 south_needle_flag=1、≥100 high_elev_enf=1；
              42 含 ≥100 东北；45/46 全部来自 S3 档（mixed_tier=='S3'）
* 输出:
    f4_sample.parquet     抽样清单（uid/lon/lat/层/flag；**不含标签**）
    f4_parts/{acct}/cell_*.parquet   GEE 分格原始指标（断点续跑）
    f4_blind.csv          盲判表（无标签）
    f4_result.csv         merge 标签后逐点结果
    L3_result.json        一致率 / Wilson CI / 逐类混淆 / 分层 / 敏感性 / 判据效度
* 用法:
    python f4_l3.py --mode sample
    python f4_l3.py --mode worker --acct ybzljyma --shard 0/2
    python f4_l3.py --mode judge
    python f4_l3.py --mode report
* 资源: 单进程单线程；账号并发 ≤2（本机仅 2 账号健康）；内存 <8 GB
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
WORK = r'F:/lc_work/nb_national_2026/forest6'
SRC = os.path.join(WORK, 'forest6_v1.parquet')
SAMPLE = os.path.join(WORK, 'f4_sample.parquet')
PARTS = os.path.join(WORK, 'f4_parts')
BLIND = os.path.join(WORK, 'f4_blind.csv')
RESULT = os.path.join(WORK, 'f4_result.csv')
OUTJSON = os.path.join(WORK, 'L3_result.json')
FIGD = os.path.join(WORK, 'figs')

S2 = 'COPERNICUS/S2_SR_HARMONIZED'
S1 = 'COPERNICUS/S1_GRD'
SENT = -9999.0
CLOUDY_PCT = 60
CELL_DEG = 1.0
MIN_OBS = 3
YEARS = list(range(2019, 2025))
SCALE = 10                 # 原生 10 m（连续量 NDVI，不涉及类别重采样）
SEED = 20261025
N_PER_CLASS = 700
ECO_MIN = 60
SPACING = 0.25             # 度
CLASSES = [41, 42, 43, 44, 45, 46]
CLS_NAME = {41: '常绿针叶', 42: '落叶针叶', 43: '常绿阔叶', 44: '落叶阔叶',
            45: '常绿针阔混', 46: '落叶针阔混'}

# 健康账号（2026-10-25 19:07 串行 s0_sweep --force + s0b_health 实测仅此 2 个 OK）
ACCTS = ['ybzljyma', 'nnjxsn']
PIDS = {'ybzljyma': 'modern-tangent-509301-q8', 'nnjxsn': 'melodic-gamma-509323-u5'}

# ============================ PREREG：判据（判完不改）============================
# 量纲：ndvi_* 为 0-1；amp/ses/std 为 NDVI×1000
PREREG = dict(
    djf_ev_needle=0.45,      # 41 冬季 NDVI 下限
    djf_ev_broad=0.50,       # 43 冬季 NDVI 下限（按任务表 41/43 阈值差分离）
    djf_dec=0.35,            # 42/44 冬季 NDVI 上限
    djf_mix_ev=0.40,         # 45 混交"常绿占优"下限
    amp_ev=250.0,            # 常绿年振幅上限
    amp_dec=400.0,           # 落叶年振幅下限
    ses_early=100.0,         # 42 秋季早衰：aug-oct ≥100（×1000）判为早衰（落叶松）
    mix_hi=0.24,             # 5×5 窗口内 DJF NDVI>0.45 的像元占比下限（≥6/25）
    mix_lo=0.24,             # 5×5 窗口内 DJF NDVI<0.25 的像元占比下限
    mix_gs_std=120.0,        # 46 候选：生长季窗口 NDVI 标准差下限（两冠层结构）
    mix_oct_std=150.0,       # 46 候选：10 月窗口 NDVI 标准差下限（衰败期异质）
    djf_dec_46=0.40,         # 46 冬季 NDVI 上限
    fallback_djf_ev=0.40,    # 兜底分支的常绿判据
    fallback_amp_dec=300.0,  # 兜底分支的落叶判据
    fallback_djf_dec=0.42,
    min_obs=MIN_OBS,
    window_px='5x5 @10m = 50m x 50m',
)

JUDGE_RULE = """
先判混交（窗口），再判纯林（物候阈值），最后兜底。优先级自上而下，命中即停。
 0  n_obs(gs)<3 或 n_obs(djf)<3 ......................... 无法判读:data
 1  mix_ev  = (w_hi>=0.24) and (w_lo>=0.24)              50m 窗口内冬季"仍绿"与"已枯"共存
    mix_dec = (w_gs_std>=120) and (w_oct_std>=150) and (djf<0.40)
 2  mix_ev and djf>=0.40 ................................ 45 常绿针阔混
    mix_dec ............................................. 46 落叶针阔混
    djf>=0.50 and amp<250 ............................... 43 常绿阔叶
    djf>=0.45 and amp<250 ............................... 41 常绿针叶
    djf<0.35 and amp>400 and ses>=100 ................... 42 落叶针叶（秋季早衰）
    djf<0.35 and amp>400 ............................... 44 落叶阔叶
 3  兜底（低置信）:
    amp<400 and djf>=0.40 ....... 43/41（按 djf 0.50 分）
    amp>=300 and djf<0.42 ....... 44
    其余 ........................ 无法判读:boundary
 4  置信:
    高 = 分支命中且两季 n_obs>=6、且未触发降级
    中 = 分支命中但 n_obs 3-5，或距阈值 <10% 边界余量
    低 = 兜底分支，或高海拔(>3500m)/djf 与 djf_snowfree 差异>0.15（积雪污染），或 w_hi+w_lo 极端
"""

RESERVE = {41: ['south_needle_flag', 'high_elev_enf'], 42: ['东北'], 45: [], 46: []}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


# ============================ 抽样 ============================
def thin_spacing(df, spacing, seed, order=None):
    """贪心空间稀释：按给定顺序接受点，要求与已接受点欧氏距离（度）≥ spacing。"""
    if order is None:
        order = np.random.default_rng(seed).permutation(len(df))
    lon = df['lon'].values.astype('float64')
    lat = df['lat'].values.astype('float64')
    cell = max(spacing, 1e-6)
    grid = {}
    keep = []
    for i in order:
        gx = int(np.floor(lon[i] / cell))
        gy = int(np.floor(lat[i] / cell))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy), ()):
                    if (lon[i] - lon[j]) ** 2 + (lat[i] - lat[j]) ** 2 < spacing ** 2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            keep.append(i)
            grid.setdefault((gx, gy), []).append(i)
    return np.array(keep, dtype='int64')


def eco_quota(sub, n_total):
    """按 sqrt(点量) 分配，每生态区下限 ECO_MIN，上限为可用量。"""
    cnt = sub['eco_id'].value_counts()
    if len(cnt) == 0:
        return {}
    w = np.sqrt(cnt.values.astype('float64'))
    raw = n_total * w / w.sum()
    q = {}
    for e, r, c in zip(cnt.index, raw, cnt.values):
        q[e] = int(min(max(ECO_MIN, round(r)), c))
    return q


def eco_targets(counts, n_total, floor=ECO_MIN):
    """生态区配额：先保底 min(floor, avail)，余额按 sqrt(点量) 比例分，逐轮封顶 avail。"""
    tgt = {e: int(min(floor, c)) for e, c in counts.items()}
    remain = n_total - sum(tgt.values())
    for _ in range(200):
        if remain <= 0:
            break
        act = [e for e, c in counts.items() if tgt[e] < c]
        if not act:
            break
        w = np.sqrt(np.array([counts[e] for e in act], dtype='float64'))
        add = np.floor(remain * w / w.sum()).astype(int)
        add = np.maximum(add, 1)
        moved = 0
        for e, a in zip(act, add):
            room = counts[e] - tgt[e]
            d = int(min(a, room, remain))
            if d > 0:
                tgt[e] += d
                remain -= d
                moved += d
        if moved == 0:
            break
    return tgt


def do_sample():
    cols = ['uid', 'lon', 'lat', 'eco', 'prov', 'mixed_tier', 'relabeled', 'also_mixed_window',
            'high_elev_enf', 'south_needle_flag', 'elev_m', 'bio01', 'bio06', 'p_needle']
    df = pd.read_parquet(SRC, columns=cols + ['cls6_train'])
    d = df[df.cls6_train.isin(CLASSES)].copy()
    d['eco_id'] = d['eco'].astype(str)
    rng = np.random.default_rng(SEED)

    picked = []
    design = {}
    for c in CLASSES:
        sub = d[d.cls6_train == c].reset_index(drop=True)
        s3_bad = 0
        if c in (45, 46):                       # 45/46 必须全部来自 S3 档
            s3_bad = int((sub.mixed_tier != 'S3').sum())
            sub = sub[sub.mixed_tier == 'S3'].reset_index(drop=True)
        n = len(sub)

        # ---- 难点子集（每组一档上限 / 一档下限）
        if c == 41:
            hard_spec = [(np.where((sub.south_needle_flag == 1).values)[0], 100, 180),
                         (np.where((sub.high_elev_enf == 1).values)[0], 100, 180)]
        elif c == 42:
            hard_spec = [(np.where((sub.eco_id == '东北').values)[0], 100, 220)]
        else:
            hard_spec = []
        hard_spec = [(idx, lo, hi) for idx, lo, hi in hard_spec if len(idx)]

        counts = sub['eco_id'].value_counts().to_dict()
        tgt = eco_targets(counts, N_PER_CLASS)          # 生态区：floor 60 + sqrt 比例

        lon = sub['lon'].values; lat = sub['lat'].values
        eco_arr = sub.eco_id.values
        grid, g2 = {}, {}
        cnt = {e: 0 for e in counts}
        keep, used = [], np.zeros(n, dtype=bool)
        SP2 = 0.15

        def _try(i, sp, g):
            gx = int(np.floor(lon[i] / sp)); gy = int(np.floor(lat[i] / sp))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in g.get((gx + dx, gy + dy), ()):
                        if (lon[i] - lon[j]) ** 2 + (lat[i] - lat[j]) ** 2 < sp ** 2:
                            return False
            g.setdefault((gx, gy), []).append(i)
            return True

        def add(i, sp_g=None):
            ok = _try(i, SPACING, grid)
            if ok:
                if sp_g is not None:
                    sp_g.setdefault((int(np.floor(lon[i] / SP2)),
                                     int(np.floor(lat[i] / SP2))), []).append(i)
                keep.append(i); used[i] = True
                cnt[eco_arr[i]] = cnt.get(eco_arr[i], 0) + 1
            return ok

        orders = []
        for idx, lo, hi in hard_spec:
            o = idx.copy(); rng.shuffle(o); orders.append(o[:hi])

        # 1) 难点子集轮转取点
        ptr = 0; alive = True
        while alive and len(keep) < N_PER_CLASS:
            alive = False
            for o in orders:
                if ptr < len(o):
                    alive = True
                    i = int(o[ptr])
                    if not used[i]:
                        add(i)
            ptr += 1
        # 1b) 硬下限保底：不足 lo 的组，从**全量候选**继续补（0.15° 间隔，不受生态区配额）
        extra = {}
        for gi, (idx, lo, hi) in enumerate(hard_spec):
            if int(used[idx].sum()) >= lo:
                continue
            pool = idx[~used[idx]].copy()
            rng.shuffle(pool)
            for i in pool:
                if int(used[idx].sum()) >= lo or len(keep) >= N_PER_CLASS:
                    break
                i = int(i)
                if used[i]:
                    continue
                add(i, g2)
            extra[str(gi)] = int(used[idx].sum())

        # 2) 常规层：生态区轮转（rank/sqrt(count)）受 target 封顶
        rank = np.zeros(n, dtype='float64')
        for e in counts:
            idx = np.where(eco_arr == e)[0]
            idx = idx[~used[idx]]
            if not len(idx):
                continue
            rng.shuffle(idx)
            rank[idx] = np.arange(len(idx), dtype='float64')
        norm = np.sqrt(np.array([counts[e] for e in eco_arr], dtype='float64'))
        score = rank / np.maximum(norm, 1.0) + rng.random(n) * 0.999
        for i in np.argsort(score, kind='stable'):
            if len(keep) >= N_PER_CLASS:
                break
            if used[i] or cnt.get(eco_arr[i], 0) >= tgt.get(eco_arr[i], 0):
                continue
            add(i)

        # 3) 补足：先硬子集剩余，再按最缺生态区
        relaxed = False
        if len(keep) < N_PER_CLASS:
            relaxed = True
            for gi, o in enumerate(orders):
                for i in o:
                    if len(keep) >= N_PER_CLASS:
                        break
                    i = int(i)
                    if not used[i]:
                        add(i, g2)
            rest = np.where(~used)[0]
            rest = rest[np.argsort(rank[rest] / np.maximum(norm[rest], 1.0))]
            for i in rest:
                if len(keep) >= N_PER_CLASS:
                    break
                if not used[i]:
                    add(i, g2)

        cur = sub.iloc[np.array(keep, dtype='int64')].drop_duplicates('uid').reset_index(drop=True)
        cur['cls_label'] = c
        cur['sample_relaxed'] = np.int8(1 if relaxed else 0)
        picked.append(cur)

        ec = cur['eco_id'].value_counts().to_dict()
        design[str(c)] = dict(
            available=int(n), selected=int(len(cur)), relaxed=bool(relaxed),
            s3_violation=int(s3_bad), target=tgt,
            by_eco={k: int(v) for k, v in ec.items()},
            available_by_eco={k: int(v) for k, v in counts.items()},
            eco_below_min=[k for k, v in ec.items() if v < min(ECO_MIN, counts.get(k, 0))],
            hard=dict(
                south_needle=int((cur.south_needle_flag == 1).sum()),
                high_elev_enf=int((cur.high_elev_enf == 1).sum()),
                northeast=int((cur.eco_id == '东北').sum()),
                orig_45_46=int(cur['cls6_train'].isin([45, 46]).sum()),
                relabeled=int(cur['relabeled'].sum()),
                also_mixed_window=int(cur['also_mixed_window'].sum())),
            elev_med=round(float(cur['elev_m'].median()), 0))
        emit('cls %d 可用 %6d → 抽 %3d（放宽=%s）｜生态 %s'
             % (c, n, len(cur), relaxed, {k: int(v) for k, v in ec.items()}))

    out = pd.concat(picked, ignore_index=True)
    # 盲判清单：绝不带 cls6_train
    keep = ['uid', 'lon', 'lat', 'cls_label', 'eco_id', 'prov', 'sample_relaxed',
            'high_elev_enf', 'south_needle_flag', 'elev_m']
    s = out[keep].copy()
    s['cls_label'] = s.pop('cls_label').astype('int16')   # 抽样层身份（不是待判标签）
    s['layer'] = s['cls_label']                            # 层名，判读时不用
    s.to_parquet(SAMPLE, index=False)
    json.dump(dict(seed=SEED, n_per_class=N_PER_CLASS, spacing_deg=SPACING, eco_min=ECO_MIN,
                   design=design, n_total=int(len(s)),
                   cells=int(s.assign(c=(np.floor(s.lon / CELL_DEG).astype(int).astype(str) + '_' +
                                         np.floor(s.lat / CELL_DEG).astype(int).astype(str))).c.nunique())),
              open(os.path.join(WORK, 'f4_sample_design.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    emit('写出 %s：%d 点' % (SAMPLE, len(s)))


# ============================ GEE 取数 ============================
def init_ee(acct):
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
        os.environ.pop(k, None)
    import socket
    port = 7890
    for p in (7890, 7897, 7891):
        s = socket.socket()
        s.settimeout(0.6)
        try:
            if s.connect_ex(('127.0.0.1', p)) == 0:
                port = p
                break
        finally:
            s.close()
    url = 'socks5h://127.0.0.1:%d' % port
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        os.environ[k] = url
    os.environ['NO_PROXY'] = ''
    sys.path.insert(0, os.path.join(PROJ, '代码', '8.全国生产'))
    import prod_conf as C
    C.apply_account_env(acct)
    import ee
    pid = PIDS.get(acct) or C.ACCOUNTS[acct]['proj']
    ee.Initialize(project=pid)
    try:
        ee.data.setDeadline(420000)
    except Exception:
        pass
    emit('GEE 就绪 acct=%s proj=%s proxy=%s' % (acct, pid, url))
    return ee


def s2_col(ee, roi, cloud=CLOUDY_PCT):
    return (ee.ImageCollection(S2).filterBounds(roi)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud)))


def ndvi_of(img):
    """w1 口径：median(B8,B4) → NDVI，×1000，保持自然掩膜（不 unmask）"""
    med = img.select(['B8', 'B4']).median()
    return med.normalizedDifference(['B8', 'B4']).multiply(1000.0).toFloat()


def ndmi_of(img):
    """NDMI=(B8-B11)/(B8+B11) —— 针叶 SWIR 更高 → NDMI 更低（针/阔旁证轴）"""
    med = img.select(['B8', 'B11']).median()
    return med.normalizedDifference(['B8', 'B11']).toFloat()


def window_dates(kind):
    if kind == 'djf':
        return [('%d-12-01' % (y - 1), '%d-03-01' % y) for y in YEARS]
    if kind == 'gs':
        return [('%d-06-01' % y, '%d-09-01' % y) for y in YEARS]
    if kind == 'aug':
        return [('%d-08-01' % y, '%d-09-01' % y) for y in YEARS]
    if kind == 'oct':
        return [('%d-10-01' % y, '%d-11-01' % y) for y in YEARS]
    raise ValueError(kind)


def build_image(ee, roi):
    col = s2_col(ee, roi).map(
        lambda im: im.updateMask(im.select('SCL').neq(3)
                                 .And(im.select('SCL').neq(8))
                                 .And(im.select('SCL').neq(9))))
    col_ns = s2_col(ee, roi).map(
        lambda im: im.updateMask(im.select('SCL').neq(3)
                                 .And(im.select('SCL').neq(8))
                                 .And(im.select('SCL').neq(9))
                                 .And(im.select('SCL').neq(11))))   # 额外剔雪

    def comp(kind, source, tag):
        ws = window_dates(kind)
        c = None
        for a, b in ws:
            p = source.filterDate(a, b)
            c = p if c is None else c.merge(p)
        nd = ndvi_of(c).rename(tag)
        n = c.select('B8').count().rename(tag + 'n').toFloat()
        return nd, n

    gs, gsn = comp('gs', col, 'gs')
    djf, djfn = comp('djf', col, 'djf')
    djfns, djfnsn = comp('djf', col_ns, 'djfns')
    aug, augn = comp('aug', col, 'aug')
    oct, octn = comp('oct', col, 'oct')
    ndmi, _ = comp('gs', col, 'ndmi_gs')
    # NDMI 单独重建（避免 count 冗余）
    cgs = None
    for a, b in window_dates('gs'):
        p = col.filterDate(a, b)
        cgs = p if cgs is None else cgs.merge(p)
    ndmi = ndmi_of(cgs).rename('ndmi_gs')

    # ---- 50 m 窗口（5×5 @10m）统计
    # 占比一律按**窗口内有效像元数**归一化（不是 /25），否则云/海岸会系统性压低
    k5 = ee.Kernel.square(radius=2, units='pixels', normalize=False)

    def frac(img, thr, mode, tag):
        m = img.mask()
        valid = m.unmask(0)
        hit = (img.gt(thr) if mode == 'gt' else img.lt(thr)).unmask(0)
        num = hit.reduceNeighborhood(ee.Reducer.mean(), k5)
        den = valid.reduceNeighborhood(ee.Reducer.mean(), k5)
        return num.divide(den.max(0.04)).clamp(0, 1).rename(tag)

    def nstat(img, red, tag):
        return (img.reduceNeighborhood(red, k5).rename(tag))

    w_hi = frac(djf, 450, 'gt', 'w_hi')          # 冬季 5×5 内"仍绿"占比（常绿冠层）
    w_lo = frac(djf, 250, 'lt', 'w_lo')          # 冬季 5×5 内"已枯"占比（落叶冠层）
    w_gs_std = nstat(gs, ee.Reducer.stdDev(), 'w_gs_std')
    w_oct_std = nstat(oct, ee.Reducer.stdDev(), 'w_oct_std')
    w_djf_std = nstat(djf, ee.Reducer.stdDev(), 'w_djf_std')
    w_djf_mean = nstat(djf, ee.Reducer.mean(), 'w_djf_mean')
    w_gs_mean = nstat(gs, ee.Reducer.mean(), 'w_gs_mean')
    w_nvalid = nstat(djf.mask().unmask(0), ee.Reducer.sum(), 'w_nvalid')

    bands = [gs, gsn, djf, djfn, djfns, djfnsn, aug, augn, oct, octn, ndmi,
             w_hi, w_lo, w_djf_std, w_gs_std, w_oct_std, w_djf_mean, w_gs_mean, w_nvalid]

    # ---- S1 VH（DJF，dB）
    s1 = (ee.ImageCollection(S1).filterBounds(roi)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .select('VH'))
    c1 = None
    for a, b in window_dates('djf'):
        p = s1.filterDate(a, b)
        c1 = p if c1 is None else c1.merge(p)
    vh = c1.median().rename('vh_djf')
    vhn = c1.count().rename('vh_djf_n').toFloat()
    bands += [vh, vhn]

    img = ee.Image.cat(bands)
    return img.unmask(SENT)


BANDS = ['gs', 'gsn', 'djf', 'djfn', 'djfns', 'djfnsn', 'aug', 'augn', 'oct', 'octn', 'ndmi_gs',
         'w_hi', 'w_lo', 'w_djf_std', 'w_gs_std', 'w_oct_std', 'w_djf_mean', 'w_gs_mean',
         'w_nvalid', 'vh_djf', 'vh_djf_n']


def reduce_cell(ee, img, sub, tag):
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {'uid': str(r.uid)})
             for r in sub.itertuples()]
    fc = ee.FeatureCollection(feats)
    last = ''
    for att in range(4):
        try:
            t0 = time.time()
            r = img.sampleRegions(collection=fc, scale=SCALE, geometries=False,
                                  tileScale=4).getInfo()['features']
            emit('   %s n=%d ok %.1fs' % (tag, len(r), time.time() - t0))
            return {str(f['properties']['uid']): f['properties'] for f in r}
        except Exception as e:
            last = str(e)[:180]
            emit('   %s retry%d %s' % (tag, att + 1, last))
            time.sleep(5 * (att + 1))
    emit('   %s FAILED %s' % (tag, last))
    return None


def run_worker(acct, shard_i, shard_n):
    pts = pd.read_parquet(SAMPLE, columns=['uid', 'lon', 'lat'])
    pts['cell'] = (np.floor(pts.lon / CELL_DEG).astype(int).astype(str) + '_'
                   + np.floor(pts.lat / CELL_DEG).astype(int).astype(str))
    cells = sorted(pts.cell.unique())
    mine = [c for i, c in enumerate(cells) if i % shard_n == shard_i]
    outd = os.path.join(PARTS, acct)
    os.makedirs(outd, exist_ok=True)
    emit('[%s] 点 %d ｜ 格 %d 总 / 本片 %d' % (acct, len(pts), len(cells), len(mine)))
    todo = [c for c in mine if not os.path.exists(os.path.join(outd, 'cell_%s.parquet' % c))]
    if not todo:
        emit('[%s] 断点续跑：全部已完成' % acct)
        return
    ee = init_ee(acct)
    t00 = time.time()
    for ci, c in enumerate(todo, 1):
        sub = pts[pts.cell == c]
        lon0, lat0 = [int(x) for x in c.split('_')]
        roi = ee.Geometry.Rectangle([lon0 - 0.02, lat0 - 0.02,
                                     lon0 + CELL_DEG + 0.02, lat0 + CELL_DEG + 0.02])
        img = build_image(ee, roi)
        got = reduce_cell(ee, img, sub, '%s %s' % (acct, c))
        if got is None:
            continue
        rows = []
        for r in sub.itertuples():
            p = got.get(str(r.uid))
            rec = {'uid': str(r.uid), 'lon': float(r.lon), 'lat': float(r.lat), 'cell': c}
            for b in BANDS:
                v = None if p is None else p.get(b)
                rec[b] = np.nan if v is None else float(v)
            rows.append(rec)
        o = pd.DataFrame(rows)
        for b in BANDS:
            o.loc[o[b] <= -9000, b] = np.nan
        o.to_parquet(os.path.join(outd, 'cell_%s.parquet' % c), index=False)
        if ci % 10 == 0:
            el = time.time() - t00
            emit('[%s] %d/%d 格  累计 %.0f min  预计总 %.0f min'
                 % (acct, ci, len(todo), el / 60, el / 60 * len(todo) / ci))
    emit('[%s] 完成，耗时 %.1f min' % (acct, (time.time() - t00) / 60))


# ============================ 判读 ============================
def load_parts():
    fs = glob.glob(os.path.join(PARTS, '*', 'cell_*.parquet'))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df['uid'] = df['uid'].astype('int64')
    return df.drop_duplicates('uid').reset_index(drop=True)


def derive(df):
    o = df.copy()
    P = PREREG
    o['ndvi_gs'] = o['gs'] / 1000.0
    o['ndvi_djf'] = o['djf'] / 1000.0
    o['ndvi_djf_ns'] = o['djfns'] / 1000.0
    o['ndvi_aug'] = o['aug'] / 1000.0
    o['ndvi_oct'] = o['oct'] / 1000.0
    o['amp'] = (o['gs'] - o['djf']).abs()
    o['ses'] = o['aug'] - o['oct']                      # ×1000，正=8月比10月绿
    o['vh_djf'] = o['vh_djf']                            # dB，旁证
    o['w_mix'] = np.minimum(o['w_hi'], o['w_lo'])
    o['w_het'] = np.maximum(o['w_gs_std'], o['w_oct_std'])
    return o


def judge_one(r):
    """盲判：只用指标，不看标签。返回 (结论, 置信, 证据)"""
    if not isinstance(r, dict):
        r = r._asdict()
    P = PREREG
    gs, djf, amp = r['ndvi_gs'], r['ndvi_djf'], r['amp']
    ses = r['ses']
    w_hi, w_lo = r['w_hi'], r['w_lo']
    w_gs_std, w_oct_std = r['w_gs_std'], r['w_oct_std']
    nobs = min(r['gsn'] if np.isfinite(r['gsn']) else 0,
               r['djfn'] if np.isfinite(r['djfn']) else 0)

    if not np.isfinite(gs) or not np.isfinite(djf):
        return 'nan', 'nan', 'NDVI 缺测'
    if nobs < P['min_obs']:
        return 'nan', 'nan', 'n_obs<3 (gs=%.0f djf=%.0f)' % (r['gsn'] if np.isfinite(r['gsn']) else -1,
                                                            r['djfn'] if np.isfinite(r['djfn']) else -1)

    # 混交（窗口）
    mix_ev = (w_hi >= P['mix_hi']) and (w_lo >= P['mix_lo'])
    mix_dec = (w_gs_std >= P['mix_gs_std']) and (w_oct_std >= P['mix_oct_std']) and (djf < P['djf_dec_46'])

    cls, conf = None, None
    ev = []
    ev.append('GS%.2f/DJF%.2f/amp%.0f/aug%.2f/oct%.2f' % (gs, djf, amp, r['ndvi_aug'], r['ndvi_oct']))
    ev.append('win hi%.2f lo%.2f gsσ%.0f octσ%.0f' % (w_hi, w_lo, w_gs_std, w_oct_std))

    if mix_ev and djf >= P['djf_mix_ev']:
        cls = 45
        conf = 'high' if (w_hi >= 0.32 and w_lo >= 0.32) else 'mid'
        ev.append('50m窗口冬季同时见绿(%.0f%%)与枯(%.0f%%)→常绿成分占优' % (w_hi * 100, w_lo * 100))
    elif mix_dec:
        cls = 46
        conf = 'mid' if (w_gs_std >= 150 and w_oct_std >= 180) else 'low'
        ev.append('冬季全枯且窗口生长季/衰败期高异质(gsσ%.0f octσ%.0f)→两冠层落叶混交' % (w_gs_std, w_oct_std))
    elif djf >= P['djf_ev_broad'] and amp < P['amp_ev']:
        cls = 43
        conf = 'high' if (djf >= 0.55 and amp < 200 and nobs >= 6) else 'mid'
        ev.append('冬季仍绿 DJF%.2f>0.50 且振幅小 %.0f<250 → 常绿（阔叶档）' % (djf, amp))
    elif djf >= P['djf_ev_needle'] and amp < P['amp_ev']:
        cls = 41
        conf = 'high' if (nobs >= 6 and amp < 200) else 'mid'
        ev.append('冬季仍绿 DJF%.2f>0.45 且振幅小 %.0f<250 → 常绿（针叶档）' % (djf, amp))
    elif djf < P['djf_dec'] and amp > P['amp_dec'] and ses >= P['ses_early']:
        cls = 42
        conf = 'high' if ses >= 150 else 'mid'
        ev.append('冬季枯 DJF%.2f<0.35 振幅大 %.0f>400 且秋季早衰 ses=%.0f≥100 → 落叶针叶' % (djf, amp, ses))
    elif djf < P['djf_dec'] and amp > P['amp_dec']:
        cls = 44
        conf = 'high' if (djf < 0.28 and amp > 500) else 'mid'
        ev.append('冬季枯 DJF%.2f<0.35 振幅大 %.0f>400 无早衰(ses=%.0f) → 落叶阔叶' % (djf, amp, ses))
    elif amp < 400 and djf >= P['fallback_djf_ev']:
        cls = 43 if djf >= P['djf_ev_broad'] else 41
        conf = 'low'
        ev.append('兜底：常绿倾向 amp%.0f<400 DJF%.2f' % (amp, djf))
    elif amp >= P['fallback_amp_dec'] and djf < P['fallback_djf_dec']:
        cls = 44
        conf = 'low'
        ev.append('兜底：落叶倾向 amp%.0f≥300 DJF%.2f<0.42' % (amp, djf))
    else:
        return 'nan', 'nan', '边界带 amp%.0f DJF%.2f GS%.2f' % (amp, djf, gs)

    # 降级：积雪污染 / 高海拔 / 极端窗口
    notes = []
    if np.isfinite(r['ndvi_djf_ns']) and abs(r['ndvi_djf_ns'] - djf) > 0.15:
        conf = 'low'
        notes.append('剔雪DJF%.2f 与含雪差>0.15（积雪污染）' % r['ndvi_djf_ns'])
    if np.isfinite(r['w_hi']) and (w_hi + w_lo > 0.95) and cls in (41, 43, 42, 44):
        conf = 'low'
        notes.append('窗口两极占比过高(%.2f)' % (w_hi + w_lo))
    ev.append('; '.join(notes) if notes else '')
    return str(cls), conf, ' | '.join([x for x in ev if x])


def do_judge():
    got = load_parts()
    smp = pd.read_parquet(SAMPLE)
    if not len(got):
        raise SystemExit('无 GEE 产出')
    d = derive(got)
    m = smp.merge(d, on='uid', how='left', suffixes=('', '_g'))
    emit('抽样 %d ｜ 取数成功 %d ｜ 缺 %d'
         % (len(m), int(m['gs'].notna().sum()), int(m['gs'].isna().sum())))
    res = [judge_one(r) for r in m.to_dict('records')]
    m['结论'] = [x[0] for x in res]
    m['置信'] = [x[1] for x in res]
    m['证据'] = [x[2] for x in res]
    blind = m[['uid', 'lon', 'lat', 'ndvi_gs', 'ndvi_djf', 'amp', 'ndvi_oct', 'ndvi_aug', 'vh_djf',
               'w_djf_std', 'w_mix', '结论', '置信', '证据']].copy()
    blind = blind.rename(columns={'w_djf_std': '窗口NDVI方差_std_x1000', 'w_mix': '窗口NDVI双峰度'})
    blind.to_csv(BLIND, index=False, encoding='utf-8-sig', float_format='%.3f')
    emit('写出盲判表 %s（%d 行）' % (BLIND, len(blind)))
    emit('结论分布 %s' % blind['结论'].value_counts().to_dict())
    emit('置信分布 %s' % blind['置信'].value_counts().to_dict())
    m.to_parquet(os.path.join(WORK, '_f4_judged.parquet'), index=False)


# ============================ 报告 ============================
def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 4), round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def rate_table(sub, ycol='correct'):
    r = {}
    for c, g in sub.groupby('cls6_train'):
        v = g[g[ycol].notna()]
        n = int(len(v)); k = int(v[ycol].sum())
        r[str(c)] = dict(n=n, correct=k, rate=round(k / n, 4) if n else None,
                         ci=list(wilson(k, n)[1:]) if n else None)
    return r


def do_report():
    m = pd.read_parquet(os.path.join(WORK, '_f4_judged.parquet'))
    src = pd.read_parquet(SRC, columns=['uid', 'cls6_train', 'cls6_orig', 'mixed_tier', 'relabeled',
                                        'also_mixed_window', 'high_elev_enf', 'south_needle_flag',
                                        'eco', 'prov', 'amp2023', 'djf2023', 'p_needle', 'elev_m'])
    m = m.drop(columns=[c for c in ('cls_label', 'eco', 'prov', 'elev_m') if c in m.columns])
    m = m.merge(src, on='uid', how='left', suffixes=('', '_src'))
    m['pred'] = pd.to_numeric(m['结论'], errors='coerce')
    m['correct'] = np.where(m['pred'].notna(), (m['pred'] == m['cls6_train']).astype(float), np.nan)
    # 轴级（常绿 41/43 | 落叶 42/44 | 混交 45/46）
    AX = {41: 'EV', 42: 'DEC', 43: 'EV', 44: 'DEC', 45: 'MIX', 46: 'MIX'}
    m['axis_true'] = m['cls6_train'].map(AX)
    m['axis_pred'] = m['pred'].map(AX)
    m['correct_axis'] = np.where(m['pred'].notna(), (m['axis_pred'] == m['axis_true']).astype(float), np.nan)
    m.to_csv(RESULT, index=False, encoding='utf-8-sig', float_format='%.3f')

    S = dict(generated=pd.Timestamp.now().isoformat(timespec='seconds'),
             role='L3 独立影像验证（doc84 §4.3 验收口径）',
             source=dict(s2=S2, s1=S1, years='%d-%d' % (YEARS[0], YEARS[-1]),
                         scl_mask='SCL not in {3,8,9}（w1 生产口径）；djf_ns 额外剔 11(雪)',
                         scale_m=SCALE, n_per_class=N_PER_CLASS, seed=SEED,
                         spacing_deg=SPACING, eco_min=ECO_MIN),
             prereg=PREREG, judge_rule=JUDGE_RULE,
             blind_protocol='判定函数只接收 S2/S1 指标，不接收 cls6_train；f4_blind.csv 无标签列；'
                            '标签仅在 report 阶段由 uid 左连 forest6_v1.parquet 加入')

    n_all = int(len(m))
    judged = m[m['pred'].notna()]
    S['three_state'] = dict(
        n_sample=n_all, n_judged=int(len(judged)), n_unjudged=int(n_all - len(judged)),
        share_unjudged=round((n_all - len(judged)) / n_all, 4),
        unjudged_reason={str(k): int(v) for k, v in m.loc[m['pred'].isna(), '证据'].str.split('|')
                         .str[-1].str.strip().str.slice(0, 24).value_counts().items()},
        by_class={str(c): dict(n=int((m.cls6_train == c).sum()),
                               judged=int(((m.cls6_train == c) & m['pred'].notna()).sum()),
                               unjudged=int(((m.cls6_train == c) & m['pred'].isna()).sum()),
                               nan_pred=int(((m.cls6_train == c) & (m['结论'] == 'nan')).sum()))
                  for c in CLASSES},
        conf_dist={str(k): int(v) for k, v in m['置信'].value_counts().items()})

    # ---- 主指标：总体一致率（6 类精确）
    k = int(m['correct'].sum()); n = int(m['correct'].notna().sum())
    p, lo, hi = wilson(k, n)
    S['overall_6class'] = dict(n=n, correct=k, rate=p, ci95=[lo, hi],
                              note='**验收口径**（doc84 §4.3 L3 ≥0.85）；判读=预注册物候规则，非人工目视')
    # 加权（按类点数加权 = 等价于总体，另给按可用类等权）
    per = rate_table(m)
    S['per_class_6class'] = per
    S['macro_6class'] = round(float(np.mean([v['rate'] for v in per.values() if v['rate'] is not None])), 4)
    S['weighted_by_n'] = p

    # ---- 轴级（常绿/落叶/混交 三态）
    ka = int(m['correct_axis'].sum()); na = int(m['correct_axis'].notna().sum())
    S['overall_axis3'] = dict(n=na, correct=ka, rate=round(ka / na, 4) if na else None,
                              ci95=list(wilson(ka, na)[1:]),
                              note='41/43 合为"常绿"、42/44 合为"落叶"、45/46 合为"混交"')

    # ---- 低置信处理
    hm = m[m['置信'].isin(['high', 'mid']) & m['pred'].notna()]
    kh, nh = int(hm['correct'].sum()), len(hm)
    S['high_mid_conf_only'] = dict(n=nh, correct=kh, rate=round(kh / nh, 4) if nh else None,
                                   ci95=list(wilson(kh, nh)[1:]),
                                   coverage=round(nh / n_all, 4))
    hi_ = m[(m['置信'] == 'high') & m['pred'].notna()]
    S['high_conf_only'] = dict(n=len(hi_), correct=int(hi_['correct'].sum()),
                               rate=round(float(hi_['correct'].mean()), 4) if len(hi_) else None,
                               ci95=list(wilson(int(hi_['correct'].sum()), len(hi_))[1:]),
                               coverage=round(len(hi_) / n_all, 4))

    # ---- 逐类混淆矩阵
    cm = pd.crosstab(m['cls6_train'], m['pred'].fillna(-1).astype(int), margins=True)
    S['confusion_6class'] = {str(i): {str(j): int(cm.loc[i, j]) for j in cm.columns}
                             for i in cm.index}
    # 判成"nan"单独列
    S['pred_distribution'] = {str(k2): int(v) for k2, v in m['pred'].fillna(-1).astype(int)
                              .value_counts().sort_index().items()}

    # ---- 分层
    def strat(col):
        out = {}
        for g, s in m.groupby(col):
            v = s[s['correct'].notna()]
            if not len(v):
                out[str(g)] = dict(n=0)
                continue
            kk = int(v['correct'].sum())
            out[str(g)] = dict(n=int(len(v)), correct=kk, rate=round(kk / len(v), 4),
                               ci95=list(wilson(kk, len(v))[1:]))
        return out
    S['by_eco'] = strat('eco')
    S['by_prov'] = strat('prov')
    # 逐类 × 生态区
    S['by_class_eco'] = {str(c): strat_g for c, strat_g in
                         ((c, {str(e): dict(n=int(len(s)), rate=round(float(s['correct'].mean()), 4))
                               for e, s in g.groupby('eco') if s['correct'].notna().any()})
                          for c, g in m.groupby('cls6_train'))}

    # ---- 难点子集
    def subset(mask, name):
        s = m[mask & m['correct'].notna()]
        kk = int(s['correct'].sum())
        ka2 = int(s['correct_axis'].fillna(0).sum())
        return dict(name=name, n=int(len(s)), correct=kk,
                    rate=round(kk / len(s), 4) if len(s) else None,
                    ci95=list(wilson(kk, len(s))[1:]) if len(s) else None,
                    axis3_rate=round(ka2 / len(s), 4) if len(s) else None,
                    pred_dist={str(a): int(b) for a, b in s['pred'].value_counts().items()})
    S['hard_subsets'] = dict(
        south_needle=subset((m.cls6_train == 41) & (m.south_needle_flag == 1), '41 南方针叶'),
        high_elev_enf=subset((m.cls6_train == 41) & (m.high_elev_enf == 1), '41 高海拔ENF(青藏)'),
        northeast_42=subset((m.cls6_train == 42) & (m.eco == '东北'), '42 东北兴安落叶松'),
        cls45=subset(m.cls6_train == 45, '45 常绿针阔混'),
        cls46=subset(m.cls6_train == 46, '46 落叶针阔混'),
        cls45_46=subset(m.cls6_train.isin([45, 46]), '45+46 合计'),
        pure_41_44=subset(m.cls6_train.isin([41, 42, 43, 44]), '41-44 纯林'),
        s3_only=subset(m.mixed_tier == 'S3', 'S3 档（全部 45/46）'),
        relabeled=subset(m.relabeled == 1, 'relabeled（纯林被改判为混交）'))

    # ---- 阈值敏感性
    sens = {}

    def variant(name, override, note=''):
        g = dict(PREREG)                       # 必须深拷贝快照（PREREG 是模块级 dict，别名会互相清空）
        PREREG.update(override)
        try:
            pr = [judge_one(r)[0] for r in m.to_dict('records')]
        finally:
            PREREG.clear(); PREREG.update(g)
        pr = pd.to_numeric(pd.Series(pr), errors='coerce')
        ok = pr.notna()
        acc = float((pr[ok] == m['cls6_train'][ok]).mean()) if ok.any() else None
        acc_axis = float((pr[ok].map(AX) == m['axis_true'][ok]).mean()) if ok.any() else None
        rate45 = float((pr[m.cls6_train == 45] == 45).mean())
        rate46 = float((pr[m.cls6_train == 46] == 46).mean())
        sens[name] = dict(change=override or 'baseline', note=note,
                          coverage=round(float(ok.mean()), 4),
                          overall6=round(acc, 4) if acc is not None else None,
                          axis3=round(acc_axis, 4) if acc_axis is not None else None,
                          rate_45=round(rate45, 4), rate_46=round(rate46, 4),
                          flipped=int((pr.fillna(-1) != m['pred'].fillna(-1)).sum()))
    variant('baseline', {})
    variant('amp_ev_300', dict(amp_ev=300.0), '常绿振幅上限 250→300')
    variant('amp_dec_350', dict(amp_dec=350.0), '落叶振幅下限 400→350')
    variant('djf_dec_040', dict(djf_dec=0.40), '落叶冬季 NDVI 上限 0.35→0.40')
    variant('no_mix_window', dict(mix_hi=1.01, mix_lo=1.01, mix_gs_std=9e9),
            '去掉窗口混交判据（45/46 不再可能被命中）')
    variant('no_senescence', dict(ses_early=-9e9), '去掉秋季早衰条件（42/44 不再可分）')
    variant('ses_150', dict(ses_early=150.0), '早衰阈值 100→150')
    variant('mix_lo_015', dict(mix_lo=0.15, mix_hi=0.15), '窗口双峰门槛 0.24→0.15（更易判混交）')
    variant('mix_lo_035', dict(mix_lo=0.35, mix_hi=0.35), '窗口双峰门槛 0.24→0.35（更难判混交）')
    S['threshold_sensitivity'] = sens
    S['_no_vh_in_rule'] = '主判据不使用 VH（doc82 §4：北方冬季 VH 崩到 −25.6 dB）；VH 仅随盲判表提交作旁证'

    # ---- 判据效度自检：常绿 vs 落叶 能否分开（ROC/AUC，用 amp 与 djf）
    def auc(score, y):
        d = pd.DataFrame({'s': score, 'y': y}).dropna()
        if d['y'].nunique() < 2:
            return None, 0
        s = d['s'].values; yy = d['y'].values.astype(int)
        order = np.argsort(s)
        r = np.empty(len(s), dtype='float64')
        r[order] = np.arange(1, len(s) + 1)
        # 处理并列
        d2 = pd.DataFrame({'s': s, 'r': r})
        r = d2.groupby('s')['r'].transform('mean').values
        n1 = yy.sum(); n0 = len(yy) - n1
        return round((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), 4), int(len(d))

    pur = m[m.cls6_train.isin([41, 42, 43, 44]) & m['pred'].notna()].copy()
    ever = pur['cls6_train'].isin([41, 43]).astype(int)
    a_djf, n_djf = auc(pur['ndvi_djf'], ever)
    a_amp, n_amp = auc(-pur['amp'], ever)
    a_ndmi, n_ndmi = auc(pur['ndmi_gs'], ever)
    # 针叶 vs 阔叶（常绿组内 / 落叶组内）
    ev = pur[pur.cls6_train.isin([41, 43])]
    de = pur[pur.cls6_train.isin([42, 44])]
    needle_all = pur['cls6_train'].isin([41, 42]).astype(int)
    a_ndmi_needle, n_ndmi_needle = auc(-pur['ndmi_gs'], needle_all)
    a_ndmi_needle_ev, n_ev_nd = auc(-ev['ndmi_gs'], ev['cls6_train'].eq(41).astype(int))
    a_ndmi_needle_de, n_de_nd = auc(-de['ndmi_gs'], de['cls6_train'].eq(42).astype(int))
    S['criterion_validity'] = dict(
        question='预注册判据能否真正分开 常绿/落叶，以及 针叶/阔叶？',
        evergreen_vs_deciduous=dict(
            by_winter_ndvi=dict(auc=a_djf, n=n_djf, expect='>0.90 才算有效'),
            by_amplitude=dict(auc=a_amp, n=n_amp),
            by_ndmi_swir=dict(auc=a_ndmi, n=n_ndmi)),
        needle_vs_broad=dict(
            all_4_pure=dict(via_ndmi_auc=a_ndmi_needle, n=n_ndmi_needle),
            within_evergreen=dict(via_ndmi_auc=a_ndmi_needle_ev, n=n_ev_nd),
            within_deciduous=dict(via_ndmi_auc=a_ndmi_needle_de, n=n_de_nd)),
        median_by_class={str(c): dict(
            djf=round(float(g['ndvi_djf'].median()), 3), gs=round(float(g['ndvi_gs'].median()), 3),
            amp=round(float(g['amp'].median()), 0), ses=round(float(g['ses'].median()), 0),
            ndmi=round(float(g['ndmi_gs'].median()), 3),
            w_hi=round(float(g['w_hi'].median()), 2), w_lo=round(float(g['w_lo'].median()), 2),
            w_gs_std=round(float(g['w_gs_std'].median()), 0),
            w_oct_std=round(float(g['w_oct_std'].median()), 0),
            n=int(len(g))) for c, g in m[m['pred'].notna()].groupby('cls6_train')},
        verdict=None)
    v = S['criterion_validity']
    ok_ev = (a_djf is not None and a_djf >= 0.90) and (a_amp is not None and a_amp >= 0.85)
    ok_nd = all(x is not None and x >= 0.85 for x in (a_ndmi_needle, a_ndmi_needle_ev, a_ndmi_needle_de))
    v['verdict'] = ('常绿/落叶：可分（%.3f/%.3f）；针/阔：%s' %
                    (a_djf or -1, a_amp or -1, '可分' if ok_nd else '不可分（与 doc54 E15 AUC 0.724 一致）'))
    v['evergreen_axis_valid'] = bool(ok_ev)
    v['needle_axis_valid'] = bool(ok_nd)
    v['implication'] = ('若 needle_axis_valid=False，则 41↔43、42↔44 的同轴互错不是判读错误而是'
                        '判据在针/阔轴上无分辨力 → 六类精确一致率的天花板受此限制')

    # ---- 与生产口径 amp2023/djf2023 的一致性（口径校验）
    cc = m[['amp', 'amp2023', 'ndvi_djf', 'djf2023']].dropna()
    if len(cc):
        S['calibration_vs_production'] = dict(
            n=int(len(cc)),
            amp=dict(median_diff=round(float((cc['amp'] - cc['amp2023']).median()), 2),
                     mae=round(float((cc['amp'] - cc['amp2023']).abs().mean()), 2),
                     corr=round(float(np.corrcoef(cc['amp'], cc['amp2023'])[0, 1]), 4)),
            djf=dict(median_diff=round(float((cc['ndvi_djf'] * 1000 - cc['djf2023']).median()), 2),
                     corr=round(float(np.corrcoef(cc['ndvi_djf'] * 1000, cc['djf2023'])[0, 1]), 4)),
            note='本轮为 2019-2024 合并窗口、scale=10；生产 amp2023/djf2023 为 2023 单年、scale=20')

    # ---- 结论
    S['headline'] = dict(
        overall_6class=S['overall_6class']['rate'],
        overall_6class_ci95=S['overall_6class']['ci95'],
        overall_axis3=S['overall_axis3']['rate'],
        gate='L3 ≥0.85',
        gate_pass=bool(p is not None and p >= 0.85),
        per_class={k2: v['rate'] for k2, v in per.items()},
        cls45_46_pass=bool(all(per.get(str(c), {}).get('rate') is not None and
                               per[str(c)]['rate'] >= 0.85 for c in (45, 46)))
        if all(str(c) in per for c in (45, 46)) else False)

    json.dump(S, open(OUTJSON, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    emit('写出 %s' % OUTJSON)
    emit('总体6类一致率 %.4f [%.4f-%.4f] (n=%d)' % (p, lo, hi, n))
    emit('轴级3类一致率 %s' % S['overall_axis3']['rate'])
    emit('逐类 %s' % json.dumps({k2: v['rate'] for k2, v in per.items()}, ensure_ascii=False))
    emit('判据效度 %s' % v['verdict'])
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['sample', 'worker', 'judge', 'report'])
    ap.add_argument('--acct', default='')
    ap.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    if a.mode == 'sample':
        do_sample()
    elif a.mode == 'worker':
        si, sn = a.shard.split('/')
        run_worker(a.acct, int(si), int(sn))
    elif a.mode == 'judge':
        do_judge()
    else:
        do_report()


if __name__ == '__main__':
    main()

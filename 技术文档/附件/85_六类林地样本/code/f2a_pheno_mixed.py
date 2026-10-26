# -*- coding: utf-8 -*-
"""f2a_pheno_mixed.py — 45/46 混交林样本的 Sentinel-2 物候补采（与既有 amp2023/djf2023 同口径）

* 输入（只读）:
    forest6/forest_pool_v1.parquet（472,307 行；取 cls6∈{45,46} 共 6,864 点）
    forest6/_calib_pts.parquet（口径校验对照点：非混交且已有 amp2023/djf2023）
* 规则源（口径唯一权威，逐行复刻）:
    F:/lc_work/v31_exp/code/w1_forest_fetch.py  —— amp2023/djf2023 的实际生产脚本
        S2 = COPERNICUS/S2_SR_HARMONIZED
        DJF 窗口 = (Y-1)-12-01 ~ Y-03-01      （2023 → 2022-12-01 / 2023-03-01）
        GS  窗口 = Y-06-01 ~ Y-09-01          （"生长季 6-9 月"，与任务书一致）
        SCL 掩膜 = 剔 3(阴影)/8(云中概率)/9(云高概率)
        CLOUDY_PIXEL_PERCENTAGE < 60
        合成 = 先对 B8/B4 反射率取 median，再 normalizedDifference（**不是**逐景 NDVI 后 median）
        量纲 = ×1000（**非 ×10000**；已实测 amp2023 == abs(jja-djf) 恒等，corr=1.000000，maxabs=0.0000）
        amp  = abs(jja - djf)                  （w2_forest_qc.py 第 110 行）
        采样 = sampleRegions(scale=20, geometries=False, tileScale=4)
    + 任务书额外要求的去云：QA60 bit10/bit11、SCL 11(雪)/6(水)
        → 仅在 --strict-extra 下作为**旁证**计算，用于量化口径偏移；主值一律用上述 w1 口径
* 门槛：2023 年某季有效观测 <3 → 退化为 2019+2023 两年合并，标 pheno_year_src
* 输出：forest6/mixed_pheno.parquet、forest6/f2a_stats.json、forest6/f2a_calib.csv
        forest6/pheno_parts/{tag}/{acct}/cell_*.parquet（断点续跑）
* 用法：python f2a_pheno_mixed.py --mode calib            # 只跑口径校验（先跑这个）
        python f2a_pheno_mixed.py --mode full             # 全量 6,864 点（内部起 2 进程分账号）
        python f2a_pheno_mixed.py --mode worker --acct <a> --shard 0/2 --tag mixed
        python f2a_pheno_mixed.py --mode report
* 资源：每进程单线程；最多 2 个并发账号（本机当前仅 2 个健康账号）
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
WORK = r'F:/lc_work/nb_national_2026/forest6'
POOL = os.path.join(WORK, 'forest_pool_v1.parquet')
CALIB_PTS = os.path.join(WORK, '_calib_pts.parquet')
PARTS = os.path.join(WORK, 'pheno_parts')
OUT_PARQ = os.path.join(WORK, 'mixed_pheno.parquet')
OUT_STATS = os.path.join(WORK, 'f2a_stats.json')
OUT_CALIB = os.path.join(WORK, 'f2a_calib.csv')

S2 = 'COPERNICUS/S2_SR_HARMONIZED'
SENT = -9999.0
SCALE = 20                 # 与 w1_forest_fetch.py 一致
CLOUDY_PCT = 60            # 与 w1 一致
CELL_DEG = 1.0             # 1° 格切分（任务书硬要求）
MIN_OBS = 3                # 有效观测门槛，<3 退化到两年合并
GS_YEARS = [2019, 2023]
UNITS = 'x1000 (ndvi*1000; amp = abs(gs - djf))'

# 健康账号：2026-10-25 17:03 串行 s0_sweep --force + s0b_health 实测仅此 2 个 OK，
# 其余 21 个 restricted-mode、2 个无项目权限。二者不在 SAME_DOMAIN_PAIRS 内，可并行。
ACCTS = ['ybzljyma', 'nnjxsn']
PIDS = {'ybzljyma': 'modern-tangent-509301-q8', 'nnjxsn': 'melodic-gamma-509323-u5'}

BANDS = ['gs23', 'gs23n', 'djf23', 'djf23n', 'gsP', 'gsPn', 'djfP', 'djfPn']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


# ------------------------------ GEE ------------------------------
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
    C.apply_account_env(acct)          # 同时设 HOME 与 USERPROFILE
    import ee
    pid = PIDS.get(acct) or C.ACCOUNTS[acct]['proj']
    ee.Initialize(project=pid)
    try:
        ee.data.setDeadline(300000)
    except Exception:
        pass
    emit('GEE 就绪 acct=%s proj=%s proxy=%s scale=%d cloudy<%d'
         % (acct, pid, url, SCALE, CLOUDY_PCT))
    return ee


def build_collection(ee, roi):
    col = (ee.ImageCollection(S2)
           .filterBounds(roi)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PCT)))

    def msk(im):
        scl = im.select('SCL')
        return im.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)))

    return col.map(msk)


def strict_collection(ee, roi):
    """任务书额外要求：QA60 bit10/bit11 + SCL 11(雪)/6(水) —— 仅作旁证"""
    col = (ee.ImageCollection(S2)
           .filterBounds(roi)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_PCT)))

    def msk(im):
        scl = im.select('SCL')
        qa = im.select('QA60')
        bad = (scl.eq(3).Or(scl.eq(6)).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(11))
               .Or(qa.bitwiseAnd(1 << 10).neq(0))
               .Or(qa.bitwiseAnd(1 << 11).neq(0)))
        return im.updateMask(bad.Not())

    return col.map(msk)


def season_band(col, d0, d1, tag):
    """w1 口径：先 median 反射率再 NDVI，×1000，unmask 哨兵"""
    c = col.filterDate(d0, d1)
    med = c.select(['B8', 'B4']).median()
    nd = (med.normalizedDifference(['B8', 'B4']).multiply(1000.0)
          .unmask(SENT).toFloat().rename(tag))
    cnt = c.select('B8').count().rename(tag + 'n').toFloat()
    return nd, cnt


def windows_for(year):
    """与 w1_forest_fetch.py 完全一致"""
    return (('%d-12-01' % (year - 1), '%d-03-01' % year),      # DJF
            ('%d-06-01' % year, '%d-09-01' % year))            # GS (6-9 月)


def build_image(ee, roi, strict=False):
    col = strict_collection(ee, roi) if strict else build_collection(ee, roi)
    bands = []
    djf23w, gs23w = windows_for(2023)
    g23, g23n = season_band(col, gs23w[0], gs23w[1], 'gs23')
    d23, d23n = season_band(col, djf23w[0], djf23w[1], 'djf23')
    bands += [g23, g23n, d23, d23n]
    # 两年合并（2019 与 2023 四窗并集，同一季节合并后取 median）
    gs_all = col.filterDate('%d-06-01' % 2019, '%d-09-01' % 2019).merge(
        col.filterDate(gs23w[0], gs23w[1]))
    djf_w19 = windows_for(2019)[0]
    djf_all = col.filterDate(djf_w19[0], djf_w19[1]).merge(
        col.filterDate(djf23w[0], djf23w[1]))

    def band_from(c, tag):
        med = c.select(['B8', 'B4']).median()
        nd = (med.normalizedDifference(['B8', 'B4']).multiply(1000.0)
              .unmask(SENT).toFloat().rename(tag))
        cnt = c.select('B8').count().rename(tag + 'n').toFloat()
        return nd, cnt

    gp, gpn = band_from(gs_all, 'gsP')
    dp, dpn = band_from(djf_all, 'djfP')
    bands += [gp, gpn, dp, dpn]
    return ee.Image.cat(bands)


def reduce_cell(ee, img, sub, rid_col, tag):
    """一个 1° 格 + 一次 reduce：失败重试；返回 {rid: props}"""
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                        {rid_col: str(getattr(r, rid_col))})
             for r in sub.itertuples()]
    fc = ee.FeatureCollection(feats)
    last = ''
    for att in range(4):
        try:
            t0 = time.time()
            r = img.sampleRegions(collection=fc, scale=SCALE, geometries=False,
                                  tileScale=4).getInfo()['features']
            emit('   %s n=%d ok %.1fs' % (tag, len(r), time.time() - t0))
            return {str(f['properties'][rid_col]): f['properties'] for f in r}
        except Exception as e:
            last = str(e)[:160]
            emit('   %s retry%d %s' % (tag, att + 1, last))
            time.sleep(5 * (att + 1))
    emit('   %s FAILED %s' % (tag, last))
    return None


# ------------------------------ worker ------------------------------
def run_worker(acct, shard_i, shard_n, tag, strict=False):
    pts_path = CALIB_PTS if tag == 'calib' else None
    if tag == 'calib':
        pts = pd.read_parquet(pts_path)
        rid = 'uid'
    else:
        df = pd.read_parquet(POOL, columns=['uid', 'lon', 'lat', 'cls6', 'prov', 'eco', 'tier',
                                           'code_master'])
        pts = df[df.cls6.isin([45, 46])].reset_index(drop=True)
        rid = 'uid'
    pts['cell'] = (np.floor(pts.lon / CELL_DEG).astype(int).astype(str) + '_'
                   + np.floor(pts.lat / CELL_DEG).astype(int).astype(str))
    cells = sorted(pts.cell.unique())
    mine = [c for i, c in enumerate(cells) if i % shard_n == shard_i]
    outd = os.path.join(PARTS, tag + ('_strict' if strict else ''), acct)
    os.makedirs(outd, exist_ok=True)
    emit('[%s/%s] 点 %d ｜ 格 %d/%d（本片 %d 格）strict=%s'
         % (tag, acct, len(pts), len(mine), len(cells), len(mine), strict))

    todo = []
    for c in mine:
        fp = os.path.join(outd, 'cell_%s.parquet' % c)
        if not os.path.exists(fp):
            todo.append(c)
    if not todo:
        emit('[%s/%s] 全部已完成（断点续跑）' % (tag, acct))
        return
    ee = init_ee(acct)
    t00 = time.time()
    for ci, c in enumerate(todo, 1):
        sub = pts[pts.cell == c]
        lon0, lat0 = [int(x) for x in c.split('_')]
        roi = ee.Geometry.Rectangle([lon0 - 0.05, lat0 - 0.05,
                                     lon0 + CELL_DEG + 0.05, lat0 + CELL_DEG + 0.05])
        img = build_image(ee, roi, strict=strict)
        got = reduce_cell(ee, img, sub, rid, '%s %s' % (acct, c))
        if got is None:
            continue
        rows = []
        for r in sub.itertuples():
            p = got.get(str(getattr(r, rid)))
            rec = {rid: str(getattr(r, rid)), 'lon': float(r.lon), 'lat': float(r.lat),
                   'cell': c}
            for b in BANDS:
                v = None if p is None else p.get(b)
                rec[b] = np.nan if v is None else float(v)
            rows.append(rec)
        o = pd.DataFrame(rows)
        for b in BANDS:
            o.loc[o[b] <= -9000, b] = np.nan
        o.to_parquet(os.path.join(outd, 'cell_%s.parquet' % c), index=False)
        if ci % 5 == 0:
            emit('[%s/%s] %d/%d 格，累计 %.0fs' % (tag, acct, ci, len(todo), time.time() - t00))
    emit('[%s/%s] 完成，耗时 %.1f min' % (tag, acct, (time.time() - t00) / 60))


# ------------------------------ 汇总 ------------------------------
def load_parts(tag, strict=False):
    d = os.path.join(PARTS, tag + ('_strict' if strict else ''))
    fs = glob.glob(os.path.join(d, '*', 'cell_*.parquet'))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df['uid'] = df['uid'].astype('int64')          # GEE 侧为字符串键，回本地还原整型
    return df.drop_duplicates('uid').reset_index(drop=True)


def derive(df):
    """按 2023 → 两年合并 的退化规则落最终指标（量纲 ×1000，amp=abs）"""
    out = df.copy()
    ok23 = (out.gs23n >= MIN_OBS) & (out.djf23n >= MIN_OBS) & out.gs23.notna() & out.djf23.notna()
    okP = (out.gsPn >= MIN_OBS) & (out.djfPn >= MIN_OBS) & out.gsP.notna() & out.djfP.notna()
    out['ndvi_gs'] = np.where(ok23, out.gs23, np.where(okP, out.gsP, np.nan))
    out['ndvi_djf'] = np.where(ok23, out.djf23, np.where(okP, out.djfP, np.nan))
    out['n_obs_gs'] = np.where(ok23, out.gs23n, np.where(okP, out.gsPn, np.nan))
    out['n_obs_djf'] = np.where(ok23, out.djf23n, np.where(okP, out.djfPn, np.nan))
    out['pheno_year_src'] = np.where(ok23, '2023', np.where(okP, '2019+2023', ''))
    out['amp'] = (out.ndvi_gs - out.ndvi_djf).abs()
    out['djf'] = out.ndvi_djf
    out['ndvi_gs'] = out.ndvi_gs / 1000.0
    out['ndvi_djf'] = out.ndvi_djf / 1000.0
    # status
    st = np.full(len(out), 'ok', dtype=object)
    st[out.pheno_year_src == '2019+2023'] = 'ok_pooled_2019_2023'
    nobs = out[['gs23n', 'djf23n', 'gsPn', 'djfPn']].fillna(0)
    st[(nobs.max(axis=1) == 0)] = 'no_s2_obs'
    st[(nobs.max(axis=1) > 0) & (out.pheno_year_src == '')] = 'low_obs_both_years'
    st[out.gs23.isna() & out.djf23.isna() & out.gsP.isna() & out.djfP.isna()] = 'no_s2_obs'
    out['status'] = st
    return out


def calib_report():
    got = load_parts('calib')
    ref = pd.read_parquet(CALIB_PTS)
    if not len(got):
        raise SystemExit('calib 无产出')
    d = derive(got)
    m = ref.merge(d[['uid', 'ndvi_gs', 'ndvi_djf', 'amp', 'djf', 'n_obs_gs', 'n_obs_djf',
                     'pheno_year_src', 'status']],
                  on='uid', how='inner', suffixes=('_old', '_new'))
    m['amp_old'] = m['amp2023']
    m['djf_old'] = m['djf2023']
    res = {}
    for nm, a, b in [('amp', 'amp_old', 'amp'), ('djf', 'djf_old', 'djf')]:
        s = m[[a, b]].dropna()
        diff = s[b] - s[a]
        res[nm] = dict(n=int(len(s)),
                       median_diff=round(float(diff.median()), 2),
                       mean_diff=round(float(diff.mean()), 2),
                       mae=round(float(diff.abs().mean()), 2),
                       p90_abs_diff=round(float(diff.abs().quantile(.90)), 2),
                       corr=round(float(s.corr().iloc[0, 1]), 5),
                       spearman=round(float(s.corr(method='spearman').iloc[0, 1]), 5),
                       slope=round(float(np.polyfit(s[a], s[b], 1)[0]), 4),
                       old_median=round(float(s[a].median()), 1),
                       new_median=round(float(s[b].median()), 1))
    gate = dict(n_ok=res['amp']['n'] >= 200,
                median_diff_ok=abs(res['amp']['median_diff']) <= 100,
                corr_ok=res['amp']['corr'] >= 0.9)
    gate['PASS'] = all(gate.values())
    m['amp_diff'] = m['amp'] - m['amp_old']
    m['djf_diff'] = m['djf'] - m['djf_old']
    m[['uid', 'lon', 'lat', 'cls6', 'amp_old', 'amp', 'amp_diff', 'djf_old', 'djf', 'djf_diff',
       'n_obs_gs', 'n_obs_djf', 'pheno_year_src', 'status']].to_csv(
        OUT_CALIB, index=False, encoding='utf-8-sig', float_format='%.3f')
    emit('口径校验 amp: n=%d 中位差=%.2f 相关=%.4f MAE=%.2f → %s'
         % (res['amp']['n'], res['amp']['median_diff'], res['amp']['corr'], res['amp']['mae'],
            'PASS' if gate['PASS'] else 'FAIL'))
    # 旁证：严格掩膜（QA60+snow+water）在同一批点上的偏移
    gs = load_parts('calib', strict=True)
    if len(gs):
        ds = derive(gs)
        ms = ref.merge(ds[['uid', 'amp', 'djf']], on='uid', how='inner')
        d2 = (ms['amp'] - ms['amp2023']).dropna()
        res['strict_mask_variant'] = dict(
            n=int(len(d2)), median_diff=round(float(d2.median()), 2),
            mae=round(float(d2.abs().mean()), 2),
            corr=round(float(ms[['amp', 'amp2023']].dropna().corr().iloc[0, 1]), 5),
            note='任务书要求的 QA60+snow(11)+water(6) 掩膜；仅旁证，主值不采用')
    return res, gate, m


def full_report(t0=None):
    got = load_parts('mixed')
    df = pd.read_parquet(POOL, columns=['uid', 'lon', 'lat', 'cls6', 'prov', 'eco', 'tier'])
    mix = df[df.cls6.isin([45, 46])].reset_index(drop=True)
    if not len(got):
        raise SystemExit('mixed 无产出')
    d = derive(got)
    out = mix.merge(d[['uid', 'ndvi_gs', 'ndvi_djf', 'amp', 'djf', 'n_obs_gs', 'n_obs_djf',
                       'pheno_year_src', 'status']], on='uid', how='left')
    out['status'] = out.status.fillna('not_attempted')
    cols = ['uid', 'lon', 'lat', 'cls6', 'ndvi_gs', 'ndvi_djf', 'amp', 'djf',
            'n_obs_gs', 'n_obs_djf', 'pheno_year_src', 'status']
    out[cols].to_parquet(OUT_PARQ, index=False)

    stats = dict(generated=pd.Timestamp.now().isoformat(timespec='seconds'),
                 unit=UNITS, scale=SCALE, cloudy_pct=CLOUDY_PCT,
                 windows=dict(gs_2023='2023-06-01/2023-09-01',
                              djf_2023='2022-12-01/2023-03-01',
                              gs_2019='2019-06-01/2019-09-01',
                              djf_2019='2018-12-01/2019-03-01'),
                 scl_mask='SCL != 3,8,9（w1_forest_fetch.py 原口径）',
                 composite='median(B8,B4) → NDVI → ×1000',
                 min_obs=MIN_OBS,
                 n_input=int(len(mix)),
                 n_attempted=int(out.status.ne('not_attempted').sum()),
                 status_counts={str(k): int(v) for k, v in out.status.value_counts().items()},
                 year_src_counts={str(k): int(v) for k, v in
                                  out.pheno_year_src.fillna('').value_counts().items()})
    okm = out[out.status.isin(['ok', 'ok_pooled_2019_2023'])]
    stats['n_success'] = int(len(okm))
    stats['n_fail'] = int(len(out) - len(okm))
    stats['fail_reasons'] = {str(k): int(v) for k, v in
                             out.loc[~out.index.isin(okm.index), 'status'].value_counts().items()}
    stats['success_rate'] = round(float(len(okm)) / len(mix), 4)

    dist = {}
    for c in [45, 46]:
        s = okm[okm.cls6 == c]
        for col in ['amp', 'djf']:
            v = s[col].dropna()
            dist['%d_%s' % (c, col)] = dict(
                n=int(len(v)), median=round(float(v.median()), 1),
                p25=round(float(v.quantile(.25)), 1), p75=round(float(v.quantile(.75)), 1),
                p10=round(float(v.quantile(.10)), 1), p90=round(float(v.quantile(.90)), 1),
                mean=round(float(v.mean()), 1))
    stats['dist_by_cls6'] = dist
    stats['n_obs_median'] = dict(gs=float(okm.n_obs_gs.median()),
                                 djf=float(okm.n_obs_djf.median()))
    stats['by_prov'] = {p: dict(n=int(len(s)),
                                ok=int(s.status.isin(['ok', 'ok_pooled_2019_2023']).sum()),
                                amp_med=round(float(s.amp.median()), 1) if s.amp.notna().any() else None)
                       for p, s in out.groupby('prov')}
    # 分账号
    acct_stats = {}
    for a in ACCTS:
        fs = glob.glob(os.path.join(PARTS, 'mixed', a, 'cell_*.parquet'))
        n = 0
        for f in fs:
            n += len(pd.read_parquet(f, columns=['uid']))
        acct_stats[a] = dict(cells=len(fs), points=n)
    stats['by_account'] = acct_stats
    if t0:
        stats['elapsed_min'] = round((time.time() - t0) / 60, 1)
    # 口径校验结果并入
    if os.path.exists(OUT_CALIB):
        cb = pd.read_csv(OUT_CALIB)
        r = {}
        for nm, a, b in [('amp', 'amp_old', 'amp'), ('djf', 'djf_old', 'djf')]:
            s = cb[[a, b]].dropna()
            diff = s[b] - s[a]
            r[nm] = dict(n=int(len(s)), median_diff=round(float(diff.median()), 2),
                         mae=round(float(diff.abs().mean()), 2),
                         corr=round(float(s.corr().iloc[0, 1]), 5),
                         old_median=round(float(s[a].median()), 1),
                         new_median=round(float(s[b].median()), 1))
        stats['calibration'] = r
    # 任务书要求的 strict 掩膜旁证（QA60 + 雪 + 水）
    sv = os.path.join(WORK, 'f2a_strict_variant.json')
    if os.path.exists(sv):
        stats['strict_mask_variant'] = json.load(open(sv, encoding='utf-8'))
    # doc84 §4.2 拆分规则在实测物候上的适用率（关键诊断）
    ok = out[out.status.isin(['ok', 'ok_pooled_2019_2023'])]
    r45 = (ok.amp < 350) & (ok.djf > 450)
    r46 = (ok.amp > 400) | (ok.djf < 350)
    stats['doc84_4_2_rule_applicability'] = {
        'rule_evergreen': 'amp<350 AND djf>450',
        'rule_deciduous': 'amp>400 OR djf<350',
        'by_cls6': {str(c): dict(n=int(len(g)),
                                 evergreen_rule=int(((g.amp < 350) & (g.djf > 450)).sum()),
                                 deciduous_rule=int(((g.amp > 400) | (g.djf < 350)).sum()),
                                 uncertain=int((~((g.amp < 350) & (g.djf > 450)) &
                                                ~((g.amp > 400) | (g.djf < 350))).sum()),
                                 both=int((((g.amp < 350) & (g.djf > 450)) &
                                           ((g.amp > 400) | (g.djf < 350))).sum()))
                    for c, g in ok.groupby('cls6')},
        'overall': dict(n=int(len(ok)), evergreen_rule=int(r45.sum()),
                        deciduous_rule=int(r46.sum()),
                        uncertain=int((~r45 & ~r46).sum()), both=int((r45 & r46).sum())),
        'note': '若 cls45（本应常绿占优）绝大多数落入落叶规则，说明 amp/djf 单靠物候无法支撑 45/46 拆分'}
    # 与既有四类纯林的分位对照（同量纲可比性展示）
    try:
        pl = pd.read_parquet(POOL, columns=['cls6', 'amp2023', 'djf2023'])
        ref_q = {}
        for c in [41, 42, 43, 44]:
            s = pl[pl.cls6 == c][['amp2023', 'djf2023']].dropna()
            ref_q[str(c)] = dict(n=int(len(s)), amp_p25=round(float(s.amp2023.quantile(.25)), 1),
                                 amp_med=round(float(s.amp2023.median()), 1),
                                 amp_p75=round(float(s.amp2023.quantile(.75)), 1),
                                 djf_med=round(float(s.djf2023.median()), 1))
        for c in [45, 46]:
            s = ok[ok.cls6 == c]
            ref_q[str(c)] = dict(n=int(len(s)), amp_p25=round(float(s.amp.quantile(.25)), 1),
                                 amp_med=round(float(s.amp.median()), 1),
                                 amp_p75=round(float(s.amp.quantile(.75)), 1),
                                 djf_med=round(float(s.djf.median()), 1), source='newly_fetched')
        stats['amp_by_class_reference'] = ref_q
    except Exception as e:
        stats['amp_by_class_reference'] = None
    json.dump(stats, open(OUT_STATS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1,
              default=str)
    emit('写出 %s（%d 行）｜成功 %d / 失败 %d' % (OUT_PARQ, len(out), len(okm), len(out) - len(okm)))
    emit(json.dumps({k: stats[k] for k in ('n_success', 'n_fail', 'fail_reasons',
                                           'year_src_counts', 'success_rate')},
                    ensure_ascii=False))
    return stats


def spawn(tag, nshard, strict=False):
    accts = ACCTS[:nshard]
    procs = []
    for i, a in enumerate(accts):
        cmd = [sys.executable, os.path.abspath(__file__), '--mode', 'worker',
               '--acct', a, '--shard', '%d/%d' % (i, len(accts)), '--tag', tag]
        if strict:
            cmd.append('--strict-extra')
        lf = os.path.join(WORK, 'pheno_parts', ('%s_%s.log' % (tag, a)))
        os.makedirs(os.path.dirname(lf), exist_ok=True)
        procs.append((a, subprocess.Popen(cmd, stdout=open(lf, 'w', encoding='utf-8'),
                                          stderr=subprocess.STDOUT), lf))
        emit('起进程 %s → %s' % (a, lf))
    for a, p, lf in procs:
        p.wait()
        emit('进程 %s 退出码 %s' % (a, p.returncode))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='full',
                    choices=['calib', 'full', 'worker', 'report', 'calib-strict'])
    ap.add_argument('--acct', default='')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--tag', default='mixed')
    ap.add_argument('--strict-extra', action='store_true')
    ap.add_argument('--nshard', type=int, default=2)
    a = ap.parse_args()

    if a.mode == 'worker':
        i, n = [int(x) for x in a.shard.split('/')]
        run_worker(a.acct, i, n, a.tag, strict=a.strict_extra)
    elif a.mode == 'calib':
        t0 = time.time()
        if not len(load_parts('calib')):
            spawn('calib', a.nshard)
        res, gate, _ = calib_report()
        json.dump(dict(calibration=res, gates=gate,
                       elapsed_min=round((time.time() - t0) / 60, 1)),
                  open(os.path.join(WORK, 'f2a_calib_gate.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1, default=str)
        print(json.dumps(dict(calibration=res, gates=gate), ensure_ascii=False, indent=1))
        if not gate['PASS']:
            emit('口径校验 FAIL —— 按任务书要求停止，不继续全量')
    elif a.mode == 'calib-strict':
        spawn('calib', a.nshard, strict=True)
        _ = calib_report()
    elif a.mode == 'full':
        t0 = time.time()
        spawn('mixed', a.nshard)
        full_report(t0)
    else:
        full_report()


if __name__ == '__main__':
    main()

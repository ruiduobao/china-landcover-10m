# -*- coding: utf-8 -*-
"""f7_final.py — 扩样混交候选点取 AEF64 + 构建最终六类林地样本交付集

* 输入（只读）
    F:/lc_work/nb_national_2026/forest6/mixed_expanded.parquet        70,000 混交候选（45:35000 / 46:35000）
    F:/lc_work/nb_national_2026/forest6/forest6_v1.parquet            472,307 六类池（含 AEF64 + cls6_train）
    F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet                 2,118,721 母库年度子集（AEF64 权威副本）
    Z:/nb_national_2026/fcs10_dl/tiles/{TILE}_fcs10.tif               FCS10 2023 原生 10 m 瓦片（补算 500 m 窗口）
* 特征源
    GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL  2023 年，64 波段 A00..A63，10 m
* 规则源
    技术文档/84_全国六类林地样本汇总.md §4.2（45/46 由邻域针叶码定义）/ §4.4（分生态区配额）
    r6_aef_fetch.py（AEF 取样口径：ee.Image.sampleRegions(scale=10, tileScale=4, geometries=False)）
    本任务书 B1–B5（质量分档 / 组装 / 分布保证 / 卫生检查）
* AEF 取数工程结论（2026-10-25 本机实测，见 _f7_probe/_diag/_bench）
    - 全国点或 15,000 点塞进一个 FC → Computation timed out（耗时 343–731 s 后失败），必须分块
    - 单 1° 格（≤361 点，全国 853 格）→ 每格 4.5–6.8 s，全量外推 373 min，超时预算
    - 4° 块（≤2,059 点，全国 87 块）→ 每块 18–86 s，pts/s 35，外推 ~59 min；两账号并发 → ~30 min
    - 自适应：4° → 失败降 2° → 1° → 0.5°，幂等 checkpoint，超时即降级重试
    - 请求间**结果确定性**已验证（同点重复请求 max|diff|=0）
    - 校验口径（关键）：forest6_v1 的 AEF 是 f1_pool 用 lon/lat 四舍五入 6 位做 key 从 r7 贴过来的，
      而 r7 原始坐标是 15 位 → 两者差 ~2.5 cm；在 10 m 像元边界平局处会翻到相邻像元。
      实测：按 forest6_v1 存的 6 位坐标重取 → 400 点中 17 点差>1e-6（max 0.069）；
            按 r7 原始坐标重取       → 400 点 0 点差>1e-6（max 2.1e-8）。
      ⇒ 方法本身正确（无网格/波段顺序错），参考端的坐标取整造成子像元相位差。两道门都报。
* 门槛
    AEF 完整率 100%（无 NaN）；Gate-B（r7 原始坐标）max|diff| ≤ 1e-6 且 0 点超阈
    否则立即停止并写 _f7_gate.json（不组装）
* 输出
    mixed_expanded_aef.parquet      70,000 × (19 + 64 + 分档列)
    forest6_core.parquet            版本1：高纯度（A 档）
    forest6_full.parquet            版本2：量大（含 B 档，保 quality_tier）
    forest6_south_needle_excluded.parquet   41/42 中 south_needle_flag==1 的点（doc82 无判别效度，单独放）
    f7_stats.json                   全部实测数字
    _f7_aef_parts/                  逐块 checkpoint（幂等断点续跑）
    _f7_gate.json                   AEF 校验门结果
* 用法
    python f7_final.py --fetch --shard 0 --nshard 2     # 取数（可两进程并行，各绑一个账号，总并发=2）
    python f7_final.py --fetch --shard 1 --nshard 2
    python f7_final.py --gate                           # AEF 校验（≥200 点，双口径）
    python f7_final.py --assemble                       # 组装两版本 + 卫生检查 + 配额
    python f7_final.py                                  # = fetch(单进程) + gate + assemble
* 资源：单进程 < 4 GB；GEE 账号并发 ≤2（两个不同域账号，非 s8xpcl1w+hte4021n / ppzynq+r45smj3u 对）
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.stdout.reconfigure(encoding='utf-8')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '8.全国生产'))
import prod_conf as C

for _k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
    os.environ.pop(_k, None)
os.environ['HTTP_PROXY'] = C.PROXY_URL
os.environ['HTTPS_PROXY'] = C.PROXY_URL
os.environ['ALL_PROXY'] = C.PROXY_URL
os.environ['GEE_PROXY'] = C.PROXY_URL

import numpy as np
import pandas as pd

F6 = r'F:/lc_work/nb_national_2026/forest6'
PARTS = os.path.join(F6, '_f7_aef_parts')
MIX = os.path.join(F6, 'mixed_expanded.parquet')
POOL = os.path.join(F6, 'forest6_v1.parquet')
R7 = r'F:/lc_work/年度子集_含稀有类/r7_train_2023.parquet'
TILEDIR = r'Z:/nb_national_2026/fcs10_dl/tiles'
PLAN_CSV = os.path.join(PROJ, '代码', '8.全国生产', 'plan', 'tiles.csv')
OUT_MIX = os.path.join(F6, 'mixed_expanded_aef.parquet')
OUT_CORE = os.path.join(F6, 'forest6_core.parquet')
OUT_FULL = os.path.join(F6, 'forest6_full.parquet')
OUT_SN = os.path.join(F6, 'forest6_south_needle_excluded.parquet')
OUT_STATS = os.path.join(F6, 'f7_stats.json')
GATE = os.path.join(F6, '_f7_gate.json')

A = ['A%02d' % i for i in range(64)]
# 2026-10-25 s0_sweep/s0b_health 串行复检：23 个候选里只有这 2 个健康（其余 restricted-mode）
ACCTS = [('ybzljyma', 'modern-tangent-509301-q8'), ('nnjxsn', 'melodic-gamma-509323-u5')]
BLOCK_DEG = 4.0           # 起始块大小（已基准：87 块 / ≤2,059 点）
BLOCK_CAP = 3000          # 单块点数上限（远低于 15,000 超时阈值）
MIN_DEG = 0.5             # 自适应降级下限

# 生态区（与 f1_pool.py / s2_sample_national.py 逐字一致）
ECO = {
    '东北': ['黑龙江省', '吉林省', '辽宁省'],
    '华北': ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区'],
    '西北': ['陕西省', '甘肃省', '宁夏回族自治区', '新疆维吾尔自治区'],
    '青藏': ['青海省', '西藏自治区'],
    '西南': ['四川省', '重庆市', '云南省', '贵州省'],
    '华中': ['河南省', '湖北省', '湖南省', '江西省', '安徽省'],
    '华东': ['江苏省', '浙江省', '上海市', '山东省', '福建省', '台湾省'],
    '华南': ['广东省', '广西壮族自治区', '海南省', '香港特别行政区', '澳门特别行政区'],
}
PROV2ECO = {p: e for e, ps in ECO.items() for p in ps}
ECO_ORDER = ['东北', '华北', '西北', '青藏', '西南', '华中', '华东', '华南']

# doc84 §4.4 分区配额（生态区 → {类: 最低点数}）
QUOTA = {
    '东北': {41: 5000, 42: 5000, 44: 5000, 45: 5000, 46: 5000},
    '华北': {41: 3000, 42: 3000, 44: 3000, 45: 3000, 46: 3000},
    '西北': {41: 2000, 42: 2000, 44: 2000},
    '青藏': {41: 2000, 42: 2000},
    '西南': {41: 3000, 42: 3000, 43: 3000, 44: 3000, 45: 3000},
    '华中': {41: 3000, 43: 3000, 44: 3000, 45: 3000, 46: 3000},
    '华东': {41: 2000, 43: 2000, 44: 2000, 45: 2000},
    '华南': {43: 2000, 44: 2000, 45: 2000, 46: 2000},
}
SIX = [41, 42, 43, 44, 45, 46]


def emit(m):
    print(time.strftime('[%H:%M:%S] ') + str(m), flush=True)


# ================================================================ GEE
def init_ee(prefer=None):
    import ee
    order = list(ACCTS)
    if prefer:
        order.sort(key=lambda x: 0 if x[0] == prefer else 1)
    for a, pid in order:
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                C.apply_account_env(a)
                ee.Initialize(project=pid)
                msg = ' '.join(str(x.message) for x in ws)
                if 'restricted' in msg or 'quota' in msg:
                    emit('  acct %s RESTRICTED' % a)
                    continue
            emit('  using %s / %s' % (a, pid))
            return ee, a
        except Exception as e:
            emit('  acct %s FAIL %s' % (a, repr(e)[:140]))
    raise RuntimeError('no usable GEE account')


def aef_image(ee):
    return (ee.ImageCollection(C.AEF).filterDate('2023-01-01', '2024-01-01')
            .mosaic().select(A))


def fetch_points(ee, img, lon, lat, pid, scale=10):
    """一次 sampleRegions → (values[n,64], n_returned)。r6 验证过的口径。"""
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(a), float(b)]), {'pid': int(c)})
                               for a, b, c in zip(lon, lat, pid)])
    got = img.sampleRegions(collection=fc, scale=scale, geometries=False,
                            tileScale=4).getInfo()
    g = {int(f['properties']['pid']): f['properties'] for f in got['features']}
    out = np.full((len(pid), 64), np.nan, 'float64')
    for i, c in enumerate(pid):
        p = g.get(int(c))
        if p is not None:
            out[i] = [p.get(b, np.nan) for b in A]
    return out, len(g)


def fetch_block(ee, img, d, deg, tag):
    """自适应取数：deg 起，超时/失败降级到 deg/2，直到 MIN_DEG。返回 (values, log)"""
    tries = []
    while deg >= MIN_DEG:
        t0 = time.time()
        try:
            v, nret = fetch_points(ee, img, d.lon.values, d.lat.values, d.pid.values)
            fin = int(np.isfinite(v).all(1).sum())
            tries.append(dict(deg=deg, ok=True, returned=nret, finite=fin,
                              secs=round(time.time() - t0, 1)))
            return v, tries
        except Exception as e:
            tries.append(dict(deg=deg, ok=False, secs=round(time.time() - t0, 1),
                              err=repr(e)[:120]))
            emit('    %s deg=%g 失败(%.0fs) %s → 降级' % (tag, deg, time.time() - t0, repr(e)[:90]))
            deg = deg / 2.0
    return None, tries


def cmd_fetch(shard=0, nshard=1, prefer=None):
    os.makedirs(PARTS, exist_ok=True)
    t0 = time.time()
    m = pd.read_parquet(MIX)
    m['pid'] = np.arange(len(m))
    m['bx'] = np.floor(m.lon / BLOCK_DEG).astype(int)
    m['by'] = np.floor(m.lat / BLOCK_DEG).astype(int)
    blocks = sorted(m.groupby(['bx', 'by']).size().index.tolist())
    blocks = [b for i, b in enumerate(blocks) if i % nshard == shard]
    emit('总 %d 点｜本分片 %d/%d，块 %d（起始 %.1f°）'
         % (len(m), shard, nshard, len(blocks), BLOCK_DEG))

    # 超 CAP 的块先对半切
    todo, split = [], []
    for b in blocks:
        n = int(((m.bx == b[0]) & (m.by == b[1])).sum())
        (split if n > BLOCK_CAP else todo).append(b)
    if split:
        emit('超 %d 点上限的块 %d 个 → 2° 子块' % (BLOCK_CAP, len(split)))
        for b in split:
            d = m[(m.bx == b[0]) & (m.by == b[1])]
            d = d.assign(bx2=np.floor(d.lon / 2).astype(int), by2=np.floor(d.lat / 2).astype(int))
            for b2 in d.groupby(['bx2', 'by2']).size().index:
                # 用 2° 坐标做 block id（与 4° 块不冲突：前缀不同）
                todo.append(('s', int(b2[0]), int(b2[1])))

    done = {f[:-8] for f in os.listdir(PARTS) if f.startswith('blk_') and f.endswith('.parquet')}

    def block_key(b):
        return ('blk_s_%d_%d' % (b[1], b[2])) if b[0] == 's' else ('blk_g_%d_%d' % (b[0], b[1]))

    left = [b for b in todo if block_key(b) not in done]
    emit('块 %d（已完成 %d，待取 %d，断点续跑）' % (len(todo), len(todo) - len(left), len(left)))
    ee, acct = init_ee(prefer)
    img = aef_image(ee)
    log, n_done, npts = [], 0, 0
    for b in left:
        if b[0] == 's':
            d = m[(np.floor(m.lon / 2).astype(int) == b[1]) &
                  (np.floor(m.lat / 2).astype(int) == b[2])]
            start_deg = 2.0
        else:
            d = m[(m.bx == b[0]) & (m.by == b[1])]
            start_deg = BLOCK_DEG
        key = block_key(b)
        fp = os.path.join(PARTS, key + '.parquet')
        if os.path.exists(fp):
            continue
        if not len(d):
            continue
        v, tries = fetch_block(ee, img, d, start_deg, key)
        if v is None:
            log.append(dict(block=key, n=len(d), ok=False, tries=tries))
            emit('  %s 全部降级失败 n=%d' % (key, len(d)))
            continue
        df = pd.DataFrame(v, columns=A).astype('float32')
        df.insert(0, 'pid', d.pid.values)
        df.to_parquet(fp, index=False)
        fin = int(np.isfinite(v).all(1).sum())
        log.append(dict(block=key, n=len(d), ok=True, finite=fin, tries=tries))
        n_done += 1
        npts += fin
        if n_done <= 3 or n_done % 10 == 0:
            emit('  块 %d/%d %s n=%d finite=%d 累计 %d 点 已用 %.1f min 最近 %.0fs'
                 % (n_done, len(todo), key, len(d), fin, npts, (time.time() - t0) / 60,
                    sum(x['secs'] for x in tries)))
    emit('分片 %d 完成：新取 %d 块 / %d 点，%.1f min' % (shard, n_done, npts, (time.time() - t0) / 60))
    json.dump(dict(acct=acct, shard=shard, nshard=nshard, blocks=len(todo), log=log,
                   elapsed_min=round((time.time() - t0) / 60, 1)),
              open(os.path.join(F6, '_f7_fetch_shard%d.json' % shard), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)


# ================================================================ 校验门
def build_mixed_with_aef():
    """拼 checkpoint → mixed_expanded_aef.parquet；返回 (df, 取数统计)"""
    parts = [f for f in os.listdir(PARTS) if f.startswith('blk_') and f.endswith('.parquet')]
    if not parts:
        raise SystemExit('无 checkpoint，先跑 --fetch')
    aef = pd.concat([pd.read_parquet(os.path.join(PARTS, f)) for f in parts],
                    ignore_index=True).drop_duplicates('pid')
    m = pd.read_parquet(MIX)
    m['pid'] = np.arange(len(m))
    d = m.merge(aef, on='pid', how='left', validate='one_to_one')
    assert len(d) == len(m), '拼接行数不符 %d vs %d' % (len(d), len(m))
    aef_ok = d[A].notna().all(1)
    fetch = dict(parts=len(parts), rows=len(d), aef_complete=int(aef_ok.sum()),
                 aef_missing=int((~aef_ok).sum()),
                 success_rate=round(float(aef_ok.mean()), 6))
    return d, fetch


def cmd_gate():
    """AEF 校验：两道口径 + 独立源对照"""
    d, fetch = build_mixed_with_aef()
    if fetch['aef_missing']:
        emit('⚠ AEF 缺失 %d 点（%.4f%%）' % (fetch['aef_missing'], 100 * (1 - fetch['success_rate'])))
    ee, acct = init_ee(ACCTS[0][0])
    img = aef_image(ee)
    out = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), acct=acct, fetch=fetch)

    # --- 参考池：forest6_v1 + r7 原始坐标
    f6 = pd.read_parquet(POOL, columns=['uid', 'lon', 'lat'] + A)
    r7 = pd.read_parquet(R7, columns=['lon', 'lat'])
    r7['k'] = r7.lon.round(6).astype(str) + '_' + r7.lat.round(6).astype(str)
    r7 = r7.drop_duplicates('k').rename(columns={'lon': 'lon_r7', 'lat': 'lat_r7'})
    f6['k'] = f6.lon.round(6).astype(str) + '_' + f6.lat.round(6).astype(str)
    ref = f6.merge(r7[['k', 'lon_r7', 'lat_r7']], on='k', how='inner')
    ref = ref.sample(min(300, len(ref)), random_state=20261025).reset_index(drop=True)
    ref['pid'] = np.arange(len(ref))
    emit('校验点 %d（forest6_v1 ∩ r7，随机抽）' % len(ref))
    stored = ref[A].to_numpy('float64')

    v_at_f6, _ = fetch_points(ee, img, ref.lon.values, ref.lat.values, ref.pid.values)
    v_at_r7, _ = fetch_points(ee, img, ref.lon_r7.values, ref.lat_r7.values, ref.pid.values)
    for lab, v in (('gateA_at_forest6_coords', v_at_f6), ('gateB_at_r7_coords', v_at_r7)):
        ok = np.isfinite(v).all(1)
        dd = np.abs(v[ok] - stored[ok])
        ptm = np.nanmax(dd, axis=1)
        out[lab] = dict(n=int(ok.sum()), max_abs=float(np.nanmax(dd)),
                        mean_abs=float(np.nanmean(dd)),
                        n_pt_gt_1e6=int((ptm > 1e-6).sum()),
                        frac_pt_gt_1e6=round(float((ptm > 1e-6).mean()), 6),
                        corr=round(float(np.corrcoef(v[ok].ravel(), stored[ok].ravel())[0, 1]), 8))
        emit('%s: n=%d max|diff|=%.3g mean=%.2e 超阈点 %d/%d corr=%.8f'
             % (lab, int(ok.sum()), np.nanmax(dd), np.nanmean(dd),
                int((ptm > 1e-6).sum()), int(ok.sum()), out[lab]['corr']))

    # --- 本批 70,000 的独立自检：与 r7 在最近点比较（同 1″ 内），确认不是常量/错列
    m = d
    r7f = pd.read_parquet(R7, columns=['lon', 'lat'] + A)
    # 空间分箱找最近（0.002° 桶 + 邻桶）
    B = 0.002
    r7f['bx'] = np.floor(r7f.lon / B).astype(int)
    r7f['by'] = np.floor(r7f.lat / B).astype(int)
    grid = {}
    for (bx, by), g in r7f.groupby(['bx', 'by']):
        grid[(bx, by)] = (g.lon.values, g.lat.values, g[A].to_numpy('float64'))
    samp = m.sample(500, random_state=7)
    diffs, pairs = [], 0
    for r in samp.itertuples():
        bx, by = int(np.floor(r.lon / B)), int(np.floor(r.lat / B))
        best, bd = None, 1e9
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                g = grid.get((bx + dx, by + dy))
                if g is None:
                    continue
                dd = (g[0] - r.lon) ** 2 + (g[1] - r.lat) ** 2
                i = int(np.argmin(dd))
                if dd[i] < bd:
                    bd, best = dd[i], g[2][i]
        if best is not None and bd < (B ** 2):
            diffs.append(np.abs(np.array([getattr(r, a) for a in A], 'float64') - best))
            pairs += 1
    if pairs:
        dd = np.array(diffs)
        out['new_points_vs_nearest_r7'] = dict(
            n=pairs, max_abs=float(np.nanmax(dd)), mean_abs=float(np.nanmean(dd)),
            note='最近邻（≤0.002°，约 220 m）——非同点，仅证明取值非常量/非错列；'
                 'AEF 10 m 尺度下邻点自然不同')

    # --- 判定
    gateB = out['gateB_at_r7_coords']
    verdict = 'PASS' if (gateB['max_abs'] <= 1e-6 and gateB['n_pt_gt_1e6'] == 0) else 'FAIL'
    out['verdict'] = verdict
    out['gateA_note'] = ('gateA 用 forest6_v1 自身存储的 6 位小数坐标重取；其 AEF 实际来自 r7 未取整坐标'
                         '（f1_pool 按 6 位 key 贴值，两点相距 ~2.5 cm）→ 像元边界平局翻格，'
                         '系统性产生少量 >1e-6 差异。方法验证以 gateB 为准。')
    json.dump(out, open(GATE, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('AEF 校验判定 %s → _f7_gate.json' % verdict)
    if verdict != 'PASS':
        emit('‼ Gate-B 未通过，按纪律停止组装')
        raise SystemExit(2)
    return out


# ================================================================ 分档
def mixed_tier(v):
    """frac_needle_500m → M_A/M_B/M_C。边界用半开区间以复现任务书给出的 22,938/20,169/26,893。"""
    v = np.asarray(v, 'float32')
    A_ = (v >= np.float32(0.25)) & (v < np.float32(0.75))
    B_ = ((v >= np.float32(0.10)) & (v < np.float32(0.25))) | \
         ((v >= np.float32(0.75)) & (v < np.float32(0.90)))
    return np.where(A_, 'A', np.where(B_, 'B', 'C'))


def frac_needle_500m_of(df):
    """对给定点补算 500 m（101×101 @10m）窗口针叶占比；tile 由 plan 2° 格推得。"""
    L_N = np.zeros(256, bool)
    L_N[[71, 72, 81, 82]] = True
    L_B = np.zeros(256, bool)
    L_B[[51, 52, 61, 62]] = True
    import rasterio
    from rasterio.windows import Window
    out = np.full(len(df), np.nan, np.float32)
    tv = np.array([int('%02d%02d' % (a, b)) for a, b in
                   zip(np.floor((df.lon.values - 73) / 2).astype(int),
                       np.floor((df.lat.values - 18) / 2).astype(int))])
    for t in np.unique(tv):
        path = os.path.join(TILEDIR, 'T%04d_fcs10.tif' % t)
        if not os.path.exists(path):
            continue
        ii = np.flatnonzero(tv == t)
        try:
            with rasterio.open(path) as ds:
                inv = ~ds.transform
                xx, yy = inv * (df.lon.values[ii], df.lat.values[ii])
                rr = np.floor(yy).astype(np.int64)
                cc = np.floor(xx).astype(np.int64)
                for k, (ri, ci) in enumerate(zip(rr, cc)):
                    try:
                        a = ds.read(1, window=Window(int(ci) - 50, int(ri) - 50, 101, 101),
                                    boundless=True, fill_value=0)
                    except Exception:
                        continue
                    nl = int(L_N[a].sum())
                    nb = int(L_B[a].sum())
                    if nl + nb > 0:
                        out[ii[k]] = nl / (nl + nb)
        except Exception:
            continue
    return out


def strat_sample(df, cap, cls_col='cls6', eco_col='eco', seed=42):
    """按生态区按比例抽到 cap（最大余数法），保证每区至少 1 点。"""
    if len(df) <= cap:
        return df
    g = df.groupby(eco_col).size()
    raw = g / g.sum() * cap
    base = np.floor(raw).astype(int).clip(lower=1)
    # 调整到 cap
    while base.sum() > cap:
        k = base[base > 1].idxmax()
        base[k] -= 1
    while base.sum() < cap:
        rem = (raw - base).sort_values(ascending=False)
        placed = False
        for k in rem.index:
            if base[k] < g[k]:
                base[k] += 1
                placed = True
                break
        if not placed:
            break
    rng = np.random.RandomState(seed)
    parts = []
    for k, n in base.items():
        sub = df[df[eco_col] == k]
        parts.append(sub.sample(min(n, len(sub)), random_state=rng.randint(1 << 30)))
    return pd.concat(parts).sort_index()


def quota_floor(cls, eco):
    """doc84 §4.4 给该生态区该类规定的最低点数（未规定=0）。"""
    return QUOTA.get(eco, {}).get(int(cls), 0)


def strat_sample_quota(df, cls, target, seed=42, eco_col='eco'):
    """配额感知分层抽样：先满足各生态区最低配额（doc84 §4.4），余量按可得点数比例分配。

    * 与"手工平衡"的区别：不做任何逐区手工加减；配额只用于**抽样权重下限**，
      最终能否达标仍由 pool 实际可得点数决定，未达标一律在 stats 里列出。
    * target = max(15_000, 该类配额总和)：避免"全局上限"人为制造配额失败。
    """
    g = df.groupby(eco_col).size()
    alloc = {e: min(quota_floor(cls, e), int(n)) for e, n in g.items()}
    total = sum(alloc.values())
    if total < target:
        rem_cap = {e: int(g[e]) - alloc[e] for e in g.index}
        pool = sum(rem_cap.values())
        if pool > 0:
            need = min(target - total, pool)
            for e in g.index:
                alloc[e] += int(round(need * rem_cap[e] / pool))
    # 校正到 <= target 且 <= 可得
    for e in sorted(g.index, key=lambda x: -alloc[x]):
        over = sum(alloc.values()) - target
        if over <= 0:
            break
        cut = min(over, alloc[e])
        alloc[e] -= cut
    rng = np.random.RandomState(seed)
    parts = [df[df[eco_col] == e].sample(n, random_state=rng.randint(1 << 30))
             for e, n in alloc.items() if n > 0]
    return pd.concat(parts).sort_index() if parts else df.iloc[:0]


def quota_check(df, ver):
    """doc84 §4.4 逐条配额；返回 (rows, unmet)"""
    rows, unmet = [], []
    for eco, need in QUOTA.items():
        for cls, q in need.items():
            n = int(((df.eco == eco) & (df.cls6 == cls)).sum())
            rows.append(dict(version=ver, eco=eco, cls6=cls, n=n, quota=q,
                             pass_=bool(n >= q)))
            if n < q:
                unmet.append(dict(eco=eco, cls6=cls, n=n, quota=q, gap=q - n))
    return rows, unmet


def hygiene(df, tag):
    h = {}
    h['rows'] = int(len(df))
    keys = df.lon.round(6).astype(str) + '_' + df.lat.round(6).astype(str)
    h['dup_coord_6dp'] = int(keys.duplicated().sum())
    h['dup_uid'] = int(df.uid.duplicated().sum())
    num = [c for c in ['lon', 'lat'] + A if c in df.columns]
    h['nan_cells'] = int(df[num].isna().sum().sum())
    h['nan_rows'] = int(df[num].isna().any(axis=1).sum())
    h['coord_out_of_bbox'] = int(((df.lon < 73) | (df.lon > 136) |
                                  (df.lat < 17) | (df.lat > 54)).sum())
    h['coord_range'] = dict(lon=[float(df.lon.min()), float(df.lon.max())],
                            lat=[float(df.lat.min()), float(df.lat.max())])
    va = df[A].to_numpy('float64')
    h['aef_missing_rows'] = int((~np.isfinite(va).all(1)).sum())
    h['aef_min'] = float(np.nanmin(va))
    h['aef_max'] = float(np.nanmax(va))
    h['aef_out_of_-1_1'] = int(((va < -1) | (va > 1)).sum())
    h['pass'] = bool(h['dup_coord_6dp'] == 0 and h['dup_uid'] == 0 and h['nan_rows'] == 0 and
                     h['coord_out_of_bbox'] == 0 and h['aef_missing_rows'] == 0 and
                     h['aef_out_of_-1_1'] == 0)
    return h


def overlap_rates(df):
    """与 v2 / v30 / opt_v1 的坐标重叠率。

    分母口径说明：`pct_of_delivered` = 本交付集的点有多大比例能在这张参照表里精确命中
    （这才是"本集与既有数据重叠率"）；`pct_of_ref` = 参照表自身有多大比例被本集命中。
    """
    out = {}
    srcs = {'v2': r'F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet',
            'v30': r'F:/lc_work/nb_national_2026/opt/needle_broad_veg30_v1.parquet',
            'opt_v1': r'F:/lc_work/nb_national_2026/opt/needle_broad_opt_v1.parquet'}
    base6 = set(df.lon.round(6).astype(str) + '_' + df.lat.round(6).astype(str))
    base5 = set(df.lon.round(5).astype(str) + '_' + df.lat.round(5).astype(str))
    for k, p in srcs.items():
        if not os.path.exists(p):
            out[k] = dict(error='missing')
            continue
        d = pd.read_parquet(p, columns=['lon', 'lat'])
        k6 = set(d.lon.round(6).astype(str) + '_' + d.lat.round(6).astype(str))
        k5 = set(d.lon.round(5).astype(str) + '_' + d.lat.round(5).astype(str))
        h6, h5 = len(base6 & k6), len(base5 & k5)
        out[k] = dict(ref_rows=int(len(d)),
                      hit_6dp=h6,
                      pct_of_delivered=round(100 * h6 / max(len(base6), 1), 4),
                      hit_5dp=h5,
                      pct_of_delivered_5dp=round(100 * h5 / max(len(base5), 1), 4),
                      pct_of_ref=round(100 * h6 / max(len(k6), 1), 4))
        del d
    return out


# ================================================================ 组装
def cmd_assemble(core_legacy='A'):
    t0 = time.time()
    CORE_LEGACY = core_legacy
    d, fetch = build_mixed_with_aef()
    if fetch['aef_missing']:
        emit('⚠ AEF 缺失 %d 点 → 先跑 --fetch 补齐' % fetch['aef_missing'])
    stats = dict(generated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                 script=os.path.abspath(__file__),
                 rule_source='技术文档/84 §4.2/§4.4 + 任务书 B1–B5',
                 aef_fetch=fetch)

    # ---------- B1 混交分档 ----------
    d['mixed_tier'] = mixed_tier(d.frac_needle_500m.values)
    d['quality_tier'] = d['mixed_tier']
    d['tier_src'] = 'derived_M_' + d['mixed_tier']
    d['has_mixedcode'] = (d.n_mixedcode_win > 0).astype(np.int8)
    d['v30_is_mixed'] = (d.v30_group == '针阔混交林').astype(np.int8)
    d['legacy_mixed'] = np.int8(0)
    d['eco'] = d.prov.map(PROV2ECO).fillna('未匹配')
    d['src'] = 'mixed_expand'
    d['gid'] = 'M' + d.uid.astype(str)
    d = d.drop(columns=['pid'])
    d.to_parquet(OUT_MIX, index=False)
    emit('写 mixed_expanded_aef.parquet %d 行 × %d 列' % (len(d), d.shape[1]))
    tier_cnt = d.mixed_tier.value_counts().to_dict()
    stats['mixed_tiers'] = dict(
        counts={k: int(v) for k, v in tier_cnt.items()},
        convention='A=[0.25,0.75) B=[0.10,0.25)∪[0.75,0.90) C=<0.10∪>=0.90'
                   '（半开；复现任务书 22,938/20,169/26,893。恰在 0.75 的 19 点归 B，'
                   '恰在 0.25 的 10 点归 A）',
        by_class={str(k): {kk: int(vv) for kk, vv in v.items()}
                  for k, v in pd.crosstab(d.cls6_expand, d.mixed_tier).T.to_dict().items()},
        has_mixedcode=int(d.has_mixedcode.sum()),
        v30_is_mixed=int(d.v30_is_mixed.sum()),
        by_tier_flags={t: dict(n=int((d.mixed_tier == t).sum()),
                               has_mixedcode=int(((d.mixed_tier == t) & (d.has_mixedcode == 1)).sum()),
                               v30_is_mixed=int(((d.mixed_tier == t) & (d.v30_is_mixed == 1)).sum()))
                       for t in ('A', 'B', 'C')})

    # ---------- B2 纯林分档 ----------
    cols_p = ['uid', 'lon', 'lat', 'cls6_train', 'tier', 'also_mixed_window', 'south_needle_flag',
              'high_elev_enf', 'eco', 'prov', 'legacy_mixed'] + A
    p = pd.read_parquet(POOL, columns=cols_p)
    p = p[p.cls6_train.isin([41, 42, 43, 44])].copy()
    p['eco'] = p.prov.map(PROV2ECO).fillna('未匹配')
    p['cls6'] = p.cls6_train.astype(np.int16)
    is_sn = (p.south_needle_flag == 1)
    p['pure_tier'] = np.where((p.also_mixed_window == 0) & p.tier.isin(['gold', 'silver']),
                              'A', 'B')
    pure_all = dict(total=int(len(p)),
                    P_A=int((p.pure_tier == 'A').sum()), P_B=int((p.pure_tier == 'B').sum()),
                    south_needle_41_42=int(is_sn.sum()),
                    high_elev_enf=int((p.high_elev_enf == 1).sum()),
                    by_class={str(c): dict(total=int((p.cls6 == c).sum()),
                                           P_A=int(((p.cls6 == c) & (p.pure_tier == 'A')).sum()),
                                           P_B=int(((p.cls6 == c) & (p.pure_tier == 'B')).sum()),
                                           south_needle=int(((p.cls6 == c) & is_sn).sum()))
                              for c in (41, 42, 43, 44)})
    stats['pure_tiers'] = pure_all
    sn = p[is_sn].copy()
    sn['quality_tier'] = 'south_needle'
    sn['src'] = 'forest6_v1'
    sn['gid'] = 'P' + sn.uid.astype(str)
    sn['excluded_reason'] = 'south_needle_flag=1（doc82：FCS10 南方针叶无判别效度）'
    sn = sn.drop(columns=['cls6_train', 'pure_tier'])
    sn.to_parquet(OUT_SN, index=False)
    emit('写 %s：%d 行（41/42 南方针叶，已从两版本排除）' % (os.path.basename(OUT_SN), len(sn)))

    pu = p[~is_sn].copy()          # 可用纯林（已排除南方针叶）
    pu['src'] = 'forest6_v1'
    pu['gid'] = 'P' + pu.uid.astype(str)
    pu['quality_tier'] = pu.pure_tier
    pu['tier_src'] = pu.tier
    pu['frac_needle_500m'] = np.nan
    pu['has_mixedcode'] = np.int8(0)
    pu['v30_is_mixed'] = np.int8(0)
    pu = pu.drop(columns=['cls6_train', 'pure_tier', 'tier'])

    # ---------- 既有混交点（cls6_train 45/46）----------
    lg = pd.read_parquet(POOL, columns=['uid', 'lon', 'lat', 'cls6_train', 'tier', 'eco', 'prov',
                                        'legacy_mixed', 'south_needle_flag', 'high_elev_enf'] + A)
    lg = lg[lg.cls6_train.isin([45, 46])].copy()
    lg['cls6'] = lg.cls6_train.astype(np.int16)
    lg['eco'] = lg.prov.map(PROV2ECO).fillna('未匹配')
    emit('既有混交点 %d，补算 500 m 窗口针叶占比…' % len(lg))
    cache = os.path.join(F6, '_f7_legacy500m.parquet')
    if os.path.exists(cache):
        c = pd.read_parquet(cache)
        if len(c) == len(lg) and set(c.uid) == set(lg.uid):
            lg['frac_needle_500m'] = c.set_index('uid').reindex(lg.uid).frac_needle_500m.values
            emit('  命中缓存 _f7_legacy500m.parquet')
    if 'frac_needle_500m' not in lg or lg.frac_needle_500m.isna().all():
        lg['frac_needle_500m'] = frac_needle_500m_of(lg)
        pd.DataFrame({'uid': lg.uid, 'frac_needle_500m': lg.frac_needle_500m}).to_parquet(
            cache, index=False)
    lg['mixed_tier'] = mixed_tier(lg.frac_needle_500m.fillna(0.5).values)
    lg.loc[lg.frac_needle_500m.isna(), 'mixed_tier'] = 'NA'
    lg['has_mixedcode'] = (lg.legacy_mixed == 1).astype(np.int8)
    lg['v30_is_mixed'] = np.int8(0)      # forest6_v1 无 v30_group 列，此处如实置 0 并注明
    lg['src'] = 'forest6_v1'
    lg['gid'] = 'P' + lg.uid.astype(str)
    lg['quality_tier'] = lg['mixed_tier']     # 统一用 A/B/C/NA，便于下游按 quality_tier 过滤
    lg['tier_src'] = lg.tier
    lg = lg.drop(columns=['cls6_train', 'tier'])
    stats['legacy_mixed_500m'] = dict(
        n=int(len(lg)), na=int((lg.mixed_tier == 'NA').sum()),
        counts={k: int(v) for k, v in lg.mixed_tier.value_counts().items()},
        note='forest6_v1 无 v30_group 列 → 这些点 v30_is_mixed 置 0（缺列，非真值 0）；'
             'has_mixedcode 用 legacy_mixed（母库 91/92 码）代替')
    emit('既有混交点 500 m 分档 %s（NA=%d）'
         % (lg.mixed_tier.value_counts().to_dict(), int((lg.mixed_tier == 'NA').sum())))

    # ---------- 版本 1：core（高纯度 A 档）----------
    # 任务书两条要求互相冲突：「41–44 取 P_A 全部」vs「目标每类 ≥0.8–1.5 万」。
    # P_A 实际每类 3.5–10 万，全取会让 core(34.6 万) 比 full 更大，违背"版本2 量大"。
    # 处置：按 doc84 §4.4 配额感知抽样，每类取 max(15,000, 该类配额总和)；
    #       A 档不足 10,000 时才用 B 补。P_A 总量与实际取用都写进 stats。
    core_parts, core_log = [], {}
    for c in (41, 42, 43, 44):
        a_ = pu[(pu.cls6 == c) & (pu.quality_tier == 'A')]
        b_ = pu[(pu.cls6 == c) & (pu.quality_tier == 'B')]
        floor = sum(QUOTA.get(e, {}).get(c, 0) for e in ECO_ORDER)
        target = max(15000, floor)
        if len(a_) >= 10000:
            take = strat_sample_quota(a_, c, target, seed=300 + c)
            core_log[str(c)] = dict(P_A_available=int(len(a_)), P_A_used=int(len(take)),
                                    P_B_topup=0, quota_floor=floor, target=target,
                                    capped=bool(len(a_) > target),
                                    all_A=True)
        else:
            need = 10000 - len(a_)
            take = pd.concat([a_, b_.sample(min(need, len(b_)), random_state=41)])
            core_log[str(c)] = dict(P_A_available=int(len(a_)), P_A_used=int(len(a_)),
                                    P_B_topup=int(min(need, len(b_))), quota_floor=floor,
                                    target=target, capped=False, all_A=False)
        core_parts.append(take)
    for c in (45, 46):
        a_ = d[(d.cls6_expand == c) & (d.mixed_tier == 'A')].copy()
        a_['cls6'] = c
        lg_c = lg[lg.cls6 == c]
        # 任务书：core 的 45/46 = M_A 全部 + 既有 9,301 点；但同段又要求 core「全部是 A 档」。
        # 两者冲突（既有点里 3,152 是 B、2,242 是 C）。core 定义为高纯度 → 默认只收既有 A 档点；
        # --core-legacy all 可切换成字面的「全部 9,301」。两种口径的点数都记入 stats。
        lg_used = lg_c if CORE_LEGACY == 'all' else lg_c[lg_c.quality_tier == 'A']
        core_log[str(c)] = dict(M_A=int(len(a_)), legacy_available=int(len(lg_c)),
                                legacy_used=int(len(lg_used)),
                                legacy_mode=CORE_LEGACY,
                                legacy_by_tier={k: int(v) for k, v in
                                                lg_c.quality_tier.value_counts().items()},
                                legacy_B_C_in_full_only=int(
                                    len(lg_c) - len(lg_c[lg_c.quality_tier == 'A'])))
        core_parts.append(pd.concat([a_, lg_used], ignore_index=True))
    core = pd.concat(core_parts, ignore_index=True)

    # ---------- 版本 2：full（A+B，每类 ≤40,000，配额感知分层）----------
    full_parts, full_log = [], {}
    for c in (41, 42, 43, 44):
        sub = pu[pu.cls6 == c]
        take = strat_sample_quota(sub, c, 40000, seed=100 + c)
        full_log[str(c)] = dict(pool=int(len(sub)), taken=int(len(take)),
                                by_tier={k: int(v) for k, v in take.quality_tier.value_counts().items()})
        full_parts.append(take)
    for c in (45, 46):
        sub = pd.concat([d[(d.cls6_expand == c) & (d.mixed_tier.isin(['A', 'B']))].assign(cls6=c),
                         lg[lg.cls6 == c]], ignore_index=True)
        take = strat_sample_quota(sub, c, 40000, seed=200 + c)
        full_log[str(c)] = dict(pool=int(len(sub)), taken=int(len(take)),
                                by_tier={k: int(v) for k, v in take.quality_tier.value_counts().items()})
        full_parts.append(take)
    full = pd.concat(full_parts, ignore_index=True)

    # 统一列序
    base_cols = ['gid', 'uid', 'src', 'cls6', 'lon', 'lat', 'eco', 'prov', 'quality_tier',
                 'mixed_tier', 'frac_needle_500m', 'has_mixedcode', 'v30_is_mixed',
                 'tier_src', 'also_mixed_window', 'south_needle_flag', 'high_elev_enf',
                 'legacy_mixed'] + A
    for df in (core, full):
        for c in base_cols:
            if c not in df.columns:
                df[c] = np.nan
    core = core[base_cols].copy()
    full = full[base_cols].copy()
    core['has_mixedcode'] = core.has_mixedcode.fillna(0).astype('int8')
    full['has_mixedcode'] = full.has_mixedcode.fillna(0).astype('int8')
    core['v30_is_mixed'] = core.v30_is_mixed.fillna(0).astype('int8')
    full['v30_is_mixed'] = full.v30_is_mixed.fillna(0).astype('int8')
    core.to_parquet(OUT_CORE, index=False)
    full.to_parquet(OUT_FULL, index=False)
    emit('写 forest6_core.parquet %d 行 ｜ forest6_full.parquet %d 行' % (len(core), len(full)))

    # ---------- 统计 ----------
    def describe(df, ver):
        by_cls = df.cls6.value_counts().sort_index()
        rows, unmet = quota_check(df, ver)
        return dict(
            rows=int(len(df)),
            by_class={str(k): int(v) for k, v in by_cls.items()},
            by_tier={k: int(v) for k, v in df.quality_tier.value_counts().items()},
            by_class_tier={str(k): {kk: int(vv) for kk, vv in v.items()}
                           for k, v in pd.crosstab(df.cls6, df.quality_tier).T.to_dict().items()},
            by_eco={e: {str(c): int(((df.eco == e) & (df.cls6 == c)).sum()) for c in SIX}
                    for e in ECO_ORDER},
            by_prov={p: int(v) for p, v in df.prov.value_counts().items()},
            quota_rows=rows,
            quota_unmet=unmet,
            quota_pass=bool(not unmet),
            mixed_quality={str(c): dict(
                n=int((df.cls6 == c).sum()),
                from_expanded=int(((df.cls6 == c) & (df.src == 'mixed_expand')).sum()),
                from_legacy=int(((df.cls6 == c) & (df.src == 'forest6_v1')).sum()),
                M_A=int(((df.cls6 == c) & (df.mixed_tier == 'A')).sum()),
                M_B=int(((df.cls6 == c) & (df.mixed_tier == 'B')).sum()),
                M_C=int(((df.cls6 == c) & (df.mixed_tier == 'C')).sum()),
                A_from_expanded=int(((df.cls6 == c) & (df.mixed_tier == 'A') &
                                     (df.src == 'mixed_expand')).sum()),
                has_mixedcode=int(((df.cls6 == c) & (df.has_mixedcode == 1)).sum()),
                v30_is_mixed=int(((df.cls6 == c) & (df.v30_is_mixed == 1)).sum()))
                for c in (45, 46)},
            hygiene=hygiene(df, ver),
            overlap=overlap_rates(df))

    stats['core'] = describe(core, 'core')
    stats['core_selection_log'] = core_log
    stats['full'] = describe(full, 'full')
    stats['full_selection_log'] = full_log
    stats['quota_rules'] = {e: {str(k): v for k, v in q.items()} for e, q in QUOTA.items()}
    stats['outputs'] = dict(mixed=OUT_MIX, core=OUT_CORE, full=OUT_FULL,
                            south_needle=OUT_SN, stats=OUT_STATS)
    stats['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(stats, open(OUT_STATS, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    emit('写 f7_stats.json｜core %d / full %d｜配额 core %s full %s'
         % (len(core), len(full), stats['core']['quota_pass'], stats['full']['quota_pass']))
    for ver in ('core', 'full'):
        if stats[ver]['quota_unmet']:
            emit('  %s 未达标：%s' % (ver, ', '.join(
                '%s/%s %d<%d' % (u['eco'], u['cls6'], u['n'], u['quota'])
                for u in stats[ver]['quota_unmet'])))
    emit('完成 %.1f min' % ((time.time() - t0) / 60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fetch', action='store_true')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--acct', default=None)
    ap.add_argument('--core-legacy', choices=['A', 'all'], default='A',
                    help='core 的 45/46 既有点收哪些：A=只要既有 A 档（默认，保高纯度）；all=字面全部 9301')
    a = ap.parse_args()
    nodef = not (a.fetch or a.gate or a.assemble)
    if a.fetch or nodef:
        cmd_fetch(a.shard, a.nshard, a.acct)
    if a.gate or nodef:
        cmd_gate()
    if a.assemble or nodef:
        cmd_assemble(a.core_legacy)


if __name__ == '__main__':
    main()

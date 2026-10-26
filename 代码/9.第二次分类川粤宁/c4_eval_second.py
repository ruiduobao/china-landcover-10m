# -*- coding: utf-8 -*-
"""c4_eval_second.py — 第二次分类（川/粤/宁 40 瓦）分省评估：口径与第一次 eval_by_prov.py 完全一致

* 输入：留出点 F:/lc_work/v31_exp/data/yearly/eval_<year>_w{1..5}.parquet（row_id/lon/lat/class_new/src）；
  分类图默认本地模式：F:/lc_work/prod2nd_2023/rasters_batch/{tile}_10m.tif（清单 plan/tiles_secondrun.csv，
  --local 可逗号分隔替换/指定，如换成第一次交付栅格做冒烟）；省界=DataV GCJ-02（经 p5common 只读加载）
* 规则源：F:/lc_work/prod5p_2023/evalprep/eval_by_prov.py（第一次评估口径逐条照搬）；
  screening_decision §七.3（按 src 来源分层强制分报）；AGENTS 硬约束 4/5（留出池、<30 点只报合并级）
* 门槛：评估点逐 rid 排除 = ∩旧池 train2023_clean(row_id) ∪ ∩新池 train2023_v2（point_id 字符串键
  或 (lon,lat) 6 位位置键，两键都排）∪ ∩错位清单；两池交叉计数显式打印（含 0）；
  类码 0/>250 计 miss 不计指标；n<30 类只作合并级参考；GEE 用量记账落 prod2nd/logs（只读纪律：不写 prod5p/v31_exp）
* 输出：F:/lc_work/prod2nd_2023/reports/eval2/eval2_<省简称>_<year>.json/.md
        F:/lc_work/prod2nd_2023/reports/eval2/compare_run1_run2_<省简称>.md（--compare）
        F:/lc_work/prod2nd_2023/reports/eval2/selftest_<省简称>.json（--self-test）
* 用法：
    python 代码/9.第二次分类川粤宁/c4_eval_second.py --prov 粤 --compare          # 生产就绪后正式跑
    python 代码/9.第二次分类川粤宁/c4_eval_second.py --prov all --year 2023 --windows w1,w2,w3,w4,w5
    python 代码/9.第二次分类川粤宁/c4_eval_second.py --self-test --local <tif,...> [--self-test-ref <tif,...>]
    python 代码/9.第二次分类川粤宁/c4_eval_second.py --compare-only --prov 粤
"""
import os, sys, io, json, time, argparse

sys.dont_write_bytecode = True       # 只读纪律：不往 p5common / v31_common 目录写 __pycache__
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')    # p5common：省界/别名/分层口径/代理
sys.path.insert(0, r'F:/lc_work/v31_exp/code')          # v31_common：metrics/to_v31/to_macro
import p5common as PC
import v31_common as VC
import numpy as np
import pandas as pd
import rasterio
from shapely import contains_xy, prepare

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
ROOT1 = r'F:/lc_work/prod5p_2023'                     # 第一次（只读）
ROOT2 = r'F:/lc_work/prod2nd_2023'                    # 第二次
EVAL_DIR = r'F:/lc_work/v31_exp/data/yearly'          # 留出点（只读）
PLAN2 = os.path.join(ROOT2, 'plan', 'tiles_secondrun.csv')
RASTERS2 = os.path.join(ROOT2, 'rasters_batch')
POOL1 = os.path.join(ROOT1, 'data', 'train2023_clean.parquet')   # 第一次训练池（双保险排除）
POOL2 = os.path.join(ROOT2, 'data', 'train2023_v2.parquet')      # 第二次训练池（无 row_id）
MISALIGNED = os.path.join(PROJ, '数据/本地处理/全国清洗训练/错位样本剔除清单.parquet')
RESULTS = os.path.join(ROOT2, 'reports', 'eval2')
LOGS2 = os.path.join(ROOT2, 'logs')
RUN1_DIRS = [os.path.join(ROOT1, 'evalprep', 'results'),
             r'D:/生产_2023五省/成品/评估']              # 第一次结果的已知落盘处（探测用）
PROV2 = ['川', '粤', '宁']                              # 第二次 40 瓦覆盖的省（plan 口径）
FCS10_TAG = 'fcs10系(src以glc_fcs10开头)'
NONFCS_TAG = '非fcs10系'


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def eecu_log(tool, acct, op, quantity, est_eecu_h, note=''):
    """GEE 用量台账落 prod2nd/logs（**不**用 PC.eecu_log：那会写 prod5p_2023）。"""
    os.makedirs(LOGS2, exist_ok=True)
    fp = os.path.join(LOGS2, 'eecu_tooling_2nd.csv')
    new = not os.path.exists(fp)
    import csv
    with open(fp, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['time', 'tool', 'acct', 'op', 'quantity', 'est_eecu_h', 'note'])
        w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), tool, acct, op, quantity,
                    '%.4f' % est_eecu_h, note])


def parse_args():
    ap = argparse.ArgumentParser(description='第二次分类·分省评估（口径同第一次 eval_by_prov.py）')
    ap.add_argument('--prov', default='all', help='省简称/全名（川/粤/宁）或 all 或逗号分隔')
    ap.add_argument('--local', default='', help='本地分类栅格（逗号分隔）；空=按 plan/tiles_secondrun.csv 自动组盘')
    ap.add_argument('--asset', default='', help='（可选）GEE 资产，逗号分隔；第二次目前走本地模式')
    ap.add_argument('--acct', default='', help='读 --asset 的 GEE 账号')
    ap.add_argument('--year', type=int, default=2023)
    ap.add_argument('--windows', default='w1,w2,w3,w4,w5')
    ap.add_argument('--chunk', type=int, default=4000, help='GEE 逐点直查分块（--asset 模式）')
    ap.add_argument('--self-test', action='store_true',
                    help='本地采样冒烟自检：随机点走 sample_local 路径 vs 独立仿射读数；有 --self-test-ref 再比一致率')
    ap.add_argument('--self-test-n', type=int, default=400)
    ap.add_argument('--self-test-ref', default='', help='（可选）参考栅格（逗号分隔），如第一次交付 tif')
    ap.add_argument('--compare', action='store_true', help='评估后生成第一次 vs 第二次对比表')
    ap.add_argument('--compare-only', action='store_true', help='跳过评估，只用已有第二次 json 生成对比表')
    ap.add_argument('--pool2-id-key', default='both', choices=['both', 'pos', 'id'],
                    help='新池 train2023_v2 排除键：both=point_id 键∪位置键（默认）｜pos=只用位置键｜id=只用 point_id 键')
    ap.add_argument('--out-suffix', default='',
                    help='输出文件名后缀（冒烟留痕用，如 _smoke；正式跑留空）')
    return ap.parse_args()


# ---------------- 第二次瓦片计划（plan/tiles_secondrun.csv，字段同第一次 tiles_frozen.csv） ----------------
def read_plan2():
    import csv as _csv
    rows = []
    with open(PLAN2, encoding='utf-8') as f:
        for r in _csv.DictReader(f):
            rows.append(dict(tile=r['tile'], ix=int(r['ix']), iy=int(r['iy']),
                             x0=float(r['x0']), y0=float(r['y0']),
                             x1=float(r['x1']), y1=float(r['y1']),
                             land_frac=float(r['land_frac']), prov=r['prov']))
    return rows


def tiles_needed(df, plan_rows):
    """留出点 → 所依赖的瓦片集合（与第一次同口径：这些瓦片就绪才能完成该省评估）。"""
    need = {}
    for _, r in df.iterrows():
        for p in plan_rows:
            if p['x0'] - 1e-9 <= r.lon <= p['x1'] + 1e-9 and \
               p['y0'] - 1e-9 <= r.lat <= p['y1'] + 1e-9:
                need.setdefault(p['tile'], 0)
                need[p['tile']] += 1
                break
    return dict(sorted(need.items(), key=lambda x: -x[1]))


def locals_for_prov(prov_short, plan_rows):
    """第二次成品路径自动组盘；返回 (existing list, missing tile list)。"""
    want = [p['tile'] for p in plan_rows if p['prov'] == prov_short]
    have, miss = [], []
    for t in want:
        fp = os.path.join(RASTERS2, '%s_10m.tif' % t)
        (have if os.path.exists(fp) else miss).append(fp if os.path.exists(fp) else t)
    return have, miss


# ---------------- 归省（与第一次 eval_by_prov.assign_prov 完全一致） ----------------
def assign_prov(lons, lats, provs):
    """WGS84 点 → 省全名（无匹配=''）。GCJ 边界敏感点（距省界 <0.005°）另计。"""
    from shapely.geometry import Point
    geoms = {}
    for p in provs:
        full = PC.PROV_ALIAS.get(p, p)
        g = PC.prov_geom(full)
        prepare(g)
        geoms[full] = g
    out = np.array([''] * len(lons), dtype=object)
    mind = np.full(len(lons), 9e9)
    for full, g in geoms.items():
        hit = contains_xy(g, lons, lats)
        out[hit] = full
        if hit.any():
            from shapely import distance
            d = distance(g.boundary, [Point(x, y) for x, y in
                                      zip(lons[hit], lats[hit])])
            mind[hit] = np.minimum(mind[hit], d)
    return out, mind


# ---------------- 采样：本地 / GEE（与第一次同口径） ----------------
def sample_local(fps, lons, lats, chunk=200000):
    """多本地栅格按序补查（首中为准）；返回 (vals,None=未覆盖, 0)。类别栅格只允许直查，不做重采样。"""
    datasets = []
    for fp in [s.strip() for s in fps if s.strip()]:
        ds = rasterio.open(fp)
        assert ds.crs is not None and ds.crs.to_epsg() == 4326, '%s 非 EPSG:4326' % fp
        datasets.append((fp, ds))
    vals = [None] * len(lons)
    try:
        for a0 in range(0, len(lons), chunk):
            idx = list(range(a0, min(len(lons), a0 + chunk)))
            for fp, ds in datasets:
                todo = [i for i in idx if vals[i] is None]
                if not todo:
                    continue
                b = ds.bounds
                pts = {i: (lons[i], lats[i]) for i in todo
                       if b.left <= lons[i] <= b.right and b.bottom <= lats[i] <= b.top}
                if not pts:
                    continue
                for i, v in zip(pts.keys(), ds.sample(list(pts.values()))):
                    if v is not None and v[0] is not None:
                        vals[i] = int(v[0])
    finally:
        for _, ds in datasets:
            ds.close()
    return vals, 0


def asset_bounds(ee, aid):
    try:
        g = ee.Image(aid).geometry().bounds().getInfo()
        ring = g['coordinates'][0]
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        return min(xs), min(ys), max(xs), max(ys)
    except Exception as e:
        emit('asset_bounds %s 失败 %s' % (aid, str(e)[:80]))
        return None


def sample_gee(ee, assets, lons, lats, chunk):
    """逐点直查资产类码（sampleRegions→CSV）；返回 (vals list,None=miss, n_req)。与第一次同实现。"""
    vals = [None] * len(lons)
    n_req = 0
    import requests
    boxes = []
    for aid in assets:
        b = asset_bounds(ee, aid)
        boxes.append((aid, b))
        emit('  资产 %s bbox=%s' % (aid, '无几何' if b is None else
                                    tuple(round(v, 3) for v in b)))
    for a0 in range(0, len(lons), chunk):
        idx = list(range(a0, min(len(lons), a0 + chunk)))
        feats_all = {i: ee.Feature(ee.Geometry.Point([float(lons[i]), float(lats[i])]),
                                   {'i': i}) for i in idx}
        for aid, b in boxes:
            got = [i for i in idx
                   if vals[i] is None and
                   (b is None or (b[0] <= lons[i] <= b[2] and b[1] <= lats[i] <= b[3]))]
            if not got:
                continue
            fc = ee.FeatureCollection([feats_all[i] for i in got])
            out = (ee.Image(aid).select('class').sampleRegions(
                collection=fc, properties=['i'], scale=10, tileScale=4))
            url = out.getDownloadURL(filetype='csv', selectors=['i', 'class'])
            r = requests.get(url, proxies=PC.get_proxy(), timeout=300)
            r.raise_for_status()
            n_req += 1
            df = pd.read_csv(io.StringIO(r.text))
            for _, row in df.iterrows():
                vals[int(row['i'])] = int(row['class'])
    return vals, n_req


# ---------------- 本地冒烟自检（第二次无资产；替代第一次的 asset vs tif 路径校验） ----------------
def self_test_local(fps, n, rng, ref_fps=''):
    """随机点：① 走 sample_local（评估同路径）② 独立仿射读数（手算 row/col 后 1×1 窗读）
    独立读数与 sample_local 必须 100% 一致，否则说明采样路径/坐标系有问题。
    给了 --self-test-ref 再算 与参考栅格 的可比一致率（等同第一次 --self-test 口径）。"""
    from rasterio.windows import Window
    with rasterio.open(fps[0]) as ds:
        b = ds.bounds
    lons = rng.uniform(b.left + 0.02, b.right - 0.02, size=n)
    lats = rng.uniform(b.bottom + 0.02, b.top - 0.02, size=n)
    vals, _ = sample_local(fps, lons, lats)
    cover = sum(1 for v in vals if v is not None)
    # 独立读数（显式仿射逆变换 + 1×1 窗），首个命中为准
    import warnings
    manual = [None] * n
    for fp in fps:
        with rasterio.open(fp) as ds:
            T = ds.transform
            for i in range(n):
                if manual[i] is not None:
                    continue
                col = int((lons[i] - T.c) / T.a)
                row = int((lats[i] - T.f) / T.e)
                if 0 <= row < ds.height and 0 <= col < ds.width:
                    with warnings.catch_warnings():   # rasterio+numpy2.5 的 shape 弃用告警（只影响 1×1 读）
                        warnings.simplefilter('ignore', DeprecationWarning)
                        v = ds.read(1, window=Window(col, row, 1, 1))
                    if v.size:
                        manual[i] = int(v[0, 0])
    ok = bad = miss = 0
    for a, m in zip(vals, manual):
        if a is None or m is None:
            miss += 1
        elif a == m:
            ok += 1
        else:
            bad += 1
    res = dict(n=n, cover=int(cover), manual_agree=ok, manual_bad=bad,
               manual_miss=miss, manual_rate=round(ok / max(1, ok + bad), 4))
    emit('本地自检①：sample_local vs 独立仿射读数 → 覆盖 %d/%d，一致 %d / 不一致 %d / 不可比 %d（一致率 %.4f）'
         % (cover, n, ok, bad, miss, res['manual_rate']))
    # 构成直方图（类码 top10，人工核对是否落在 24 类码表内）
    hist = {}
    for v in vals:
        if v is not None:
            hist[int(v)] = hist.get(int(v), 0) + 1
    top = sorted(hist.items(), key=lambda x: -x[1])[:10]
    res['hist_top10'] = [[int(k), int(v)] for k, v in top]
    emit('类码构成 top10（类码:点数）：%s' % ', '.join('%d:%d' % (k, v) for k, v in top))
    names = VC.V31_NAMES()
    emit('  其中非 24 类码（0/调色板外）计数：%d；>250：%d' % (
        sum(c for c, v in hist.items() if c not in [int(k) for k in names]),
        sum(v for c, v in hist.items() if c > 250)))
    if ref_fps:
        rvals, _ = sample_local(ref_fps, lons, lats)
        r_ok = r_bad = r_miss = 0
        for lv, sv in zip(vals, rvals):
            if sv is None or sv <= 0 or sv > 250:
                r_miss += 1
            elif (lv > 250 and sv > 250) or (lv is not None and lv <= 250 and lv == sv):
                r_ok += 1
            else:
                r_bad += 1
        res['ref_n'] = len(ref_fps)
        res['ref_agree'] = r_ok
        res['ref_bad'] = r_bad
        res['ref_miss'] = r_miss
        res['ref_rate'] = round(r_ok / max(1, r_ok + r_bad), 4)
        emit('本地自检②：本地 tif vs 参考栅格 → 一致 %d / 不一致 %d / 参考缺失 %d（一致率 %.4f）'
             % (r_ok, r_bad, r_miss, res['ref_rate']))
    return res


# ---------------- 排除清单（双池 + 错位） ----------------
def pos_key(lon, lat, nd=6):
    """(lon,lat) 6 位位置键 → int64（无碰撞：偏移后各 4e8 窗口）。"""
    import numpy as np
    a = np.round(np.asarray(lon, dtype=float) * 10 ** nd).astype('int64') + 200_000_000
    b = np.round(np.asarray(lat, dtype=float) * 10 ** nd).astype('int64') + 100_000_000
    return a * 400_000_000 + b


def build_exclusions(ev, pool2_keys='both'):
    """评估点防污染排除（双池 + 错位）。

    键的含义（实测坑：train2023_v2 的 point_id **不唯一**——121,992 个 id 各对应两个
    相距可达数百公里的不同点，且这些 id 全部也在旧池 row_id 空间里，故 point_id 单键会
    产生假命中；6 位位置键才是新池的可靠键，两键并用 + 差额显式报告）：
      ① 旧池 train2023_clean：row_id（与第一次口径完全一致）
      ② 新池 train2023_v2：point_id 字符串键（对留出点 row_id）+ (lon,lat) 6 位位置键
    pool2_keys: both（默认，两键都排）| pos（只用位置键）| id（只用 point_id 键）
    返回 (mask_keep, detail)：所有交叉计数显式给出（含 0），pid_only 附假命中证据。
    """
    rid_int = ev.row_id.astype('int64').to_numpy()
    rid_str = np.array([str(int(v)) for v in rid_int], dtype=object)
    ev_pos = pos_key(ev.lon.to_numpy(), ev.lat.to_numpy())
    detail = {}
    # ① 旧池 train2023_clean：row_id（主键，第一次口径）
    p1 = pd.read_parquet(POOL1, columns=['row_id', 'lon', 'lat'])
    m1 = np.isin(rid_int, np.unique(p1.row_id.astype('int64').to_numpy()))
    detail['excl_v1_rid'] = int(m1.sum())
    # 旧池位置键（诊断用：第一次口径未排，仅上报，不扩排除）
    detail['excl_v1_pos_diag'] = int(np.isin(
        ev_pos, pos_key(p1.lon.to_numpy(), p1.lat.to_numpy())).sum())
    del p1
    # ② 新池 train2023_v2
    p2 = pd.read_parquet(POOL2, columns=['point_id', 'lon', 'lat'])
    detail['n_pool2'] = int(len(p2))
    p2_pid = p2.point_id.astype(str)
    p2_nn = pd.to_numeric(p2_pid, errors='coerce').isna().to_numpy()
    # 键①：point_id 字符串键（留出点 row_id 字符串 ∈ 池 point_id 集合）
    pid_pool_set = set(p2_pid.tolist())
    m2id = np.array([s in pid_pool_set for s in rid_str])
    detail['excl_v2_pid'] = int(m2id.sum())
    detail['n_pool2_pid_dup_rows'] = int(len(p2) - p2_pid.nunique())
    detail['n_pool2_nonnum_id_rows'] = int(p2_nn.sum())
    # 键②：6 位位置键（按评估点 lon/lat bbox 预筛全池后求交）
    lo0, lo1 = float(ev.lon.min()) - 0.001, float(ev.lon.max()) + 0.001
    la0, la1 = float(ev.lat.min()) - 0.001, float(ev.lat.max()) + 0.001
    mask_bb = ((p2.lon >= lo0) & (p2.lon <= lo1) & (p2.lat >= la0) & (p2.lat <= la1)).to_numpy()
    pool_keys = pos_key(p2.lon.to_numpy()[mask_bb], p2.lat.to_numpy()[mask_bb])
    m2pos = np.isin(ev_pos, pool_keys)
    detail['excl_v2_pos'] = int(m2pos.sum())
    detail['n_pool2_in_bbox'] = int(mask_bb.sum())
    # 两键交叉（显式打印，含 0；必须一致——不一致即为 point_id 重号假命中）
    detail['excl_v2_pid_only'] = int((m2id & ~m2pos).sum())
    detail['excl_v2_pos_only'] = int((m2pos & ~m2id).sum())
    detail['excl_v2_both'] = int((m2id & m2pos).sum())
    detail['pid_xor_pos_eval_pts'] = int((m2id ^ m2pos).sum())
    if detail['excl_v2_pid_only']:
        # 假命中证据：这些留出点 id 在池里的行数 + 与留出点的最小坐标距离（度）
        bad = ev[~m2pos & m2id]
        cnt, mind = [], []
        for _, r in bad.iterrows():
            sub = p2[p2_pid == str(int(r.row_id))]
            cnt.append(len(sub))
            if len(sub):
                d = np.min(np.hypot(sub.lon.to_numpy() - r.lon, sub.lat.to_numpy() - r.lat))
                mind.append(float(d))
        detail['pid_only_pool_rows_mean'] = round(float(np.mean(cnt)), 2) if cnt else 0.0
        detail['pid_only_min_dist_deg'] = round(float(np.min(mind)), 3) if mind else None
        detail['pid_only_min_dist_max_deg'] = round(float(np.max(mind)), 3) if mind else None
    else:
        detail['pid_only_pool_rows_mean'] = 0.0
        detail['pid_only_min_dist_deg'] = None
        detail['pid_only_min_dist_max_deg'] = None
    del p2
    # ③ 错位清单
    mis = pd.read_parquet(MISALIGNED, columns=['row_id'])
    mm = np.isin(rid_int, np.unique(mis.row_id.astype('int64').to_numpy()))
    detail['excl_misaligned'] = int(mm.sum())
    # ④ 新池排除掩码（按 pool2_keys 口径）+ 双池交叉
    if pool2_keys == 'pos':
        m2 = m2pos
    elif pool2_keys == 'id':
        m2 = m2id
    else:
        m2 = m2pos | m2id
    detail['pool2_keys_used'] = pool2_keys
    keep = ~(m1 | m2 | mm)
    detail['n_keep'] = int(keep.sum())
    detail['excl_total'] = int((~keep).sum())
    detail['excl_v1_only'] = int((m1 & ~(m2 | mm)).sum())
    detail['excl_v2_only'] = int((m2 & ~(m1 | mm)).sum())
    detail['excl_v1_v2_both'] = int((m1 & m2).sum())
    # 位置键口径下的评估池（用于诊断 rid/id 键的额外扣除量是否假命中）
    detail['n_keep_pos_only'] = int((~(m1 | m2pos | mm)).sum())
    return keep, detail


# ---------------- 指标与报告（与第一次 fmt_metrics 逐行一致） ----------------
def fmt_metrics(m, names, macro_names, t24, p24, label):
    m9 = VC.metrics(VC.to_macro(t24), VC.to_macro(p24))
    L = ['### %s（n=%d）' % (label, m['n']), '',
         '| 口径 | OA | macroF1 | 9大类 OA |', '|---|---|---|---|',
         '| 值 | %.4f | %.4f | %.4f |' % (m['OA'], m['macroF1'], m9['OA']), '',
         '逐类 F1（n=真值点数；**n<30 只作合并级参考，不作类级结论**）：', '',
         '| 类码 | 名称 | n | F1 | PA | UA |', '|---|---|---|---|---|---|']
    for c in sorted(int(k) for k in m['per']):
        d = m['per'][c]
        flag = ' ⚠n<30' if d['n'] < 30 else ''
        L.append('| %d | %s | %d%s | %.3f | %.3f | %.3f |' % (
            c, names.get(str(c), '?'), d['n'], flag, d['f1'], d['pa'], d['ua']))
    return L


def strata_block(t24, p24, isf):
    """三层指标（全点 / fcs10 / 非fcs10），口径与第一次一致。"""
    st = {}
    m_all = VC.metrics(t24, p24)
    st['all'] = dict(n=m_all['n'], OA=m_all['OA'], macroF1=m_all['macroF1'],
                     m9=VC.metrics(VC.to_macro(t24), VC.to_macro(p24))['OA'],
                     per=m_all['per'])
    st['all']['m9_per'] = VC.metrics(VC.to_macro(t24), VC.to_macro(p24))['per']
    if isf.sum() >= 10:
        m_f = VC.metrics(t24[isf], p24[isf])
        st['fcs10'] = dict(n=m_f['n'], OA=m_f['OA'], macroF1=m_f['macroF1'],
                           m9=VC.metrics(VC.to_macro(t24[isf]), VC.to_macro(p24[isf]))['OA'],
                           per=m_f['per'])
    if (~isf).sum() >= 10:
        m_n = VC.metrics(t24[~isf], p24[~isf])
        st['nonfcs10'] = dict(n=m_n['n'], OA=m_n['OA'], macroF1=m_n['macroF1'],
                              m9=VC.metrics(VC.to_macro(t24[~isf]), VC.to_macro(p24[~isf]))['OA'],
                              per=m_n['per'])
    return st


# ---------------- 对比：第一次 vs 第二次 ----------------
def probe_run1(tag, year):
    """探测第一次结果 json（短名/全名/别名 × 已知落盘处）。返回命中的所有候选路径。"""
    full = PC.PROV_ALIAS.get(tag, tag)
    cand_names = []
    for k, v in PC.PROV_ALIAS.items():
        if v == full and k not in cand_names:
            cand_names.append(k)
    if full not in cand_names:
        cand_names.append(full)
    if PC.PROV_SHORT.get(full, full) not in cand_names:
        cand_names.insert(0, PC.PROV_SHORT.get(full, full))
    hits = []
    for d in RUN1_DIRS:
        for nm in cand_names:
            fp = os.path.join(d, 'eval_%s_%d.json' % (nm, year))
            if os.path.exists(fp) and fp not in hits:
                hits.append(fp)
    return hits


def _g(strata, stratum, key):
    d = strata.get(stratum)
    return None if not d else d.get(key)


def compare_prov(tag, year, suffix=''):
    """第二次 vs 第一次 对比表：OA 系列 Δ + 逐类 F1 Δ（按类码联合，缺席标注）。"""
    fp2 = os.path.join(RESULTS, 'eval2_%s_%d%s.json' % (tag, year, suffix))
    if not os.path.exists(fp2):
        emit('对比跳过：缺第二次结果 %s（先跑评估或以 --prov %s 评估）' % (fp2, tag))
        return None
    hits = probe_run1(tag, year)
    if not hits:
        emit('对比跳过：%s 探测不到第一次结果 json（已探测目录 %s）' % (tag, RUN1_DIRS))
        return None
    d2 = VC.jload(fp2, {})
    d1 = VC.jload(hits[0], {})
    if len(hits) > 1:
        emit('第一次结果多候选（取首个）：%s' % ' ｜ '.join(hits))
    names = VC.V31_NAMES()
    rows = [('全点 OA', 'all', 'OA'), ('全点 macroF1', 'all', 'macroF1'),
            ('9 大类 OA', 'all', 'm9'),
            ('非fcs10 OA', 'nonfcs10', 'OA'), ('非fcs10 macroF1', 'nonfcs10', 'macroF1'),
            ('fcs10 OA', 'fcs10', 'OA'), ('fcs10 macroF1', 'fcs10', 'macroF1')]
    L = ['# 第一次 vs 第二次 · %s · %d（同一评估口径，仅分类图与训练池排除不同）' % (
        d2.get('prov', tag), year), '']
    L += ['',
          '- 第一次：`%s`' % hits[0],
          '  - 分类图：%s ｜ 训练池排除：train2023_clean(row_id) ｜ n_pool=%s，覆盖 %s%%' % (
              (d1.get('local') or d1.get('assets') or '—'), d1.get('n_pool'),
              d1.get('coverage_pct')),
          '- 第二次：`%s`' % fp2,
          '  - 分类图：%s（%d 幅）｜ 训练池排除：v2 双键 + 旧池 rid ｜ n_pool=%s，覆盖 %s%%' % (
              '、'.join(os.path.basename(x) for x in (d2.get('local_list') or [d2.get('local', '—')])),
              len(d2.get('local_list') or []), d2.get('n_pool'), d2.get('coverage_pct')),
          '',
          '> **Δ = 第二次 − 第一次**；n 不同时 Δ 含构成（覆盖面/评估池）效应，不能全归因模型。',
          '> 来源分层为强制口径（screening_decision §七.3）；全点 OA 受 FCS10 标注惯例主导。',
          '',
          '| 口径 | 第一次 n | 第一次 | 第二次 n | 第二次 | Δ |',
          '|---|---|---|---|---|---|']
    for label, s, k in rows:
        n1, n2 = _g(d1.get('strata', {}), s, 'n'), _g(d2.get('strata', {}), s, 'n')
        v1, v2 = _g(d1.get('strata', {}), s, k), _g(d2.get('strata', {}), s, k)
        if v1 is None or v2 is None:
            L.append('| %s | %s | %s | %s | %s | — |' % (
                label, n1 if n1 is not None else '—', '%.4f' % v1 if v1 is not None else '缺席',
                n2 if n2 is not None else '—', '%.4f' % v2 if v2 is not None else '缺席'))
        else:
            dl = '%+.4f' % (v2 - v1)
            warn = '' if n1 == n2 else ' ⚠n不等'
            L.append('| %s | %d | %.4f | %d | %.4f | %s%s |' % (label, n1, v1, n2, v2, dl, warn))
    # 逐类 F1 Δ（24 类，按类码联合）
    per1 = (d1.get('strata', {}).get('all', {}) or {}).get('per', {}) or {}
    per2 = (d2.get('strata', {}).get('all', {}) or {}).get('per', {}) or {}
    ks = sorted(set(int(k) for k in per1) | set(int(k) for k in per2))
    L += ['', '### 逐类 F1 Δ（24 类口径，按类码联合；缺席类标注）', '',
          '| 类码 | 名称 | 第一次 n | 第一次 F1 | 第二次 n | 第二次 F1 | ΔF1 | 备注 |',
          '|---|---|---|---|---|---|---|---|']
    for c in ks:
        a, b = per1.get(str(c)), per2.get(str(c))
        note = []
        if a is None:
            note.append('第一次缺席（该类真值点为 0 或被合并）')
        if b is None:
            note.append('第二次缺席')
        if (a and a.get('n', 0) < 30) and (b and b.get('n', 0) < 30):
            note.append('双侧 n<30 仅合并级参考')
        elif (a and a.get('n', 0) < 30) or (b and b.get('n', 0) < 30):
            note.append('⚠一侧 n<30')
        if a and b:
            df1 = b['f1'] - a['f1']
            L.append('| %d | %s | %d | %.3f | %d | %.3f | %+.3f | %s |' % (
                c, names.get(str(c), '?'), a['n'], a['f1'], b['n'], b['f1'], df1,
                '；'.join(note)))
        else:
            L.append('| %d | %s | %s | %s | %s | %s | — | %s |' % (
                c, names.get(str(c), '?'),
                a['n'] if a else '—', '%.3f' % a['f1'] if a else '—',
                b['n'] if b else '—', '%.3f' % b['f1'] if b else '—', '；'.join(note)))
    # 差异解读提示（基于实测值，避免口径漂移）
    oa1, oa2 = _g(d1.get('strata', {}), 'all', 'OA'), _g(d2.get('strata', {}), 'all', 'OA')
    if oa1 is not None and oa2 is not None:
        L += ['', '### 读数提示', '']
        L.append('- 全点 OA %.4f → %.4f（Δ %+.4f）；若第二次覆盖率低（<100%%），该 Δ 只代表已覆盖子区。'
                 % (oa1, oa2, oa2 - oa1))
        n1f, n2f = _g(d1.get('strata', {}), 'nonfcs10', 'n'), _g(d2.get('strata', {}), 'nonfcs10', 'n')
        if n1f != n2f:
            L.append('- 非 fcs10 层 n 不等（%s → %s）：评估池构成有变，分层 Δ 需同 n 对照后再引用。' % (n1f, n2f))
        miss2 = d2.get('tiles_missing') or []
        if miss2:
            L.append('- ⚠ 第二次缺瓦片 %d 幅（%s）：覆盖率未满，属**部分覆盖口径**。' % (
                len(miss2), ', '.join(miss2[:12])))
        src1 = str(d1.get('local') or d1.get('assets') or '')
        src2 = '、'.join(os.path.basename(x) for x in (d2.get('local_list') or [])) or \
            str(d2.get('local') or '')
        if src1 and src2 and src1 != src2:
            L.append('- ⚠ 两次分类图来源形态不同（第一次：`%s`；第二次：%s）——实测同一批瓦片与交付合幅'
                     '成品在广东有 117/2912 点取值不同（类码 9↔10、1↔9、3↔1，且远离瓦片边界，非拼接/重采样'
                     '伪影）→ 瓦片口径与合幅口径本身可差 ±0.4pp；若要严格同形对比，等第二次分省/合幅'
                     '裁剪产物就绪后用 --local 换成同一形态重跑。' % (
                         os.path.basename(src1), src2 if src2 else '—'))
        L.append('- 口径纪律：单次种子噪声 ±1.5 pp、<2 pp 的单次差异不可信（AGENTS 硬约束 12）。')
    fp = os.path.join(RESULTS, 'compare_run1_run2_%s%s.md' % (tag, suffix))
    with open(fp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    emit('→ %s' % fp)
    return fp


# ---------------- 主流程 ----------------
def main():
    a = parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    names = VC.V31_NAMES()
    macro_names = VC.MACRO_NAMES()
    plan_rows = read_plan2()
    # 省列表：短名（川/粤/宁）
    if a.prov == 'all':
        tags = list(PROV2)
    else:
        tags = []
        for p in [s.strip() for s in a.prov.split(',') if s.strip()]:
            full = PC.PROV_ALIAS.get(p, p)
            tags.append(PC.PROV_SHORT.get(full, full))
    tags = [t for t in tags if t]
    if a.compare_only:
        for t in tags:
            compare_prov(t, a.year, a.out_suffix)
        return
    if a.asset and not a.acct:
        raise SystemExit('--asset 模式必须给 --acct（GEE 账号，如 e5h08k）')
    emit('第二次分类评估：省=%s ｜ year=%d ｜ windows=%s ｜ 模式=%s' % (
        ','.join(tags), a.year, a.windows, 'asset' if a.asset else 'local'))

    # 1) 留出点汇总（与第一次同口径）
    windows = [w.strip() for w in a.windows.split(',') if w.strip()]
    frames = []
    for w in windows:
        fp = os.path.join(EVAL_DIR, 'eval_%d_%s.parquet' % (a.year, w))
        if not os.path.exists(fp):
            emit('缺表跳过 %s' % fp)
            continue
        d = pd.read_parquet(fp, columns=['row_id', 'lon', 'lat', 'class_new', 'src'])
        d['win'] = w
        frames.append(d)
    assert frames, '无可用留出表'
    ev = pd.concat(frames, ignore_index=True)
    n0 = len(ev)

    # 2) 排除（双池 + 错位；交叉计数显式打印）
    keep, xd = build_exclusions(ev, a.pool2_id_key)
    emit('留出点 %d → 排除：∩旧池(train2023_clean) rid=%d ｜ ∩新池(train2023_v2) point_id 键=%d ｜ '
         '∩新池 6 位位置键=%d ｜ ∩错位清单=%d → 评估池（保留）%d' % (
             n0, xd['excl_v1_rid'], xd['excl_v2_pid'], xd['excl_v2_pos'],
             xd['excl_misaligned'], int(keep.sum())))
    emit('两池交叉核对（显式含 0）：仅旧池命中=%d ｜ 仅新池命中=%d ｜ 两池同时命中=%d ｜ '
         '新池键不一致=%d（point_id 独有 %d / 位置键独有 %d）｜ 新池 %d 点中 %d 落在评估点 bbox ｜ '
         '旧池位置键诊断=%d ｜ 新池 id 重号行=%d、非数字 id 行=%d ｜ 生效新池键=%s（位置键口径评估池=%d）' % (
             xd['excl_v1_only'], xd['excl_v2_only'], xd['excl_v1_v2_both'],
             xd['pid_xor_pos_eval_pts'], xd['excl_v2_pid_only'], xd['excl_v2_pos_only'],
             xd['n_pool2'], xd['n_pool2_in_bbox'], xd['excl_v1_pos_diag'],
             xd['n_pool2_pid_dup_rows'], xd['n_pool2_nonnum_id_rows'],
             xd['pool2_keys_used'], xd['n_keep_pos_only']))
    if xd['excl_v2_pid'] == 0 and xd['excl_v2_pos'] == 0:
        emit('说明：新池与留出点交叉为 0（两键均为 0，符合"留出点未进训练池"预期）')
    if xd['excl_v2_pid_only']:
        emit('⚠ point_id 键独有命中 %d 点＝假命中风险（train2023_v2 point_id 重号：同 id 两行坐标'
             '相差可达数十度，最小距离 %.3f–%.3f°）；位置键口径保留 %d 点。'
             '如需与位置键口径对照，用 --pool2-id-key pos 重跑。' % (
                 xd['excl_v2_pid_only'], xd['pid_only_min_dist_deg'] or 0,
                 xd['pid_only_min_dist_max_deg'] or 0, xd['n_keep_pos_only']))
    ev = ev[keep].reset_index(drop=True)

    # 3) 归省（省界集合与第一次一致：五省，保证窗×省矩阵与边界口径不漂移）
    prov_assign, bdist = assign_prov(ev.lon.to_numpy(), ev.lat.to_numpy(), list(PC.PROV_SHORT))
    ev['prov'] = prov_assign
    ev['near_boundary'] = bdist < 0.005
    mat = ev.groupby(['win', 'prov']).size().unstack(fill_value=0)
    mat = mat.rename(columns={'': '省外'})
    emit('窗×省 交叠矩阵：\n%s' % mat.to_string())

    # 4) 自检（第二次无资产 → 本地模式冒烟）
    if a.self_test:
        fps = [s.strip() for s in (a.local or '').split(',') if s.strip()]
        if not fps:
            fps = []
            for t in tags:
                have, _ = locals_for_prov(t, plan_rows)
                fps += have
        if not fps:
            emit('自检失败：无可用本地栅格（--local 未给且 rasters_batch 为空）')
            return
        ref = [s.strip() for s in a.self_test_ref.split(',') if s.strip()]
        res = self_test_local(fps, a.self_test_n, np.random.RandomState(7), ref)
        res['local_list'] = fps
        res['ref_list'] = ref
        VC.jsave(res, os.path.join(RESULTS, 'selftest_%s.json' % tags[0]))
        emit('→ %s' % os.path.join(RESULTS, 'selftest_%s.json' % tags[0]))
        return

    # 5) 逐省评估
    for tag in tags:
        full = PC.PROV_ALIAS.get(tag, tag)
        ev_all = ev[ev.prov == full].reset_index(drop=True)
        if len(ev_all) == 0:
            emit('%s 无留出点（窗口未覆盖该省，如第一次的福建）' % tag)
            continue
        need = tiles_needed(ev_all, plan_rows)
        # 分类图
        fps = [s.strip() for s in (a.local or '').split(',') if s.strip()]
        miss_tiles = []
        if not fps and not a.asset:
            have, miss = locals_for_prov(tag, plan_rows)
            fps, miss_tiles = have, miss
            emit('%s 自动组盘：就绪 %d 幅 / 缺 %d 幅（缺=%s）' % (
                tag, len(have), len(miss), ','.join(miss[:12]) + ('...' if len(miss) > 12 else '')))
        assets = [s.strip() for s in a.asset.split(',') if s.strip()]
        if not fps and not assets:
            emit('%s 无可读分类图（第二次栅格未就绪或缺 %s）——跳过；先跑生产或 --local 指定' % (tag, RASTERS2))
            continue
        emit('%s 评估依赖瓦片 %d 幅（覆盖全部留出点需要）：%s' % (
            tag, len(need), json.dumps(need, ensure_ascii=False)))
        t0 = time.time()
        if assets:
            ee, pid = VC.ctx(a.acct)
            vals, n_req = sample_gee(ee, assets, ev_all.lon.to_numpy(),
                                     ev_all.lat.to_numpy(), a.chunk)
            eecu_log('c4_eval_second', a.acct, 'sampleRegions_holdout',
                     '%d pts x %d req' % (len(ev_all), n_req), round(0.0025 * max(1, n_req), 4),
                     '留出点直查（估算）')
        else:
            vals, n_req = sample_local(fps, ev_all.lon.to_numpy(), ev_all.lat.to_numpy())
        emit('%s 预测取值完成 %d 点（%.1f 分钟）' % (tag, len(ev_all), (time.time() - t0) / 60))

        ev_all = ev_all.assign(pred=vals)
        ev_all['miss'] = ev_all.pred.isna() | (ev_all.pred <= 0) | (ev_all.pred > 250)
        scored = ev_all[~ev_all.miss].reset_index(drop=True)
        cov = round(100 * len(scored) / max(1, len(ev_all)), 1)
        emit('覆盖口径：留出点 %d，落在已给图幅内 %d（%.1f%%）——指标只对覆盖内点计' % (
            len(ev_all), len(scored), cov))
        t24 = VC.to_v31(scored.class_new.to_numpy(int))
        p24 = scored.pred.to_numpy(int)
        isf = PC.is_fcs10(scored.src).to_numpy()
        st = strata_block(t24, p24, isf)

        out = dict(prov=full, prov_tag=tag, year=a.year, run='second',
                   assets=assets, local=(fps[0] if len(fps) == 1 else
                                         (','.join(fps) if fps else a.local)),
                   local_list=fps, n_local=len(fps), tiles_missing=miss_tiles,
                   train_pool_excl='v1_rid ∪ v2(%s) ∪ misaligned' % a.pool2_id_key,
                   n_holdout_raw=n0, excl_in_train=xd['excl_v1_rid'],
                   excl_misaligned=xd['excl_misaligned'], excl_detail=xd,
                   n_pool=len(ev_all), n_miss=int(ev_all.miss.sum()),
                   n_out_of_asset=int((ev_all.pred.isna()).sum()),
                   n_scored=len(scored), coverage_pct=cov,
                   near_boundary_pts=int(ev_all.near_boundary.sum()),
                   window_prov_matrix={w: {c: int(v) for c, v in row.items()}
                                       for w, row in mat.iterrows()},
                   tiles_needed=need, strata=st)
        fpj = os.path.join(RESULTS, 'eval2_%s_%d%s.json' % (tag, a.year, a.out_suffix))
        VC.jsave(out, fpj)

        # md 报告
        src_note = ('GEE 资产: %s' % ', '.join(assets)) if assets else \
            ('本地 %d 幅: %s' % (len(fps), ', '.join(os.path.basename(x) for x in fps)))
        m_all = VC.metrics(t24, p24)
        L = ['# 第二次分类 · 分省评估 · %s · %d（开发集口径，与第一次 eval_by_prov.py 一致）' % (full, a.year), '',
             '> 分类图：%s' % src_note,
             '> 留出池=五窗固定留出 ∩ 省界（逐 rid 排除**旧池 rid ∪ 新池 point_id 键 ∪ 新池 6 位位置键** ∪ 错位清单）',
             '> **来源分层为强制口径**（screening_decision §七.3）：全点 OA 会偏向 FCS10 标注惯例',
             '> 类码 0/缺失计 miss 不计指标；n<30 类只作合并级参考；GCJ-02 边界 %d 点敏感（距省界<0.005°，归属可能翻转）' %
             out['near_boundary_pts'], '',
             '| 步骤 | 点数 |', '|---|---|',
             '| 五窗留出原始 | %d |' % n0,
             '| 排除 ∩旧池(train2023_clean) rid | %d |' % xd['excl_v1_rid'],
             '| 排除 ∩新池(train2023_v2) point_id 键 | %d |' % xd['excl_v2_pid'],
             '| 排除 ∩新池(train2023_v2) 6 位位置键 | %d |' % xd['excl_v2_pos'],
             '| 排除 ∩错位清单 | %d |' % xd['excl_misaligned'],
             '| %s 评估池 | %d |' % (full, len(ev_all)),
             '| 其中 miss/出图幅 | %d |' % out['n_miss'],
             '| 计入指标（覆盖 %.1f%%） | %d |' % (cov, len(scored)), '',
             '**覆盖该省全部留出点所需瓦片（就绪前无法完成完整评估）**：%s' %
             (json.dumps(need, ensure_ascii=False) if need else '—'), '']
        if miss_tiles:
            L.append('⚠ 第二次缺瓦片 %d 幅：%s（本次为部分覆盖口径）' % (
                len(miss_tiles), ', '.join(miss_tiles[:16])))
            L.append('')
        L.append('两池交叉核对（显式含 0）：仅旧池命中 %d ｜ 仅新池(生效键 %s)命中 %d ｜ 两池同时命中 %d ｜ '
                 '新池 point_id 键 %d 点 / 位置键 %d 点（两键不一致 %d = point_id 重号假命中风险 %d + '
                 '位置键独有 %d）' % (
                     xd['excl_v1_only'], xd['pool2_keys_used'], xd['excl_v2_only'],
                     xd['excl_v1_v2_both'], xd['excl_v2_pid'], xd['excl_v2_pos'],
                     xd['pid_xor_pos_eval_pts'], xd['excl_v2_pid_only'], xd['excl_v2_pos_only']))
        if xd['excl_v2_pid_only']:
            L.append('')
            L.append('⚠ **point_id 键假命中**：train2023_v2 有 %d 行 point_id 重号（一个 id 对应两个'
                     '不同坐标的点，本次命中点两行相距 %.3f–%.3f°）；单靠 point_id 键会多扣 %d 点。'
                     '位置键口径评估池 = %d（本报告生效口径=%s）。' % (
                         xd['n_pool2_pid_dup_rows'], xd['pid_only_min_dist_deg'] or 0,
                         xd['pid_only_min_dist_max_deg'] or 0, xd['excl_v2_pid_only'],
                         xd['n_keep_pos_only'], xd['pool2_keys_used']))
        L.append('')
        if 'fcs10' in st:
            L += fmt_metrics(st['fcs10'], names, macro_names, t24[isf], p24[isf], FCS10_TAG)
            L += ['']
        if 'nonfcs10' in st:
            L += fmt_metrics(st['nonfcs10'], names, macro_names, t24[~isf], p24[~isf], NONFCS_TAG)
            L += ['']
        L += fmt_metrics(m_all, names, macro_names, t24, p24, '全点（仅供参考，受 fcs10 惯例主导）')
        L += ['']
        if cov < 100:
            L.append('⚠ **部分覆盖口径**：仅 %.1f%% 留出点落在已就绪图幅内，OA/逐类 F1 只代表'
                     '已覆盖子区，不等于全省口径；补齐依赖瓦片后须重跑。' % cov)
        if len(scored) and 0.70 <= m_all['OA'] <= 0.95:
            L.append('合理性：全点 OA %.4f 在预期带（0.70–0.95）内。' % m_all['OA'])
        elif len(scored):
            L.append('⚠ 全点 OA %.4f 超出预期带（0.70–0.95），需查：图幅/点错位、分层构成、miss 集中类。'
                     % m_all['OA'])
        fpm = os.path.join(RESULTS, 'eval2_%s_%d%s.md' % (tag, a.year, a.out_suffix))
        with open(fpm, 'w', encoding='utf-8') as f:
            f.write('\n'.join(L) + '\n')
        emit('→ %s 与同名 .json' % fpm)
        for s, d in st.items():
            emit('  %-9s n=%-6d OA=%.4f mF1=%.4f m9OA=%.4f' % (
                s, d['n'], d['OA'], d['macroF1'], d['m9']))

        # 6) 对比第一次
        if a.compare or a.compare_only:
            compare_prov(tag, a.year, a.out_suffix)


if __name__ == '__main__':
    main()

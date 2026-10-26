# -*- coding: utf-8 -*-
"""t1_prep.py — T-SC 四川森林坡向受控分类小实验：范围/臂/参考/防泄漏准备（**不跑 RF**）

* 输入（全部只读）：
    - 二轮冻结池  F:/lc_work/prod2nd_2023/data/train2023_v2.parquet（cl 24 码 + lon/lat + A00..A63 + src）
    - 五窗开发参考 F:/lc_work/v31_exp/data/yearly/eval_2023_w2.parquet（w2 秦岭窗；class_new 30 码 + AEF）
    - 森林 strict 视图 F:/lc_work/v31_exp/data/forest_qc/out/forest_samples_v1_strict.parquet（class4 24 码）
    - 瓦片分区   F:/lc_work/prod2nd_2023/plan/tiles_secondrun.csv（2° 生产瓦 bbox）
    - 细 DEM     F:/lc_work/v31_exp/data/eco_gate/_dem/dem_T*.tif（SRTMGL1 60m，川 24 瓦，只读复用）
    - doc54 判读真值 F:/lc_work/v31_exp/data/m3/expA_judged.csv、expA2_judged.csv（开发用，非独立真值）
* 规则源：doc70 v1.0（20 分钟小区域实验规范）§2.1/§4.2/§4.3/§6.1/§6.2；doc69 §5.3/§8.2；
    v31_map.json 24 码、macro9.json 9 大类；坡向口径 asp∈[135,225]=阳坡、[315,360)∪[0,45)=阴坡
* 关键实现（防泄漏，doc70 §4.2/§6.1）：
    - 评价点（小框内 w2∩川）按 0.25° 空间块稳定哈希拆 dev(60%)/conf(40%)，同块不拆；
    - **训练框内落在任一评价块的点，一律从训练候选剔除；基线与候选同一规则**；
    - 训练候选 = 瓦片±2° 窗 ∩ 有 60m DEM 支持（否则 T-STRATA 不可比）；DEM 仅川 24 瓦 → 实际窗=相邻 DEM 瓦并集。
* 门槛：本脚本**无分类**，只做范围/臂/参考/防泄漏登记，产出可复算计数。
* 输出：F:/lc_work/improve_2023_pilot_20261015/terrain/out/
    t1_registration.json、t1_frames.csv、t1_arms_<frame>_seed<seed>.npz（供 t2_rf_eval.py 直接训练）
* 用法：python t1_prep.py [--frames F1,F2] [--pad 2.0] [--cap 20000] [--block 0.25] [--dev-frac 0.6]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True

WORK = r'F:/lc_work/improve_2023_pilot_20261015/terrain'
OUTD = os.path.join(WORK, 'out')
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
CODE9 = os.path.join(PROJ, '代码', '9.第二次分类川粤宁')
sys.path.insert(0, CODE9)
sys.path.insert(0, r'F:/lc_work/prod5p_2023/mosaic')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')

POOL2 = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
STRICT = r'F:/lc_work/v31_exp/data/forest_qc/out/forest_samples_v1_strict.parquet'
EVAL_W2 = r'F:/lc_work/v31_exp/data/yearly/eval_2023_w2.parquet'
TILES = r'F:/lc_work/prod2nd_2023/plan/tiles_secondrun.csv'
DEMD = r'F:/lc_work/v31_exp/data/eco_gate/_dem'
DOC54_A = r'F:/lc_work/v31_exp/data/m3/expA_judged.csv'
DOC54_A2 = r'F:/lc_work/v31_exp/data/m3/expA2_judged.csv'

FEATS = ['A%02d' % i for i in range(64)]
FOREST = (4, 5, 6, 7, 8)
ELEV_BANDS = [0, 500, 1000, 1500, 2500, 3500, 4500, 9999]   # 7 带（doc70 §4.3 高程带）

# ---- 候选小框（w2∩川，位于单一 DEM 瓦内；doc70 §2.1 小框优先 0.5–1°，由参考支持决定）----
CAND_FRAMES = {
    'F1': dict(tile='T1607', bbox=(105.5, 32.0, 106.5, 33.0)),   # n=418 森林189 ENF90/EBF66
    'F2': dict(tile='T1706', bbox=(107.0, 31.0, 108.0, 32.0)),   # n=206 森林120 ENF32/EBF66
    'F3': dict(tile='T1606', bbox=(106.0, 31.0, 107.0, 32.0)),   # n=194 森林 82
}


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def md5_of(fp):
    h = hashlib.md5()
    with open(fp, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def identity(fp):
    st = os.stat(fp)
    return {'path': fp, 'size': st.st_size,
            'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
            'md5': md5_of(fp)}


def code_hash():
    h = hashlib.md5()
    with open(os.path.abspath(__file__), 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------- 地形（米制换算，口径同 t2_pool_aspect.py / terrain REPORT） ----------------
def aspect_class(asp, slope):
    """0=阳坡(135-225) 1=阴坡(315-360∪0-45) 2=其它 3=平(slope<=5)。"""
    c = np.full(asp.shape, 2, np.int8)
    c[(asp >= 135) & (asp <= 225)] = 0
    c[(asp <= 45) | (asp >= 315)] = 1
    c[slope <= 5] = 3
    return c


def elev_band(el):
    return np.clip(np.digitize(el, ELEV_BANDS) - 1, 0, len(ELEV_BANDS) - 2).astype(np.int8)


class DEMIndex:
    def __init__(self, demd=DEMD):
        import rasterio
        self.idx = {}
        self.demd = demd
        for f in sorted(os.listdir(demd)):
            if f.startswith('dem_T') and f.endswith('.tif'):
                with rasterio.open(os.path.join(demd, f)) as s:
                    self.idx[f[4:-4]] = (s.bounds, s.transform, s.width, s.height)
        self._cache = {}

    def _read(self, tile):
        import rasterio
        if tile not in self._cache:
            with rasterio.open(os.path.join(self.demd, 'dem_%s.tif' % tile)) as s:
                self._cache[tile] = s.read(1)
        return self._cache[tile]

    def tiles_for_bbox(self, bbox):
        x0, y0, x1, y1 = bbox
        out = []
        for t, (b, _, _, _) in self.idx.items():
            if b.left < x1 and b.right > x0 and b.bottom < y1 and b.top > y0:
                out.append(t)
        return sorted(out)

    def sample_any(self, lon, lat):
        """在任一 DEM 瓦内取最近 cell → (elev, slope, asp)；瓦外 NaN。3×3 中心差分、米制。"""
        el_out = np.full(lon.shape, np.nan, np.float32)
        sl_out = np.full(lon.shape, np.nan, np.float32)
        as_out = np.full(lon.shape, np.nan, np.float32)
        for t, (b, tfm, W, H) in self.idx.items():
            m = (lon >= b.left) & (lon < b.right) & (lat > b.bottom) & (lat <= b.top) & ~np.isfinite(el_out)
            if not m.any():
                continue
            els = self._read(t)
            dlat, dlon = -tfm.e, tfm.a
            col = ((lon[m] - tfm.c) / dlon).astype(np.int64).clip(1, W - 2)
            row = ((tfm.f - lat[m]) / dlat).astype(np.int64).clip(1, H - 2)
            la = np.deg2rad(lat[m])
            m_lat = 111132.92 - 559.82 * np.cos(2 * la) + 1.175 * np.cos(4 * la)
            m_lon = 111412.84 * np.cos(la) - 93.5 * np.cos(3 * la)
            dzdx = (els[row, col + 1] - els[row, col - 1]).astype(np.float32) / 2.0 / (dlon * m_lon)
            dzdy = -(els[row + 1, col] - els[row - 1, col]).astype(np.float32) / 2.0 / (dlat * m_lat)
            el_out[m] = els[row, col].astype(np.float32)
            sl_out[m] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
            as_out[m] = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360.0) % 360.0
        return el_out, sl_out, as_out

    def frame_area_strata(self, bbox):
        """框内（跨 DEM 瓦）逐 cell 的 (aspect3 × elevband) 面积权重（cos(lat) 加权）。"""
        x0, y0, x1, y1 = bbox
        area = np.zeros((4, len(ELEV_BANDS) - 1), np.float64)
        for t in self.tiles_for_bbox(bbox):
            b, tfm, W, H = self.idx[t]
            els = self._read(t).astype(np.float32)
            lats = tfm.f - (-tfm.e) * (np.arange(H) + 0.5)
            la = np.deg2rad(lats)
            m_lat = 111132.92 - 559.82 * np.cos(2 * la) + 1.175 * np.cos(4 * la)
            m_lon = 111412.84 * np.cos(la) - 93.5 * np.cos(3 * la)
            sx = tfm.a * m_lon[:, None]
            sy = (-tfm.e) * m_lat[:, None]
            gE = np.gradient(els, axis=1) / sx
            gN = -np.gradient(els, axis=0) / sy
            slope = np.degrees(np.arctan(np.sqrt(gE * gE + gN * gN))).astype(np.float32)
            asp = ((np.degrees(np.arctan2(-gE, -gN)) + 360.0) % 360.0).astype(np.float32)
            lons = tfm.c + tfm.a * (np.arange(W) + 0.5)
            m = ((lons[None, :] >= x0) & (lons[None, :] < x1) &
                 (lats[:, None] >= y0) & (lats[:, None] < y1))
            if not m.any():
                continue
            a3 = aspect_class(asp[m], slope[m])
            eb = elev_band(els[m])
            w = np.cos(np.deg2rad(lats[:, None].repeat(W, 1)[m])) * (tfm.a * (-tfm.e))
            for i in range(4):
                for j in range(len(ELEV_BANDS) - 1):
                    area[i, j] += w[(a3 == i) & (eb == j)].sum()
        return area


# ---------------- 空间块（0.25° 固定，doc70 §6.1；稳定哈希分 dev/conf） ----------------
def block_key(lon, lat, bs):
    return (np.floor(np.asarray(lon) / bs).astype(np.int64),
            np.floor(np.asarray(lat) / bs).astype(np.int64))


def block_keys_str(lon, lat, bs):
    i, j = block_key(lon, lat, bs)
    return ['%d,%d' % (a, b) for a, b in zip(i, j)]


def block_split(keys_str, dev_frac):
    """按块稳定哈希（blake2b）排序，前 dev_frac 比例的**块**为 dev，其余 conf；同块不拆。"""
    ub = sorted(set(keys_str))
    h = {b: int.from_bytes(hashlib.blake2b(b.encode(), digest_size=8).digest(), 'big') for b in ub}
    ub.sort(key=lambda b: h[b])
    ndev = int(round(dev_frac * len(ub)))
    dev_blocks = set(ub[:ndev])
    return np.array([b in dev_blocks for b in keys_str])


# ---------------- 抽样 ----------------
def per_class_counts(lab, cap):
    """基线每类 count：n_k = min(count_k, round(count_k*cap/N))，总量≈cap（自然比例）。"""
    n = len(lab)
    cnt = {int(c): int(v) for c, v in zip(*np.unique(lab, return_counts=True))}
    return {c: min(v, int(round(v * cap / n))) for c, v in cnt.items()}


def select_baseline(lab, counts, seed):
    """类内均匀随机，每类取 counts[c] 个。返回选中索引（原始数组下标）。"""
    rng = np.random.RandomState(seed)
    keep = []
    for c, k in counts.items():
        idx = np.flatnonzero(lab == c)
        keep.append(idx if k >= len(idx) else rng.permutation(idx)[:k])
    out = np.concatenate(keep)
    out.sort()
    return out


def hamilton(weights, total):
    """最大余数法：把 total 按 weights 分配到 len(weights) 桶。"""
    w = np.asarray(weights, np.float64)
    if w.sum() <= 0 or total <= 0:
        return np.zeros(len(w), np.int64)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(np.int64)
    rem = int(total - base.sum())
    if rem > 0:
        for o in np.argsort(-(raw - base))[:rem]:
            base[o] += 1
    return base


def select_terrain(lab, counts, seed, a3, eb, frame_area):
    """T-STRATA（固定 1 种规则）：同每类 count；类内按 (aspect3×elevband) 分层，
    各层目标分配 ∝ 框内地形面积（frame_area，非类自身分布）；层内随机；不足额按盈余层容量补。
    保证每类恰取 k 个、无重复。"""
    rng = np.random.RandomState(seed + 101)
    neb = len(ELEV_BANDS) - 1
    strata = a3.astype(np.int64) * neb + eb.astype(np.int64)   # 0..27
    fw = frame_area.flatten()
    keep = []
    for c, k in counts.items():
        idx = np.flatnonzero(lab == c)
        if k >= len(idx):
            keep.append(idx)
            continue
        st = strata[idx]
        present = [int(s) for s in np.unique(st)]
        wt = np.zeros(len(fw), np.float64)
        for s in present:
            wt[s] = fw[s]
        if wt.sum() <= 0:
            for s in present:
                wt[s] = 1.0
        alloc = hamilton(wt, k)
        pools = {s: rng.permutation(idx[st == s]) for s in present}   # 每层打乱（不重复切片）
        taken = {s: min(int(alloc[s]), len(pools[s])) for s in present}
        chosen = [pools[s][:taken[s]] for s in present if taken[s] > 0]
        short = k - sum(taken.values())
        guard = 0
        while short > 0 and guard < 100:
            guard += 1
            left = {s: len(pools[s]) - taken[s] for s in present}
            tot = sum(v for v in left.values() if v > 0)
            if tot <= 0:
                break
            for s in sorted(left, key=lambda x: -left[x]):
                if short <= 0:
                    break
                if left[s] <= 0:
                    continue
                extra = min(left[s], max(1, int(round(short * left[s] / tot))))
                chosen.append(pools[s][taken[s]:taken[s] + extra])
                taken[s] += extra
                short -= extra
        cc = np.concatenate(chosen) if chosen else np.array([], np.int64)
        if len(cc) < k:   # 极端兜底：从该类剩余点随机补齐
            rest = np.setdiff1d(idx, cc, assume_unique=False)
            cc = np.concatenate([cc, rng.permutation(rest)[:k - len(cc)]])
        keep.append(cc[:k])
    out = np.concatenate(keep)
    out.sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='F1,F2')
    ap.add_argument('--pad', type=float, default=2.0, help='训练框=瓦片±pad（doc70 §2.1 = 2.0）')
    ap.add_argument('--cap', type=int, default=20000)
    ap.add_argument('--block', type=float, default=0.25)
    ap.add_argument('--dev-frac', type=float, default=0.6)
    ap.add_argument('--seeds', default='7,11,23')
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]

    import pandas as pd
    import pyarrow.parquet as pq
    import v31_common as VC
    import c4_eval_second as C4
    import p5common as PC

    reg = {'experiment_id': 'T-SC', 'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'script': os.path.abspath(__file__), 'code_hash': code_hash(),
           'spec': 'doc70 v1.0 §2.1/§4.2/§4.3/§6.1/§6.2；doc69 §5.3',
           'class_map': 'v31_map.json 24 码（新池 cl 已是 24 码；评价 class_new 经 to_v31）',
           'seeds': seeds, 'n_train_limit': a.cap, 'train_pad_deg': a.pad,
           'block_deg': a.block, 'dev_frac': a.dev_frac,
           'rf_params': dict(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000, n_jobs=6),
           'deadline_s': 1200, 'arms': ['baseline', 'T-STRATA', 'F-VIEW'],
           'inputs': {}}
    for nm, fp in (('pool2', POOL2), ('strict', STRICT), ('eval_w2', EVAL_W2),
                   ('tiles', TILES), ('doc54_A', DOC54_A), ('doc54_A2', DOC54_A2)):
        reg['inputs'][nm] = identity(fp)
    emit('输入身份已记录')

    dem = DEMIndex()
    reg['dem_tiles'] = sorted(dem.idx.keys())

    T = pd.read_csv(TILES)
    Tmap = {r.tile: (r.x0, r.y0, r.x1, r.y1) for r in T.itertuples()}

    # ---- 参考池：w2 ∩ 四川（开发参考，非独立真值）----
    e = pd.read_parquet(EVAL_W2, columns=['row_id', 'lon', 'lat', 'class_new', 'src'] + FEATS)
    pf, bdist = C4.assign_prov(e.lon.to_numpy(), e.lat.to_numpy(), list(PC.PROV_SHORT))
    e['prov'] = pf
    sc = e[e.prov == '四川省'].reset_index(drop=True)
    sc['t24'] = VC.to_v31(sc.class_new.to_numpy().astype(np.int64))
    sc['is_fcs10'] = PC.is_fcs10(sc.src).to_numpy()
    emit('w2∩川 开发参考点 n=%d（fcs10 %d / 非fcs10 %d）' % (
        len(sc), int(sc.is_fcs10.sum()), int((~sc.is_fcs10).sum())))

    # ---- 池 / strict 读入（只读）----
    t = pq.read_table(POOL2, columns=['lon', 'lat', 'cl', 'src'] + FEATS)
    plon = np.asarray(t['lon']); plat = np.asarray(t['lat']); pcl = np.asarray(t['cl']).astype(np.int64)
    X = np.empty((t.num_rows, 64), np.float32)
    for i in range(64):
        X[:, i] = t.column(FEATS[i]).to_numpy()
    del t
    emit('新池装载 n=%d' % len(pcl))

    ts = pq.read_table(STRICT, columns=['lon', 'lat', 'class4'] + FEATS)
    slon = np.asarray(ts['lon']); slat = np.asarray(ts['lat'])
    scl = np.asarray(ts['class4']).astype(np.int64)
    Xs = np.empty((ts.num_rows, 64), np.float32)
    for i in range(64):
        Xs[:, i] = ts.column(FEATS[i]).to_numpy()
    del ts
    emit('strict 森林视图 n=%d（类4/5/6/7）' % len(scl))

    # ---- doc54 参考（开发诊断）----
    a1 = pd.read_csv(DOC54_A); a2 = pd.read_csv(DOC54_A2)
    ref = pd.concat([a1[a1.region == 'w2'], a2[a2.region == 'w2']])
    ref = ref[ref.truth_code.isin([4, 5, 6, 7, 8])]
    reg['doc54_w2_forest'] = {
        'n': int(len(ref)),
        'by_class': {str(k): int(v) for k, v in ref.truth_code.value_counts().items()},
        'conf': {str(k): int(v) for k, v in ref.conf.value_counts().items()},
        'note': '视觉针/阔标签 conf 以低为主；doc54 针/阔 AUC 仅 0.724 → 仅开发诊断，非独立真值'}

    frames_out = []
    for fid in [s.strip() for s in a.frames.split(',') if s.strip()]:
        spec = CAND_FRAMES[fid]
        tile = spec['tile']; fbox = spec['bbox']
        x0, y0, x1, y1 = fbox
        # 评价集 = w2∩川∩框
        ie = np.flatnonzero((sc.lon.values >= x0) & (sc.lon.values < x1) &
                            (sc.lat.values >= y0) & (sc.lat.values < y1))
        ev = sc.iloc[ie].reset_index(drop=True)
        n_for = int(ev.t24.isin(FOREST).sum())
        ev['block'] = block_keys_str(ev.lon.values, ev.lat.values, a.block)
        ev['split'] = np.where(block_split(ev.block.tolist(), a.dev_frac), 'dev', 'conf')
        eval_blocks = set(ev.block.unique())

        # 训练框 = 瓦片 ± pad；候选 = 框内 ∩ 有 DEM 支持 ∩ 非评价块
        tx0, ty0, tx1, ty1 = Tmap[tile]
        wx0, wy0, wx1, wy1 = tx0 - a.pad, ty0 - a.pad, tx1 + a.pad, ty1 + a.pad
        iw = np.flatnonzero((plon >= wx0) & (plon < wx1) & (plat >= wy0) & (plat < wy1))
        wbk = block_keys_str(plon[iw], plat[iw], a.block)
        leak = np.array([k in eval_blocks for k in wbk])
        iw = iw[~leak]
        n_excl = int(leak.sum())
        el, sl, asp = dem.sample_any(plon[iw], plat[iw])
        ok = np.isfinite(el) & np.isfinite(sl) & np.isfinite(asp)
        iw = iw[ok]; el = el[ok]; sl = sl[ok]; asp = asp[ok]
        lab = pcl[iw]
        a3 = aspect_class(asp, sl); eb = elev_band(el)
        dem_tiles = dem.tiles_for_bbox((wx0, wy0, wx1, wy1))

        # 评价点地形
        eel, esl, easp = dem.sample_any(ev.lon.values, ev.lat.values)
        ev['elev'] = eel; ev['slope'] = esl; ev['asp'] = easp
        ev['a3'] = aspect_class(np.nan_to_num(easp, nan=0.0), np.nan_to_num(esl, nan=0.0))
        ev['dem_ok'] = np.isfinite(easp)

        frame_area = dem.frame_area_strata(fbox)

        # strict 森林候选（窗内、非泄漏块）——F-VIEW 公共支持
        sm = np.flatnonzero((slon >= wx0) & (slon < wx1) & (slat >= wy0) & (slat < wy1) &
                            np.isin(scl, [4, 5, 6, 7]))
        skey = block_keys_str(slon[sm], slat[sm], a.block)
        sm = sm[~np.array([k in eval_blocks for k in skey])]
        strict_avail = {int(c): int((scl[sm] == c).sum()) for c in (4, 5, 6, 7)}

        # 主比较：baseline vs T-STRATA 用**完整**每类 count（各≈20k，同一 count）
        counts_full = per_class_counts(lab, a.cap)
        # F-VIEW 独立 pair：forest 每类 count 降到 strict 公共上限；非森林不变；总 N 实际≤20k
        counts_common = dict(counts_full)
        capped = {}
        for c in (4, 5, 6, 7):
            if c in counts_common and counts_common[c] > strict_avail.get(c, 0):
                capped[str(c)] = [int(counts_common[c]), strict_avail.get(c, 0)]
                counts_common[c] = strict_avail.get(c, 0)
        n_full = sum(counts_full.values())
        n_common = sum(counts_common.values())
        reg_f = dict(frame=fid, tile=tile, frame_bbox=fbox, train_bbox=(wx0, wy0, wx1, wy1),
                     dem_tiles_in_window=dem_tiles,
                     n_pool_window_raw=int(len(iw) + n_excl), n_leak_excl=n_excl,
                     n_train_candidates=int(len(iw)),
                     n_eval_pts=int(len(ev)), n_eval_forest=n_for,
                     n_eval_dem_ok=int(ev.dem_ok.sum()),
                     n_eval_dev=int((ev.split == 'dev').sum()),
                     n_eval_conf=int((ev.split == 'conf').sum()),
                     n_eval_blocks=int(len(eval_blocks)),
                     n_eval_forest_dev=int(ev[(ev.split == 'dev')].t24.isin(FOREST).sum()),
                     n_eval_forest_conf=int(ev[(ev.split == 'conf')].t24.isin(FOREST).sum()),
                     n_eval_fcs10=int(ev.is_fcs10.sum()),
                     n_eval_nonfcs10=int((~ev.is_fcs10).sum()),
                     strict_avail=strict_avail, common_support_capped=capped,
                     class_counts_full=counts_full, n_train_full=n_full,
                     class_counts_common=counts_common, n_train_common=n_common,
                     eval_class_counts={str(k): int(v) for k, v in ev.t24.value_counts().items()},
                     frame_area_strata=frame_area.tolist())
        if reg_f['n_eval_conf'] < 30:
            reg_f['conf_support_warning'] = 'conf 点 %d <30 → 确认集支持不足，仅 dev 诊断（doc70 §6.3）' % reg_f['n_eval_conf']
        frames_out.append(reg_f)
        emit('%s(%s) 评价 %d（森林 %d｜dev %d/conf %d｜DEMok %d）｜候选窗 %d → 去泄漏 %d（剔 %d）｜'
             '主对 baseline/T-STRATA 各 %d 点；F-VIEW pair 公共支持 %d 点（降 %s）' % (
                 fid, tile, len(ev), n_for, reg_f['n_eval_dev'], reg_f['n_eval_conf'],
                 reg_f['n_eval_dem_ok'], reg_f['n_pool_window_raw'], len(iw), n_excl,
                 n_full, n_common, capped or '无'))

        for seed in seeds:
            ib = select_baseline(lab, counts_full, seed)            # baseline（完整 count）
            it = select_terrain(lab, counts_full, seed, a3, eb, frame_area)  # T-STRATA（同一 count）
            ibc = select_baseline(lab, counts_common, seed)         # baseline_common（F-VIEW 同量基线）
            selX, selY, selLon, selLat, short = [], [], [], [], {}
            rng = np.random.RandomState(seed + 202)
            for c, k in counts_common.items():
                if c in (4, 5, 6, 7):
                    cand = sm[scl[sm] == c]
                    if len(cand) >= k:
                        pick = rng.permutation(cand)[:k]
                        selX.append(Xs[pick]); selY.append(np.full(k, c))
                        selLon.append(slon[pick]); selLat.append(slat[pick])
                    else:
                        short[str(c)] = int(k - len(cand))
                        pool = ibc[lab[ibc] == c]
                        selX.append(X[pool]); selY.append(np.full(len(pool), c))
                        selLon.append(plon[iw][pool]); selLat.append(plat[iw][pool])
                else:
                    pool = ibc[lab[ibc] == c]
                    selX.append(X[pool]); selY.append(np.full(len(pool), c))
                    selLon.append(plon[iw][pool]); selLat.append(plat[iw][pool])
            fvLon = np.concatenate(selLon); fvLat = np.concatenate(selLat)
            _, fvsl, fvasp = dem.sample_any(fvLon, fvLat)
            fva3 = aspect_class(np.nan_to_num(fvasp, nan=0.0), np.nan_to_num(fvsl, nan=0.0))
            reg_f.setdefault('strict_short', {})['seed%d' % seed] = short
            np.savez_compressed(
                os.path.join(OUTD, 't1_arms_%s_seed%d.npz' % (fid, seed)),
                base_X=X[iw][ib], base_y=lab[ib], base_idx=ib,
                terr_X=X[iw][it], terr_y=lab[it], terr_idx=it,
                bc_X=X[iw][ibc], bc_y=lab[ibc], bc_idx=ibc,
                fv_X=np.concatenate(selX).astype(np.float32),
                fv_y=np.concatenate(selY).astype(np.int64),
                base_a3=a3[ib], terr_a3=a3[it], bc_a3=a3[ibc], fv_a3=fva3,
                a3=a3, eb=eb, asp=asp, elev=el, slope=sl,
                train_lon=plon[iw], train_lat=plat[iw],
                eval_X=ev[FEATS].to_numpy(np.float32),
                eval_lon=ev.lon.values, eval_lat=ev.lat.values, eval_y=ev.t24.values,
                eval_a3=ev.a3.values, eval_dem_ok=ev.dem_ok.values.astype(np.int8),
                eval_split=(ev.split.values == 'dev').astype(np.int8),
                eval_is_fcs10=ev.is_fcs10.values.astype(np.int8),
                eval_block=np.array(ev.block.tolist()),
                frame_area=frame_area)
        emit('  → t1_arms_%s_seed{%s}.npz 已写' % (fid, ','.join(map(str, seeds))))

    reg['frames'] = frames_out
    reg['reference_inventory'] = {
        'primary_dev_reference': 'w2∩川 五窗开发点（FCS10 系教师标签，多轮开发使用 → 非独立真值）',
        'independent_conifer_broadleaf': '无。doc54 A/A2 仅 48 个 w2 森林视觉点、conf 以低为主、针/阔 AUC 0.724 → 仅开发诊断',
        'phenology_label': 'amp2019/amp2023 只分常绿/落叶，不能判针/阔',
        'aspect_diag_npz': 'records_川.npz 无 AEF，不可当真值（仅地形/类码诊断）',
    }
    reg['gaps'] = [
        '无独立针/阔真值：所有"误差下降"均为开发参考口径，不判伪影已修',
        'w2 开发参考为 FCS10 系教师标签；OA 会被 fcs10 层机械主导，须分层报',
        'DEM 仅川 24 瓦（60m）：训练窗虽设瓦片±2°，实际候选=相邻 DEM 瓦并集，需明报',
        '本地 sklearn RF ≠ GEE smileRF；20k 抽样为统计复现，非 EE randomColumn 逐点复现',
        '评价点 w2∩川∩小框；训练候选已按 0.25° 评价块全剔泄漏（基线/候选同规则）',
    ]
    with open(os.path.join(OUTD, 't1_registration.json'), 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=1, default=float)
    pd.DataFrame([{k: v for k, v in r.items()
                   if k not in ('class_counts_full', 'class_counts_common',
                                'frame_area_strata', 'strict_short')}
                  for r in frames_out]).to_csv(os.path.join(OUTD, 't1_frames.csv'),
                                               index=False, encoding='utf-8-sig')
    emit('→ out/t1_registration.json + t1_frames.csv')


if __name__ == '__main__':
    main()

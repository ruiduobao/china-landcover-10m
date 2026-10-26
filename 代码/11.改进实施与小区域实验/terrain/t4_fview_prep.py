# -*- coding: utf-8 -*-
"""t4_fview_prep.py — F-VIEW 独立对（新空间框）准备：**不跑 RF**，供主 agent 批准

* 目的：doc70 §4.3 F-VIEW = 新机制（非 T-STRATA 同权重）；独立 common-support pair。
* 条件（主 agent 冻结）：
    - 训练 count = **strict 与旧全量（v2 池）共同支持** min(pool_count, strict_avail)，森林类降幅如实报；
    - **非森林完全同选择**（两臂共用同一非森林点）；
    - **不依预测挑 strict 规则**（strict 视图来源固定，见下"字段证据"）；
    - 用 **F1/F2 以外的新空间框**，confirm 块独立未看过；
    - 固定 3 seed：baseline_common vs strict_common。
* 输入（只读）：v2 池、strict 视图、eval_2023_w2、tiles_secondrun、DEM（同 t1_prep）
* 输出：out/t4_fview_registration.json、out/t4_fview_frames.csv、out/t4_fview_arms_<F>_seed<seed>.npz
* 用法：python t4_fview_prep.py [--frames G1] [--pad 2.0] [--cap 20000] [--block 0.25] [--dev-frac 0.6]
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUTD = os.path.join(HERE, 'out')

import t1_prep as T1   # 复用 DEMIndex / 块 / 抽样 / 地形口径（只读）

POOL2 = T1.POOL2
STRICT = T1.STRICT
EVAL_W2 = T1.EVAL_W2
TILES = T1.TILES
FEATS = T1.FEATS
FOREST = (4, 5, 6, 7)

# 新候选框（F1/F2 以外的瓦片；仅按标签 n 与 DEM 覆盖选）
CAND = {
    'G1': dict(tile='T1707', bbox=(107.0, 32.0, 108.0, 33.0)),   # n=199 森林114 ENF42/EBF60/DBF12
    'G2': dict(tile='T1606', bbox=(106.0, 31.0, 107.0, 32.0)),   # n=194 森林 82 ENF51/EBF29
    'G3': dict(tile='T1605', bbox=(105.5, 29.0, 106.5, 30.0)),   # 备用
}


def seen_conf_blocks(outd):
    """已看过的 conf 块（orchard 4 区 + terrain F1/F2），统一为 'lon,lat' 键。"""
    seen = set()
    ofp = r'F:/lc_work/improve_2023_pilot_20261015/orchard/prereg/regions_frozen.json'
    src = {}
    if os.path.exists(ofp):
        od = json.load(open(ofp, encoding='utf-8'))
        for rk, rv in od.get('regions', {}).items():
            for b in rv.get('conf_blocks', []):
                seen.add(b.replace('_', ',')); src[b.replace('_', ',')] = 'orchard_%s' % rk
    for fr in ('F1', 'F2'):
        fp = os.path.join(outd, 't1_arms_%s_seed7.npz' % fr)
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=False)
            bl = z['eval_block']; sp = z['eval_split']
            for b in set(bl[sp == 0].tolist()):
                seen.add(str(b)); src[str(b)] = 'terrain_%s' % fr
    return seen, src


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='G1')
    ap.add_argument('--pad', type=float, default=2.0)
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

    reg = {'experiment': 'F-VIEW (new frame)', 'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'script': os.path.abspath(__file__), 'spec': 'doc70 §4.3 F-VIEW / §6.1 独立 confirm',
           'seeds': seeds, 'n_train_limit': a.cap, 'block_deg': a.block, 'dev_frac': a.dev_frac,
           'arms': ['baseline_common', 'strict_common'],
           'rules': {
               'count': 'min(v2池类count, strict视图类count)（共同支持，森林类降幅如实报）',
               'nonforest': '两臂共用同一非森林点（完全同选择）',
               'strict_view': 'forest_samples_v1_strict.parquet（doc65 clean/strict；来源固定，不依预测挑规则）',
               'new_frame': 'F1(T1607)/F2(T1706) 以外；confirm 块未看过'},
           'inputs': {}}
    for nm, fp in (('pool2', POOL2), ('strict', STRICT), ('eval_w2', EVAL_W2), ('tiles', TILES)):
        reg['inputs'][nm] = T1.identity(fp)

    dem = T1.DEMIndex()
    T = pd.read_csv(TILES)
    Tmap = {r.tile: (r.x0, r.y0, r.x1, r.y1) for r in T.itertuples()}

    e = pd.read_parquet(EVAL_W2, columns=['row_id', 'lon', 'lat', 'class_new', 'src'] + FEATS)
    pf, _ = C4.assign_prov(e.lon.to_numpy(), e.lat.to_numpy(), list(PC.PROV_SHORT))
    sc = e[pf == '四川省'].reset_index(drop=True)
    sc['t24'] = VC.to_v31(sc.class_new.to_numpy().astype(np.int64))
    sc['is_fcs10'] = PC.is_fcs10(sc.src).to_numpy()

    t = pq.read_table(POOL2, columns=['lon', 'lat', 'cl'] + FEATS)
    plon = np.asarray(t['lon']); plat = np.asarray(t['lat']); pcl = np.asarray(t['cl']).astype(np.int64)
    X = np.empty((t.num_rows, 64), np.float32)
    for i in range(64):
        X[:, i] = t.column(FEATS[i]).to_numpy()
    del t
    ts = pq.read_table(STRICT, columns=['lon', 'lat', 'class4', 'src', 'tier', 'aef_year'] + FEATS)
    slon = np.asarray(ts['lon']); slat = np.asarray(ts['lat'])
    scl = np.asarray(ts['class4']).astype(np.int64)
    ssrc = np.asarray(ts['src'].to_pylist()); stier = np.asarray(ts['tier'].to_pylist())
    syear = np.asarray(ts['aef_year'])
    Xs = np.empty((ts.num_rows, 64), np.float32)
    for i in range(64):
        Xs[:, i] = ts.column(FEATS[i]).to_numpy()
    del ts
    # strict 为八年并集：只保留 aef_year==2023 以与 v2 池（2023）对齐
    y23 = np.isfinite(syear) & (syear == 2023)
    n_y23_drop = int((~y23).sum())
    slon, slat, scl, ssrc, stier, Xs = slon[y23], slat[y23], scl[y23], ssrc[y23], stier[y23], Xs[y23]
    emit('v2池 n=%d ｜ strict 森林视图 n=%d（aef_year==2023 后；剔非2023 %d）' % (len(pcl), len(scl), n_y23_drop))
    reg['strict_field_evidence'] = {
        'file': STRICT,
        'n_total_after_year2023': int(len(scl)),
        'n_non2023_dropped': n_y23_drop,
        'class4_counts': {str(k): int(v) for k, v in zip(*np.unique(scl, return_counts=True))},
        'src_top': {str(k): int(v) for k, v in pd.Series(ssrc).value_counts().head(6).items()},
        'tier_counts': {str(k): int(v) for k, v in pd.Series(stier).value_counts().items()},
        'note': 'class4=24码（4/5/6/7）；src 为旧教师来源目录名；tier=质量层；已按 aef_year==2023 对齐；不含针/阔真值（doc54 AUC0.724）'}

    frames_out = []
    seen_blocks, seen_src = seen_conf_blocks(OUTD)
    reg['seen_conf_blocks_registered'] = {b: seen_src[b] for b in sorted(seen_blocks)}
    emit('已登记已看 conf 块 %d 个（orchard 4 区 + terrain F1/F2）' % len(seen_blocks))
    for fid in [s.strip() for s in a.frames.split(',') if s.strip()]:
        spec = CAND[fid]; tile = spec['tile']; x0, y0, x1, y1 = spec['bbox']
        ie = np.flatnonzero((sc.lon.values >= x0) & (sc.lon.values < x1) &
                            (sc.lat.values >= y0) & (sc.lat.values < y1))
        ev = sc.iloc[ie].reset_index(drop=True)
        ev['block'] = T1.block_keys_str(ev.lon.values, ev.lat.values, a.block)
        ev['split'] = np.where(T1.block_split(ev.block.tolist(), a.dev_frac), 'dev', 'conf')
        eval_blocks = set(ev.block.unique())
        conf_blocks = set(ev.block[ev.split == 'conf'].tolist())
        # 确认独立性：本框 conf 块 ∩ 已看 conf 块（需=0）
        overlap_blocks = sorted(conf_blocks & seen_blocks)
        poskey = np.array(['%.6f_%.6f' % (lo, la) for lo, la in zip(ev.lon.values, ev.lat.values)])
        ev['poskey'] = poskey

        tx0, ty0, tx1, ty1 = Tmap[tile]
        wx0, wy0, wx1, wy1 = tx0 - a.pad, ty0 - a.pad, tx1 + a.pad, ty1 + a.pad
        iw = np.flatnonzero((plon >= wx0) & (plon < wx1) & (plat >= wy0) & (plat < wy1))
        wkey = T1.block_keys_str(plon[iw], plat[iw], a.block)
        leak_self = np.array([k in eval_blocks for k in wkey])
        leak_seen = np.array([k in seen_blocks for k in wkey])   # 相同训练块排除（已看 conf 块）
        iw = iw[~(leak_self | leak_seen)]
        n_excl_self = int(leak_self.sum()); n_excl_seen = int((leak_seen & ~leak_self).sum())
        el, sl, asp = dem.sample_any(plon[iw], plat[iw])
        ok = np.isfinite(el) & np.isfinite(sl) & np.isfinite(asp)
        iw = iw[ok]; el = el[ok]; sl = sl[ok]; asp = asp[ok]
        lab = pcl[iw]
        a3 = T1.aspect_class(asp, sl); eb = T1.elev_band(el)

        sm = np.flatnonzero((slon >= wx0) & (slon < wx1) & (slat >= wy0) & (slat < wy1) &
                            np.isin(scl, list(FOREST)))
        skey = T1.block_keys_str(slon[sm], slat[sm], a.block)
        sm = sm[~np.array([k in eval_blocks for k in skey])]
        strict_avail = {int(c): int((scl[sm] == c).sum()) for c in FOREST}
        pool_avail = {int(c): int((lab == c).sum()) for c in FOREST}

        counts_full = T1.per_class_counts(lab, a.cap)
        counts_common = dict(counts_full)
        reduction = {}
        for c in FOREST:
            if c in counts_common:
                before = counts_common[c]
                counts_common[c] = min(before, strict_avail.get(c, 0))
                if counts_common[c] < before:
                    reduction[str(c)] = [int(before), int(counts_common[c])]
        n_full = sum(counts_full.values()); n_common = sum(counts_common.values())

        reg_f = dict(frame=fid, tile=tile, frame_bbox=spec['bbox'], train_bbox=(wx0, wy0, wx1, wy1),
                     n_eval=int(len(ev)),
                     n_eval_forest=int(ev.t24.isin(FOREST + (8,)).sum()),
                     n_eval_dev=int((ev.split == 'dev').sum()),
                     n_eval_conf=int((ev.split == 'conf').sum()),
                     n_eval_blocks=int(len(eval_blocks)),
                     n_eval_fcs10=int(ev.is_fcs10.sum()),
                     n_eval_nonfcs10=int((~ev.is_fcs10).sum()),
                     n_eval_conf_nonfcs10=int(((ev.split == 'conf') & (~ev.is_fcs10)).sum()),
                     n_conf_class4=int(((ev.split == 'conf') & (ev.t24 == 4) & (~ev.is_fcs10)).sum()),
                     n_eval_class={str(k): int(v) for k, v in ev.t24.value_counts().items()},
                     strict_avail_window=strict_avail, pool_avail_window=pool_avail,
                     counts_full=counts_full, n_train_full=n_full,
                     counts_common=counts_common, n_train_common=n_common,
                     forest_reduction=reduction,
                     n_pool_window_raw=int(len(iw) + n_excl_self + n_excl_seen),
                     n_leak_excl_self_evalblock=n_excl_self,
                     n_leak_excl_seen_confblock=n_excl_seen,
                     n_train_candidates=int(len(iw)),
                     eval_blocks_list=sorted(eval_blocks),
                     conf_blocks_list=sorted(conf_blocks),
                     confirm_independence=dict(
                         overlap_blocks_with_seen=overlap_blocks,
                         n_overlap_points=int(np.isin(ev.block.tolist(), overlap_blocks).sum()),
                         seen_src={b: seen_src.get(b, '?') for b in overlap_blocks},
                         verdict='PASS(0)' if not overlap_blocks else 'FAIL'))
        if reg_f['n_eval_conf'] < 30:
            reg_f['conf_support_warning'] = 'conf 点 %d <30 → 确认集支持不足' % reg_f['n_eval_conf']
        frames_out.append(reg_f)
        emit('%s(%s) 评价 %d（森林 %d｜dev %d/conf %d｜块 %d）｜共同支持训练 %d（原 %d，森林降 %s）' % (
            fid, tile, reg_f['n_eval'], reg_f['n_eval_forest'], reg_f['n_eval_dev'],
            reg_f['n_eval_conf'], reg_f['n_eval_blocks'], n_common, n_full, reduction or '无'))

        for seed in seeds:
            ibc = T1.select_baseline(lab, counts_common, seed)
            selX, selY, short = [], [], {}
            rng = np.random.RandomState(seed + 202)
            for c, k in counts_common.items():
                if c in FOREST:
                    cand = sm[scl[sm] == c]
                    if len(cand) >= k:
                        pick = rng.permutation(cand)[:k]
                        selX.append(Xs[pick]); selY.append(np.full(k, c))
                    else:
                        short[str(c)] = int(k - len(cand))
                        pool = ibc[lab[ibc] == c]
                        selX.append(X[pool]); selY.append(np.full(len(pool), c))
                else:
                    pool = ibc[lab[ibc] == c]      # 非森林两臂同选择
                    selX.append(X[pool]); selY.append(np.full(len(pool), c))
            reg_f.setdefault('strict_short', {})['seed%d' % seed] = short
            eel, esl, easp = dem.sample_any(ev.lon.values, ev.lat.values)
            np.savez_compressed(
                os.path.join(OUTD, 't4_fview_arms_%s_seed%d.npz' % (fid, seed)),
                bc_X=X[iw][ibc], bc_y=lab[ibc],
                sc_X=np.concatenate(selX).astype(np.float32),
                sc_y=np.concatenate(selY).astype(np.int64),
                eval_X=ev[FEATS].to_numpy(np.float32),
                eval_lon=ev.lon.values, eval_lat=ev.lat.values, eval_y=ev.t24.values,
                eval_a3=T1.aspect_class(np.nan_to_num(easp, nan=0.0), np.nan_to_num(esl, nan=0.0)),
                eval_split=(ev.split.values == 'dev').astype(np.int8),
                eval_is_fcs10=ev.is_fcs10.values.astype(np.int8),
                eval_block=np.array(ev.block.tolist()), eval_row_id=np.asarray(ev.row_id))
        emit('  → t4_fview_arms_%s_seed{%s}.npz 已写' % (fid, ','.join(map(str, seeds))))

    reg['frames'] = frames_out
    with open(os.path.join(OUTD, 't4_fview_registration.json'), 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=1, default=float)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('counts_full', 'counts_common', 'strict_short')}
                  for r in frames_out]).to_csv(os.path.join(OUTD, 't4_fview_frames.csv'),
                                               index=False, encoding='utf-8-sig')
    emit('→ out/t4_fview_registration.json + t4_fview_frames.csv')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""t4b_fview_prep_g2.py — F-VIEW 独立对（新空间框 G2）**修复版**准备：不跑 RF

* 为什么需要本脚本（相对 t4_fview_prep.py 的唯一差异＝一个纯工程 bug 修复）：
    t4_fview_prep.py 在组装 strict_common 臂时对**非森林点**写了 `X[pool]`，而 X 是
    全局池特征矩阵、pool 是**窗内相对下标**（正确写法 `X[iw][pool]`）。基线臂
    `X[iw][ibc]` 是对的 → 两臂非森林行**不是同一批点**，违反冻结规则
    "非森林两臂完全同选择"。因此 strict 臂在 t4 版里训练的是错行的非森林样本，
    其结果不可用。本脚本按 doc70 §4.1"允许修纯工程 bug，但保留原运行记录、注明修复后
    重跑、禁止悄悄覆盖旧结果"执行：旧 npz/注册件**原样保留**（out/），本脚本输出到
    新目录 out/fview_g2/，并在注册件里记录 v1→v2 差异与自检证据。
* 规则源：doc70 §4.3 F-VIEW / §6.1 独立 confirm / §7 采纳门；主 agent 冻结判据。
* 冻结条件（与 t4 相同，逐条自检并在不符时 rc!=0 退出）：
    - 每类 count 两臂**完全相同**；非森林两臂**同点（同 pool 行号）**；
    - 森林取 min(v2池类count, strict视图类count)，strict 视图来源固定、不依预测挑；
    - 新框（F1/F2 以外），confirm 块与已看 conf 块交集 = 0；参考点 ID 交集 = 0；
    - 固定 3 seed：7/11/23。
* 输出（新目录，不覆盖 out/ 旧件）：
    out/fview_g2/t4_fview_registration_G2v2.json
    out/fview_g2/t4_fview_frames_G2v2.csv
    out/fview_g2/t4_fview_arms_G2_seed<seed>.npz（含 bc_idx/sc_idx/sc_src 训练行号留痕）
    out/fview_g2/t4_fview_idcheck_G2v2.json（ID/块级独立性证据）
* 用法：python t4b_fview_prep_g2.py [--outdir out/fview_g2] [--v1-dir out]
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

import t1_prep as T1   # 复用 DEMIndex / 块 / 抽样口径（只读）

POOL2 = T1.POOL2
STRICT = T1.STRICT
EVAL_W2 = T1.EVAL_W2
TILES = T1.TILES
FEATS = T1.FEATS
FOREST = (4, 5, 6, 7)
ORCH_RES = r'F:/lc_work/improve_2023_pilot_20261015/orchard/results'

CAND = {'G2': dict(tile='T1606', bbox=(106.0, 31.0, 107.0, 32.0))}   # n=194 森林82 ENF51/EBF29


def emit(m):
    print('[%s] %s' % (time.strftime('%m-%d %H:%M:%S'), m), flush=True)


def poskeys(lon, lat):
    return np.array(['%.6f_%.6f' % (lo, la) for lo, la in zip(lon, lat)])


def seen_conf_ids(outd_v1):
    """已看过的 conf 参考点：orchard 4 区（pred CSV，含 row_id）+ terrain F1/F2（npz 位置键）。

    返回 (rowid_set, poskey_set, detail)。row_id 空间用 eval_w2 的 (lon,lat)→row_id 映射校验。
    """
    import pandas as pd
    e = pd.read_parquet(EVAL_W2, columns=['row_id', 'lon', 'lat'])
    pk = poskeys(e.lon.values, e.lat.values)
    pk2rid = {}
    for k, r in zip(pk.tolist(), e.row_id.tolist()):
        pk2rid.setdefault(k, set()).add(int(r))
    rowids, posk, detail = set(), set(), {}
    if os.path.isdir(ORCH_RES):
        for fn in sorted(os.listdir(ORCH_RES)):
            if not fn.endswith('_baseline_pred.csv'):
                continue
            d = pd.read_csv(os.path.join(ORCH_RES, fn))
            d = d[d.split == 'conf']
            if not len(d):
                continue
            posk |= set(poskeys(d.lon.values, d.lat.values).tolist())
            rowids |= set(int(x) for x in d.row_id.tolist())
            detail[fn.replace('_baseline_pred.csv', '')] = dict(n_conf=int(len(d)), id_src='pred_csv')
    for fr in ('F1', 'F2'):
        fp = os.path.join(outd_v1, 't1_arms_%s_seed7.npz' % fr)
        if not os.path.exists(fp):
            continue
        z = np.load(fp, allow_pickle=False)
        m = z['eval_split'] == 0
        k = poskeys(z['eval_lon'][m], z['eval_lat'][m]).tolist()
        posk |= set(k)
        for kk in k:
            rowids |= pk2rid.get(kk, set())
        detail['terrain_%s' % fr] = dict(n_conf=int(m.sum()), id_src='t1 npz lon/lat→eval_w2 row_id')
    return rowids, posk, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=os.path.join(HERE, 'out', 'fview_g2'))
    ap.add_argument('--v1-dir', default=os.path.join(HERE, 'out'))
    ap.add_argument('--frames', default='G2')
    ap.add_argument('--pad', type=float, default=2.0)
    ap.add_argument('--cap', type=int, default=20000)
    ap.add_argument('--block', type=float, default=0.25)
    ap.add_argument('--dev-frac', type=float, default=0.6)
    ap.add_argument('--seeds', default='7,11,23')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(',') if s.strip()]

    import pandas as pd
    import pyarrow.parquet as pq
    import v31_common as VC
    import c4_eval_second as C4
    import p5common as PC

    reg = {'experiment': 'F-VIEW (new frame, G2) — prep v2 (bug-fixed)',
           'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
           'script': os.path.abspath(__file__),
           'spec': 'doc70 §4.3 F-VIEW / §6.1 独立 confirm / §4.1 bug 修复留痕',
           'supersedes': os.path.join(a.v1_dir, 't4_fview_registration.json'),
           'bugfix_vs_v1': {
               'file': 't4_fview_prep.py',
               'defect': 'strict 臂非森林行写 X[pool]（pool=窗内相对下标，X=全局池矩阵）→ 取错行；正确 X[iw][pool]',
               'affected_rows_v1': 'strict_common 非森林 12383 行/seed（森林行 Xs[pick] 本身正确）',
               'why_blocking': '违反冻结规则"非森林两臂完全同选择"；strict 臂训练样本无效',
               'v1_kept': 'out/ 下 v1 npz/注册件原样保留，未覆盖'},
           'seeds': seeds, 'n_train_limit': a.cap, 'block_deg': a.block, 'dev_frac': a.dev_frac,
           'arms': ['baseline_common', 'strict_common'],
           'rules': {
               'count': 'min(v2池类count, strict视图类count)（共同支持，森林类降幅如实报）',
               'nonforest': '两臂共用同一非森林点（同 pool 行号；v2 已按 X[iw][pool] 修复并断言）',
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
    y23 = np.isfinite(syear) & (syear == 2023)
    n_y23_drop = int((~y23).sum())
    slon, slat, scl, ssrc, stier, Xs = slon[y23], slat[y23], scl[y23], ssrc[y23], stier[y23], Xs[y23]
    assert (syear[y23] == 2023).all(), 'strict 保留行必须全为 aef_year==2023'
    years_before = sorted(set(np.asarray(syear).tolist()))
    emit('v2池 n=%d ｜ strict 森林视图 n=%d（aef_year==2023 后；剔非2023 %d；保留年份 %s）' % (
        len(pcl), len(scl), n_y23_drop, sorted(set(syear[y23].tolist()))))
    reg['strict_year_evidence'] = {
        'rule': '只保留 aef_year==2023（非八年并集）；AEF64 取该点 2023 年特征',
        'years_present_before': years_before,
        'years_retained': sorted(set(syear[y23].tolist())),
        'n_retained': int(len(scl)), 'n_dropped_non2023': n_y23_drop,
        'assert_all_2023': True}
    reg['strict_field_evidence'] = {
        'file': STRICT,
        'n_total_after_year2023': int(len(scl)),
        'n_non2023_dropped': n_y23_drop,
        'class4_counts': {str(k): int(v) for k, v in zip(*np.unique(scl, return_counts=True))},
        'src_top': {str(k): int(v) for k, v in pd.Series(ssrc).value_counts().head(6).items()},
        'tier_counts': {str(k): int(v) for k, v in pd.Series(stier).value_counts().items()},
        'note': 'class4=24码（4/5/6/7）；已按 aef_year==2023 对齐；不含针/阔真值（doc54 AUC0.724）'}

    # 已看 conf：块级 + 点级
    seen_blocks, seen_src = set(), {}
    ofp = r'F:/lc_work/improve_2023_pilot_20261015/orchard/prereg/regions_frozen.json'
    if os.path.exists(ofp):
        od = json.load(open(ofp, encoding='utf-8'))
        for rk, rv in od.get('regions', {}).items():
            for b in rv.get('conf_blocks', []):
                seen_blocks.add(b.replace('_', ',')); seen_src[b.replace('_', ',')] = 'orchard_%s' % rk
    for fr in ('F1', 'F2'):
        fp = os.path.join(a.v1_dir, 't1_arms_%s_seed7.npz' % fr)
        if os.path.exists(fp):
            z = np.load(fp, allow_pickle=False)
            bl = z['eval_block']; sp = z['eval_split']
            for b in set(bl[sp == 0].tolist()):
                seen_blocks.add(str(b)); seen_src[str(b)] = 'terrain_%s' % fr
    seen_rid, seen_posk, seen_detail = seen_conf_ids(a.v1_dir)
    reg['seen_conf_blocks_registered'] = {b: seen_src[b] for b in sorted(seen_blocks)}
    reg['seen_conf_points_registered'] = dict(n_rowids=len(seen_rid), n_poskeys=len(seen_posk),
                                              detail=seen_detail)
    emit('已登记已看 conf：块 %d 个、点 row_id %d / 位置键 %d' % (len(seen_blocks), len(seen_rid), len(seen_posk)))

    frames_out, idcheck = [], {}
    for fid in [s.strip() for s in a.frames.split(',') if s.strip()]:
        spec = CAND[fid]; tile = spec['tile']; x0, y0, x1, y1 = spec['bbox']
        ie = np.flatnonzero((sc.lon.values >= x0) & (sc.lon.values < x1) &
                            (sc.lat.values >= y0) & (sc.lat.values < y1))
        ev = sc.iloc[ie].reset_index(drop=True)
        ev['block'] = T1.block_keys_str(ev.lon.values, ev.lat.values, a.block)
        ev['split'] = np.where(T1.block_split(ev.block.tolist(), a.dev_frac), 'dev', 'conf')
        ev['poskey'] = poskeys(ev.lon.values, ev.lat.values)
        eval_blocks = set(ev.block.unique())
        conf_blocks = set(ev.block[ev.split == 'conf'].tolist())
        overlap_blocks = sorted(conf_blocks & seen_blocks)
        g2_conf = ev[ev.split == 'conf']
        g2_conf_rid = set(int(x) for x in g2_conf.row_id.tolist())
        g2_conf_posk = set(g2_conf.poskey.tolist())
        id_inter_rid = sorted(g2_conf_rid & seen_rid)
        id_inter_posk = sorted(g2_conf_posk & seen_posk)

        tx0, ty0, tx1, ty1 = Tmap[tile]
        wx0, wy0, wx1, wy1 = tx0 - a.pad, ty0 - a.pad, tx1 + a.pad, ty1 + a.pad
        iw = np.flatnonzero((plon >= wx0) & (plon < wx1) & (plat >= wy0) & (plat < wy1))
        wkey = T1.block_keys_str(plon[iw], plat[iw], a.block)
        leak_self = np.array([k in eval_blocks for k in wkey])
        leak_seen = np.array([k in seen_blocks for k in wkey])
        iw = iw[~(leak_self | leak_seen)]
        n_excl_self = int(leak_self.sum()); n_excl_seen = int((leak_seen & ~leak_self).sum())
        el, sl, asp = dem.sample_any(plon[iw], plat[iw])
        ok = np.isfinite(el) & np.isfinite(sl) & np.isfinite(asp)
        iw = iw[ok]; el = el[ok]; sl = sl[ok]; asp = asp[ok]
        lab = pcl[iw]
        a3 = T1.aspect_class(asp, sl); eb = T1.elev_band(el)

        sm = np.flatnonzero((slon >= wx0) & (slon < wx1) & (slat >= wy0) & (slat < wy1) &
                            np.isin(scl, list(FOREST)))
        n_strict_raw = int(len(sm))
        strict_avail_raw = {int(c): int((scl[sm] == c).sum()) for c in FOREST}
        # strict 候选必须与 baseline 池**同过滤规则**（不绕过过滤新增漏点）：
        #   ① 窗内（同上）② 排除本框 eval 块（dev+conf）③ 排除已看 conf 块（orchard+terrain F1/F2）
        #   ④ 与 baseline 相同的有 DEM 支持域（baseline 池经 DEM 有效性筛选；strict 同样筛）
        skey = T1.block_keys_str(slon[sm], slat[sm], a.block)
        m_eval = np.array([k in eval_blocks for k in skey])
        m_seen = np.array([k in seen_blocks for k in skey])
        sm = sm[~(m_eval | m_seen)]
        n_strict_excl_eval = int(m_eval.sum()); n_strict_excl_seen = int((m_seen & ~m_eval).sum())
        s_el, s_sl, s_asp = dem.sample_any(slon[sm], slat[sm])
        m_dem = np.isfinite(s_el) & np.isfinite(s_sl) & np.isfinite(s_asp)
        n_strict_dem_dropped = int((~m_dem).sum())
        sm = sm[m_dem]
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
                     n_conf_class4_any=int(((ev.split == 'conf') & (ev.t24 == 4)).sum()),
                     n_conf_class6_any=int(((ev.split == 'conf') & (ev.t24 == 6)).sum()),
                     n_eval_class={str(k): int(v) for k, v in ev.t24.value_counts().items()},
                     strict_avail_window=strict_avail, pool_avail_window=pool_avail,
                     strict_avail_window_raw=strict_avail_raw, n_strict_window_raw=n_strict_raw,
                     n_strict_excl_evalblock=n_strict_excl_eval,
                     n_strict_excl_seenblock=n_strict_excl_seen,
                     n_strict_dem_dropped=n_strict_dem_dropped,
                     strict_filter_rule='与 baseline 池同规则：窗内 ∩ 非本框eval块 ∩ 非已看conf块 ∩ 有DEM支持',
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
        idcheck[fid] = dict(
            frame_bbox=spec['bbox'], tile=tile,
            conf_blocks=sorted(conf_blocks), all_blocks=sorted(eval_blocks),
            conf_intersect_seen_blocks=overlap_blocks,
            conf_rowid_intersect_seen=id_inter_rid,
            conf_poskey_intersect_seen=id_inter_posk,
            n_conf_points=int(len(g2_conf)),
            n_conf_rowids=len(g2_conf_rid), n_conf_poskeys=len(g2_conf_posk),
            seen_n_blocks=len(seen_blocks), seen_n_rowids=len(seen_rid), seen_n_poskeys=len(seen_posk),
            verdict='UNSEEN_CONFIRMED' if (not overlap_blocks and not id_inter_rid and not id_inter_posk)
                    else 'OVERLAP_REJECT')
        emit('%s(%s) 评价 %d（森林 %d｜dev %d/conf %d｜块 %d）｜共同支持训练 %d（原 %d，森林降 %s）｜conf块∩已看 %d｜conf点ID∩已看 %d' % (
            fid, tile, reg_f['n_eval'], reg_f['n_eval_forest'], reg_f['n_eval_dev'],
            reg_f['n_eval_conf'], reg_f['n_eval_blocks'], n_common, n_full, reduction or '无',
            len(overlap_blocks), len(id_inter_rid) + len(id_inter_posk)))
        if idcheck[fid]['verdict'] != 'UNSEEN_CONFIRMED':
            emit('FAIL：确认集与已看重叠 → 停止（不换框、不改判据）')
            json.dump(idcheck, open(os.path.join(a.outdir, 't4_fview_idcheck_G2v2.json'), 'w',
                                    encoding='utf-8'), ensure_ascii=False, indent=1)
            sys.exit(3)

        for seed in seeds:
            ibc = T1.select_baseline(lab, counts_common, seed)
            selX, selY, sel_src, sel_idx, short = [], [], [], [], {}
            rng = np.random.RandomState(seed + 202)
            for c, k in counts_common.items():
                if c in FOREST:
                    cand = sm[scl[sm] == c]
                    if len(cand) >= k:
                        pick = rng.permutation(cand)[:k]
                        selX.append(Xs[pick]); selY.append(np.full(k, c))
                        sel_src.append(np.ones(k, np.int8)); sel_idx.append(pick.astype(np.int64))
                    else:
                        short[str(c)] = int(k - len(cand))
                        pool = ibc[lab[ibc] == c]
                        selX.append(X[iw][pool]); selY.append(np.full(len(pool), c))
                        sel_src.append(np.zeros(len(pool), np.int8)); sel_idx.append(iw[pool].astype(np.int64))
                else:
                    pool = ibc[lab[ibc] == c]      # 非森林两臂同选择（同 pool 行号）
                    selX.append(X[iw][pool]); selY.append(np.full(len(pool), c))
                    sel_src.append(np.zeros(len(pool), np.int8)); sel_idx.append(iw[pool].astype(np.int64))
            reg_f.setdefault('strict_short', {})['seed%d' % seed] = short
            sc_X = np.concatenate(selX).astype(np.float32)
            sc_y = np.concatenate(selY).astype(np.int64)
            sc_src = np.concatenate(sel_src)
            sc_idx = np.concatenate(sel_idx)
            bc_X = X[iw][ibc]; bc_y = lab[ibc]
            bc_idx = iw[ibc].astype(np.int64)

            # ---- 冻结规则自检（不符即 FAIL，不产出可用臂）----
            nb = ~np.isin(bc_y, list(FOREST))     # baseline 非森林行
            ns_pool = (sc_src == 0) & (~np.isin(sc_y, list(FOREST)))   # strict 臂非森林行（pool 来源）
            ob_b = np.argsort(bc_idx[nb]); ob_s = np.argsort(sc_idx[ns_pool])
            idx_b_sorted = bc_idx[nb][ob_b]; idx_s_sorted = sc_idx[ns_pool][ob_s]
            X_b_sorted = np.ascontiguousarray(bc_X[nb][ob_b]); X_s_sorted = np.ascontiguousarray(sc_X[ns_pool][ob_s])
            chk = dict(
                counts_equal=bool({int(k): int(v) for k, v in zip(*np.unique(bc_y, return_counts=True))} ==
                                  {int(k): int(v) for k, v in zip(*np.unique(sc_y, return_counts=True))}),
                nonforest_same_ids=bool(set(bc_idx[nb].tolist()) == set(sc_idx[ns_pool].tolist())),
                nonforest_ids_sorted_equal=bool(np.array_equal(idx_b_sorted, idx_s_sorted)),
                nonforest_X_values_equal=bool(np.array_equal(X_b_sorted, X_s_sorted)),
                nonforest_n=int(nb.sum()),
                forest_from_strict=bool((sc_src[np.isin(sc_y, list(FOREST))] == 1).all()),
                forest_counts_equal=bool(
                    {int(c): int((bc_y == c).sum()) for c in FOREST} ==
                    {int(c): int((sc_y == c).sum()) for c in FOREST}),
                all_features_finite=bool(np.isfinite(bc_X).all() and np.isfinite(sc_X).all()),
                idx_unique=bool(len(set(bc_idx.tolist())) == len(bc_idx) and
                                len(set(sc_idx[sc_src == 0].tolist())) == int((sc_src == 0).sum()) and
                                len(set(sc_idx[sc_src == 1].tolist())) == int((sc_src == 1).sum())),
                idx_unique_note='唯一性按来源分别检查（pool 全局行号 / strict 表行号 属不同索引空间）',
                idx_in_window=bool(np.isin(bc_idx, iw).all() and
                                   np.isin(sc_idx[sc_src == 0], iw).all()),
                n_baseline=int(len(bc_y)), n_strict=int(len(sc_y)),
                short=short)
            chk['PASS'] = bool(all(chk[k] for k in (
                'counts_equal', 'nonforest_same_ids', 'nonforest_ids_sorted_equal',
                'nonforest_X_values_equal', 'forest_from_strict', 'forest_counts_equal',
                'all_features_finite', 'idx_unique', 'idx_in_window')))
            reg_f.setdefault('arm_selfcheck', {})['seed%d' % seed] = chk
            if not chk['PASS']:
                emit('FAIL：seed %d 臂自检未过 %s' % (seed, chk))
                json.dump(reg, open(os.path.join(a.outdir, 't4_fview_registration_G2v2.json'), 'w',
                                    encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
                sys.exit(4)
            emit('  seed %d 自检 PASS：count 相同、非森林同点 %d（ID 与数值逐位一致）、森林 strict %d、特征 finite' % (
                seed, chk['nonforest_n'], int(np.isin(bc_y, list(FOREST)).sum())))

            eel, esl, easp = dem.sample_any(ev.lon.values, ev.lat.values)
            out_npz = os.path.join(a.outdir, 't4_fview_arms_%s_seed%d.npz' % (fid, seed))
            np.savez_compressed(
                out_npz,
                bc_X=bc_X, bc_y=bc_y, bc_idx=bc_idx,
                sc_X=sc_X, sc_y=sc_y, sc_idx=sc_idx, sc_src=sc_src,
                eval_X=ev[FEATS].to_numpy(np.float32),
                eval_lon=ev.lon.values, eval_lat=ev.lat.values, eval_y=ev.t24.values,
                eval_a3=T1.aspect_class(np.nan_to_num(easp, nan=0.0), np.nan_to_num(esl, nan=0.0)),
                eval_split=(ev.split.values == 'dev').astype(np.int8),
                eval_is_fcs10=ev.is_fcs10.values.astype(np.int8),
                eval_block=np.array(ev.block.tolist()), eval_row_id=np.asarray(ev.row_id),
                eval_poskey=ev.poskey.values)
            # 与 v1 对比：评价点必须逐字段一致（证明框/拆分未变），臂森林部分一致、非森林 v1 为错行
            v1 = os.path.join(a.v1_dir, 't4_fview_arms_%s_seed%d.npz' % (fid, seed))
            cmp = None
            if os.path.exists(v1):
                z1 = np.load(v1, allow_pickle=False)
                nf = ~np.isin(bc_y, list(FOREST))
                cmp = dict(
                    eval_identical=bool(all(np.array_equal(z1[k], v) for k, v in (
                        ('eval_lon', ev.lon.values), ('eval_lat', ev.lat.values),
                        ('eval_y', ev.t24.values), ('eval_split', (ev.split.values == 'dev').astype(np.int8)),
                        ('eval_is_fcs10', ev.is_fcs10.values.astype(np.int8)),
                        ('eval_block', np.array(ev.block.tolist())), ('eval_row_id', np.asarray(ev.row_id))))),
                    baseline_identical=bool(np.array_equal(z1['bc_X'], bc_X) and np.array_equal(z1['bc_y'], bc_y)),
                    strict_forest_identical=bool(np.array_equal(z1['sc_X'][np.isin(z1['sc_y'], list(FOREST))],
                                                               sc_X[np.isin(sc_y, list(FOREST))])),
                    v1_nonforest_rows_wrong=bool(not np.array_equal(
                        z1['sc_X'][~np.isin(z1['sc_y'], list(FOREST))], sc_X[nf])))
            reg_f.setdefault('v1_compare', {})['seed%d' % seed] = cmp
            reg_f.setdefault('arm_hash', {})['seed%d' % seed] = dict(
                bc=hashlib.sha256(np.ascontiguousarray(bc_X).tobytes() + bc_y.tobytes()).hexdigest()[:16],
                sc=hashlib.sha256(np.ascontiguousarray(sc_X).tobytes() + sc_y.tobytes()).hexdigest()[:16],
                bc_idx=hashlib.sha256(np.sort(bc_idx).tobytes()).hexdigest()[:16],
                sc_idx_pool=hashlib.sha256(np.sort(sc_idx[sc_src == 0]).tobytes()).hexdigest()[:16],
                sc_idx_strict=hashlib.sha256(np.sort(sc_idx[sc_src == 1]).tobytes()).hexdigest()[:16],
                n=dict(bc=int(len(bc_y)), sc=int(len(sc_y)),
                       sc_pool=int((sc_src == 0).sum()), sc_strict=int((sc_src == 1).sum())))
        emit('  → %s 已写（3 seed）' % os.path.join(a.outdir, 't4_fview_arms_%s_seed*.npz' % fid))

    reg['frames'] = frames_out
    reg['idcheck'] = idcheck
    f0 = frames_out[0]
    preflight = dict(
        schema='fview_g2_preflight/v1', generated=time.strftime('%Y-%m-%d %H:%M:%S'), frame=f0['frame'],
        old_v1=dict(path=os.path.join(a.v1_dir, 't4_fview_arms_%s_seed7.npz' % f0['frame']),
                    status='INVALID_STRICT_ARM_KEPT_AS_HISTORY',
                    reason='v1 strict 臂非森林行 X[pool] 索引 bug；不删、不覆盖、不用于训练'),
        strict_source=dict(
            view=STRICT, what_changes='只做质量成员筛选（doc65 clean/strict 子集）',
            aef_year='aef_year==2023 单年（非八年并集）', years_retained=reg['strict_year_evidence']['years_retained']),
        strict_filter_rule=f0['strict_filter_rule'],
        strict_window_accounting=dict(
            n_raw_window=f0['n_strict_window_raw'], avail_raw=f0['strict_avail_window_raw'],
            excl_eval_blocks=f0['n_strict_excl_evalblock'], excl_seen_conf_blocks=f0['n_strict_excl_seenblock'],
            dropped_no_dem=f0['n_strict_dem_dropped'], avail_after=f0['strict_avail_window']),
        counts=dict(counts_full=f0['counts_full'], counts_common=f0['counts_common'],
                    n_train_full=f0['n_train_full'], n_train_common=f0['n_train_common'],
                    forest_reduction=f0['forest_reduction'],
                    note=('n_train_common=%d 为本运行产物（Σ counts_common，三 seed 相同）。'
                          'v1 注册件记 18588、audit 记 18556，差异根因＝v1 对 strict 候选**未**施加'
                          '与池相同的过滤（v1 只排 eval 块；本轮加排已看 conf 块 + DEM 支持域），'
                          '故 strict 可用森林下降、共同支持 count 随之下降；这是本轮预注册的过滤口径修正，'
                          '不是静默改动；以本产物与自检为准。') % f0['n_train_common']),
        asserts={k: v for k, v in f0.get('arm_selfcheck', {}).items()},
        v1_compare=f0.get('v1_compare'),
        id_check=idcheck)
    with open(os.path.join(a.outdir, 't4_fview_preflight_G2v2.json'), 'w', encoding='utf-8') as f:
        json.dump(preflight, f, ensure_ascii=False, indent=1, default=float)
    emit('→ %s' % os.path.join(a.outdir, 't4_fview_preflight_G2v2.json'))
    with open(os.path.join(a.outdir, 't4_fview_registration_G2v2.json'), 'w', encoding='utf-8') as f:
        json.dump(reg, f, ensure_ascii=False, indent=1, default=float)
    with open(os.path.join(a.outdir, 't4_fview_idcheck_G2v2.json'), 'w', encoding='utf-8') as f:
        json.dump(idcheck, f, ensure_ascii=False, indent=1, default=float)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('counts_full', 'counts_common', 'strict_short',
                                                          'arm_selfcheck', 'arm_hash', 'v1_compare')}
                  for r in frames_out]).to_csv(os.path.join(a.outdir, 't4_fview_frames_G2v2.csv'),
                                               index=False, encoding='utf-8-sig')
    emit('→ %s' % os.path.join(a.outdir, 't4_fview_registration_G2v2.json'))
    emit('ALL_CHECKS_PASS（ID 独立性 + 臂自检）')


if __name__ == '__main__':
    main()

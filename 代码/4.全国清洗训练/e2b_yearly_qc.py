# -*- coding: utf-8 -*-
"""
e2b_yearly_qc.py — P2/P3 核心：逐点×逐年嵌入 QC + 八套年度训练子集（评审阶段B/C/D）
* 输入: emb_parts_r7/chunkr7_*.parquet（1190 块提取成品）+ chunks_index_r7 + r7_train_validity
* 指标（评审 B3）: embedding_valid/norm、cos_to_class_center(逐类×年)、
  delta_prev/next(相邻年 L2)、delta z-score(逐类×年 median/MAD)
* 状态（评审 C3）: keep / downweight(0.5) / exclude_year(数据异常或类心离群) /
  persistent_change(连续两年强变化, 降权 0.3 待复核)
* 产出:
    sample_year_qc.parquet          全部 point-years 的 QC 表
    年度子集/r7_train_{y}.parquet   2017-2024 八套（含 64 维嵌入 + train_weight）
    年度QC_summary.json
* 用法: python e2b_yearly_qc.py   （提取完成后运行）
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
BASE = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024/数据/本地处理/全国清洗训练'
PARTS = os.path.join(BASE, 'emb_parts_r7')
WORKR = os.path.join(PROJ, '数据/本地处理/样本重建')
OUT_QC = os.path.join(BASE, 'sample_year_qc.parquet')
SUB_DIR = os.path.join(BASE, '年度子集')
os.makedirs(SUB_DIR, exist_ok=True)
FEATS = [f'A{i:02d}' for i in range(64)]
YEARS = list(range(2017, 2025))
TIER_W = {'gold': 1.0, 'silver': 0.8, 'bronze': 0.6, 'external': 0.7, 'uncovered': 0.4}

_FMAP = None


def load_year(y):
    """载入某年全部嵌入（按 emb_year 逐行筛选），返回 (row_id int64, E float32[n,64])"""
    frames = []
    for f, ys in _FMAP.items():
        if y not in ys:
            continue
        try:
            d = pd.read_parquet(f)
        except Exception as e:
            print(f'⚠️ 读取失败跳过 {os.path.basename(f)}: {str(e)[:80]}', flush=True)
            continue
        d = d[d.emb_year == y]
        if len(d):
            frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates('row_id')
    E = df[FEATS].to_numpy(np.float32)
    return df.row_id.to_numpy(np.int64), E

def main():
    t0 = time.time()
    idx = pd.read_parquet(os.path.join(BASE, 'chunks_index_r7.parquet'))
    idx_u = idx.drop_duplicates('row_id').set_index('row_id')   # row_id 在多年重复，先去重
    global _FMAP
    _FMAP = {os.path.join(PARTS, f'chunkr7_{int(cid):04d}.parquet'):
             set(int(v) for v in ys)
             for cid, ys in idx.groupby('chunk_id').year.unique().items()}
    val = pd.read_parquet(os.path.join(WORKR, 'r7_train_validity.parquet'))
    n_idx = len(idx)
    print(f'索引 point-years: {n_idx:,}', flush=True)

    # ---------- 第一遍：逐年有效性/norm/类心余弦 ----------
    per_year = {}
    for y in YEARS:
        r = load_year(y)
        if r is None:
            print(f'y{y}: 无嵌入!', flush=True)
            per_year[y] = None
            continue
        rid, E = r
        ok = np.isfinite(E).all(1) & (np.linalg.norm(E, axis=1) > 1e-6)
        nrm = np.linalg.norm(E, axis=1)
        En = E / (nrm[:, None] + 1e-12)
        meta = idx_u.loc[rid]
        cls = meta.class_new.to_numpy(int)
        cos = np.full(len(E), np.nan, dtype=np.float32)
        for c in np.unique(cls):
            m = (cls == c) & ok
            if m.sum() >= 30:
                mu = E[m].mean(0)
                mu = mu / (np.linalg.norm(mu) + 1e-12)
                cos[m] = (En[m] @ mu).astype(np.float32)
        d = pd.DataFrame({'row_id': rid, 'year': y, 'emb_valid': ok,
                          'norm': nrm.astype(np.float32), 'cos_center': cos})
        d['class_new'] = cls
        per_year[y] = d
        del E, En
        print(f'y{y}: {len(d):,} 行, 有效 {ok.sum():,}', flush=True)

    # ---------- 第二遍：相邻年 delta（按 row_id 对齐） ----------
    dv = {y: pd.Series(np.nan, index=per_year[y].row_id) if per_year[y] is not None else None
          for y in YEARS}
    Emb = {}
    for y in YEARS:
        r = load_year(y)
        Emb[y] = r  # (rid, E) 暂驻，成对用完即删
    for y in YEARS[1:]:
        if per_year[y] is None or per_year[y-1] is None or Emb[y] is None or Emb[y-1] is None:
            continue
        rid_a, E_a = Emb[y - 1]
        rid_b, E_b = Emb[y]
        common, ia, ib = np.intersect1d(rid_a, rid_b, return_indices=True)
        d = np.linalg.norm(E_b[ib] - E_a[ia], axis=1)
        dv[y] = pd.Series(d, index=common)          # 记在较晚年上
        print(f'delta {y-1}→{y}: {len(common):,} 对', flush=True)
    Emb.clear()

    for y in YEARS:
        if per_year[y] is None:
            continue
        per_year[y]['delta_prev'] = per_year[y].row_id.map(dv[y]).to_numpy(np.float32) \
            if dv[y] is not None else np.nan
        nxt = dv[y + 1] if (y + 1) in dv and dv[y + 1] is not None else None
        per_year[y]['delta_next'] = per_year[y].row_id.map(nxt).to_numpy(np.float32) \
            if nxt is not None else np.nan

    # ---------- z-score（逐类×年，median/MAD） ----------
    for col in ['delta_prev', 'delta_next']:
        for y in YEARS:
            d = per_year[y]
            if d is None or col not in d:
                continue
            x = d[col].to_numpy()
            z = np.full(len(d), np.nan, dtype=np.float32)
            for c in np.unique(d.class_new):
                m = (d.class_new == c) & np.isfinite(x)
                if m.sum() < 100:
                    continue
                med = np.median(x[m])
                mad = np.median(np.abs(x[m] - med)) * 1.4826 + 1e-9
                z[m] = (x[m] - med) / mad
            d[col + '_z'] = z

    # ---------- 逐类×年 cos 分位 ----------
    pct = {}
    for y in YEARS:
        d = per_year[y]
        if d is None:
            continue
        for c, g in d.groupby('class_new'):
            v = g.cos_center.dropna().to_numpy()
            if len(v) >= 100:
                pct[(c, y)] = (np.percentile(v, 1), np.percentile(v, 5))

    # ---------- QC 状态 ----------
    qc_frames = []
    for y in YEARS:
        d = per_year[y]
        if d is None:
            continue
        status = np.full(len(d), 'keep', dtype=object)
        reason = np.array([''] * len(d), dtype=object)
        qw = np.ones(len(d), dtype=np.float32)
        # 1) 嵌入无效
        bad = ~d.emb_valid.to_numpy(bool)
        status[bad], reason[bad], qw[bad] = 'exclude_year', 'embedding_invalid', 0.0
        # 2) 类心离群（P01 剔 / P01-P05 降权）
        cos = d.cos_center.to_numpy()
        lo1 = np.array([pct.get((c, y), (np.nan, np.nan))[0] for c in d.class_new])
        lo5 = np.array([pct.get((c, y), (np.nan, np.nan))[1] for c in d.class_new])
        m = (~bad) & np.isfinite(cos) & np.isfinite(lo1)
        ex = m & (cos < lo1)
        status[ex], reason[ex], qw[ex] = 'exclude_year', 'class_center_outlier', 0.0
        dw = m & (cos >= lo1) & (cos < lo5) & (status == 'keep')
        status[dw], reason[dw], qw[dw] = 'downweight', 'class_center_low', 0.5
        # 3) 单年突变（一侧 z>5 另一侧正常）→ 数据异常剔该年
        zp = d.delta_prev_z.to_numpy() if 'delta_prev_z' in d else np.full(len(d), np.nan)
        zn = d.delta_next_z.to_numpy() if 'delta_next_z' in d else np.full(len(d), np.nan)
        spike = (~bad) & (status == 'keep') & (
            (np.isfinite(zp) & (zp > 5) & (~np.isfinite(zn) | (zn <= 3))) |
            (np.isfinite(zn) & (zn > 5) & (~np.isfinite(zp) | (zp <= 3))))
        status[spike], reason[spike], qw[spike] = 'exclude_year', 'single_year_anomaly', 0.0
        # 4) 连续两年强变化 → 持续变化，降权待复核
        pers = (~bad) & (status == 'keep') & np.isfinite(zp) & np.isfinite(zn) & \
               (zp > 5) & (zn > 5)
        status[pers], reason[pers], qw[pers] = 'persistent_change', 'persistent_change', 0.3
        d['qc_status'], d['qc_reason'], d['qc_weight'] = status, reason, qw
        qc_frames.append(d[['row_id', 'year', 'class_new', 'emb_valid', 'norm',
                            'cos_center', 'delta_prev', 'delta_next',
                            'delta_prev_z', 'delta_next_z',
                            'qc_status', 'qc_reason', 'qc_weight']])
    qc = pd.concat(qc_frames, ignore_index=True)
    # 元数据并入（row_id 去重后按显式列合并，避免 1427 万行大合并）
    meta_u = idx_u.reset_index()[['row_id', 'lon', 'lat']].merge(
        val[['row_id', 'tier', 'src', 'src_conf', 'valid_from', 'valid_to', 'qc_scope']],
        on='row_id', how='left')
    meta_u = meta_u[meta_u.valid_from.notna()]   # 2026-09-12: 剔除已清洗出库的点（159 省界 + 8,993 生态），防幽灵行
    qc = qc.merge(meta_u[['row_id', 'lon', 'lat', 'tier', 'src', 'src_conf']],
                  on='row_id', how='inner')
    qc['train_weight'] = (qc.tier.map(TIER_W).fillna(0.5) * qc.qc_weight *
                          (0.7 + 0.3 * qc.src_conf.fillna(0.8))).astype(np.float32)
    # ---- B 方案（2026-09-13，用户拍板）：FCS10-2023 系样本跨年（year != 2023）降权 ×0.5 ----
    _is_f = qc.src.astype(str).str.lower().str.contains('fcs10')
    _cross = _is_f & (qc.year != 2023) & (qc.qc_status != 'exclude_year')
    if _cross.any():
        qc.loc[_cross, 'train_weight'] = (qc.loc[_cross, 'train_weight'] * 0.5).astype(np.float32)
        _rs = qc.loc[_cross, 'qc_reason'].fillna('').astype(str)
        _rs = _rs.where(_rs == '', _rs + ';') + 'fcs10_crossyear_w0.5'
        qc.loc[_cross, 'qc_reason'] = _rs
    print(f'[B方案] FCS10 系跨年降权 ×0.5: {int(_cross.sum()):,} point-years '
          f'({_cross.groupby(qc.year).sum().to_dict() if _cross.any() else {}})', flush=True)
    qc.to_parquet(OUT_QC, index=False)

    # ---------- 年度子集（含嵌入） ----------
    sub_summary = {}
    for y in YEARS:
        r = load_year(y)
        if r is None:
            continue
        rid, E = r
        qy = qc[(qc.year == y) & (qc.qc_status != 'exclude_year')].set_index('row_id')
        keep = np.isin(rid, qy.index.to_numpy())
        rid, E = rid[keep], E[keep]
        qy = qy.loc[rid]
        out = pd.DataFrame(E, columns=FEATS, index=rid)
        out.index.name = 'row_id'
        out = out.reset_index()
        for col in ['lon', 'lat', 'class_new', 'year', 'tier', 'src', 'src_conf',
                    'train_weight', 'qc_status']:
            out[col] = qy[col].to_numpy()
        fp = os.path.join(SUB_DIR, f'r7_train_{y}.parquet')
        out.to_parquet(fp, index=False)
        sub_summary[y] = {'n': int(len(out)), 'excluded': int((qc.year == y).sum() - len(out))}
        print(f'年度子集 y{y}: {len(out):,}', flush=True)
        del r, E, out

    summary = {
        'time': time.strftime('%Y-%m-%d %H:%M'),
        'point_years': int(len(qc)),
        'status_counts': qc.qc_status.value_counts().to_dict(),
        'weighted_points': float(qc[qc.qc_status != 'exclude_year'].train_weight.sum()),
        'yearly_subsets': sub_summary,
        'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(os.path.join(BASE, '年度QC_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: summary[k] for k in ['point_years', 'status_counts',
                                              'weighted_points']}, ensure_ascii=False))

if __name__ == '__main__':
    main()

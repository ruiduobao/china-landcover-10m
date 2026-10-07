# -*- coding: utf-8 -*-
"""
e2b_yearly_qc_f.py — 年度 QC + 八套年度训练子集（F 盘自包含版）
* 全部输入/输出在 F:\\r7_prod\\，与 Z 盘零依赖
* 用法: F:\\anaconda\\python.exe F:\\r7_prod\\e2b_yearly_qc_f.py
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd

BASE = r'F:\r7_prod'
PARTS = os.path.join(BASE, 'emb_parts_r7')
OUT_QC = os.path.join(BASE, 'sample_year_qc.parquet')
SUB_DIR = os.path.join(BASE, '年度子集')
os.makedirs(SUB_DIR, exist_ok=True)
FEATS = [f'A{i:02d}' for i in range(64)]
YEARS = list(range(2017, 2025))
TIER_W = {'gold': 1.0, 'silver': 0.8, 'bronze': 0.6, 'external': 0.7, 'uncovered': 0.4}

def load_year(y):
    frames = []
    for f in glob.glob(os.path.join(PARTS, 'chunkr7_*.parquet')):
        head = pd.read_parquet(f, columns=['emb_year'])
        if len(head) and head.emb_year.iloc[0] == y:
            frames.append(pd.read_parquet(f))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates('row_id')
    E = df[FEATS].to_numpy(np.float32)
    return df.row_id.to_numpy(np.int64), E

def main():
    t0 = time.time()
    idx = pd.read_parquet(os.path.join(BASE, 'chunks_index_r7.parquet'))
    val = pd.read_parquet(os.path.join(BASE, 'r7_train_validity.parquet'))
    print(f'索引 point-years: {len(idx):,}', flush=True)

    # 第一遍：逐年有效性/norm/类心余弦
    per_year = {}
    for y in YEARS:
        r = load_year(y)
        if r is None:
            per_year[y] = None
            continue
        rid, E = r
        ok = np.isfinite(E).all(1) & (np.linalg.norm(E, axis=1) > 1e-6)
        nrm = np.linalg.norm(E, axis=1)
        En = E / (nrm[:, None] + 1e-12)
        meta = idx.set_index('row_id').loc[rid]
        cls = meta.class_new.to_numpy(int)
        cos = np.full(len(E), np.nan, dtype=np.float32)
        for c in np.unique(cls):
            m = (cls == c) & ok
            if m.sum() >= 30:
                mu = E[m].mean(0)
                mu = mu / (np.linalg.norm(mu) + 1e-12)
                cos[m] = (En[m] @ mu).astype(np.float32)
        d = pd.DataFrame({'row_id': rid, 'year': y, 'emb_valid': ok,
                          'norm': nrm.astype(np.float32), 'cos_center': cos,
                          'class_new': cls})
        per_year[y] = d
        del E, En
        print(f'y{y}: {len(d):,} 行, 有效 {ok.sum():,}', flush=True)

    # 第二遍：相邻年 delta
    dv = {}
    Emb = {}
    for y in YEARS:
        Emb[y] = load_year(y)
    for y in YEARS[1:]:
        if Emb.get(y) is None or Emb.get(y - 1) is None:
            continue
        rid_a, E_a = Emb[y - 1]
        rid_b, E_b = Emb[y]
        common, ia, ib = np.intersect1d(rid_a, rid_b, return_indices=True)
        dv[y] = pd.Series(np.linalg.norm(E_b[ib] - E_a[ia], axis=1), index=common)
        print(f'delta {y-1}→{y}: {len(common):,} 对', flush=True)
    Emb.clear()

    for y in YEARS:
        d = per_year[y]
        if d is None:
            continue
        d['delta_prev'] = d.row_id.map(dv[y]).to_numpy(np.float32) if y in dv else np.nan
        d['delta_next'] = d.row_id.map(dv[y + 1]).to_numpy(np.float32) \
            if (y + 1) in dv else np.nan

    # z-score（逐类×年 median/MAD）
    for col in ['delta_prev', 'delta_next']:
        for y in YEARS:
            d = per_year[y]
            if d is None:
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

    # 逐类×年 cos 分位
    pct = {}
    for y in YEARS:
        d = per_year[y]
        if d is None:
            continue
        for c, g in d.groupby('class_new'):
            v = g.cos_center.dropna().to_numpy()
            if len(v) >= 100:
                pct[(c, y)] = (np.percentile(v, 1), np.percentile(v, 5))

    # QC 状态
    qc_frames = []
    for y in YEARS:
        d = per_year[y]
        if d is None:
            continue
        status = np.full(len(d), 'keep', dtype=object)
        reason = np.array([''] * len(d), dtype=object)
        qw = np.ones(len(d), dtype=np.float32)
        bad = ~d.emb_valid.to_numpy(bool)
        status[bad], reason[bad], qw[bad] = 'exclude_year', 'embedding_invalid', 0.0
        cos = d.cos_center.to_numpy()
        lo1 = np.array([pct.get((c, y), (np.nan, np.nan))[0] for c in d.class_new])
        lo5 = np.array([pct.get((c, y), (np.nan, np.nan))[1] for c in d.class_new])
        m = (~bad) & np.isfinite(cos) & np.isfinite(lo1)
        ex = m & (cos < lo1)
        status[ex], reason[ex], qw[ex] = 'exclude_year', 'class_center_outlier', 0.0
        dw = m & (cos >= lo1) & (cos < lo5) & (status == 'keep')
        status[dw], reason[dw], qw[dw] = 'downweight', 'class_center_low', 0.5
        zp = d.delta_prev_z.to_numpy() if 'delta_prev_z' in d else np.full(len(d), np.nan)
        zn = d.delta_next_z.to_numpy() if 'delta_next_z' in d else np.full(len(d), np.nan)
        spike = (~bad) & (status == 'keep') & (
            (np.isfinite(zp) & (zp > 5) & (~np.isfinite(zn) | (zn <= 3))) |
            (np.isfinite(zn) & (zn > 5) & (~np.isfinite(zp) | (zp <= 3))))
        status[spike], reason[spike], qw[spike] = 'exclude_year', 'single_year_anomaly', 0.0
        pers = (~bad) & (status == 'keep') & np.isfinite(zp) & np.isfinite(zn) & \
               (zp > 5) & (zn > 5)
        status[pers], reason[pers], qw[pers] = 'persistent_change', 'persistent_change', 0.3
        d['qc_status'], d['qc_reason'], d['qc_weight'] = status, reason, qw
        qc_frames.append(d[['row_id', 'year', 'class_new', 'emb_valid', 'norm',
                            'cos_center', 'delta_prev', 'delta_next',
                            'delta_prev_z', 'delta_next_z',
                            'qc_status', 'qc_reason', 'qc_weight']])
    qc = pd.concat(qc_frames, ignore_index=True)
    meta = idx.merge(val[['lon', 'lat', 'tier', 'src', 'src_conf', 'valid_from',
                          'valid_to', 'qc_scope']], left_on='row_id',
                     right_index=True, how='left')
    qc = qc.merge(meta[['row_id', 'year', 'lon', 'lat', 'tier', 'src', 'src_conf']],
                  on=['row_id', 'year'], how='left')
    qc['train_weight'] = (qc.tier.map(TIER_W).fillna(0.5) * qc.qc_weight *
                          (0.7 + 0.3 * qc.src_conf.fillna(0.8))).astype(np.float32)
    qc.to_parquet(OUT_QC, index=False)

    # 年度子集（含嵌入）
    sub_summary = {}
    for y in YEARS:
        r = load_year(y)
        if r is None:
            continue
        rid, E = r
        qy = qc[(qc.year == y) & (qc.qc_status != 'exclude_year')].set_index('row_id').loc[rid]
        out = pd.DataFrame(E, columns=FEATS, index=rid).reset_index()
        out.index.name = 'row_id'
        for col in ['lon', 'lat', 'class_new', 'year', 'tier', 'src', 'src_conf',
                    'train_weight', 'qc_status']:
            out[col] = qy[col].to_numpy()
        out.to_parquet(os.path.join(SUB_DIR, f'r7_train_{y}.parquet'), index=False)
        sub_summary[y] = {'n': int(len(out))}
        print(f'年度子集 y{y}: {len(out):,}', flush=True)
        del r, E, out

    summary = {'time': time.strftime('%Y-%m-%d %H:%M'),
               'point_years': int(len(qc)),
               'status_counts': qc.qc_status.value_counts().to_dict(),
               'yearly_subsets': sub_summary,
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(os.path.join(BASE, '年度QC_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))

if __name__ == '__main__':
    main()

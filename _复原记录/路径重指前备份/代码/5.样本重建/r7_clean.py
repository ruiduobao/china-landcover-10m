# -*- coding: utf-8 -*-
"""
r7_clean.py — 合并 E130-E135 缺失带补采，重建 r7
* 背景：E 盘 FCS10 缺 E130-E135 带 → 中国最东部（黑龙江抚远/牡丹江东部、吉林延边）
  在 r4 所有 FCS10 派生样本中形成 130°E 直线截断。Z 盘 zip 补采 145,359 点（r5_band_voted）。
* 门槛：
  - FCS10 灌丛层（120/121）：保留（对标产品即 FCS10，其灌丛定义为权威）；
  - E130 补采带：非盲区类需 agree≥3；盲区类用耕地/建成否决（frac<0.5）；
  - 其余 r4_pool 点：沿用 r4 规则（非盲区 agree≥3/专题豁免；盲区 FCS10 同源一致或耕地建成否决）。
* 输出: r7_pool.parquet / r7_train.parquet / r7_audit.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

STAGE = r'F:\r7_stage'
os.makedirs(STAGE, exist_ok=True)
AUDIT = {}
T0 = time.time()
BLIND = {91, 92, 140, 180, 181, 182, 183, 184, 185, 186}
THEMATIC_RE = (r'^(aomc|citrus|teamap|rubber|gmw|china_coastal|saltmarsh|'
               r'gwl_fcs30_2020_raster|global_oilpalm|sdpt)')

def log(m): print(f'[r7 {time.time()-T0:.0f}s] {m}', flush=True)
def audit(k, **kw):
    AUDIT[k] = kw
    log(f'{k}: {json.dumps(kw, ensure_ascii=False, default=int)[:260]}')

def main():
    pool = pd.read_parquet(os.path.join(C.WORK, 'r6_pool.parquet'))
    audit('input_r6_pool', n=int(len(pool)))
    # 用新生态规则重筛的灌丛层替换旧灌丛层（121 南界 25→21 找回南方点）
    src0 = pool['src'].fillna('').astype(str)
    n_old = int((src0 == 'glc_fcs10_2023_shrub').sum())
    pool = pool[src0 != 'glc_fcs10_2023_shrub'].reset_index(drop=True)
    shrub = pd.read_parquet(os.path.join(C.WORK, 'r7_fcs10_shrub.parquet'))
    audit('shrub_swap', old=int(n_old), new=int(len(shrub)))
    pool = pd.concat([pool, shrub], ignore_index=True, sort=False)
    # r6_pool 已含 E130 补采带，无需再加
    # 池级国界复检（s3 当年假设 ESRI 已裁剪，实际混入 1.5 万境外点）
    inside = G.china_contains(pool.lon.to_numpy(), pool.lat.to_numpy())
    n_before = len(pool)
    pool = pool[inside].reset_index(drop=True)
    audit('pool_boundary_recheck', removed=int(n_before - len(pool)), left=int(len(pool)))

    # 耕地/建成否决
    V = pool[['wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0']].to_numpy(np.int8)
    valid = V >= 0
    nv = valid.sum(1)
    cb = (((V == 1) | (V == 6)) & valid).sum(1)
    pool['_crop_built_frac'] = np.where(nv > 0, cb / np.maximum(nv, 1), 0.0)

    src = pool['src'].fillna('').astype(str)
    thematic = src.str.contains(THEMATIC_RE, regex=True)
    is_shrub = pool.class_new.isin([120, 121])
    is_band = src == 'glc_fcs10_2023_band130'
    blind = pool.class_new.isin(BLIND).to_numpy()
    ok_tier = pool['tier'].isin(['gold', 'silver', 'bronze'])
    ok_ext = (pool['tier'] == 'external') & (
        (pool['agree_n'].fillna(0) >= C.VOTE_KEEP_EXTERNAL_MIN) | thematic)
    # 盲区（r4_pool 的 FCS30D 来源点）：FCS10 同源一致
    xc = pd.read_parquet(os.path.join(C.WORK, 'r4_fcs10_xcheck.parquet'))
    key = (pool.loc[blind, 'lon'].round(9).astype(str) + '_' +
           pool.loc[blind, 'lat'].round(9).astype(str) + '_' +
           pool.loc[blind, 'class_new'].astype(int).astype(str))
    xckey = (xc['lon'].round(9).astype(str) + '_' + xc['lat'].round(9).astype(str) + '_' +
             xc['class_new'].astype(int).astype(str))
    mm = pd.Series(xc['match'].to_numpy(), index=xckey.to_numpy())
    fm = mm.reindex(key.to_numpy()).to_numpy()
    keep_blind = np.where(pd.isna(fm), False,
                          np.where(fm == 'same', True,
                                   np.where(fm == 'nodata',
                                            pool.loc[blind, '_crop_built_frac'].to_numpy() < 0.5,
                                            False)))
    # 补采带规则
    band_ok = np.zeros(len(pool), dtype=bool)
    bm = is_band.to_numpy()
    if bm.any():
        band_ok[bm & is_shrub.to_numpy()] = True
        band_ok[bm & blind] = pool.loc[bm & blind, '_crop_built_frac'].to_numpy() < 0.5
        band_ok[bm & ~blind & ~is_shrub.to_numpy()] = \
            pool.loc[bm & ~blind & ~is_shrub.to_numpy(), 'agree_n'].fillna(0).to_numpy() >= 3
    keep_blind_full = np.zeros(len(pool), dtype=bool)
    keep_blind_full[blind] = keep_blind
    keep = is_shrub | ok_tier | ok_ext | (blind & keep_blind_full) | band_ok
    df = pool[keep].reset_index(drop=True)
    audit('gate_r5', n=int(len(df)),
          band_kept=int((bm & keep.to_numpy()).sum()), band_total=int(bm.sum()))

    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    audit('eco', n=int(len(df)))

    tw = df['tier'].map(C.TIER_W).fillna(0.5).to_numpy()
    conf = df['src_conf'].fillna(0.5).to_numpy()
    rng = np.random.default_rng(C.SEED)
    df['_prio'] = tw * 10 + conf + rng.random(len(df)) * 1e-6
    df['train_weight'] = 1.0
    m_rice = (df['class_new'] == 12) & df['src'].fillna('').str.startswith('ne_crops')
    df.loc[m_rice, 'train_weight'] = 1.2

    from scipy.spatial import cKDTree

    def bucket_thin(sub, dist_m):
        dist_km = dist_m / 1000.0
        km = np.column_stack([sub.lon.to_numpy() * 111.32 * np.cos(np.radians(sub.lat.to_numpy())),
                              sub.lat.to_numpy() * 110.57])
        order = np.argsort(-sub['_prio'].to_numpy(), kind='stable')
        kept, keep = {}, np.zeros(len(sub), dtype=bool)
        for i in order:
            kx = int(np.floor(km[i, 0] / dist_km)); ky = int(np.floor(km[i, 1] / dist_km))
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in kept.get((kx + dx, ky + dy), ()):
                        if (km[i, 0] - km[j, 0]) ** 2 + (km[i, 1] - km[j, 1]) ** 2 < dist_km * dist_km:
                            ok = False; break
                    if not ok: break
                if not ok: break
            if ok:
                kept.setdefault((kx, ky), []).append(i); keep[i] = True
        return sub[keep]

    parts = []
    for c in sorted(df.class_new.unique()):
        sub = df[df.class_new == c]
        dist_m = C.MIN_DIST.get(int(c), C.MIN_DIST['default'])
        if len(sub) <= 50000:
            parts.append(bucket_thin(sub, dist_m))
            continue
        fine_c = (np.floor(sub.lon.to_numpy() / 0.025).astype(np.int64) * 100000 +
                  np.floor(sub.lat.to_numpy() / 0.025).astype(np.int64))
        order = np.lexsort((-sub['_prio'].to_numpy(), fine_c))
        sf = fine_c[order]
        fmask = np.ones(len(order), dtype=bool)
        fmask[1:] = sf[1:] != sf[:-1]
        L1 = sub.iloc[order[fmask]]
        km1 = np.column_stack([L1.lon.to_numpy() * 111.32 * np.cos(np.radians(L1.lat.to_numpy())),
                               L1.lat.to_numpy() * 110.57])
        prs = cKDTree(km1).query_pairs(0.5, output_type='ndarray')
        drop = np.zeros(len(L1), dtype=bool)
        pr = L1['_prio'].to_numpy()
        for i, j in prs:
            if drop[i]:
                continue
            drop[i if pr[i] >= pr[j] else j] = True
        parts.append(L1[~drop])
    df2 = pd.concat(parts, ignore_index=True)
    audit('adaptive_thin', n=int(len(df2)))

    cell_q = np.floor(df2.lon * 4).astype(np.int64) * 10000 + np.floor(df2.lat * 4).astype(np.int64)
    pk = cell_q.astype(np.int64) * 1000 + df2.class_new.to_numpy()
    tmp = pd.DataFrame({'pk': pk, 'prio': df2['_prio'].to_numpy()}).sort_values(
        ['pk', 'prio'], ascending=[True, False])
    rank = tmp.groupby('pk').cumcount().to_numpy()
    df2 = df2.iloc[rank < C.CELL_QUOTA].reset_index(drop=True)
    inside = G.china_contains(df2.lon.to_numpy(), df2.lat.to_numpy())
    df2 = df2[inside].reset_index(drop=True)
    audit('final', n=int(len(df2)),
          by_class={str(int(k)): int(v) for k, v in df2.class_new.value_counts().sort_index().items()},
          elapsed_min=round((time.time() - T0) / 60, 1))

    pool.to_parquet(os.path.join(STAGE, 'r7_pool.parquet'), index=False)
    df2.to_parquet(os.path.join(STAGE, 'r7_train.parquet'), index=False)
    json.dump(AUDIT, open(os.path.join(STAGE, 'r7_audit.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=int)
    import shutil
    for f in ['r7_pool.parquet', 'r7_train.parquet', 'r7_audit.json']:
        shutil.copyfile(os.path.join(STAGE, f), os.path.join(C.WORK, f))
    log(f'完成：r7_pool={len(pool):,} r7_train={len(df2):,}')

if __name__ == '__main__':
    main()


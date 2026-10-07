# -*- coding: utf-8 -*-
"""
r4_clean.py — 修复 120/121 谱系分歧与空间截断，重建 r4
* 依据（2026-09-08 法证）：
  1) FCS30D(2020) 的灌丛与 FCS10(2023) 系统性不一致：我们 120 池点 90% 被 FCS10 判常绿阔叶林，
     121 池点被 FCS10 判草地 35%/裸地 25%；产品投票也大量反对 → 盲区保底规则叠加
     ESRI/清华瓦片覆盖不均，造成"有覆盖区删、无覆盖区留"的硬边界（云南 100-103E、甘肃 100-102E）。
  2) 对标产品即 GLC_FCS10 → 灌丛标签以 FCS10 为权威。
* r4 措施：
  A. 类 120/121：**整体替换为 FCS10 2023 全国纯净像元重采**（r4_fcs10_shrub.parquet，
     raw 121→120、raw 120/122→121，689,435 点，空间连续无截断）；
  B. 其余盲区类：把"uncovered 保底"改为"**FCS10 同位置一致才保留**"（nodata 时用
     耕地/建成否决规则兜底），消除覆盖不均造成的偏倚；专题产品源照旧豁免；
  C. 非盲区类：沿用 r3 的 agree≥3 门槛；
  D. 空间抽稀：>5 万点类 0.025° 分层 + 500m 边界清理；≤5 万点类按类最小间距桶贪心。
* 输出: r4_pool.parquet / r4_train.parquet / r4_audit.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

STAGE = r'F:\r4_stage'
os.makedirs(STAGE, exist_ok=True)
AUDIT = {}
T0 = time.time()
BLIND = {91, 92, 140, 180, 181, 182, 183, 184, 185, 186}
THEMATIC_RE = (r'^(aomc|citrus|teamap|rubber|gmw|china_coastal|saltmarsh|'
               r'gwl_fcs30_2020_raster|global_oilpalm|sdpt)')

def log(m): print(f'[r4 {time.time()-T0:.0f}s] {m}', flush=True)
def audit(k, **kw):
    AUDIT[k] = kw
    log(f'{k}: {json.dumps(kw, ensure_ascii=False, default=int)[:260]}')

def main():
    pool = pd.read_parquet(os.path.join(C.WORK, 'r2_backup', 'r1_pool_备份_20260908.parquet'))
    audit('input_r1_pool', n=int(len(pool)))

    # ---------- A. 120/121 整体替换为 FCS10 重采 ----------
    old = pool.class_new.isin([120, 121]).sum()
    pool = pool[~pool.class_new.isin([120, 121])].reset_index(drop=True)
    shrub = pd.read_parquet(os.path.join(C.WORK, 'r4_fcs10_shrub.parquet'))
    audit('shrub_replace', dropped_fcs30d=int(old),
          added_fcs10=int(len(shrub)),
          by_class={str(int(k)): int(v) for k, v in shrub.class_new.value_counts().items()})
    pool = pd.concat([pool, shrub], ignore_index=True, sort=False)

    # ---------- B. 盲区类 FCS10 同源一致性把关 ----------
    xc = pd.read_parquet(os.path.join(C.WORK, 'r4_fcs10_xcheck.parquet'))
    blind_mask = pool.class_new.isin(BLIND).to_numpy()
    # xc 与原 pool 的盲区子集顺序一致（构建方式相同），逐行对齐
    # 原盲区子集 = r1_pool 中 class in {91,92,140,180..186,120,121}，但我们已删除 120/121，
    # 故按 (lon,lat,class_new) 精确合并更稳
    key = (pool.loc[blind_mask, 'lon'].round(9).astype(str) + '_' +
           pool.loc[blind_mask, 'lat'].round(9).astype(str) + '_' +
           pool.loc[blind_mask, 'class_new'].astype(int).astype(str))
    xckey = (xc['lon'].round(9).astype(str) + '_' + xc['lat'].round(9).astype(str) + '_' +
             xc['class_new'].astype(int).astype(str))
    match_map = pd.Series(xc['match'].to_numpy(), index=xckey.to_numpy())
    pool['_fcs10_match'] = np.nan
    pool.loc[blind_mask, '_fcs10_match'] = match_map.reindex(key.to_numpy()).to_numpy()
    n_hit = pool.loc[blind_mask, '_fcs10_match'].notna().sum()
    audit('fcs10_xcheck_joined', blind=int(blind_mask.sum()), matched=int(n_hit))

    # 耕地/建成否决（nodata 兜底）
    V = pool[['wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0']].to_numpy(np.int8)
    valid = V >= 0
    nv = valid.sum(1)
    cb = (((V == 1) | (V == 6)) & valid).sum(1)
    pool['_crop_built_frac'] = np.where(nv > 0, cb / np.maximum(nv, 1), 0.0)

    src = pool['src'].fillna('').astype(str)
    thematic = src.str.contains(THEMATIC_RE, regex=True)
    is_shrub = pool.class_new.isin([120, 121])          # FCS10 重采，直接保留
    blind = pool.class_new.isin(BLIND).to_numpy()
    ok_tier = pool['tier'].isin(['gold', 'silver', 'bronze'])
    ok_ext = (pool['tier'] == 'external') & (
        (pool['agree_n'].fillna(0) >= C.VOTE_KEEP_EXTERNAL_MIN) | thematic)
    fm = pool['_fcs10_match'].to_numpy()
    keep_blind = np.where(pd.isna(fm), False,
                          np.where(fm == 'same', True,
                                   np.where(fm == 'nodata', pool['_crop_built_frac'].to_numpy() < 0.5,
                                            False)))
    keep = is_shrub | ok_tier | ok_ext | (blind & keep_blind)
    df = pool[keep].reset_index(drop=True)
    audit('gate_r4', n=int(len(df)),
          kept_blind_by_fcs10=int((blind & keep_blind).sum()),
          dropped_blind=int((blind & ~keep_blind & ~ok_tier & ~ok_ext).sum()))

    # ---------- C. 生态 ----------
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    audit('eco', n=int(len(df)))

    # ---------- D. 优先级 + 抽稀 ----------
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
    audit('adaptive_thin', n=int(len(df2)),
          by_class={str(int(k)): int(v) for k, v in df2.class_new.value_counts().sort_index().items()})

    # ---------- E. 配额 + 国界 ----------
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
          by_tier={k: int(v) for k, v in df2.tier.value_counts().items()},
          elapsed_min=round((time.time() - T0) / 60, 1))

    pool.to_parquet(os.path.join(STAGE, 'r4_pool.parquet'), index=False)
    df2.to_parquet(os.path.join(STAGE, 'r4_train.parquet'), index=False)
    json.dump(AUDIT, open(os.path.join(STAGE, 'r4_audit.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=int)
    import shutil
    for f in ['r4_pool.parquet', 'r4_train.parquet', 'r4_audit.json']:
        shutil.copyfile(os.path.join(STAGE, f), os.path.join(C.WORK, f))
    log(f'完成：r4_pool={len(pool):,} r4_train={len(df2):,}')

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
s5_votes_pool.py — r1 步骤4：五产品投票 + 置信分层 + 全池合并 → r1_pool.parquet
* 输入: r1_base(底座) + r1_ext(外部) + r1_topup(FCS10全覆盖重采)
* 投票（教师=各点自身 class_new 的 level0，票=WC21/ESRI20/TH17/CLCD30/CN30）:
    - 底座点: 直接 join 旧 v2 的投票列（v2 ≡ v1 逐点同集，s2 已校验；确定性查表复用），
              缺票点(如 v1 之外的新增)进入"补采"分支
    - 外部/topup 点: 用 lc14 同款采样器实测 5 产品（含 3×3 邻域多样性 div）
    - CLCD100 稳定列 stab_years: 底座 join v2；外部点不查（保持 -1）
* 分层（同 lc15 规则）:
    gold = agree==5 且非边界带; silver = agree>=4; bronze = agree==3;
    uncovered = 有效票<=3 且 agree<3; reject = 其余
    external 点一律 tier='external'，agree_n 供 s6 训练门槛
* 输出: r1_pool.parquet（lon/lat/class_new/tier/year/src/src_conf/agree_n/border/
        stab_years/wc_l0/esri_l0/th_l0/clcd_l0/cn30_l0/province）
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '0.本地流水线'))
import s0_conf as C
import s1_geom as G

VOTE_COLS = ['wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0']

def teacher_l0(cls):
    return np.array([C.LEVEL0[C.CLASSES[int(c)][3]] for c in cls], dtype=np.int8)

def tier_from_votes(agree, nvote, border):
    tier = np.full(len(agree), 'reject', dtype=object)
    tier[agree >= 3] = 'bronze'
    tier[agree >= 4] = 'silver'
    gold = (agree == 5) & (~border.astype(bool))
    tier[gold] = 'gold'
    tier[(nvote <= 3) & (tier == 'reject')] = 'uncovered'
    return tier

def main():
    t0 = time.time()
    base = pd.read_parquet(C.R1_BASE)
    print(f'底座 {len(base):,}', flush=True)

    # ---------- 1) 底座 join v2 投票列（cn30 除外：v2 的 cn30_l0 是错表产物，用 raw 重算） ----------
    VOTE_COLS_V2 = ['wc_l0', 'esri_l0', 'th_l0', 'clcd_l0']
    v2 = pd.read_parquet(C.V2, columns=['lon', 'lat'] + VOTE_COLS_V2 + ['border', 'stab_years'])
    v2['k'] = list(zip(v2.lon.round(9), v2.lat.round(9)))
    base['k'] = list(zip(base.lon.round(9), base.lat.round(9)))
    vmap = v2.set_index('k')
    joined = vmap.reindex(base['k'])
    del v2
    hit = joined[VOTE_COLS_V2[0]].notna().to_numpy()
    print(f'v2 四产品票命中: {hit.sum():,} / {len(base):,}（未命中 {len(base)-hit.sum():,} 走补采）', flush=True)

    # cn30_l0：从 v2_parts/F_cn30 的 raw 值按纠正后的 G30 码表重查
    print('cn30 票重算（G30 码表）…', flush=True)
    import glob as _glob
    fparts = sorted(_glob.glob(os.path.join(os.path.dirname(C.V1), 'v2_parts', 'F_cn30*.parquet')))
    v1 = pd.read_parquet(C.V1, columns=['lon', 'lat'])
    v1['k'] = list(zip(v1.lon.round(9), v1.lat.round(9)))
    v1['row_id'] = np.arange(len(v1))
    k2rid = v1.set_index('k')['row_id']
    rid = k2rid.reindex(base['k']).to_numpy()
    del v1
    fraw = pd.concat([pd.read_parquet(f) for f in fparts], ignore_index=True)
    fraw['miss'] = (fraw['cn30_raw'] == 255)
    fraw = fraw.sort_values(['row_id', 'miss']).drop_duplicates('row_id', keep='first')
    raw_by_rid = fraw.set_index('row_id')['cn30_raw']
    cn30_raw = raw_by_rid.reindex(rid).to_numpy()
    del fraw, raw_by_rid
    base['cn30_l0'] = np.where(np.isnan(cn30_raw.astype(float)), -1,
                               C.PROD_LUT['CN30'][np.nan_to_num(cn30_raw, nan=255).astype(np.uint8)])
    n_cn = int((base['cn30_l0'] >= 0).sum())
    print(f'   cn30 有效票: {n_cn:,} / {len(base):,}', flush=True)

    # ---------- 2) 未命中点（应为 3 个 bbox 边缘点）补采 ----------
    miss_idx = np.flatnonzero(~hit)
    if len(miss_idx):
        from s5_sample_products import sample_products  # lc14 同款采样器
        pts = base.iloc[miss_idx][['lon', 'lat']].reset_index(drop=True)
        votes_m = sample_products(pts)     # dict col->array
        for col in VOTE_COLS:
            arr = joined[col].to_numpy(dtype=float)
            arr[miss_idx] = votes_m[col]
            joined[col] = arr
        for col in ['border']:
            arr = joined[col].to_numpy(dtype=float)
            arr[miss_idx] = votes_m.get('border', np.zeros(len(pts)))[...]
            joined[col] = arr
        print(f'补采完成 {len(miss_idx)} 点', flush=True)

    for col in VOTE_COLS + ['border']:
        base[col] = joined[col].fillna(-1).astype(np.int8)
    base['stab_years'] = joined['stab_years'].fillna(-1).astype(np.float32).to_numpy()
    base = base.drop(columns=['k'])
    del joined
    print(f'投票列就绪 ({time.time()-t0:.0f}s)', flush=True)

    # 底座分层 + 补溯源列
    base['src'] = 'v2_glc_fcs30d2020_7yr_stable'
    base['src_conf'] = 0.95
    tl = teacher_l0(base['class_new'].to_numpy())
    votes = base[VOTE_COLS].to_numpy(dtype=np.int8)
    agree = (votes == tl[:, None]).sum(axis=1).astype(np.int8)
    nvote = (votes >= 0).sum(axis=1).astype(np.int8)
    base['agree_n'] = agree
    base['tier'] = tier_from_votes(agree, nvote, base['border'].to_numpy())
    print('底座分层:', dict(base['tier'].value_counts()), flush=True)

    # ---------- 3) 外部 + topup + 专题产品 + 补充源 点投票（本地实测，带缓存） ----------
    frames = [base]
    for f in [C.R1_EXT, C.R1_TOPUP, C.R1_THEMATIC, os.path.join(C.WORK, 'r1_ext2.parquet')]:
        if not os.path.exists(f):
            continue
        tag = os.path.splitext(os.path.basename(f))[0]
        cache = os.path.join(C.WORK, f'votes_{tag}.parquet')
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            print(f'[缓存] {tag}: {len(df):,}', flush=True)
            frames.append(df)
            continue
        df = pd.read_parquet(f)
        from s5_sample_products import sample_products
        pts = df[['lon', 'lat']].reset_index(drop=True)
        votes_x = sample_products(pts)
        for col in VOTE_COLS:
            df[col] = votes_x[col].astype(np.int8)
        df['border'] = votes_x['border'].astype(np.int8)
        df['stab_years'] = np.full(len(df), -1, dtype=np.float32)
        df['agree_n'] = (df[VOTE_COLS].to_numpy(dtype=np.int8) ==
                         teacher_l0(df['class_new'].to_numpy())[:, None]).sum(axis=1).astype(np.int8)
        df['tier'] = 'external'
        df.to_parquet(cache, index=False)
        frames.append(df)
        print(f'{tag}: {len(df):,}, agree_n 均值 {df.agree_n.mean():.2f}', flush=True)

    pool = pd.concat(frames, ignore_index=True, sort=False)

    # ---------- 4) 省归属（沿海滩涂等省界外点就近归属兜底） ----------
    print('省归属 …', flush=True)
    prov = G.province_of(pool['lon'].to_numpy(), pool['lat'].to_numpy())
    nun_mask = prov == '未匹配'
    if nun_mask.any():
        import shapely as _sp
        names, geoms, tree = G.provinces()
        pts = _sp.points(pool.loc[nun_mask, 'lon'].to_numpy(),
                         pool.loc[nun_mask, 'lat'].to_numpy())
        nearest = tree.query_nearest(pts)   # 兼容 (n,) 与 2×n 两种返回
        if nearest.ndim == 2:
            nearest = nearest[1]
        prov[nun_mask] = np.array(names)[nearest]
        print(f'就近归属兜底 {nun_mask.sum():,} 点（如沿海滩涂）', flush=True)
    pool['province'] = prov
    print('省归属完成', flush=True)

    cols = ['lon', 'lat', 'class_new', 'tier', 'year', 'src', 'src_conf',
            'agree_n', 'border', 'stab_years'] + VOTE_COLS + ['province']
    if 'class_raw' in pool.columns:
        cols.insert(3, 'class_raw')
    if 'urban' in pool.columns:
        cols.append('urban')
    pool = pool[cols]
    pool.to_parquet(C.R1_POOL, index=False)

    summary = {'total': int(len(pool)),
               'by_tier': {k: int(v) for k, v in pool['tier'].value_counts().items()},
               'by_class': {str(k): int(v) for k, v in pool['class_new'].value_counts().sort_index().items()},
               'elapsed_min': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(C.R1_POOL.replace('.parquet', '_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=int)
    print('\nr1_pool 类别分布（前12）:')
    for k, v in pool['class_new'].value_counts().sort_values(ascending=False).head(12).items():
        print(f'  {k:>4} {C.class_name(k):<12} {v:,}')
    print('\n输出:', C.R1_POOL, f'({time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()

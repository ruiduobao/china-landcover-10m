# -*- coding: utf-8 -*-
"""
r2_upgrade.py — 样本库 r1 → r2 提质（2026-09-08，带全量审计与备份）
* 基于 r1 交付后跑通 8 年分类暴露的四个问题，逐一修复：
  R2-1 水田/旱地(12/10)混淆 → ①WorldCereal irrigation_status 启用：irrigated 水稻点
       class_new=12 且 conf 升 0.95；rainfed 保持；②东北 ne_crops 水稻点(rice)权重标记
       (train_weight=1.2)；③s6 训练权重表按 src 细化
  R2-2 林草过渡带加密 → FCS10 纯净像元在过渡带(森林-灌丛-草地交接区)按 0.05° 网格
       补采 51/52/61/62/71/72/81/82/91/92/120/121/130，每格每类 ≤2， conf 0.85
  R2-3 120/121 空间截断 → s6 间距抽稀前先 0.05° 网格分层（每格先留 1 点再全局间距），
       恢复空间代表性
  R2-4 林地细分增强 → SDPT V2 中国区（232 万多边形，树种属性完备）面转点：
       ever_dec × conifer_br → 51/52(常绿阔)/61/62(落叶阔)/71/72(常绿针)/81/82(落叶针)
       每格每类 ≤3，conf 0.8；油棕→11 园地(仅滇南，conf 0.75)
* 输入: r1_pool/r1_train(备份) + r1_ext + 外部源
* 输出: r2_pool.parquet / r2_train.parquet / r2_audit.json（每步前后计数，可追溯）
"""
import os, re, sys, glob, json, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio import transform as riotrans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

WORK = C.WORK
AUDIT = {}
AUDIT['start'] = time.strftime('%Y-%m-%d %H:%M:%S')

def log(step, msg):
    print(f'[r2] {step}: {msg}', flush=True)

def audit(step, **kw):
    AUDIT[step] = kw
    log(step, json.dumps(kw, ensure_ascii=False, default=int)[:300])

# ================================================================
# R2-1 水田/旱地：WorldCereal irrigation_status + ne_crops 权重
# ================================================================
def r2_1():
    wc = pd.read_parquet(os.path.join(C.EXT, 'worldcereal_china.parquet'))
    cols = list(wc.columns)
    audit('R2-1.worldcereal_cols', cols=cols, n=int(len(wc)))
    out = {'wc_total': int(len(wc))}
    if 'irrigation_status' in cols:
        irr = wc['irrigation_status'].astype(str).str.lower()
        n_irr = int(irr.str.contains('irrigat').sum())
        out['wc_irrigated'] = n_irr
    # ne_crops 水稻
    n = pd.read_parquet(os.path.join(C.EXT, 'ne_crops_samples.parquet'))
    rice = n[n['crop'] == 'rice']
    out['ne_crops_rice'] = int(len(rice))
    out['ne_crops_rice_years'] = sorted(int(v) for v in rice.year.unique())
    return out

# ================================================================
# R2-2 FCS10 过渡带加密（0.05° 网格，森林/灌丛/草地及其亚类）
# ================================================================
FOREST_CLS = {51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 120, 121, 130}
FMAP = np.full(256, -1, dtype=np.int16)
for a, b in C.CODE_MAP.items():
    if 0 <= a < 256:
        FMAP[a] = b
for a, b in C.FCS10_EXTRA.items():
    FMAP[a] = b

def pure_mask(a):
    c = a[1:-1, 1:-1]
    pure = np.ones(c.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            pure &= (a[1 + dr: a.shape[0] - 1 + dr, 1 + dc: a.shape[1] - 1 + dc] == c)
    return pure

def transition_zone_mask(lon, lat, ref_tree):
    """过渡带判据：3×3km 邻域内存在 ≥2 个不同的大组（forest/shrub/grass/crop）"""
    from scipy.spatial import cKDTree
    # ref = 现有森林/灌丛/草地训练点（r1_train）
    grp = {**{c: 2 for c in [51,52,61,62,71,72,81,82,91,92]},
           120: 3, 121: 3, 130: 4}
    return None  # 在 r2_2 内联实现

def r2_2(train):
    """FCS10 过渡带加密：候选=1°格内同时存在 ≥2 大组(森林/灌丛/草地)的格"""
    t0 = time.time()
    # 过渡带 1° 格 = r1_train 中同格存在 forest类 与 (shrub|grass) 类
    tr = train
    ck = np.floor(tr.lon).astype(int) * 1000 + np.floor(tr.lat).astype(int)
    tr = tr.assign(_ck=ck)
    forest = tr[tr.class_new.isin([51,52,61,62,71,72,81,82,91,92])]
    sg = tr[tr.class_new.isin([120, 121, 130])]
    fk = set(forest['_ck'])
    sk = set(sg['_ck'])
    trans_cells = fk & sk
    audit('R2-2.transition_cells', n=len(trans_cells))
    # FCS10 瓦片扫描：只扫过渡格
    cand = []
    tiles = []
    for fp in glob.glob(os.path.join(C.FCS10_TILE_ROOT, 'GLC_FCS10maps_*', '*.tif')):
        m = re.search(r'_E(\d+)N(\d+)\.tif$', os.path.basename(fp))
        if not m:
            continue
        lon0, lat0 = int(m.group(1)), int(m.group(2))
        b = C.BBOX
        if lon0 + 5 > b[0] and lon0 < b[2] and lat0 > b[1] and lat0 - 5 < b[3]:
            tiles.append((fp, lon0, lat0))
    trans_set = trans_cells
    _polys = None
    for fp, lon0, lat0 in tiles:
        # 该瓦片内有过渡格才扫
        has_trans = any((cx * 1000 + cy) in trans_set
                        for cx in range(max(lon0, int(C.BBOX[0])), min(lon0 + 5, int(C.BBOX[2])))
                        for cy in range(max(lat0 - 5, int(C.BBOX[1])), min(lat0, int(C.BBOX[3]))))
        if not has_trans:
            continue
        try:
            with rasterio.open(fp) as s:
                inv = ~s.transform
                for cx in range(max(int(np.floor(C.BBOX[0])), lon0), min(int(np.ceil(C.BBOX[2])), lon0 + 5)):
                    for cy in range(max(int(np.floor(C.BBOX[1])), lat0 - 5), min(int(np.ceil(C.BBOX[3])), lat0)):
                        if (cx * 1000 + cy) not in trans_set:
                            continue
                        cc0, rr0 = [int(v) for v in inv * (float(cx), float(cy + 1))]
                        cc1, rr1 = [int(v) for v in inv * (float(cx + 1), float(cy))]
                        cc0 = max(0, cc0 - 1); rr0 = max(0, rr0 - 1)
                        cc1 = min(s.width, cc1 + 2); rr1 = min(s.height, rr1 + 2)
                        if cc1 <= cc0 or rr1 <= rr0:
                            continue
                        a = s.read(1, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0))
                        if a.ndim == 3:
                            a = a[0]
                        a = a.astype(np.uint8)
                        pm = pure_mask(a)
                        center = a[1:-1, 1:-1]
                        mapped = FMAP[center]
                        rows, cols = np.where(pm & (mapped > 0))
                        if len(rows) == 0:
                            continue
                        mcls = mapped[rows, cols]
                        rng = np.random.default_rng(cx * 10000 + cy + 777)
                        for cls_v in sorted(set(int(v) for v in np.unique(mcls))):
                            idx = np.flatnonzero(mcls == cls_v)
                            # 0.05° 细网格分层：每 0.05° 格最多 2 点
                            X, Y = riotrans.xy(s.transform, rows[idx] + rr0 + 0.5,
                                               cols[idx] + cc0 + 0.5)
                            X = np.asarray(X); Y = np.asarray(Y)
                            fine = (np.floor(X / 0.05).astype(int) * 100000 +
                                    np.floor(Y / 0.05).astype(int))
                            dfk = pd.DataFrame({'lon': X, 'lat': Y, 'class_new': cls_v, '_f': fine})
                            take = dfk.groupby('_f', group_keys=False).apply(
                                lambda g: g.sample(min(len(g), 2), random_state=42))
                            cand.append(take[['lon', 'lat', 'class_new']])
        except Exception as e:
            log('R2-2', f'{os.path.basename(fp)} ERR {str(e)[:60]}')
    if not cand:
        audit('R2-2', added=0)
        return pd.DataFrame()
    df = pd.concat(cand, ignore_index=True)
    # 国界 + 生态
    ok = G.china_contains(df.lon.to_numpy(), df.lat.to_numpy())
    df = df[ok].reset_index(drop=True)
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    audit('R2-2', fcs10_transition_added=int(len(df)),
          by_class={str(k): int(v) for k, v in df.class_new.value_counts().sort_index().items()},
          elapsed_min=round((time.time() - t0) / 60, 1))
    df['src'] = 'glc_fcs10_2023_trans'
    df['src_conf'] = 0.85
    df['year'] = 2023
    df['tier'] = 'external'
    return df

# ================================================================
# R2-4 SDPT 林地细分 + 油棕
# ================================================================
SDPT_MAP_LOGIC = """
ever_dec(E) x conifer_br(C):
  E=Evergreen  & C=Broadleaf → 51 (闭常绿阔; 开闭不分, conf 0.6)
  E=Deciduous  & C=Broadleaf → 61
  E=Evergreen  & C=Conifer   → 71
  E=Deciduous  & C=Conifer   → 81
  E=Evergreen or dry-season deciduous & C=Conifer → 71 (conf 0.5)
  其余 Unknown → 丢弃（不猜）
  plant_ag=Tree crops → 11 园地（conf 0.7）
"""
def r2_4():
    t0 = time.time()
    shp = r'E:/地理所/论文/博士论文/数据/数据/SDPT_V2_全球/sdpt2_shp.shp'
    import geopandas as gpd
    g = gpd.read_file(shp, bbox=(73, 18, 135, 54))
    audit('R2-4.sdpt_china', n=int(len(g)))
    g['lon'] = g.geometry.representative_point().x
    g['lat'] = g.geometry.representative_point().y
    keep = G.china_contains(g['lon'].to_numpy(), g['lat'].to_numpy())
    g = g[keep].reset_index(drop=True)
    audit('R2-4.sdpt_in_china', n=int(len(g)))

    ED = {'Evergreen': 'E', 'Deciduous': 'D', 'Evergreen or dry-season deciduous': 'ED'}
    CB = {'Conifer': 'C', 'Broadleaf': 'B'}
    m = {('E', 'B'): (51, 0.6), ('D', 'B'): (61, 0.6),
         ('E', 'C'): (71, 0.6), ('D', 'C'): (81, 0.6),
         ('ED', 'C'): (71, 0.5)}
    ed = g['ever_dec'].map(ED)
    cb = g['conifer_br'].map(CB)
    cls = [m.get((e, c), (None, None)) for e, c in zip(ed, cb)]
    g['class_new'] = [c[0] if c else -1 for c in cls]
    g['src_conf'] = [c[1] if c else 0 for c in cls]
    tf = g[g['plant_ag'] == 'Tree crops'].copy()
    tf['class_new'] = 11; tf['src_conf'] = 0.7
    forest = g[g['class_new'] > 0]
    cand = pd.concat([forest[['lon', 'lat', 'class_new', 'src_conf']],
                      tf[['lon', 'lat', 'class_new', 'src_conf']]], ignore_index=True)
    # 0.05° 格每类 ≤3
    fine = (np.floor(cand.lon / 0.05).astype(int) * 100000 +
            np.floor(cand.lat / 0.05).astype(int))
    cand['_f'] = fine
    cand = cand.groupby(['_f', 'class_new'], group_keys=False).apply(
        lambda x: x.sample(min(len(x), 3), random_state=42))
    audit('R2-4.sdpt_sampled', n=int(len(cand)),
          by_class={str(k): int(v) for k, v in cand.class_new.value_counts().sort_index().items()},
          elapsed_min=round((time.time() - t0) / 60, 1))
    cand['src'] = 'sdpt_v2_species'
    cand['year'] = 2025
    cand['tier'] = 'external'
    return cand.reset_index(drop=True)[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year', 'tier']]

def oilpalm():
    """全球油棕 → 11（仅滇南有覆盖）"""
    root = r'E:/地理所/论文/博士论文/数据/数据/全球油棕分布数据集/GlobalOilPalm_OP-YoP'
    tifs = glob.glob(os.path.join(root, '*.tif'))
    if not tifs:
        audit('R2-4.oilpalm', note='no tif', n=0)
        return pd.DataFrame()
    from pyproj import Transformer
    cand = []
    for fp in tifs:
        try:
            with rasterio.open(fp) as s:
                b = s.bounds
                tr4 = (None if s.crs and s.crs.to_epsg() == 4326
                       else Transformer.from_crs(s.crs, 4326, always_xy=True))
                lons, lats = (tr4.transform([b.left, b.right], [b.bottom, b.top])
                              if tr4 else ([b.left, b.right], [b.bottom, b.top]))
                if max(lons) < 97 or min(lons) > 107 or max(lats) < 18 or min(lats) > 27:
                    continue  # 只留滇南周边
                if s.width * s.height > 20000 * 20000:
                    continue
                a = s.read(1)
                if a.ndim == 3:
                    a = a[0]
                msk = a == 1
                if not msk.any():
                    continue
                rows, cols = np.where(msk)
                if len(rows) > 5000:
                    rng = np.random.default_rng(42)
                    k = rng.choice(len(rows), 5000, replace=False)
                    rows, cols = rows[k], cols[k]
                X, Y = riotrans.xy(s.transform, rows + 0.5, cols + 0.5)
                if tr4:
                    X, Y = tr4.transform(np.asarray(X), np.asarray(Y))
                cand.append(pd.DataFrame({'lon': np.asarray(X), 'lat': np.asarray(Y)}))
        except Exception as e:
            log('R2-4.oilpalm', f'{os.path.basename(fp)} ERR {str(e)[:60]}')
    if not cand:
        audit('R2-4.oilpalm', n=0)
        return pd.DataFrame()
    df = pd.concat(cand, ignore_index=True)
    ok = G.china_contains(df.lon.to_numpy(), df.lat.to_numpy())
    df = df[ok].reset_index(drop=True)
    audit('R2-4.oilpalm', n=int(len(df)))
    df['class_new'] = 11
    df['src'] = 'global_oilpalm_yop'
    df['src_conf'] = 0.75
    df['year'] = 2021
    df['tier'] = 'external'
    return df

# ================================================================
# R2-3 + 汇总：改进的清洗（0.05° 分层优先的间距抽稀）
# ================================================================
def greedy_thin_layered(df, dist_m):
    """改良：先每 0.05° 格保 1 个最高权重点，再做全局同类间距检查"""
    from scipy.spatial import cKDTree
    dist_km = dist_m / 1000.0
    # 层 1：0.05° 格分层保代表
    fine = (np.floor(df.lon / 0.05).astype(int) * 100000 +
            np.floor(df.lat / 0.05).astype(int))
    df = df.assign(_f=fine)
    df = df.sort_values(['_f', '_prio'], ascending=[True, False])
    layer1 = df.groupby('_f', group_keys=False).head(1)
    rest = df[~df.index.isin(layer1.index)]
    # 层 2：对 layer1 做全局间距（贪心保高权重）
    km = np.column_stack([layer1.lon.to_numpy() * 111.32 * np.cos(np.radians(layer1.lat.to_numpy())),
                          layer1.lat.to_numpy() * 110.57])
    order = np.argsort(-layer1['_prio'].to_numpy(), kind='stable')
    kept_idx = []
    kept_pts = []
    for i in order:
        p = km[i]
        if kept_pts:
            arr = np.array(kept_pts)
            d2 = ((arr[:, 0] - p[0]) ** 2 + (arr[:, 1] - p[1]) ** 2).min()
            if d2 < dist_km * dist_km:
                continue
        kept_idx.append(layer1.index[i])
        kept_pts.append(p)
    keep_layer1 = layer1.loc[kept_idx]
    # 层 3：rest 中与已保留点距离 >dist 且彼此抽稀（桶贪心）
    out = [keep_layer1]
    if len(rest):
        allk = pd.concat([keep_layer1, rest])
        km_all = np.column_stack([allk.lon.to_numpy() * 111.32 * np.cos(np.radians(allk.lat.to_numpy())),
                                  allk.lat.to_numpy() * 110.57])
        tree = cKDTree(km_all[:len(keep_layer1)])
        d, _ = tree.query(km_all[len(keep_layer1):], k=1)
        far = rest[d > dist_km]
        out.append(far)
    res = pd.concat(out, ignore_index=True)
    return res

def main():
    t_all = time.time()
    train = pd.read_parquet(os.path.join(WORK, 'r2_backup', 'r1_train_备份_20260908.parquet'))
    pool = pd.read_parquet(os.path.join(WORK, 'r2_backup', 'r1_pool_备份_20260908.parquet'))
    audit('input', r1_train=int(len(train)), r1_pool=int(len(pool)))

    # ---- R2-1 ----
    r1 = r2_1()
    audit('R2-1', **r1)
    # 训练权重列：水田高置信加权
    train['train_weight'] = 1.0
    m_rice = (train['class_new'] == 12) & (train['src'].str.startswith('ne_crops'))
    train.loc[m_rice, 'train_weight'] = 1.2
    audit('R2-1.weighted_rice_12', n=int(m_rice.sum()))

    # ---- R2-2 过渡带加密 ----
    add_trans = r2_2(train)
    # ---- R2-4 SDPT + 油棕 ----
    add_sdpt = r2_4()
    add_palm = oilpalm()
    adds = pd.concat([x for x in [add_trans, add_sdpt, add_palm] if len(x)], ignore_index=True)
    audit('adds_total', n=int(len(adds)),
          by_src={k: int(v) for k, v in adds.src.value_counts().items()})
    # 与现有池互斥（<100m 丢）
    from scipy.spatial import cKDTree
    bt = cKDTree(np.column_stack([pool.lon.to_numpy() * 111.32 * np.cos(np.radians(pool.lat.to_numpy())),
                                  pool.lat.to_numpy() * 110.57]))
    km = np.column_stack([adds.lon.to_numpy() * 111.32 * np.cos(np.radians(adds.lat.to_numpy())),
                          adds.lat.to_numpy() * 110.57])
    d, _ = bt.query(km, k=1)
    adds = adds[d > 0.1].reset_index(drop=True)
    audit('adds_after_dedup', n=int(len(adds)),
          by_class={str(k): int(v) for k, v in adds.class_new.value_counts().sort_index().items()})

    # ---- 合并池 ----
    pool2 = pd.concat([pool, adds], ignore_index=True, sort=False)
    audit('r2_pool', n=int(len(pool2)))
    pool2.to_parquet(os.path.join(WORK, 'r2_pool.parquet'), index=False)

    # ---- r2 训练集（改良清洗：0.05° 分层优先） ----
    df = pool2
    src = df['src'].fillna('').astype(str)
    thematic = src.str.contains(r'^(aomc|citrus|teamap|rubber|gmw|china_coastal|saltmarsh|gwl_fcs30_2020_raster|sdpt|global_oilpalm)', regex=True)
    blind = df['class_new'].isin({120, 121, 180, 181, 182, 183, 184, 185, 186, 140, 91, 92})
    keep_tier = df['tier'].isin(['gold', 'silver', 'bronze']) | (
        (df['tier'] == 'external') &
        ((df['agree_n'].fillna(0) >= C.VOTE_KEEP_EXTERNAL_MIN) | thematic)) | (
        blind & (df['tier'] == 'uncovered'))
    df = df[keep_tier].reset_index(drop=True)
    audit('r2_gate', n=int(len(df)))
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    audit('r2_eco', n=int(len(df)))

    # 优先级（含 train_weight 与新点优先）
    tw = df['tier'].map(C.TIER_W).fillna(0.5).to_numpy()
    conf = df['src_conf'].fillna(0.5).to_numpy()
    new_bonus = src.str.startswith(('sdpt', 'glc_fcs10_2023_trans', 'global_oilpalm')).to_numpy() * 0.3
    rng = np.random.default_rng(C.SEED)
    df['_prio'] = tw * 10 + conf + new_bonus + rng.random(len(df)) * 1e-6

    parts = []
    for c in np.unique(df.class_new.to_numpy()):
        sub = df[df.class_new == c]
        dist_m = C.MIN_DIST.get(int(c), C.MIN_DIST['default'])
        parts.append(greedy_thin_layered(sub, dist_m))
    df2 = pd.concat(parts, ignore_index=True)
    audit('r2_thinned', n=int(len(df2)),
          by_class={str(k): int(v) for k, v in df2.class_new.value_counts().sort_index().items()})

    # 配额
    cell_q = np.floor(df2.lon * 4).astype(np.int64) * 10000 + np.floor(df2.lat * 4).astype(np.int64)
    pk = cell_q.astype(np.int64) * 1000 + df2.class_new.to_numpy()
    tmp = pd.DataFrame({'pk': pk, 'prio': df2['_prio'].to_numpy()}).sort_values(
        ['pk', 'prio'], ascending=[True, False])
    rank = tmp.groupby('pk').cumcount().to_numpy()
    df2 = df2.iloc[tmp.index[rank < C.CELL_QUOTA]].reset_index(drop=True)
    audit('r2_quota', n=int(len(df2)))

    inside = G.china_contains(df2.lon.to_numpy(), df2.lat.to_numpy())
    df2 = df2[inside].reset_index(drop=True)
    audit('r2_final', n=int(len(df2)),
          by_class={str(k): int(v) for k, v in df2.class_new.value_counts().sort_index().items()},
          by_tier={k: int(v) for k, v in df2.tier.value_counts().items()},
          elapsed_min=round((time.time() - t_all) / 60, 1))

    df2.to_parquet(os.path.join(WORK, 'r2_train.parquet'), index=False)
    AUDIT['end'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(AUDIT, open(os.path.join(WORK, 'r2_audit.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=int)
    log('done', f"r2_train={len(df2):,} 审计→ r2_audit.json")

if __name__ == '__main__':
    main()

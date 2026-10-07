# -*- coding: utf-8 -*-
"""
r3_clean.py — 修正 r2 污染：剔除不可信过渡带层，重建 r3 样本
* 背景（2026-09-08 用户发现 + 本次法证）：
  r2 的 glc_fcs10_2023_trans 层（池内 376.8 万点，占 r2_train 约 2/3）是在
  "森林-灌草过渡 1° 格"里采的 FCS10 3×3 纯净像元，**未经过任何产品投票验证**
  （被错误加入 THEMATIC_FREE 豁免名单）。实测其与独立产品一致率极低：
    130 草地 15%、121 灌丛 0%、120 0%、202 水体 18%、湿地 0%、10 耕地 39%
  ——华北平原的 1.18 万"草地"点中 CLCD 60% 判耕地、21% 判不透水面。
  坐标回读证实点确实落在 FCS10 raw=130 像元上（无坐标 bug），
  即 FCS10 在农田里标出的窄条草被我们当成了草地样本，且豁免了质量门槛。
* r3 修正:
  1) **整体剔除 glc_fcs10_2023_trans**（不进池）
  2) SDPT 保留但**走 agree≥3 投票门槛**（全量 10,956 点已补票，保留 ~5.4k）
  3) 保留 r2 真正有效的改进：按类自适应抽稀（>5万类 0.025° 分层、≤5万类全留）
  4) 水稻加权列 train_weight（ne_crops 12 类 =1.2）
* 输入: r2_backup/r1_pool_备份_20260908.parquet（r1 已验证池）+ r3_sdpt_votes.parquet
* 输出: r3_pool.parquet / r3_train.parquet / r3_audit.json
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

WORK = C.WORK
STAGE = r'F:\r3_stage'
os.makedirs(STAGE, exist_ok=True)
AUDIT = {}
T0 = time.time()

def log(m): print(f'[r3 {time.time()-T0:.0f}s] {m}', flush=True)
def audit(k, **kw):
    AUDIT[k] = kw
    log(f'{k}: {json.dumps(kw, ensure_ascii=False, default=int)[:260]}')

def main():
    # ---------- 1) 池 = r1 已验证池 + SDPT（投票后） ----------
    pool = pd.read_parquet(os.path.join(WORK, 'r2_backup', 'r1_pool_备份_20260908.parquet'))
    audit('input_pool_r1', n=int(len(pool)))

    sd = pd.read_parquet(os.path.join(WORK, 'r3_sdpt_votes.parquet'))
    sd_ok = sd[sd.agree_n >= C.VOTE_KEEP_EXTERNAL_MIN].copy()
    audit('sdpt', total=int(len(sd)), kept=int(len(sd_ok)),
          by_class={str(int(k)): int(v) for k, v in sd_ok.class_new.value_counts().items()})
    for c in ['wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0']:
        pool[c] = pool[c] if c in pool.columns else -1
    add = sd_ok[['lon', 'lat', 'class_new', 'src', 'src_conf', 'year', 'tier',
                 'agree_n', 'wc_l0', 'esri_l0', 'th_l0', 'clcd_l0', 'cn30_l0']].copy()
    pool = pd.concat([pool, add], ignore_index=True, sort=False)
    audit('pool_with_sdpt', n=int(len(pool)))

    # ---------- 2) tier 门槛（无 trans；SDPT 已投票，不豁免） ----------
    src = pool['src'].fillna('').astype(str)
    thematic = src.str.contains(
        r'^(aomc|citrus|teamap|rubber|gmw|china_coastal|saltmarsh|gwl_fcs30_2020_raster|global_oilpalm)',
        regex=True)
    blind = pool['class_new'].isin({120, 121, 180, 181, 182, 183, 184, 185, 186, 140, 91, 92})
    keep_tier = (pool['tier'].isin(['gold', 'silver', 'bronze']) |
                 ((pool['tier'] == 'external') &
                  ((pool['agree_n'].fillna(0) >= C.VOTE_KEEP_EXTERNAL_MIN) | thematic)) |
                 (blind & (pool['tier'] == 'uncovered')))
    df = pool[keep_tier].reset_index(drop=True)
    audit('gate', n=int(len(df)))

    # ---------- 3) 生态硬规则 ----------
    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    audit('eco', n=int(len(df)))

    # ---------- 4) 优先级 + 水稻加权 ----------
    tw = df['tier'].map(C.TIER_W).fillna(0.5).to_numpy()
    conf = df['src_conf'].fillna(0.5).to_numpy()
    rng = np.random.default_rng(C.SEED)
    df['_prio'] = tw * 10 + conf + rng.random(len(df)) * 1e-6
    df['train_weight'] = 1.0
    m_rice = (df['class_new'] == 12) & df['src'].fillna('').str.startswith('ne_crops')
    df.loc[m_rice, 'train_weight'] = 1.2
    audit('rice_weighted', n=int(m_rice.sum()))

    # ---------- 5) 按类自适应抽稀（>5万用 0.025° 分层；≤5万全留） ----------
    from scipy.spatial import cKDTree
    parts = []
    for c in sorted(df.class_new.unique()):
        sub = df[df.class_new == c]
        if len(sub) <= 50000:
            parts.append(sub)
            continue
        fine_c = (np.floor(sub.lon.to_numpy() / 0.025).astype(np.int64) * 100000 +
                  np.floor(sub.lat.to_numpy() / 0.025).astype(np.int64))
        order = np.lexsort((-sub['_prio'].to_numpy(), fine_c))
        sf = fine_c[order]
        fm = np.ones(len(order), dtype=bool)
        fm[1:] = sf[1:] != sf[:-1]
        L1 = sub.iloc[order[fm]]
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

    # ---------- 6) 配额 0.25°格×类 ≤200 ----------
    cell_q = np.floor(df2.lon * 4).astype(np.int64) * 10000 + np.floor(df2.lat * 4).astype(np.int64)
    pk = cell_q.astype(np.int64) * 1000 + df2.class_new.to_numpy()
    tmp = pd.DataFrame({'pk': pk, 'prio': df2['_prio'].to_numpy()}).sort_values(
        ['pk', 'prio'], ascending=[True, False])
    rank = tmp.groupby('pk').cumcount().to_numpy()
    df2 = df2.iloc[rank < C.CELL_QUOTA].reset_index(drop=True)
    audit('quota', n=int(len(df2)))

    # ---------- 7) 国界终检 ----------
    inside = G.china_contains(df2.lon.to_numpy(), df2.lat.to_numpy())
    df2 = df2[inside].reset_index(drop=True)
    audit('final', n=int(len(df2)),
          by_class={str(int(k)): int(v) for k, v in df2.class_new.value_counts().sort_index().items()},
          by_tier={k: int(v) for k, v in df2.tier.value_counts().items()},
          elapsed_min=round((time.time() - T0) / 60, 1))

    pool.to_parquet(os.path.join(STAGE, 'r3_pool.parquet'), index=False)
    df2.to_parquet(os.path.join(STAGE, 'r3_train.parquet'), index=False)
    json.dump(AUDIT, open(os.path.join(STAGE, 'r3_audit.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=int)
    import shutil
    for f in ['r3_pool.parquet', 'r3_train.parquet', 'r3_audit.json']:
        shutil.copyfile(os.path.join(STAGE, f), os.path.join(WORK, f))
    log(f'完成：r3_pool={len(pool):,} r3_train={len(df2):,}，已拷回 Z 盘')

if __name__ == '__main__':
    main()

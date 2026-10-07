# -*- coding: utf-8 -*-
"""
r2_fast.py — r2 清洗段快版（跳过已完成的 R2-2/R2-4，直接用 r2_pool.parquet）
* 输入: 数据/本地处理/样本重建/r2_pool.parquet（池 1295 万，已含全部新增源）
* 工作目录: F:/r1_stage/r2/（SSD），完成后拷回 Z:
* 抽稀: 0.05° 分层（向量化 sort+drop_duplicates）+ 桶贪心（纯 Python ~10 分钟）
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s0_conf as C
import s1_geom as G

STAGE = r'F:\r1_stage\r2'
os.makedirs(STAGE, exist_ok=True)
AUDIT = {}
T0 = time.time()

def log(m):
    print(f'[r2fast {time.time()-T0:.0f}s] {m}', flush=True)

def audit(k, **kw):
    AUDIT[k] = kw
    log(f'{k}: {json.dumps(kw, ensure_ascii=False, default=int)[:250]}')

def main():
    pool = pd.read_parquet(os.path.join(C.WORK, 'r2_pool.parquet'))
    audit('input', n=int(len(pool)))

    src = pool['src'].fillna('').astype(str)
    thematic = src.str.contains(r'^(aomc|citrus|teamap|rubber|gmw|china_coastal|saltmarsh|'
                                r'gwl_fcs30_2020_raster|sdpt|global_oilpalm|glc_fcs10_2023_trans)', regex=True)
    blind = pool['class_new'].isin({120, 121, 180, 181, 182, 183, 184, 185, 186, 140, 91, 92})
    keep_tier = (pool['tier'].isin(['gold', 'silver', 'bronze']) |
                 ((pool['tier'] == 'external') &
                  ((pool['agree_n'].fillna(0) >= C.VOTE_KEEP_EXTERNAL_MIN) | thematic)) |
                 (blind & (pool['tier'] == 'uncovered')))
    df = pool[keep_tier].reset_index(drop=True)
    audit('gate', n=int(len(df)))

    bad = C.eco_violation(df.lon.to_numpy(), df.lat.to_numpy(), df.class_new.to_numpy())
    df = df[~bad].reset_index(drop=True)
    audit('eco', n=int(len(df)))

    # 优先级（新源+0.05 分层内首点加权在排序里体现）
    tw = df['tier'].map(C.TIER_W).fillna(0.5).to_numpy()
    conf = df['src_conf'].fillna(0.5).to_numpy()
    bonus = src.str.startswith(('sdpt', 'glc_fcs10_2023_trans', 'global_oilpalm')).to_numpy() * 0.3
    rng = np.random.default_rng(C.SEED)
    prio = tw * 10 + conf + bonus + rng.random(len(df)) * 1e-6
    df['_prio'] = prio
    fine = (np.floor(df.lon / 0.05).astype(np.int64) * 100000 +
            np.floor(df.lat / 0.05).astype(np.int64))

    # 0.05° 分层保首点（向量化：排序后每格取第一行）
    order = np.lexsort((-df['_prio'].to_numpy(), fine))
    df_sorted = df.iloc[order]
    first_mask = np.ones(len(df_sorted), dtype=bool)
    first_mask[1:] = fine[order][1:] != fine[order][:-1]
    layer1_pos = order[first_mask]
    sel_idx = set(df.index[layer1_pos].tolist())
    # 高置信新源（SDPT/FCS10trans/油棕）直通层1
    hi = df.index[bonus >= 0.3]
    sel_idx.update(hi.tolist())
    df = df.assign(_sel=df.index.isin(sel_idx))
    log(f'层1(0.05°分层+新源直通): {df._sel.sum():,} / {len(df):,}')
    audit('layer1', n=int(df._sel.sum()))

    # 桶贪心（全部点按优先级，含层1与未选中者，公平竞争）
    km = np.column_stack([df.lon.to_numpy() * 111.32 * np.cos(np.radians(df.lat.to_numpy())),
                          df.lat.to_numpy() * 110.57])
    order2 = np.lexsort((-df['_sel'].astype(int) * 100, -df['_prio'].to_numpy()))
    kept = {}
    keep_pos = []
    for pos in order2:
        kx = int(np.floor(km[pos, 0] / 0.5)); ky = int(np.floor(km[pos, 1] / 0.5))
        okk = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in kept.get((kx + dx, ky + dy), ()):
                    if (km[pos, 0] - km[j, 0]) ** 2 + (km[pos, 1] - km[j, 1]) ** 2 < 0.25:
                        okk = False; break
                if not okk: break
            if not okk: break
        if okk:
            kept.setdefault((kx, ky), []).append(pos)
            keep_pos.append(pos)
    log(f'间距抽稀后: {len(keep_pos):,}')
    df2 = df.iloc[sorted(keep_pos)].reset_index(drop=True)
    audit('thinned', n=int(len(df2)),
          by_class={str(int(k)): int(v) for k, v in df2.class_new.value_counts().sort_index().items()})

    # 配额
    cell_q = np.floor(df2.lon * 4).astype(np.int64) * 10000 + np.floor(df2.lat * 4).astype(np.int64)
    pk = cell_q.astype(np.int64) * 1000 + df2.class_new.to_numpy()
    tmp = pd.DataFrame({'pk': pk, 'prio': df2['_prio'].to_numpy(),
                        '_sel': df2['_sel'].to_numpy()}).sort_values(
        ['pk', '_sel', 'prio'], ascending=[True, False, False])
    rank = tmp.groupby('pk').cumcount().to_numpy()
    df2 = df2.iloc[tmp.index[rank < C.CELL_QUOTA]].reset_index(drop=True)
    audit('quota', n=int(len(df2)))

    inside = G.china_contains(df2.lon.to_numpy(), df2.lat.to_numpy())
    df2 = df2[inside].reset_index(drop=True)
    audit('final', n=int(len(df2)),
          by_class={str(int(k)): int(v) for k, v in df2.class_new.value_counts().sort_index().items()},
          by_tier={k: int(v) for k, v in df2.tier.value_counts().items()},
          elapsed_min=round((time.time() - T0) / 60, 1))

    df2.to_parquet(os.path.join(STAGE, 'r2_train.parquet'), index=False)
    json.dump(AUDIT, open(os.path.join(STAGE, 'r2_audit.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2, default=int)
    # 拷回 Z 盘
    import shutil
    shutil.copyfile(os.path.join(STAGE, 'r2_train.parquet'),
                    os.path.join(C.WORK, 'r2_train.parquet'))
    shutil.copyfile(os.path.join(STAGE, 'r2_audit.json'),
                    os.path.join(C.WORK, 'r2_audit.json'))
    log(f'完成：r2_train={len(df2):,}，已拷回 Z 盘')

if __name__ == '__main__':
    main()

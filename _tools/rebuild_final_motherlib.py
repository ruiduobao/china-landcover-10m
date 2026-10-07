# -*- coding: utf-8 -*-
"""
rebuild_final_motherlib.py — 用幸存母库重算 r11 终版母库（本地，零 GEE）

背景：Z 盘格式化后，r11 修复过的终版母库（2,241,626 点 + 显式 row_id）丢失；
本地仅存 r11 之前的交付版（2,241,785 点，无 row_id）。
本脚本按 r10_purge_misplaced.py 的原始规则（s0_conf.ECO_PROVINCE + s1_geom.province_violation）
在存活副本上重算违规清单，再执行 r11_fix_motherlib.py 的修复动作，重建终版母库。

输出：
  数据/本地处理/全国清洗训练/错位样本剔除清单.parquet     （基线部分，审计留档）
  数据/本地处理/样本重建/r7_train.parquet                 （终版母库，含 row_id）
  数据/本地处理/样本重建/r7_train_validity.parquet        （同步）
"""
import os, sys, time, shutil
import numpy as np
import pandas as pd

ROOT = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
SRC_TRAIN = os.path.join(ROOT, r'数据\样本交付\r7_train.parquet')          # 交付版（r11 前）
SRC_VALID = r'F:\r7_prod\r7_train_validity.parquet'                        # 交付版 validity
OUT_MOTHER = os.path.join(ROOT, r'数据\本地处理\样本重建')
OUT_BASE = os.path.join(ROOT, r'数据\本地处理\全国清洗训练')
BOUND = os.path.join(ROOT, r'数据\边界\china_100000_full.json')

# 让 s0_conf 可在无 Z 盘环境下导入：屏蔽其对 Z 路径的 makedirs 副作用
_real_makedirs = os.makedirs
def _safe_makedirs(p, *a, **k):
    try:
        _real_makedirs(p, *a, **k)
    except Exception:
        pass
os.makedirs = _safe_makedirs

sys.path.insert(0, os.path.join(ROOT, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(ROOT, '代码', '5.样本重建'))
import s0_conf as C
import s1_geom as G
os.makedirs = _real_makedirs

C.PROJ = ROOT
C.BOUND_JSON = BOUND          # 重定向到本地省界文件
assert os.path.isfile(C.BOUND_JSON), C.BOUND_JSON
assert os.path.isfile(SRC_TRAIN), SRC_TRAIN


def main():
    t0 = time.time()
    os.makedirs(OUT_MOTHER, exist_ok=True)
    os.makedirs(OUT_BASE, exist_ok=True)

    t = pd.read_parquet(SRC_TRAIN)
    v = pd.read_parquet(SRC_VALID)
    print(f'读入交付版母库: train {len(t):,} / validity {len(v):,}', flush=True)
    assert len(t) == len(v), '行数不一致'
    assert (t.lon.to_numpy() == v.lon.to_numpy()).all(), '行不对齐'

    # ---- r10 规则：省级白名单违规判定（基线部分）----
    lon = t.lon.to_numpy(); lat = t.lat.to_numpy(); cls = t.class_new.to_numpy()
    prov = G.province_of(lon, lat)
    bad = G.province_violation(lon, lat, cls)
    ids = np.flatnonzero(bad).astype(np.int64)
    print(f'省级白名单违规（基线）: {len(ids):,} 点', flush=True)

    p = pd.DataFrame({
        'row_id': ids, 'class_new': cls[ids], 'lon': lon[ids], 'lat': lat[ids],
        'province': prov[ids],
        'src': t.src.to_numpy()[ids] if 'src' in t.columns else '',
        'origin': 'base_r7_train', 'reason': 'province_whitelist_violation'})
    for c, g in p.groupby('class_new'):
        print(f'  {c} {len(g):>5} 点: {dict(g.province.value_counts().head(5))}', flush=True)
    p.to_parquet(os.path.join(OUT_BASE, '错位样本剔除清单.parquet'), index=False)

    # ---- r11 修复：显式 row_id + 删违规行 ----
    t2 = t.copy(); v2 = v.copy()
    t2.insert(0, 'row_id', np.arange(len(t2), dtype=np.int64))
    v2.insert(0, 'row_id', np.arange(len(v2), dtype=np.int64))
    t2 = t2[~t2.row_id.isin(ids)].reset_index(drop=True)
    v2 = v2[~v2.row_id.isin(ids)].reset_index(drop=True)
    assert len(t2) == len(t) - len(ids) and len(v2) == len(v) - len(ids)
    t2.to_parquet(os.path.join(OUT_MOTHER, 'r7_train.parquet'), index=False)
    v2.to_parquet(os.path.join(OUT_MOTHER, 'r7_train_validity.parquet'), index=False)

    print(f'\n终版母库: {len(t):,} → {len(t2):,}（-{len(ids)}）', flush=True)
    print('剩余 140/183/81:',
          {int(c): int(n) for c, n in
           t2[t2.class_new.isin([140, 183, 81])].class_new.value_counts().items()}, flush=True)
    print(f'输出: {OUT_MOTHER}', flush=True)
    print(f'({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()

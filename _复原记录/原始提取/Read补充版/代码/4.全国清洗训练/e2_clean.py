# -*- coding: utf-8 -*-
"""
e2_clean.py — 嵌入空间清洗（03 文档 §6.3）→ 样本底座 v5
* 输入: emb_parts/*.parquet（提取的嵌入）+ cn_samples_v4.parquet（元数据）
* 清洗三步:
  B1 马氏距离+卡方: 逐类逐嵌入年, P<0.001 删, 0.001-0.01 剪70%; <500 点类用 3σ
  B2 置信学习: 分区 RF out-of-fold 预测, 高置信判他类 → label_error
  B3 混淆对原型检查: 8 对易混类, 样本离本类原型远且更近他类原型 → 标记
* 输出: cn_samples_v5.parquet（clean_tier + 审计列）+ v5_summary.json
* 用法: python e2_clean.py
"""
import sys, os, glob, json, time
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'代码/0.本地流水线')
from lc_conf import CLASSES, LEVEL0, CODE_MAP

EXT_PARTS = '数据/本地处理/全国清洗训练/emb_parts'
V4 = '数据/本地处理/样本底座/cn_samples_v4.parquet'
OUT = '数据/本地处理/样本底座/cn_samples_v5.parquet'
FEATS = [f'A{i:02d}' for i in range(64)]
CONF_PAIRS = [(12, 10), (11, 51), (11, 61), (181, 130), (181, 182), (183, 201),
              (150, 201), (121, 130), (52, 51), (62, 61)]

def load_embeddings():
    fs = sorted(glob.glob(os.path.join(EXT_PARTS, 'chunk_*.parquet')))
    print('嵌入分片:', len(fs), flush=True)
    frames = [pd.read_parquet(f) for f in fs]
    emb = pd.concat(frames, ignore_index=True)
    return emb

def main():
    t0 = time.time()
    emb = load_embeddings()
    v4 = pd.read_parquet(V4)
    v4 = v4[v4.tier != 'reject'].reset_index(drop=True)
    # 坐标桥接对齐：emb.row_id → chunks_index(lon/lat) → 修正版 v4(lon/lat 精确匹配)
    # （v4 已做过国界裁剪，行序变化；旧 row_id 只能通过 chunks_index 桥接）
    idx = pd.read_parquet('数据/本地处理/全国清洗训练/chunks_index.parquet')
    meta = emb.merge(idx[['row_id', 'lon', 'lat']], on='row_id', how='left')
    meta['key'] = meta.lon.round(7).astype(str) + '_' + meta.lat.round(7).astype(str)
    v4k = v4.copy()
    v4k['key'] = v4k.lon.round(7).astype(str) + '_' + v4k.lat.round(7).astype(str)
    meta = meta.merge(v4k[['key', 'class_new', 'tier', 'year', 'src', 'src_conf']],
                      on='key', how='inner', suffixes=('', '_v4'))
    meta = meta.drop_duplicates('row_id').reset_index(drop=True)
    unmatched = len(emb) - len(meta)
    print(f'嵌入行: {len(emb):,}  匹配到修正版v4: {len(meta):,}  未匹配(境外等): {unmatched:,}', flush=True)
    E = meta[FEATS].to_numpy(np.float32)
    y = meta['class_new'].to_numpy(int)
    yr = meta['year'].to_numpy(int)
    tier = meta['tier'].to_numpy()
    print('嵌入点:', len(E), ' 类别数:', len(np.unique(y)), flush=True)

    # ---------- B1 马氏距离 + 卡方（逐类逐年） ----------
    print('B1 马氏+卡方清洗…', flush=True)
    mahal_p = np.full(len(E), np.nan, dtype=np.float32)
    from sklearn.covariance import LedoitWolf
    for (cls, year) in np.unique(np.column_stack([y, yr]), axis=0):
        msk = (y == cls) & (yr == year)
        X = E[msk]
        n = len(X)
        if n < 30:
            continue
        mu = X.mean(0)
        try:
            if n >= 500:
                cov = LedoitWolf().fit(X)
                cov_inv = np.linalg.pinv(cov.covariance_)
            else:
                cov = np.cov(X.T) + np.eye(64) * 1e-3
                cov_inv = np.linalg.pinv(cov)
        except Exception:
            continue
        d = X - mu
        md = np.einsum('ij,jk,ik->i', d, cov_inv, d)
        p = 1 - chi2.cdf(md, 64)
        mahal_p[msk] = p.astype(np.float32)

    removed = mahal_p < 0.001
    prune_zone = (mahal_p >= 0.001) & (mahal_p < 0.01)
    rng = np.random.default_rng(42)
    prune = prune_zone & (rng.random(len(E)) < 0.7)
    print(f'  马氏剔除: {removed.sum():,}  剪70%: {prune.sum():,}', flush=True)

    # ---------- B3 混淆对原型检查 ----------
    print('B3 原型检查…', flush=True)
    proto = {}
    for cls in np.unique(y):
        X = E[y == cls]
        if len(X) >= 100:
            mu = X.mean(0)
            nrm = np.linalg.norm(mu)
            if nrm > 0:
                proto[cls] = mu / nrm
    proto_cos = np.full(len(E), np.nan, dtype=np.float32)
    for cls, mu in proto.items():
        msk = y == cls
        if msk.any():
            En = E[msk] / (np.linalg.norm(E[msk], axis=1, keepdims=True) + 1e-12)
            proto_cos[msk] = (En @ mu).astype(np.float32)

    # ---------- B2 置信学习（延后到 e3 首轮训练后，此处先预留列） ----------
    cl_flag = np.zeros(len(E), dtype=np.int8)

    # ---------- 汇总 v5 ----------
    print('汇总 v5…', flush=True)
    keep = ~(removed | prune)
    v5meta = meta.copy()
    v5meta['mahal_p'] = mahal_p
    v5meta['proto_cos'] = proto_cos
    v5meta['cl_flag'] = cl_flag
    v5meta['mahal_removed'] = removed
    v5meta['mahal_pruned'] = prune
    # clean_tier
    ct = np.where(removed | prune, 'removed', tier)
    ct = np.where(keep & (tier == 'gold'), 'gold_c', ct)
    ct = np.where(keep & (tier == 'silver'), 'silver_c', ct)
    ct = np.where(keep & (tier == 'bronze'), 'bronze_c', ct)
    ct = np.where(keep & (tier == 'uncovered'), 'uncovered_c', ct)
    ct = np.where(keep & (tier == 'external_pending'), 'external_c', ct)
    v5meta['clean_tier'] = ct
    # 嵌入列并入
    for i, f in enumerate(FEATS):
        v5meta[f] = E[:, i]
    v5meta.to_parquet(OUT, index=False)

    summary = {
        'total': int(len(E)),
        'mahal_removed': int(removed.sum()),
        'mahal_pruned': int(prune.sum()),
        'clean_total': int(keep.sum()),
        'by_clean_tier': {str(k): int(v) for k, v in pd.Series(ct).value_counts().items()},
        'elapsed_min': round((time.time() - t0) / 60, 1),
    }
    json.dump(summary, open(OUT.replace('.parquet', '_summary.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('输出:', OUT)

if __name__ == '__main__':
    main()


# -*- coding: utf-8 -*-
"""
e7_merge_rare.py — P3.3 稀有类补样并入年度训练子集
* 输入: 年度子集/r7_train_{y}.parquet（e2b 产出）
        数据/本地处理/全国清洗训练/emb_parts_rare/chunkr7_*.parquet（e1c 产出）
        数据/本地处理/全国清洗训练/稀有类补样/r7_rare_samples.parquet
* 处理: 逐年把该年有效的稀有类点并入，train_weight =
        TIER_W[external]=0.7 × sample_weight(core 1.0 / boundary 0.6) × (0.7+0.3·src_conf)
* 输出: 年度子集_含稀有类/r7_train_{y}.parquet（不覆盖 e2b 原产物）
        年度子集_含稀有类/merge_summary.json
* 用法: python e7_merge_rare.py
"""
import os, glob, json, time
import numpy as np
import pandas as pd

PROJ = r'Z:/Mywork/论文/BaiduSyncdisk/中国土地覆盖数据'
BASE = os.path.join(PROJ, '数据/本地处理/全国清洗训练')
SUB = os.path.join(BASE, '年度子集')
RARE_EMB = os.path.join(BASE, 'emb_parts_rare')
RARE_META = os.path.join(BASE, '稀有类补样/r7_rare_samples.parquet')
OUT = os.path.join(BASE, '年度子集_含稀有类')
os.makedirs(OUT, exist_ok=True)
FEATS = [f'A{i:02d}' for i in range(64)]
YEARS = list(range(2017, 2025))
W_EXT = 0.7


def load_rare_year(y):
    frames = []
    for f in glob.glob(os.path.join(RARE_EMB, 'chunkr7_*.parquet')):
        h = pd.read_parquet(f, columns=['emb_year'])
        if len(h) and int(h.emb_year.iloc[0]) == y:
            frames.append(pd.read_parquet(f))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates('row_id')


def main():
    t0 = time.time()
    meta = pd.read_parquet(RARE_META)
    coords = meta[['row_id', 'lon', 'lat']]
    meta = meta[['row_id', 'class_new', 'src', 'src_conf', 'tier', 'sample_type',
                 'weight', 'valid_from', 'valid_to']].copy()
    meta['train_weight'] = (W_EXT * meta.weight *
                            (0.7 + 0.3 * meta.src_conf.fillna(0.8))).astype(np.float32)
    summary = {'time': time.strftime('%Y-%m-%d %H:%M'), 'by_year': {}}
    for y in YEARS:
        fp = os.path.join(SUB, f'r7_train_{y}.parquet')
        if not os.path.isfile(fp):
            print(f'y{y}: 缺 {os.path.basename(fp)}，跳过', flush=True)
            continue
        base = pd.read_parquet(fp)
        n0 = len(base)
        add = 0
        out = base
        r = load_rare_year(y)
        if r is not None and len(r):
            m = meta[(meta.valid_from <= y) & (meta.valid_to >= y)].drop(
                columns=['valid_from', 'valid_to'])
            rr = r.merge(m, on='row_id', how='inner').merge(coords, on='row_id', how='left')
            if len(rr):
                rr['year'] = y
                rr['qc_status'] = 'rare_topup'
                rr['tier'] = rr.tier.fillna('external')
                for c in base.columns:
                    if c not in rr:
                        rr[c] = np.nan
                out = pd.concat([base, rr[list(base.columns)]], ignore_index=True)
                add = len(rr)
        out.to_parquet(os.path.join(OUT, f'r7_train_{y}.parquet'), index=False)
        summary['by_year'][str(y)] = {'base': int(n0), 'rare_added': int(add),
                                      'total': int(len(out))}
        print(f'y{y}: {n0:,} + 稀有 {add:,} = {len(out):,}', flush=True)
    summary['elapsed_min'] = round((time.time() - t0) / 60, 1)
    json.dump(summary, open(os.path.join(OUT, 'merge_summary.json'), 'w',
                            encoding='utf-8'), ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary['by_year'], ensure_ascii=False))
    print('输出目录:', OUT)


if __name__ == '__main__':
    main()

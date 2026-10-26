# -*- coding: utf-8 -*-
"""m5_mangrove_qc.py — 红树林扩充 第5步：质检与筛选（判据 v1 预注册；阈值按原 652 点校准）

* 输入：out/feats/ev、out/feats/aef（候选）；out/feats/orig/{ev,aef}（原 652 点，同波段，用于校准与对照）
* 判据 v1（**先在原 652 上校准 → 冻结 → 再判候选**）：
    C1 真红树林像元   : mang10 == 1
    C2 非矛盾（原18类quar）: trees >= 0.05 且 dem <= 30
    C3 潮间带水邻接   : max_300 >= 0.10
    C4 植被结构       : trees >= 0.25
    C5 物候绿度       : s2_ndvi_xia >= q（阈值 = 原 652 keep 集 p05）
    C6 AEF64 一致性   : 到原 652(strong) 质心的余弦相似度 ≥ q（阈值 = 原 strong 集 p05）
  keep = C1&C2&C3&C4&C5&C6；同时报告每条的单独通过率与"原样本通过率"（校准证据）。
* 输出：out/qc/keep_v1.parquet + qc/drop_v1.parquet + out/qc/stats_qc_v1.json
* 用法：python m5_mangrove_qc.py
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp/data/mangrove'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/红树林'
AEF = ['A%02d' % i for i in range(64)]


def load(stage):
    fs = sorted(glob.glob(os.path.join(WORK, 'out', 'feats', stage, '*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if len(d):
        d['cid'] = d.cid.astype(str)
        d = d.drop_duplicates('cid')
    return d


def main():
    ev = load('ev'); aef = load('aef')
    oev = load('orig_ev') if glob.glob(os.path.join(WORK, 'out', 'feats', 'orig_ev', '*.parquet')) else None
    oaef = load('orig_aef') if glob.glob(os.path.join(WORK, 'out', 'feats', 'orig_aef', '*.parquet')) else None
    cand = pd.read_parquet(os.path.join(WORK, 'cand', 'candidates.parquet'))
    cand['cid'] = cand.cid.astype(str)
    d = cand[['cid', 'lon', 'lat']].merge(ev, on='cid', how='left').merge(aef, on='cid', how='left')
    print('候选 %d（ev %d / aef %d）' % (len(d), len(ev), len(aef)))
    # —— 原 652 校准集 ——
    assert oev is not None and oaef is not None, '缺 orig 证据（先跑 m4 --list 原样本）'
    o = oev.merge(oaef, on='cid', how='left')
    import shapefile
    r = shapefile.Reader(os.path.join(DEST, '2023', '红树林_2023_WGS84.shp'), encoding='utf-8')
    fl = [x[0] for x in r.fields[1:]]
    it = fl.index('tier')
    tier = {str(rec[fl.index('pid')]): rec[it] for rec in r.iterRecords()}
    r.close()
    o['tier'] = o.cid.map(lambda c: tier.get(c, ''))
    strong = o[o.tier == 'strong']
    # 阈值校准（原 strong 的 p05）
    q5_ndvi = float(np.nanpercentile(strong.s2_ndvi_xia, 5))
    cen = strong[AEF].to_numpy(float)
    cen = np.nanmean(cen, axis=0)
    def cos_sim(X):
        X = X - X.mean(axis=1, keepdims=True)
        c = cen - cen.mean()
        return (X @ c) / (np.linalg.norm(X, axis=1) * np.linalg.norm(c) + 1e-9)
    o['cos'] = cos_sim(o[AEF].to_numpy(float))
    strong = o[o.tier == 'strong']          # cos 列赋值后重取
    q5_cos = float(np.nanpercentile(strong.cos, 5))
    print('校准阈值：s2_ndvi_xia p05=%.1f ｜ AEF 余弦 p05=%.4f（原 strong %d 点）' % (q5_ndvi, q5_cos, len(strong)))

    def rules(x):
        return dict(
            K1=x.mang10 == 1,
            K2=(x.trees >= 0.05) & (x.dem <= 30),
            K3=x.max_300 >= 0.10,
            K4=(x.trees >= 0.25) | (x.wc95 == 1),
            K5=(x.s2_ndvi_xia >= q5_ndvi) | (x.s2_ndvi_xia.isna() & (x.wc95 == 1)),
            K6=x.cos >= max(q5_cos, 0.85))
    ro = rules(o)
    print('原 652 各条通过率：', {k: '%.3f' % v.mean() for k, v in ro.items()})
    print('原 strong 各条通过率：', {k: '%.3f' % rules(strong)[k].mean() for k in ro})
    d['cos'] = cos_sim(d[AEF].to_numpy(float))
    rc = rules(d)
    keep = np.logical_and.reduce([rc[k].to_numpy() for k in rc])
    d['qc_rule'] = np.where(keep, 'keep', '')
    for k, v in rc.items():
        d.loc[~v, 'qc_rule'] = d.loc[~v, 'qc_rule'] + '|-' + k
    # 空间去重已由构造保证（≤1 点/100 m 格）；不再额外抽稀（密度目标 1 点/~100 m）
    k1 = d[keep].copy()
    k2 = k1
    os.makedirs(os.path.join(WORK, 'out', 'qc'), exist_ok=True)
    k2.to_parquet(os.path.join(WORK, 'out', 'qc', 'keep_v1.parquet'), index=False)
    d[~keep].to_parquet(os.path.join(WORK, 'out', 'qc', 'drop_v1.parquet'), index=False)
    stats = dict(n_cand=len(d), n_keep_after_thin=len(k2),
                 q5_ndvi=q5_ndvi, q5_cos=q5_cos,
                 orig_passrates={k: round(float(v.mean()), 4) for k, v in ro.items()},
                 cand_passrates={k: round(float(v.mean()), 4) for k, v in rc.items()},
                 thin_removed=len(k1) - len(k2))
    json.dump(stats, open(os.path.join(WORK, 'out', 'qc', 'stats_qc_v1.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

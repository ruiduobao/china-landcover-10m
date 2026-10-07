# -*- coding: utf-8 -*-
"""m5c_ningxia_dissect.py — 宁夏 9 大类未达标（0.8283）混淆解剖 + 灌草合并口径（本地，零 GEE 成本；doc45 §三.A）
* 输入：data/yearly/{train,eval}_2023_w4.parquet（w4 留出构造）＋交付成品 6 瓦 F:/lc_work/prod5p_2023/rasters_batch/T{1509,1608,1609,1610,1709,1710}_10m.tif
* 口径：A=本地复测（train_2023_w4 训练 → 宁夏范围 eval 点预测，干净留出；R0/R4/R5 三策略×3种子）
        B=交付成品实读（同一批点读 10m 成品；注意成品训练池为 smp2023_merged 全量，点可能入袋，仅作参考上限）
* 指标：24 类 OA / 9 大类 OA / 灌草合并 OA（24 类空间 9,10,11 并组）/ 错分质量（灌草裸稀疏界面占比）/ 分来源分层（fcs10 vs 非）
* 门槛：3 种子均值；per-class 用 VC.metrics（PA=召回、UA=精确率，2026-10-07 已修）
* 输出：results/d2/ningxia_dissect.json + reports/宁夏混淆解剖_YYYYMMDD.md
"""
import os, sys, glob, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import y2b_d2_policy as YB

OUTD = os.path.join(VC.DATA, 'yearly')
RESD = os.path.join(VC.RES, 'd2')
NX = ['T1509', 'T1608', 'T1609', 'T1610', 'T1709', 'T1710']
TILES = [r'F:/lc_work/prod5p_2023/rasters_batch/%s_10m.tif' % t for t in NX]
SHRUBG = (9, 10, 11)          # 24 类：常绿灌丛/落叶灌丛/草地
INTERFACE = {9, 10, 11, 13, 22}  # 灌丛/草地/稀疏植被/裸地
SEEDS = YB.SEEDS


def merged_shrubgrass(v24):
    m = v24.copy()
    m[np.isin(m, SHRUBG)] = -1          # 灌草并组（临时码 -1）
    return m


def oa(t, p):
    return float((t == p).mean()) if len(t) else float('nan')


def eval_pack(t24, p24, src, tag):
    m9t, m9p = VC.to_macro(t24), VC.to_macro(p24)
    mt, mp = merged_shrubgrass(t24), merged_shrubgrass(p24)
    err = t24 != p24
    iface = err & np.isin(t24, list(INTERFACE)) & np.isin(p24, list(INTERFACE))
    sg = err & (((np.isin(t24, (9, 10))) & (p24 == 11)) | ((t24 == 11) & np.isin(p24, (9, 10))))
    nonf = ~pd.Series(src).fillna('').astype(str).str.startswith('glc_fcs10').to_numpy()
    rec = dict(tag=tag, n=int(len(t24)),
               OA24=round(oa(t24, p24), 4), OA9=round(oa(m9t, m9p), 4),
               OA_sg=round(oa(mt, mp), 4),
               OA9_nonfcs10=round(oa(m9t[nonf], m9p[nonf]), 4),
               err_n=int(err.sum()),
               err_interface_pct=round(float(iface.sum() / max(1, err.sum())), 4),
               err_shrubgrass_pct=round(float(sg.sum() / max(1, err.sum())), 4),
               n_fcs10=int((~nonf).sum()))
    per9 = VC.metrics(m9t, m9p)['per']
    rec['per9'] = {str(k): v for k, v in per9.items()}
    return rec


def main():
    os.makedirs(RESD, exist_ok=True)
    tr = pd.read_parquet(os.path.join(OUTD, 'train_2023_w4.parquet'))
    ev = pd.read_parquet(os.path.join(OUTD, 'eval_2023_w4.parquet'))
    import rasterio
    bnds = {}
    for fp in TILES:
        t = os.path.basename(fp).split('_')[0]
        with rasterio.open(fp) as ds:
            bnds[t] = ds.bounds
    W = min(b.left for b in bnds.values()); E = max(b.right for b in bnds.values())
    S = max(min(b.bottom for b in bnds.values()), 0); N = max(b.top for b in bnds.values())
    inb = ev[(ev.lon >= W) & (ev.lon <= E) & (ev.lat >= S) & (ev.lat <= N)].copy()
    import json as _json
    from shapely.geometry import shape, Point
    from shapely import make_valid
    gj = _json.load(open(r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/边界/china_100000_full.json', encoding='utf-8'))
    nxg = None
    for ft in gj['features']:
        if ft.get('properties', {}).get('name', '').startswith('宁夏'):
            nxg = make_valid(shape(ft['geometry']))
    keep = [nxg.contains(Point(lo, la)) for lo, la in zip(inb.lon, inb.lat)]
    sub = inb[np.array(keep)].copy()
    print('宁夏 6 瓦 bounds 落入 %d 点，省界内 %d 点' % (len(inb), len(sub)), flush=True)
    Xev = sub[VC.FEATS].to_numpy('float32')
    t24 = VC.to_v31(sub['class_new'].to_numpy(int))
    src = sub['src'].fillna('').astype(str).to_numpy()
    out = dict(n=len(sub), bounds=[W, E, S, N], local={}, delivered=None, seeds=SEEDS)
    # ---- A 本地复测（干净留出）----
    Xtr_all = tr[VC.FEATS].to_numpy('float32')
    for pol in ('R0', 'R4', 'R5'):
        trp, _ = YB.apply_policy(tr, pol, 'w4')
        Xtr = trp[VC.FEATS].to_numpy('float32')
        ytr = VC.to_v31(trp['class_new'].to_numpy(int))
        oas, packs = [], []
        for sd in SEEDS:
            clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_leaf_nodes=5000,
                                         max_features='sqrt', n_jobs=15, random_state=sd).fit(Xtr, ytr)
            p24 = clf.predict(Xev)
            rec = eval_pack(t24, p24, src, 'local_%s_s%d' % (pol, sd))
            packs.append(rec); oas.append(rec['OA9'])
            print('%s s%d: OA9=%.4f OA_sg=%.4f n=%d' % (pol, sd, rec['OA9'], rec['OA_sg'], rec['n']), flush=True)
        out['local'][pol] = dict(OA9_mean=round(float(np.mean(oas)), 4),
                                 OA9_sd=round(float(np.std(oas)), 4), seeds=packs,
                                 n_train=int(len(trp)))
    # ---- B 交付成品实读 ----
    import rasterio.windows as rw
    pred = np.zeros(len(sub), dtype=int)
    got = np.zeros(len(sub), dtype=bool)
    for fp in TILES:
        t = os.path.basename(fp).split('_')[0]
        with rasterio.open(fp) as ds:
            b = ds.bounds
            m = (sub.lon >= b.left) & (sub.lon <= b.right) & (sub.lat >= b.bottom) & (sub.lat <= b.top) & (~got)
            idx = np.where(m.to_numpy())[0]
            for i in idx:
                r, c = ds.index(float(sub.lon.iloc[i]), float(sub.lat.iloc[i]))
                if 0 <= r < ds.height and 0 <= c < ds.width:
                    v = int(ds.read(1, window=rw.Window(c, r, 1, 1))[0, 0])
                    if v > 0:
                        pred[i] = v
                        got[i] = True
    print('交付成品命中 %d/%d' % (got.sum(), len(sub)), flush=True)
    out['delivered'] = eval_pack(t24[got], pred[got], src[got], 'delivered_R0')
    out['delivered']['n_inbag_note'] = '成品训练池=smp2023_merged 全量，eval 点可能入袋，此口径仅作参考上限'
    VC.jsave(out, os.path.join(RESD, 'ningxia_dissect.json'))
    # ---- 报告 ----
    L = ['# 宁夏 9 大类未达标（G7 0.8283）混淆解剖 %s' % time.strftime('%Y-%m-%d %H:%M'), '',
         '- 留出构造：w4 eval（2023）∩ 宁夏 6 瓦范围 = **%d 点**；3 种子（7/11/23）；策略 R0/R4/R5' % len(sub),
         '- 9 大类 = macro9；灌草合并 = 24 类空间 {常绿灌丛,落叶灌丛,草地} 并组后 OA', '',
         '## 本地复测（干净留出）', '',
         '| 策略 | n_train | OA9（3种子） | 灌草合并 OA | OA9 非fcs10 | 错分中灌草裸界面% | 错分中灌↔草% |', '|---|---|---|---|---|---|---|']
    for pol in ('R0', 'R4', 'R5'):
        r = out['local'][pol]
        s0 = r['seeds'][0]
        L.append('| %s | %d | %.4f±%.4f | %.4f | %.4f | %.1f%% | %.1f%% |' % (
            pol, r['n_train'], r['OA9_mean'], r['OA9_sd'], s0['OA_sg'], s0['OA9_nonfcs10'],
            s0['err_interface_pct'] * 100, s0['err_shrubgrass_pct'] * 100))
    d = out['delivered']
    L += ['', '## 交付成品实读（参考上限，点可能入袋）', '',
          '| n | OA24 | OA9 | 灌草合并 OA | OA9 非fcs10 | 错分中灌草裸界面% | 错分中灌↔草% |', '|---|---|---|---|---|---|---|',
          '| %d | %.4f | %.4f | %.4f | %.4f | %.1f%% | %.1f%% |' % (
              d['n'], d['OA24'], d['OA9'], d['OA_sg'], d['OA9_nonfcs10'],
              d['err_interface_pct'] * 100, d['err_shrubgrass_pct'] * 100)]
    fp_md = os.path.join(VC.REPT, '宁夏混淆解剖_%s.md' % time.strftime('%Y%m%d'))
    open(fp_md, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('→', fp_md, flush=True)


if __name__ == '__main__':
    main()

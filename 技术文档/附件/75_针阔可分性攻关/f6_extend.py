# -*- coding: utf-8 -*-
"""f6_extend.py — 把 FCS10 纠偏验证扩展到多生态区瓦片（跨省/跨林型）

* 目的：验证 §7.7 的两瓦片结果是否普适，并检查东北 DNF 区（坡向对立最强 OR 20–41）
* 输入（只读）：
    AEF64 + v2 池标签：F:/lc_work/prod2nd_2023/data/train2023_v2.parquet
    GLC_FCS10-2023：   F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10/{tile}_fcs10.tif
    DEM(1km 全国)：     F:/lc_work/v31_exp/data/eco_gate/_dem/srtm1000_chn.tif
* 规则：沿用 f0_prep/f1_train 的 Arm-A / Arm-B 定义（Arm-B：森林针阔轴换 FCS10，常绿落叶轴随 FCS10）
* 门槛：nearest 取样；同点集同特征同参数同种子；2km 空间块 CV
* 输出：exp/fcs10_retrain/extend_{tile}.json + extend_summary.json
* 用法：python f6_extend.py [--tiles T2712 T1801 ...]
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
NB = r'F:/lc_work/nb_sep_2026'
sys.path.insert(0, os.path.join(NB, 'aspect'))
sys.path.insert(0, os.path.join(NB, 'raster'))
import terrain_lib as TL
import xw2

POOL = r'F:/lc_work/prod2nd_2023/data/train2023_v2.parquet'
EXT = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles'
DEM = r'F:/lc_work/v31_exp/data/eco_gate/_dem/srtm1000_chn.tif'
AEF = [f'A{i:02d}' for i in range(64)]

LEAF24 = {4: 2, 5: 2, 6: 1, 7: 1, 8: 3}
NEW24 = {(1, 1): 6, (1, 2): 7, (2, 1): 4, (2, 2): 5, (3, 3): 8}
FCS_LEAF2 = {1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2), 5: (3, 3)}

# 跨生态区代表瓦片：tile -> (名称, x0,y0,x1,y1)
TILES = {
    'T2712': ('黑/吉 长白山系', 127.0, 42.0, 129.0, 44.0),
    'T2813': ('黑 小兴安岭', 129.0, 44.0, 131.0, 46.0),
    'T2815': ('黑 张广才岭北', 129.0, 48.0, 131.0, 50.0),
    'T1902': ('粤 粤北', 111.0, 22.0, 113.0, 24.0),
    'T2002': ('粤 粤中', 113.0, 22.0, 115.0, 24.0),
    'T1505': ('川 川南', 103.0, 28.0, 105.0, 30.0),
    'T1406': ('川 川西', 101.0, 30.0, 103.0, 32.0),
    'T1609': ('宁 贺兰山南', 105.0, 36.0, 107.0, 38.0),
    'T1610': ('宁 银川北', 105.0, 38.0, 107.0, 40.0),
    'T2204': ('闽 闽北', 117.0, 26.0, 119.0, 28.0),
}


def sample_raster(fp, lons, lats):
    out = np.full(len(lons), -9999, dtype=np.int32)
    if not os.path.exists(fp):
        return out
    with rasterio.open(fp) as s:
        rows, cols = rasterio.transform.rowcol(s.transform, lons, lats)
        rows = np.asarray(rows); cols = np.asarray(cols)
        ok = (rows >= 0) & (cols >= 0) & (rows < s.height) & (cols < s.width)
        if ok.any():
            r0, r1 = rows[ok].min(), rows[ok].max()
            c0, c1 = cols[ok].min(), cols[ok].max()
            sub = s.read(1, window=Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1))
            out[ok] = sub[rows[ok] - r0, cols[ok] - c0]
    return out


def add_terrain(df, box):
    lon0, lat0, lon1, lat1 = box
    pad = 0.02
    with rasterio.open(DEM) as s:
        w = s.window(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)
        w = w.round_offsets().round_lengths()
        res_deg = abs(s.transform[0])
        dem = s.read(1, window=w).astype(np.float64)
        dem = np.where(dem <= -32000, np.nan, dem)
        row_off = int(w.row_off); col_off = int(w.col_off)
        H, W = dem.shape
        xs = s.transform[2] + (col_off + np.arange(W) + 0.5) * s.transform[0]
        ys = s.transform[5] + (row_off + np.arange(H) + 0.5) * s.transform[4]
    if np.isnan(dem).any():
        from scipy.ndimage import generic_filter
        m = np.isnan(dem)
        def nanmean(v):
            c = v[~np.isnan(v)]
            return c.mean() if c.size else 0.0
        fill = generic_filter(np.nan_to_num(dem), nanmean, size=3, mode='nearest')
        dem[m] = fill[m]
    lat = float((lat0 + lat1) / 2)
    asp, slp = TL.aspect_slope(dem, res_deg, lat)
    ys_c = ys[1:-1]; xs_c = xs[1:-1]
    ri = np.abs(df['lat'].values[:, None] - ys_c[None, :]).argmin(axis=1)
    ci = np.abs(df['lon'].values[:, None] - xs_c[None, :]).argmin(axis=1)
    df['aspect'] = asp[ri, ci]
    df['slope'] = slp[ri, ci]
    df['asp_cls'] = TL.aspect_class(df['aspect'].values)
    return df


def build_armb(df):
    cl_a = df['cl'].values.astype(np.int16)
    cl_b = cl_a.copy()
    isf = df['cl'].isin([4, 5, 6, 7, 8]).values
    f10 = df['fcs10_leaf'].values
    ok = isf & (f10 >= 1) & (f10 <= 5)
    for i in np.where(ok)[0]:
        nb, ed = FCS_LEAF2[int(f10[i])]
        cl_b[i] = NEW24[(nb, ed)]
    return cl_b, int(ok.sum()), int(isf.sum())


def spatial_folds(df, km=2.0, nfold=5):
    kx = np.floor(df.lon.values / (km / 111.32 / np.cos(np.radians(df.lat.values)))).astype(np.int64)
    ky = np.floor(df.lat.values / (km / 111.32)).astype(np.int64)
    key = kx * 100000 + ky
    u = np.unique(key)
    rng = np.random.default_rng(20261017)
    rng.shuffle(u)
    folds = np.zeros(len(key), dtype=np.int8)
    for fi, blk in enumerate(np.array_split(u, nfold)):
        folds[np.isin(key, blk)] = fi
    return folds


def eval_pair(y_true, y_pred, cls, f10_leaf):
    """以 FCS10 为参考：森林4类 macro-F1 + 叶型轴 recall/precision + 各森林类指标
    注：命名采用标准口径
       recall(PA) = 参考=FCS10 该类 中预测也判该类的比例（分母=参考）
       precision(UA)= 预测该类 中 FCS10 也是该类的比例（分母=预测）
    项目历史脚本曾把两者命名为 pa/ua 互换，本脚本用标准名以免混淆。
    """
    out = {}
    m = np.isin(y_true, cls)
    yt, yp = y_true[m], y_pred[m]
    if len(yt) > 0:
        out['train_label_forest4_macro_f1'] = round(float(
            f1_score(yt, yp, labels=cls, average='macro', zero_division=0)), 4)
        out['n_forest_train'] = int(m.sum())
    # 以 FCS10 为参考（f10_leaf: 1=ENF 2=DNF 3=EBF 4=DBF 5=MIX, -1=非林/未覆盖）
    pred_enf = y_pred == 6; pred_ebf = y_pred == 4
    ref_enf = f10_leaf == 1; ref_ebf = f10_leaf == 3
    pred_nb = np.isin(y_pred, [6, 7]); ref_nb = np.isin(f10_leaf, [1, 2])
    pred_br = np.isin(y_pred, [4, 5]); ref_br = np.isin(f10_leaf, [3, 4])
    def pr(pred, ref):
        prec = float((pred & ref).sum() / max(pred.sum(), 1)) if pred.sum() else None
        rec = float((pred & ref).sum() / max(ref.sum(), 1)) if ref.sum() else None
        return prec, rec
    e_prec, e_rec = pr(pred_enf, ref_enf)
    b_prec, b_rec = pr(pred_ebf, ref_ebf)
    n_prec, n_rec = pr(pred_nb, ref_nb)
    out.update({'ENF_precision_vs_FCS10': None if e_prec is None else round(e_prec, 4),
                'ENF_recall_vs_FCS10': None if e_rec is None else round(e_rec, 4),
                'EBF_precision_vs_FCS10': None if b_prec is None else round(b_prec, 4),
                'EBF_recall_vs_FCS10': None if b_rec is None else round(b_rec, 4),
                'needle_precision_vs_FCS10': None if n_prec is None else round(n_prec, 4),
                'needle_recall_vs_FCS10': None if n_rec is None else round(n_rec, 4),
                'n_pred_enf': int(pred_enf.sum()), 'n_ref_enf': int(ref_enf.sum()),
                'enf_share_of_forest_pred': round(float(pred_enf.sum() / max(1, (pred_enf | pred_ebf).sum())), 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', nargs='+', default=list(TILES.keys()))
    ap.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 23])
    a = ap.parse_args()
    pool = pd.read_parquet(POOL, columns=['lon', 'lat', 'cl'] + AEF)
    print('池', len(pool))

    summary = {}
    for tile in a.tiles:
        if tile not in TILES:
            print('跳过未知瓦', tile); continue
        name, x0, y0, x1, y1 = TILES[tile]
        t0 = time.time()
        m = ((pool.lon >= x0) & (pool.lon < x1) & (pool.lat >= y0) & (pool.lat < y1)).values
        df = pool[m].reset_index(drop=True)
        if len(df) < 500:
            print(f'[{tile}] {name}: 点太少 {len(df)}，跳过'); continue
        # FCS10
        fp = os.path.join(EXT, 'fcs10', f'{tile}_fcs10.tif')
        df['fcs10'] = sample_raster(fp, df.lon.values, df.lat.values)
        df['fcs10_leaf'] = xw2.lut('fcs10')[np.clip(df.fcs10.values, 0, 255)].astype(np.int8)
        df.loc[df.fcs10 < 0, 'fcs10_leaf'] = -1
        cover = float((df.fcs10 > 0).mean() * 100)
        df = add_terrain(df, (x0, y0, x1, y1))
        cl_b, n_relab, n_isf = build_armb(df)
        df['cl_b'] = cl_b
        print(f'[{tile}] {name}: n={len(df)} FCS10覆盖={cover:.1f}% 森林={n_isf} 改标={n_relab}')

        X = df[AEF].to_numpy(np.float32)
        folds = spatial_folds(df)
        res = dict(tile=tile, name=name, n=len(df), fcs10_cover_pct=round(cover, 2),
                   n_forest=n_isf, n_relab=n_relab, seeds=a.seeds, arms={})
        for arm, col in [('armA', 'cl'), ('armB', 'cl_b')]:
            y = df[col].to_numpy()
            per_seed = []
            oof_all = np.zeros(len(df), dtype=np.int16)
            for seed in a.seeds:
                oof = np.zeros(len(df), dtype=np.int16)
                for fi in range(5):
                    tr = folds != fi; te = ~tr
                    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2,
                                                max_leaf_nodes=5000, n_jobs=13, random_state=seed)
                    rf.fit(X[tr], y[tr])
                    oof[te] = rf.predict(X[te])
                met = eval_pair(y, oof, [4, 5, 6, 7], df['fcs10_leaf'].values)
                # 阴阳坡 OR（用 oof 预测）
                sub = df[df.slope > 5].copy()
                sub['pred'] = oof[df.slope > 5]
                es = int(((sub.pred == 6) & (sub.asp_cls == 2)).sum())
                eu = int(((sub.pred == 6) & (sub.asp_cls == 1)).sum())
                bs = int(((sub.pred == 4) & (sub.asp_cls == 2)).sum())
                bu = int(((sub.pred == 4) & (sub.asp_cls == 1)).sum())
                met['or_enf_ebf'] = round((es / eu) / (bs / bu), 3) if (eu > 0 and bu > 0 and bs > 0) else None
                met['or_cells'] = dict(enf=[es, eu], ebf=[bs, bu])
                per_seed.append(met)
                oof_all = oof
            # 聚合
            agg = {}
            for k in ['train_label_forest4_macro_f1','needle_precision_vs_FCS10','needle_recall_vs_FCS10','ENF_precision_vs_FCS10','ENF_recall_vs_FCS10','EBF_precision_vs_FCS10','EBF_recall_vs_FCS10','enf_share_of_forest_pred','n_ref_enf']:
                vals = [p[k] for p in per_seed if p.get(k) is not None]
                if vals:
                    agg[k] = round(float(np.mean(vals)), 4) if k != 'n_forest' else int(np.mean(vals))
            ors = [p['or_enf_ebf'] for p in per_seed if p.get('or_enf_ebf') is not None]
            agg['or_enf_ebf_median'] = round(float(np.median(ors)), 3) if ors else None
            res['arms'][arm] = agg
        summary[tile] = res
        json.dump(res, open(os.path.join(HERE, f'extend_{tile}.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        A, B = res['arms']['armA'], res['arms']['armB']
        print('  armA: 针prec=%s 针rec=%s ENF占比=%s OR=%s' % (
            A.get('needle_precision_vs_FCS10'), A.get('needle_recall_vs_FCS10'),
            A.get('enf_share_of_forest_pred'), A.get('or_enf_ebf_median')))
        print('  armB: 针prec=%s 针rec=%s ENF占比=%s OR=%s' % (
            B.get('needle_precision_vs_FCS10'), B.get('needle_recall_vs_FCS10'),
            B.get('enf_share_of_forest_pred'), B.get('or_enf_ebf_median')))
        print(f'  用时 {time.time()-t0:.0f}s')

    json.dump(summary, open(os.path.join(HERE, 'extend_summary.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n-> extend_summary.json')
    # 汇总表
    print('\n%-8s %-12s %6s %8s %8s %8s %8s %8s %8s' % ('tile', 'name', 'cover', 'A_F1', 'B_F1', 'A_针PA', 'B_针PA', 'A_OR', 'B_OR'))
    for t, r in summary.items():
        A, B = r['arms']['armA'], r['arms']['armB']
        print('%-8s %-12s %5.1f%% %8s %8s %8s %8s %8s %8s' % (
            t, r['name'][:12], r['fcs10_cover_pct'],
            A.get('forest4_macro_f1'), B.get('forest4_macro_f1'),
            A.get('needle_PA'), B.get('needle_PA'),
            A.get('or_enf_ebf_median'), B.get('or_enf_ebf_median')))


if __name__ == '__main__':
    main()

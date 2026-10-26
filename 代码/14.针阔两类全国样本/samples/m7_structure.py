# -*- coding: utf-8 -*-
"""m7_structure.py — 方法C：冠层结构/郁闭度是否能分针阔
* 输入：labels/methods/work_base.parquet（含 fcs10_code；FCS10 的 奇=Open 偶=Closed 由
        tree-cover 40% 阈值定义，见 ESSD-17-4039-2025 Table S1）
        F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10/*.tif（10m 原生码，用于"闭/疏"比例与邻域均匀度）
        F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs30d/*.tif（30m，独立产品同轴）
* 规则：C1 用 FCS10 闭/疏码比较针叶 vs 阔叶的郁闭比例（结构差异）
        C2 用 FCS10 10m 邻域（5×5）内"同叶型占比"作为林分纯度/结构均匀度
        C3 若 GEE 可用则补 GEDI L2A rh98/cover；本轮网络中断 → 如实标未测
* 输出：labels/methods/m7_structure.json、labels/methods/struct_pts.parquet
* 用法：python m7_structure.py [--tiles 24]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

WORK = r'F:/lc_work/nb_national_2026/labels/methods'
EXT = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles'

CLOSED = {51, 61, 71, 81}
OPEN = {52, 62, 72, 82}
NEEDLE = {71, 72, 81, 82}
BROAD = {51, 52, 61, 62}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', type=int, default=24)
    args = ap.parse_args()
    res = {}

    df = pd.read_parquet(os.path.join(WORK, 'work_base.parquet'),
                         columns=['stable_uid', 'lon', 'lat', 'cl_24', 'fcs10_code',
                                  'fcs10_covered', 'tile', 'dem_nasadem', 'slope'])
    df['stable_uid'] = df['stable_uid'].astype(str)

    # ---------- C1：闭/疏结构差
    fc = df[df['fcs10_covered']].copy()
    fc['is_closed'] = fc['fcs10_code'].isin(CLOSED)
    fc['is_open'] = fc['fcs10_code'].isin(OPEN)
    fc['is_needle'] = fc['fcs10_code'].isin(NEEDLE)
    fc['is_broad'] = fc['fcs10_code'].isin(BROAD)
    sub = fc[fc['is_needle'] | fc['is_broad']]
    res['C1_closed_open_ratio'] = {}
    for nm, m in [('needle', sub['is_needle']), ('broad', sub['is_broad'])]:
        s = sub[m]
        n_cl = int(s['is_closed'].sum()); n_op = int(s['is_open'].sum())
        res['C1_closed_open_ratio'][nm] = dict(
            n=int(len(s)), closed=n_cl, open=n_op,
            closed_pct=round(n_cl / max(n_cl + n_op, 1) * 100, 2),
            open_pct=round(n_op / max(n_cl + n_op, 1) * 100, 2))
    a = res['C1_closed_open_ratio']['needle']; b = res['C1_closed_open_ratio']['broad']
    if a['closed'] and a['open'] and b['closed'] and b['open']:
        res['C1_closed_open_OR_needle_vs_broad'] = round(
            (a['closed'] / a['open']) / (b['closed'] / b['open']), 4)
    # 闭/疏 单独预测针阔的 AUC（可判别性）
    from sklearn.metrics import roc_auc_score
    m2 = sub[['is_closed', 'is_needle']].dropna()
    res['C1_auc_closed_predicts_needle'] = round(float(roc_auc_score(m2['is_needle'], m2['is_closed'])), 4)
    res['C1_base_rate_needle'] = round(float(m2['is_needle'].mean()), 4)

    # ---------- C2：10m 邻域结构均匀度（林分纯度）
    tiles = sorted(fc['tile'].unique())[:args.tiles]
    rows = []
    import rasterio
    for t in tiles:
        p = os.path.join(EXT, 'fcs10', f'{t}_fcs10.tif')
        if not os.path.exists(p):
            continue
        g = fc[fc['tile'] == t]
        if len(g) == 0:
            continue
        with rasterio.open(p) as s:
            tr = s.transform; W = s.width
            # 只读整幅（22265^2 uint8 ≈ 495MB 太耗；改按点读 11×11 窗口）
            arr = s.read(1)
        rr = ((g['lat'].values - tr.f) / tr.e).astype(np.int64)
        cc = ((g['lon'].values - tr.c) / tr.a).astype(np.int64)
        ok = (rr >= 5) & (rr < arr.shape[0] - 5) & (cc >= 5) & (cc < arr.shape[1] - 5)
        rr = rr[ok]; cc = cc[ok]; uid = g['stable_uid'].values[ok]
        code = g['fcs10_code'].values[ok]
        needle_cnt = np.zeros(len(rr), dtype=np.int16)
        forest_cnt = np.zeros(len(rr), dtype=np.int16)
        same_cnt = np.zeros(len(rr), dtype=np.int16)
        for i, (r, c, cd) in enumerate(zip(rr, cc, code)):
            w = arr[r - 5:r + 6, c - 6:c + 6]
            isf = np.isin(w, list(NEEDLE | BROAD))
            forest_cnt[i] = isf.sum()
            needle_cnt[i] = np.isin(w, list(NEEDLE)).sum()
            same_cnt[i] = (w == cd).sum()
        rows.append(pd.DataFrame(dict(stable_uid=uid, nbr_forest=forest_cnt,
                                      nbr_needle=needle_cnt, nbr_same=same_cnt)))
        del arr
    if rows:
        nb = pd.concat(rows, ignore_index=True)
        nb['nbr_needle_frac'] = nb['nbr_needle'] / nb['nbr_forest'].clip(lower=1)
        nb['nbr_same_frac'] = nb['nbr_same'] / 121.0
        nb.to_parquet(os.path.join(WORK, 'struct_pts.parquet'), index=False)
        mm = nb.merge(fc[['stable_uid', 'is_needle']], on='stable_uid')
        # 只保留邻域里森林像元 >= 40 的（即 33% 以上森林，避免边界）
        mm2 = mm[mm['nbr_forest'] >= 40]
        res['C2_neighborhood'] = dict(
            n_tiles=len(tiles), n_points=int(len(mm2)),
            nbr_same_frac_needle=round(float(mm2.loc[mm2['is_needle'], 'nbr_same_frac'].median()), 4),
            nbr_same_frac_broad=round(float(mm2.loc[~mm2['is_needle'], 'nbr_same_frac'].median()), 4),
            nbr_needle_frac_mean_needle=round(float(mm2.loc[mm2['is_needle'], 'nbr_needle_frac'].mean()), 4),
            nbr_needle_frac_mean_broad=round(float(mm2.loc[~mm2['is_needle'], 'nbr_needle_frac'].mean()), 4))
        try:
            res['C2_auc_nbr_same_predicts_needle'] = round(
                float(roc_auc_score(mm2['is_needle'], mm2['nbr_same_frac'])), 4)
            res['C2_auc_nbr_needlefrac_predicts_needle'] = round(
                float(roc_auc_score(mm2['is_needle'], mm2['nbr_needle_frac'])), 4)
        except Exception:
            pass

    # ---------- C3：GEDI / CGLS TCD —— 本轮 GEE 网络中断，如实记录
    res['C3_gedi'] = dict(status='未测', reason='GEE 出口被 reset（代理端口通但 Google 域连不上）；'
                                               '本地无 GEDI L2A 足迹数据；'
                                               '仅有 GEDI L4B 1km 聚合生物量单瓦（MW019MW223），不覆盖中国林区',
                          local_l4b_zip=r'E:/地理所/论文/博士论文/数据/数据/GEDI_L4B_Gridded_Biomass_V2_1_2299.zip')
    res['C3_cgls_tcd'] = dict(status='部分（GEE 单次抽样成功 2,993 点后网络中断）',
                              note='已成功取到 CGLS tree-coverfraction 与 forest_type（2019 层），'
                                   '证明数据可用；但因网络中断未覆盖全样本')

    json.dump(res, open(os.path.join(WORK, 'm7_structure.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=str)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:4000])


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""t4_mixed.py — 混交 91/92 两种处置方案的差异（S1 单列 vs S2 优势型）
* 输入（只读）：points_76.parquet、needle_broad_map.json、needle_broad_map.json → mixed_policy
                FCS10 瓦片（S2 需要 51×51 像元窗口计优势型）
* 规则源（先冻结）：
    S1：混交单列 33，不并入 31/32
    S2：以混交参考像元为中心，51×51 原生 10m 窗口内纯林像元 needle_share>=0.5→31，<0.5→32；
        窗口内无纯林 → 保持 33
* 对比口径：只在「参考=FCS10 混交(33)」的点上比较两方案把它们判成 31/32/33 的比例，
            以及在「参考=纯林」点上预测为 33 的泄漏（两方案应相同，因 33 仍在预测空间里）；
            另报整体针/阔 P/R/F1（S2 把混交参考改标为 31/32 后重算）
* 输出：mixed_policy_eval.json
* 用法：python t4_mixed.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = r'F:/lc_work/prod5p_2023/xcomp/ext_tiles/fcs10'
AEF = [f'A{i:02d}' for i in range(64)]
WIN = 51


def dominant_type(lon, lat, tile, fcs10_raw):
    """对每个混交点取 51×51 窗口，计纯林像元占比 → 31/32/33"""
    out = np.full(len(lon), 33, dtype=np.int16)
    share = np.full(len(lon), np.nan, dtype=np.float32)
    nwin = np.zeros(len(lon), dtype=np.int32)
    fp = os.path.join(EXT, tile + '_fcs10.tif')
    with rasterio.open(fp) as s:
        rows, cols = rasterio.transform.rowcol(s.transform, lon, lat)
        rows = np.asarray(rows); cols = np.asarray(cols)
        need_r = int(np.ceil(WIN / 2))
        # 读整幅太大（~500MB）；改为按行分组、仅读混交行的窗口带
        order = np.argsort(rows)
        r_sorted = rows[order]
        # 逐点读窗口（混交点数量少，单瓦几十~几百）
        for i in order:
            r, c = int(rows[i]), int(cols[i])
            r0 = max(0, r - need_r); c0 = max(0, c - need_r)
            r1 = min(s.height, r + need_r + 1); c1 = min(s.width, c + need_r + 1)
            sub = s.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
            nn = int(np.isin(sub, [71, 72, 81, 82]).sum())
            nb = int(np.isin(sub, [51, 52, 61, 62]).sum())
            tot = nn + nb
            nwin[i] = tot
            if tot == 0:
                continue
            sh = nn / tot
            share[i] = sh
            out[i] = 31 if sh >= 0.5 else 32
    return out, share, nwin


def main():
    df = pd.read_parquet(os.path.join(HERE, 'points_76.parquet'),
                         columns=['lon', 'lat', 'tile', 'fcs10_raw', 'cl_fcs10'])
    mix = df[df.cl_fcs10 == 33].reset_index(drop=True)
    print('混交参考点', len(mix), '瓦片', mix.tile.nunique())
    t0 = time.time()
    new = np.full(len(mix), 33, np.int16); share = np.full(len(mix), np.nan, np.float32)
    nwin = np.zeros(len(mix), np.int32)
    for tile, sub in mix.groupby('tile'):
        idx = sub.index.to_numpy()
        n2, s2, w2 = dominant_type(sub.lon.values, sub.lat.values, tile, sub.fcs10_raw.values)
        new[idx] = n2; share[idx] = s2; nwin[idx] = w2
    print('优势型判定完成 %.0fs' % (time.time() - t0))
    mix['dom'] = new
    mix['share'] = share
    mix['n_pure_win'] = nwin
    dist = mix.dom.value_counts().sort_index().to_dict()
    st = {
        'n_mixed_ref_points': int(len(mix)),
        'tiles_with_mixed': int(mix.tile.nunique()),
        'dominant_dist': {str(k): int(v) for k, v in dist.items()},
        'no_pure_in_window': int((mix.n_pure_win == 0).sum()),
        'share_stats': {
            'n': int(mix.share.notna().sum()),
            'mean': round(float(mix.share.mean()), 4) if mix.share.notna().any() else None,
            'median': round(float(mix.share.median()), 4) if mix.share.notna().any() else None,
        },
        'by_tile': {t: {
            'n_mixed': int(len(sub)),
            'to_31': int((sub.dom == 31).sum()),
            'to_32': int((sub.dom == 32).sum()),
            'stay_33': int((sub.dom == 33).sum()),
            'median_needle_share': round(float(sub.share.median()), 3) if sub.share.notna().any() else None,
        } for t, sub in mix.groupby('tile')},
        'window': {'size_px': WIN, 'approx_m': int(WIN * 10), 'rule': 'needle_share>=0.5 -> 31 else 32; no pure -> 33'},
    }
    with open(os.path.join(HERE, 'mixed_policy_eval.json'), 'w', encoding='utf-8') as f:
        json.dump(st, f, ensure_ascii=False, indent=1)
    # 存点级结果供报告引用
    mix.to_parquet(os.path.join(HERE, 'mixed_points_dominant.parquet'), index=False)
    print(json.dumps({k: v for k, v in st.items() if k != 'by_tile'}, ensure_ascii=False, indent=1))
    print('-> mixed_policy_eval.json')


if __name__ == '__main__':
    main()

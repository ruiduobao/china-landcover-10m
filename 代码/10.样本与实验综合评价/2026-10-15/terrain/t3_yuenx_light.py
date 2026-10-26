# -*- coding: utf-8 -*-
"""t3_yuenx_light.py — 广东/宁夏 轻量对照：两版类码 vs 1km 高程（无山体尺度 DEM，不做坡向）

* 输入（只读）：二版 F:/lc_work/prod2nd_2023/delivery/成品/分省/{广东,宁夏}_2023_10m.tif
    一版 D:/生产_2023五省/成品/分省/{广东,宁夏}_2023_10m.tif（同 grid）
    高程 F:/lc_work/v31_exp/data/eco_gate/_dem/srtm1000_chn.tif（0.01°≈1km，SRTM 汇总；**非山体尺度**）
* 规则源：v31_map.json 24 类码；高程带 [0,200,500,1000,1500,2500,9999]
* 门槛：无（诊断）；抽样=每省 150 个 256×256 原生 10m 窗（随机内部块）
* 输出：terrain/tables_粤宁_light.json + 控制台
* 用法：python t3_yuenx_light.py
"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
OUTD = r'F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/terrain'
DEM1K = r'F:/lc_work/v31_exp/data/eco_gate/_dem/srtm1000_chn.tif'
PROVS = {
    '广东': (r'F:/lc_work/prod2nd_2023/delivery/成品/分省/广东_2023_10m.tif',
             r'D:/生产_2023五省/成品/分省/广东_2023_10m.tif'),
    '宁夏': (r'F:/lc_work/prod2nd_2023/delivery/成品/分省/宁夏_2023_10m.tif',
             r'D:/生产_2023五省/成品/分省/宁夏_2023_10m.tif')}
EDGES = [0, 200, 500, 1000, 1500, 2500, 9999]
NAME = {1: '草本旱地', 2: '乔灌园地', 3: '灌溉耕地', 4: '常绿阔叶', 5: '落叶阔叶', 6: '常绿针叶',
        9: '常绿灌丛', 10: '落叶灌丛', 11: '草地', 21: '不透水面', 22: '裸地', 23: '水体', 16: '湖河滩地'}
SEED = 20261015


def main():
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import reproject, Resampling
    rng = np.random.default_rng(SEED)
    out = {'note': '1km 高程仅作高程带控制；两省本地无山体尺度 DEM，坡向问题不评估',
           'elev_edges': EDGES}
    for prov, (c2fp, c1fp) in PROVS.items():
        with rasterio.open(c2fp) as s2, rasterio.open(c1fp) as s1:
            assert s2.shape == s1.shape, (s2.shape, s1.shape)
            t2, t1 = s2.transform, s1.transform
            assert max(abs(t2.a - t1.a), abs(t2.e - t1.e), abs(t2.c - t1.c), abs(t2.f - t1.f)) < 1e-8, (t2, t1)
            if t2.c != t1.c or t2.f != t1.f:
                print('  [注] 两版 transform 尾差 <1e-8°，按同窗直读（网格等价）')
            H, W = s2.shape
            tfm = s2.transform
            nb = 0
            cls2 = np.array([], np.int16); cls1 = np.array([], np.int16); elev = np.array([], np.float32)
            tries = 0
            while nb < 150 and tries < 600:
                tries += 1
                w = Window(int(rng.integers(0, W - 256)), int(rng.integers(0, H - 256)), 256, 256)
                a2 = s2.read(1, window=w)
                if (a2 > 0).mean() < 0.5:
                    continue
                a1 = s1.read(1, window=w)
                wt = rasterio.windows.transform(w, tfm)
                el = np.empty((256, 256), np.float32)
                with rasterio.open(DEM1K) as sd:
                    reproject(rasterio.band(sd, 1), el, src_transform=sd.transform, src_crs=sd.crs,
                              dst_transform=wt, dst_crs=s2.crs, resampling=Resampling.bilinear)
                m = a2 > 0
                cls2 = np.concatenate([cls2, a2[m].astype(np.int16)])
                cls1 = np.concatenate([cls1, a1[m].astype(np.int16)])
                elev = np.concatenate([elev, el[m]])
                nb += 1
        n = cls2.size
        eb = np.digitize(elev, EDGES[1:-1])
        res = {'n_px': int(n), 'windows': nb,
               'class_px_2nd': {str(int(c)): int((cls2 == c).sum()) for c in np.unique(cls2)},
               'class_px_1st': {str(int(c)): int((cls1 == c).sum()) for c in np.unique(cls1)}}
        # 逐类高程分布 + 各高程带林型构成
        perch = {}
        for c in [4, 5, 6, 9, 10, 11]:
            m = cls2 == c
            if m.sum() < 500:
                continue
            perch[str(c)] = {'n': int(m.sum()), 'elev_p10_p50_p90': [int(np.percentile(elev[m], p)) for p in (10, 50, 90)],
                             'by_band_pct': [round(100 * float((m & (eb == b)).sum()) / m.sum(), 1) for b in range(len(EDGES) - 1)]}
        res['per_class_elev_2nd'] = perch
        bandmix = []
        for b in range(len(EDGES) - 1):
            m = eb == b
            if m.sum() < 2000:
                continue
            fs = m & np.isin(cls2, (4, 5, 6))
            bandmix.append({'band': [EDGES[b], EDGES[b + 1]], 'n_px': int(m.sum()),
                            'forest_pct': round(100 * float(fs.sum()) / m.sum(), 1),
                            'EBF_share_of_forest': round(100 * float((fs & (cls2 == 4)).sum()) / max(fs.sum(), 1), 1),
                            'ENF_share_of_forest': round(100 * float((fs & (cls2 == 6)).sum()) / max(fs.sum(), 1), 1),
                            'DBF_share_of_forest': round(100 * float((fs & (cls2 == 5)).sum()) / max(fs.sum(), 1), 1)})
        res['forest_mix_by_elev_2nd'] = bandmix
        # 两版转移（森林三类）
        trans = {}
        for a in (4, 5, 6):
            for b in (4, 5, 6):
                m = (cls1 == a) & (cls2 == b)
                if m.sum() >= 200:
                    trans['%d->%d' % (a, b)] = int(m.sum())
        res['transition_4_5_6'] = trans
        out[prov] = res
        print('## %s n_px=%.2fM windows=%d' % (prov, n / 1e6, nb))
        print('  二版逐类高程(10/50/90):')
        for k, v in perch.items():
            print('    cls%2s %-6s n=%8d elev=%s band_pct=%s' % (k, NAME.get(int(k), ''), v['n'], v['elev_p10_p50_p90'], v['by_band_pct']))
        print('  高程带森林构成:')
        for r in bandmix:
            print('    %s m: forest=%.1f%% 其中 EBF=%.1f%% ENF=%.1f%% DBF=%.1f%%' %
                  (r['band'], r['forest_pct'], r['EBF_share_of_forest'], r['ENF_share_of_forest'], r['DBF_share_of_forest']))
        print('  两版转移 4/5/6:', trans)
    json.dump(out, open(os.path.join(OUTD, 'tables_粤宁_light.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    print('已写 tables_粤宁_light.json')


if __name__ == '__main__':
    main()

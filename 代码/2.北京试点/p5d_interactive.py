# -*- coding: utf-8 -*-
"""
p5d_interactive.py — 小区域 8 年逐年分类·交互式直下（免 batch 排队）
* 适用: 区域 ≤0.6°×0.5°（uint8 10m 单年 ~20MB < 50MB GEE 上限）
* 每年: 年度嵌入 mosaic → classify(decisionTreeEnsemble) → clip(region) → getDownloadURL(GEO_TIFF)
* 输出: {out_dir}/raster/{task_prefix}_{year}.tif （p7 直接可用的分年成品）
* 全国生产不适用此脚本（面积超限），仍走 p5 batch；本脚本用于流程验证/抽查。
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

def main():
    import requests
    e, pid = PC.load_account('zsi8emo')   # 必须在构造任何 ee 对象之前初始化
    import ee
    cfg = json.load(open(os.path.join(PC.OUT_DIR, 'models_pilot', 'trees_bj.json'), encoding='utf-8'))
    clf = ee.Classifier.decisionTreeEnsemble(cfg['trees'])
    FEATS_M = cfg['features']
    # 小窗口交互直下用 bbox 矩形（≤32768 像元边）；行政区界仅 batch 用
    lon0, lat0, lon1, lat1 = PC.BBOX
    REG = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
    out_dir = os.path.join(PC.OUT_DIR, 'raster')
    os.makedirs(out_dir, exist_ok=True)

    # 单账号完成全部年份（ee 全局态禁止进程内切换账号；交互式不占 batch 并发）
    e = ee  # 已由 load_account 初始化

    results = {}
    lon0, lat0, lon1, lat1 = PC.BBOX
    STEP = 0.25   # 0.25° 分块（uint8 ~23MB < 50MB 上限）
    for i, year in enumerate(PC.YEARS):
        t0 = time.time()
        parts = []
        nx = int(np.ceil((lon1 - lon0) / STEP))
        ny = int(np.ceil((lat1 - lat0) / STEP))
        for a in range(nx):
            for b in range(ny):
                x0 = lon0 + a * STEP; x1 = min(lon1, x0 + STEP)
                y0 = lat0 + b * STEP; y1 = min(lat1, y0 + STEP)
                fp = os.path.join(out_dir, f'{PC.TASK_PREFIX}_{year}_p{a}{b}.tif')
                if os.path.exists(fp) and os.path.getsize(fp) > 1000:
                    parts.append(fp)
                    continue
                blk = e.Geometry.Rectangle([x0, y0, x1, y1])
                emb = (e.ImageCollection(PC.EMB_COL)
                       .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
                       .filterBounds(blk).mosaic().select(FEATS_M))
                cls = (emb.classify(clf).rename('class').uint8())
                url = cls.getDownloadURL(dict(region=blk, scale=10, crs='EPSG:4326',
                                              format='GEO_TIFF'))
                r = requests.get(url, proxies=PC.PROXY, timeout=1200)
                r.raise_for_status()
                open(fp, 'wb').write(r.content)
                parts.append(fp)
                time.sleep(1)
        print(f'{year}: {nx*ny} 块 {sum(os.path.getsize(f) for f in parts)/1e6:.0f} MB '
              f'({time.time()-t0:.0f}s)', flush=True)
        results[year] = parts
    print('\n8 年交互式分类完成:', len(results), '期 →', out_dir)

if __name__ == '__main__':
    main()

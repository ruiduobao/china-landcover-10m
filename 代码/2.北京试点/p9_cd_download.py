# -*- coding: utf-8 -*-
"""
p9_cd_download.py — 变化检测实验·逐年嵌入 L2 距离图下载（小窗口）
* 对 (2017,2018)...(2023,2024) 相邻年对 + (2017,2024) 端点：
  d = ||emb(y2) - emb(y1)||₂（64 维全波段），GEE 端逐块算好只下 1 波段 float32
* 0.1° 分块（1 波段 ~5MB，无树评估不触发内存限制）；429/503 退避；断点续传
* 输出: 数据/本地处理/北京试点S/cd/{PAIR}_p{a}{b}.tif
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC

def main():
    import requests
    e, pid = PC.load_account('zsi8emo')
    import ee
    lon0, lat0, lon1, lat1 = PC.BBOX
    STEP = 0.1
    out_dir = os.path.join(PC.OUT_DIR, 'cd')
    os.makedirs(out_dir, exist_ok=True)

    pairs = [(y, y + 1) for y in PC.YEARS[:-1]] + [(PC.YEARS[0], PC.YEARS[-1])]
    col_cache = {}
    def emb_img(y1, y2):
        key = (y1, y2)
        if key not in col_cache:
            e1 = (ee.ImageCollection(PC.EMB_COL).filterDate(f'{y1}-01-01', f'{y1+1}-01-01')
                  .filterBounds(ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])).mosaic()
                  .select(PC.FEATS_ALL))
            e2 = (ee.ImageCollection(PC.EMB_COL).filterDate(f'{y2}-01-01', f'{y2+1}-01-01')
                  .filterBounds(ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])).mosaic()
                  .select(PC.FEATS_ALL))
            d = e2.subtract(e1).pow(2).reduce(ee.Reducer.sum()).sqrt().clamp(0, 30)
            col_cache[key] = d.toFloat().rename('l2')
        return col_cache[key]

    nx = int(np.ceil((lon1 - lon0) / STEP))
    ny = int(np.ceil((lat1 - lat0) / STEP))
    total = len(pairs) * nx * ny
    n = 0
    t0 = time.time()
    for (y1, y2) in pairs:
        pair = f'{y1}_{y2}'
        for a in range(nx):
            for b in range(ny):
                n += 1
                fp = os.path.join(out_dir, f'{pair}_p{a}{b}.tif')
                if os.path.exists(fp) and os.path.getsize(fp) > 500:
                    continue
                blk = ee.Geometry.Rectangle([lon0 + a * STEP, lat0 + b * STEP,
                                             min(lon1, lon0 + (a+1) * STEP),
                                             min(lat1, lat0 + (b+1) * STEP)])
                img = emb_img(y1, y2).clip(blk)
                ok = False
                for att in range(6):
                    try:
                        url = img.getDownloadURL(dict(region=blk, scale=10,
                                                      crs='EPSG:4326', format='GEO_TIFF'))
                        r = requests.get(url, proxies=PC.PROXY, timeout=600)
                        if r.status_code == 200:
                            open(fp, 'wb').write(r.content)
                            ok = True
                            break
                        wait = 15 * (att + 1)
                        print(f'  {pair} p{a}{b} HTTP {r.status_code} 退避{wait}s', flush=True)
                        time.sleep(wait)
                    except Exception as ex:
                        wait = 15 * (att + 1)
                        print(f'  {pair} p{a}{b} EXC {str(ex)[:60]} 退避{wait}s', flush=True)
                        time.sleep(wait)
                if not ok:
                    raise RuntimeError(f'{pair} p{a}{b} 重试 6 次失败')
                time.sleep(2)
        print(f'{pair} 完成 ({n}/{total}, {time.time()-t0:.0f}s)', flush=True)
    print('全部距离图下载完成 →', out_dir)

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
p6_download.py — SUCCEEDED 任务资产 → 本地 GeoTIFF（getDownloadURL 分瓦）
* 北京 ~1.68万km²，10m UInt8 ≈ 130MB 未压缩，压缩后可单瓦或 0.25° 分瓦
* 输出: 数据/本地处理/北京试点/raster/CNLC10_BJ_{year}.tif
"""
import sys, os, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot_common as PC
import numpy as np

OUT = os.path.join(PC.OUT_DIR, 'raster')
os.makedirs(OUT, exist_ok=True)
TASKS = os.path.join(PC.OUT_DIR, 'tasks.jsonl')

def download_year(ee, year, asset, acct):
    import requests
    img = ee.Image(asset)
    # 0.25° 分瓦（实测 50MB GEE 上限：0.55°×0.85°=221MB 超限；0.25²≈23MB OK）
    lon0, lat0, lon1, lat1 = PC.BBOX
    parts = []
    steps_lon = int(np.ceil((lon1 - lon0) / 0.25))
    steps_lat = int(np.ceil((lat1 - lat0) / 0.25))
    n = 0
    for i in range(steps_lon):
        for j in range(steps_lat):
            x0 = lon0 + i * 0.25; x1 = min(lon1, x0 + 0.25)
            y0 = lat0 + j * 0.25; y1 = min(lat1, y0 + 0.25)
            if x1 - x0 < 0.01 or y1 - y0 < 0.01:
                continue
            fp = os.path.join(OUT, f'CNLC10_BJ_{year}_p{i}{j}.tif')
            n += 1
            if os.path.exists(fp) and os.path.getsize(fp) > 1000:
                parts.append(fp)
                continue
            region = ee.Geometry.Rectangle([x0, y0, x1, y1])
            url = img.getDownloadURL(dict(
                region=region, scale=10, crs='EPSG:4326',
                format='GEO_TIFF'))
            print(f'  {year} p{i}{j}: 下载…')
            r = requests.get(url, proxies=PC.PROXY, timeout=1200)
            r.raise_for_status()
            open(fp, 'wb').write(r.content)
            print(f'  {year} p{i}{j}: {len(r.content)/1e6:.1f} MB')
            parts.append(fp)
            time.sleep(1)
    return parts

def main():
    tasks = json.load(open(TASKS))
    by_year = {}
    for t in tasks:
        # 取每年最新一次提交（重提覆盖）
        by_year[t['year']] = t
    PC.load_account('zsi8emo')  # 下载用交互式，任意账号可读自己资产；跨账号需资产公开或用所属账号
    import ee
    ok_years = []
    for year, t in sorted(by_year.items()):
        if t['status'] != 'SUCCEEDED':
            print(f"[skip] {year}: {t['status']}")
            continue
        PC.load_account(t['account'])
        try:
            parts = download_year(ee, year, t['asset'], t['account'])
            ok_years.append(year)
        except Exception as e:
            print(f"[ERR] {year}: {str(e)[:150]}")
    print('\n已下载年份:', ok_years)

if __name__ == '__main__':
    main()

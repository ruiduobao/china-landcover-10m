# -*- coding: utf-8 -*-
"""s2_timeseries.py — 哨兵-2 生长季时序材料（免密钥 STAC）+ 物候指标

* 目的：为作物细类判别提供"有日期"的物候证据（高分底图只给结构，不给日期）
* 输入：--points points.csv（id/lon/lat）；--year/--start/--end 生长季窗口；--max-cloud 云量上限
* 规则源：references/地类判读图谱.md 第 2/3 节（区域农时与指标口径）；references/影像源与瓦片手册.md 第 4 节
* 门槛：优先 eo:cloud_cover 低的景；窗口内按时间均匀取 --n-dates 景；点必须落在景的栅格范围内
        （经纬度先投影到该景 CRS 再判断，否则会读到空窗口——本项目实测坑）；
        单点无有效景 → 记 no_scene，不编造
* 输出：<outdir>/s2/<id>_时序.png（真彩缩略条 + 日期 + NDVI/NDWI 曲线）
        <outdir>/s2_metrics.csv（n/dates/ndvi_mean/amp/peak/peak_date/ndwi_min/淹水证据）
* 用法：python s2_timeseries.py --points points.csv --outdir ./run1 --year 2023 --start 05-01 --end 09-30
        python s2_timeseries.py --points points.csv --outdir ./run1 --n-dates 6 --max-cloud 30 --workers 6
幂等：<id>_时序.png 已存在且 metrics 有记录则跳过（--force 重写）。
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

STAC = 'https://earth-search.aws.element84.com/v1/search'
UA = {'User-Agent': 'Mozilla/5.0'}
HALF_M = 400          # 取 800 m 见方窗口（足够体现地物背景，又不拖慢）
THUMB = 84            # 每期缩略图边长


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def stac_search(lon, lat, dt_range, max_cloud, limit=40):
    body = {'collections': ['sentinel-2-l2a'],
            'intersects': {'type': 'Point', 'coordinates': [lon, lat]},
            'datetime': dt_range,
            'query': {'eo:cloud_cover': {'lt': max_cloud}},
            'limit': limit}
    req = urllib.request.Request(STAC, headers={**UA, 'Content-Type': 'application/json'},
                                 data=json.dumps(body).encode())
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open(req, timeout=60) as r:
        return json.loads(r.read())['features']


def contains(feat, lon, lat):
    """用 WGS84 footprint 几何筛景（零文件打开——实测每开一个 COG 要 4–5 s，不能拿来筛选）。"""
    try:
        from shapely.geometry import shape, Point
        return shape(feat['geometry']).contains(Point(lon, lat))
    except Exception:
        return True


def flood_tag(nw, opt):
    """淹水判据：NDWI(绿,近红外) 为正=水体/淹水。看**早期**（前 40% 期次）是否出现正值——
    移栽/泡田窗口在生长季前期；用 min 是反的（植被期 NDWI 本来就负）。"""
    k = max(1, int(round(len(nw) * 0.4)))
    early = nw[:k]
    mx_any, mx_early = float(np.nanmax(nw)), float(np.nanmax(early))
    if mx_early > 0.05:
        return 'yes'
    if mx_any > 0.05:
        return 'mid'          # 淹水出现在中后期（如晚稻/灌溉事件），需结合农时看
    if mx_any > -0.05:
        return 'weak'
    return 'no'


def pick_scenes(feats, n):
    """优先云量低，再按时间均匀抽 n 景（去重同一天）。"""
    seen, uniq = set(), []
    for f in sorted(feats, key=lambda x: (x['properties'].get('eo:cloud_cover', 100))):
        d = f['properties']['datetime'][:10]
        if d in seen:
            continue
        seen.add(d)
        uniq.append(f)
    uniq.sort(key=lambda x: x['properties']['datetime'])
    if len(uniq) <= n:
        return uniq
    idx = np.linspace(0, len(uniq) - 1, n).round().astype(int)
    return [uniq[i] for i in sorted(set(idx))]


def read_window(ds, lon, lat, res_m=30):
    from pyproj import Transformer
    from rasterio.windows import from_bounds
    tr = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    if not (ds.bounds.left <= x <= ds.bounds.right and ds.bounds.bottom <= y <= ds.bounds.top):
        return None
    w = from_bounds(x - HALF_M, y - HALF_M, x + HALF_M, y + HALF_M, ds.transform)
    px = max(8, int(2 * HALF_M / res_m))
    return ds.read(1, window=w, out_shape=(px, px), boundless=True, fill_value=0).astype('float32')


def read_band(href, lon, lat, res_m):
    import rasterio
    with rasterio.open(href) as ds:
        return read_window(ds, lon, lat, res_m)


def read_scene(assets, lon, lat, res_m):
    """单景四波段并行读（S3 跨境单文件 open 4–5 s，串行会拖垮——实测）。"""
    from concurrent.futures import ThreadPoolExecutor
    out = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {k: ex.submit(read_band, assets[k]['href'], lon, lat, res_m)
                for k in ('red', 'green', 'blue', 'nir') if k in assets}
        for k, fu in futs.items():
            try:
                out[k] = fu.result()
            except Exception:
                out[k] = None
    return out


def one_point(row, opt):
    pid = str(row[opt.id_col])
    fp = os.path.join(opt.outdir, 's2', pid + '_时序.png')
    if os.path.exists(fp) and not opt.force:
        return {'id': pid, 'skip': True}
    lon, lat = float(row[opt.lon_col]), float(row[opt.lat_col])
    dt = '%d-%sT00:00:00Z/%d-%sT23:59:59Z' % (opt.year, opt.start, opt.year, opt.end)
    try:
        feats = stac_search(lon, lat, dt, opt.max_cloud)
    except Exception as e:
        return {'id': pid, 'error': 'STAC %s' % type(e).__name__}
    feats = [f for f in feats if contains(f, lon, lat)]
    picks = pick_scenes(feats, opt.n_dates)
    t0 = time.time()
    rows, thumbs = [], []
    for f in picks:
        d = read_scene(f['assets'], lon, lat, opt.res_m)
        R, N, G, B = d.get('red'), d.get('nir'), d.get('green'), d.get('blue')
        if R is None or N is None or R.mean() <= 0:
            continue
        R = np.where(R > 0, R, np.nan)
        N = np.where(N > 0, N, np.nan)
        ok = np.isfinite(R) & np.isfinite(N)
        if ok.sum() < 0.2 * ok.size:
            continue
        # 逐窗云/亮目标掩膜（亮度启发式：红与近红外同时接近窗口高分位 → 云/雪/亮裸）
        br = ok & (R > np.nanpercentile(R[ok], 88)) & (N > np.nanpercentile(N[ok], 88))
        cloud_frac = float(br.sum()) / float(ok.sum())
        if cloud_frac > 0.5:
            continue          # 窗口大半是云，弃用该期（避免污染物候曲线）
        ndvi = np.full(R.shape, np.nan)
        ndvi[ok] = (N[ok] - R[ok]) / np.maximum(N[ok] + R[ok], 1)
        if G is not None:
            G = np.where(G > 0, G, np.nan)
            okw = np.isfinite(G) & np.isfinite(N)
            ndwi = np.full(R.shape, np.nan)
            ndwi[okw] = (G[okw] - N[okw]) / np.maximum(G[okw] + N[okw], 1)
        else:
            ndwi = np.full(R.shape, np.nan)
        ok2 = ok & ~br
        ndvi_v = np.full(R.shape, np.nan); ndvi_v[ok2] = ndvi[ok2]
        rows.append(dict(date=f['properties']['datetime'][:10],
                         cloud=f['properties'].get('eo:cloud_cover'),
                         cloud_win=round(cloud_frac, 2),
                         ndvi=float(np.nanmedian(ndvi_v)), ndwi=float(np.nanmedian(ndwi))))
        # 真彩缩略（2–98% 拉伸）
        rgb = np.dstack([np.nan_to_num(R), np.nan_to_num(G if G is not None else R),
                         np.nan_to_num(B if B is not None else R)])
        lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
        rgb = np.clip((rgb - lo) / max(hi - lo, 1) * 255, 0, 255).astype('uint8')
        thumbs.append((f['properties']['datetime'][5:10], Image.fromarray(rgb).resize((THUMB, THUMB), Image.BILINEAR)))
    if not rows:
        return {'id': pid, 'error': 'no_scene（窗口内无有效景）'}
    nd = np.array([r['ndvi'] for r in rows], dtype='float32')
    nw = np.array([r['ndwi'] for r in rows], dtype='float32')
    imax = int(np.nanargmax(nd))
    m = dict(id=pid, n=len(rows), dates=';'.join(r['date'] for r in rows),
             clouds=';'.join('%.0f/%.2f' % (r['cloud'] or 0, r.get('cloud_win', 0)) for r in rows),
             ndvi_mean=round(float(np.nanmean(nd)), 3), ndvi_amp=round(float(np.nanmax(nd) - np.nanmin(nd)), 3),
             ndvi_peak=round(float(nd[imax]), 3), ndvi_peak_date=rows[imax]['date'],
             ndwi_min=round(float(np.nanmin(nw)), 3), ndwi_max=round(float(np.nanmax(nw)), 3),
             ndwi_first=round(float(nw[0]), 3), ndwi_argmax_date=rows[int(np.nanargmax(nw))]['date'],
             flood_evidence=flood_tag(nw, opt),
             first_ndvi=round(float(nd[0]), 3), last_ndvi=round(float(nd[-1]), 3))
    # 画：顶部真彩条 + 日期 + 下方 NDVI/NDWI 曲线
    W = max(len(thumbs) * (THUMB + 4) + 4, 320)
    H = THUMB + 130
    im = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f14 = None
    for cand in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
                 'C:/Windows/Fonts/simsun.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
        if os.path.exists(cand):
            try:
                f14 = ImageFont.truetype(cand, 13)
                break
            except Exception:
                pass
    if f14 is None:
        f14 = ImageFont.load_default()
    for i, (dtxt, t) in enumerate(thumbs):
        x0 = 4 + i * (THUMB + 4)
        im.paste(t, (x0, 18))
        d.text((x0 + 2, 3), dtxt, fill=(0, 0, 0), font=f14)
    # NDVI 曲线
    y0, y1 = THUMB + 30, H - 26
    xs = [4 + i * (THUMB + 4) + THUMB // 2 for i in range(len(nd))]
    d.line((4, y0 + (y1 - y0) * 0.7, W - 4, y0 + (y1 - y0) * 0.7), fill=(200, 200, 200))
    pts = [(xs[i], y1 - (nd[i] - (-0.2)) / 1.2 * (y1 - y0)) for i in range(len(nd))]
    d.line([tuple(p) for p in pts], fill=(0, 150, 0), width=2)
    for (x, y), v in zip(pts, nd):
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(0, 150, 0))
    pts2 = [(xs[i], y1 - (nw[i] - (-0.6)) / 1.2 * (y1 - y0)) for i in range(len(nw))]
    d.line([tuple(p) for p in pts2], fill=(0, 90, 220), width=1)
    d.text((6, y0 - 14), 'NDVI 绿 / NDWI 蓝（点=日期）  amp=%.2f peak=%s' % (m['ndvi_amp'], m['ndvi_peak_date']),
           fill=(0, 0, 0), font=f14)
    im.save(fp)
    return m


def main():
    ap = argparse.ArgumentParser(description='哨兵时序材料（免密钥 STAC）')
    ap.add_argument('--points', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--id-col', default='point_id')
    ap.add_argument('--lon-col', default='lon')
    ap.add_argument('--lat-col', default='lat')
    ap.add_argument('--year', type=int, default=2023)
    ap.add_argument('--start', default='05-01')
    ap.add_argument('--end', default='09-30')
    ap.add_argument('--n-dates', type=int, default=5)
    ap.add_argument('--max-cloud', type=float, default=30)
    ap.add_argument('--workers', type=int, default=3, help='并发点数；总连接≈workers×4，实测 12 左右最优（>24 会互相拖慢）')
    ap.add_argument('--res-m', type=int, default=30, help='读窗重采样分辨率（米）；30=够判绿度、更快')
    ap.add_argument('--force', action='store_true')
    opt = ap.parse_args()
    os.makedirs(os.path.join(opt.outdir, 's2'), exist_ok=True)
    rows = list(csv.DictReader(open(opt.points, encoding='utf-8-sig')))
    todo = [r for r in rows if opt.force or not os.path.exists(
        os.path.join(opt.outdir, 's2', str(r[opt.id_col]) + '_时序.png'))]
    emit('点位 %d，待取 %d（%d 生长季 %s–%s，≤%d 期，云<%.0f%%）' % (
        len(rows), len(todo), opt.year, opt.start, opt.end, opt.n_dates, opt.max_cloud))
    out, n, t0 = [], 0, time.time()
    cols = ['id', 'n', 'ndvi_mean', 'ndvi_amp', 'ndvi_peak', 'ndvi_peak_date',
            'ndwi_max', 'ndwi_min', 'ndwi_first', 'ndwi_argmax_date', 'flood_evidence',
            'first_ndvi', 'last_ndvi', 'dates', 'clouds']

    def flush(write_header=False):
        fp2 = os.path.join(opt.outdir, 's2_metrics.csv')
        prev = {}
        if os.path.exists(fp2) and not write_header:
            prev = {r['id']: r for r in csv.DictReader(open(fp2, encoding='utf-8-sig'))}
        for mm in out:
            if not mm.get('error') and not mm.get('skip'):
                prev[mm['id']] = mm
        with open(fp2, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for k in sorted(prev):
                w.writerow({c: prev[k].get(c, '') for c in cols})

    with ThreadPoolExecutor(max_workers=opt.workers) as ex:
        for m in ex.map(lambda r: one_point(r, opt), todo):
            n += 1
            out.append(m)
            if m.get('error'):
                emit('  FAIL %s %s' % (m['id'], m['error']))
            if n % 5 == 0:
                emit('  进度 %d/%d（%.0fs）' % (n, len(todo), time.time() - t0))
                flush()
    fp = os.path.join(opt.outdir, 's2_metrics.csv')
    flush()
    ok = sum(1 for m in out if not m.get('error') and not m.get('skip'))
    emit('完成：%d 点出材料（%.0fs）→ %s；指标表 %s' % (ok, time.time() - t0, os.path.join(opt.outdir, 's2'), fp))


if __name__ == '__main__':
    main()

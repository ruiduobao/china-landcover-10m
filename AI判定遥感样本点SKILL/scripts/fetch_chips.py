# -*- coding: utf-8 -*-
"""fetch_chips.py — 样本点高分影像切片抓取（Esri 主源 / Google 备源）

* 输入：--points points.csv（必含 id 列、lon、lat；WGS84 经纬度）
* 规则源：references/影像源与瓦片手册.md（瓦片数学、源选择、并发与代理）
* 门槛：3×3 瓦片拼图后按点裁 --crop 窗口（点必落在中心瓦片内，裁窗数学上不会越界）；
        单块瓦片失败重试 2 次 → 换备源；逐点记录实际源与失败原因；断点续抓（切片已存在跳过）
* 输出：<outdir>/chips/<id>.png（默认 512×512；十字丝中心留空 + n 米目标框 + 编号）
        <outdir>/fetch_log.json（每点 source/zoom/mpp/时间/错误）
* 用法：python fetch_chips.py --points points.csv --outdir ./run1
        python fetch_chips.py --points points.csv --outdir ./run1 --zoom 19 --crop 768 --workers 8
        python fetch_chips.py --points points.csv --outdir ./run1 --source google --proxy socks5h://127.0.0.1:7890
幂等：同名切片存在即跳过；--force 重抓。
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
from PIL import Image, ImageDraw, ImageFont

ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
GOOG = 'http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}'
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
GAP = 24          # 十字丝中心留空（半径 px），保证靶心可见
FONT_CANDIDATES = ['C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def lonlat_to_px(lon, lat, z):
    """经纬度 → 该缩放级的全局像素坐标（Web Mercator）。"""
    n = 256.0 * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def mpp_at(lat, z):
    """该纬度、该缩放级的米/像素。"""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


def opener(proxy=None):
    if proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_tile(z, x, y, src, op):
    url = ESRI.format(z=z, y=y, x=x) if src == 'esri' else GOOG.format(x=x, y=y, z=z)
    req = urllib.request.Request(url, headers=UA)
    with op.open(req, timeout=30) as r:
        return r.read()


def crop_window(px, py, tx, ty, crop):
    """把全局像素 (px,py) 映射到 3×3 拼图（原点 = 中心瓦片左上一块）内的裁剪框。"""
    ox = int(round(px - (tx - 1) * 256)) - crop // 2
    oy = int(round(py - (ty - 1) * 256)) - crop // 2
    return max(0, min(768 - crop, ox)), max(0, min(768 - crop, oy))


def draw_marks(im, pid, half_px, label=True):
    d = ImageDraw.Draw(im)
    c = im.width // 2
    W = im.width
    for x0, y0, x1, y1 in ((0, c, c - GAP, c), (c + GAP, c, W, c), (c, 0, c, c - GAP), (c, c + GAP, c, W)):
        d.line((x0, y0, x1, y1), fill=(255, 220, 0), width=2)
    if half_px:
        d.rectangle((c - half_px, c - half_px, c + half_px, c + half_px), outline=(0, 255, 255), width=2)
    if label:
        font = None
        for fp in FONT_CANDIDATES:
            if os.path.exists(fp):
                font = ImageFont.truetype(fp, max(14, W // 23))
                break
        if font is None:
            font = ImageFont.load_default()
        tb = d.textbbox((8, 6), pid, font=font)
        d.rectangle((tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3), fill=(0, 0, 0))
        d.text((8, 6), pid, font=font, fill=(255, 255, 0))
    return im


def one_point(row, opt, cache):
    pid = str(row[opt.id_col])
    fp = os.path.join(opt.outdir, 'chips', pid + '.png')
    if os.path.exists(fp) and not opt.force:
        return {'id': pid, 'skip': True}
    try:
        lon, lat = float(row[opt.lon_col]), float(row[opt.lat_col])
    except Exception as e:
        return {'id': pid, 'error': 'lon/lat 解析失败: %s' % e}
    z = opt.zoom
    px, py = lonlat_to_px(lon, lat, z)
    tx, ty = int(px // 256), int(py // 256)
    srcs = ['esri', 'google'] if opt.source == 'auto' else [opt.source]
    mos = Image.new('RGB', (768, 768))
    used, errs = set(), []
    for i in range(-1, 2):
        for j in range(-1, 2):
            key = (z, tx + i, ty + j)
            blob = cache.get(key)
            if blob is None:
                for k, s in enumerate(srcs + srcs):        # 每源两轮重试
                    try:
                        blob = fetch_tile(z, tx + i, ty + j, s, opt._opener)
                        cache[key] = blob
                        break
                    except Exception as e:
                        errs.append('%s:%s' % (s, type(e).__name__))
                        time.sleep(0.5)
                if blob is None:
                    return {'id': pid, 'error': 'tile %s/%d/%d 全部源失败' % (z, tx + i, ty + j)}
            try:
                mos.paste(Image.open(io.BytesIO(blob)).convert('RGB'), ((i + 1) * 256, (j + 1) * 256))
            except Exception as e:
                return {'id': pid, 'error': 'decode: %s' % e}
    ox, oy = crop_window(px, py, tx, ty, opt.crop)
    im = mos.crop((ox, oy, ox + opt.crop, oy + opt.crop))
    half = int(round(opt.box_m / mpp_at(lat, z) / 2)) if opt.box_m else 0
    im = draw_marks(im, pid, max(5, half) if opt.box_m else 0)
    im.save(fp)
    return {'id': pid, 'source': '/'.join(sorted(srcs)) if len(srcs) == 1 else 'auto',
            'zoom': z, 'mpp': round(mpp_at(lat, z), 3), 'box_m': opt.box_m,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'retries': len(errs)}


def main():
    ap = argparse.ArgumentParser(description='样本点高分影像切片抓取')
    ap.add_argument('--points', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--id-col', default='point_id')
    ap.add_argument('--lon-col', default='lon')
    ap.add_argument('--lat-col', default='lat')
    ap.add_argument('--zoom', type=int, default=18, help='18≈0.5m、19≈0.25m（视源覆盖）')
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--box-m', type=float, default=10.0, help='目标框边长（米）；0=不画（如 Landsat 30m 场景填 30）')
    ap.add_argument('--source', default='auto', choices=['auto', 'esri', 'google'])
    ap.add_argument('--proxy', default=None, help='如 socks5h://127.0.0.1:7890（默认直连）')
    ap.add_argument('--workers', type=int, default=8, help='实测甜点 8–10，>24 明显劣化')
    ap.add_argument('--force', action='store_true')
    opt = ap.parse_args()
    opt._opener = opener(opt.proxy)
    os.makedirs(os.path.join(opt.outdir, 'chips'), exist_ok=True)
    rows = list(csv.DictReader(open(opt.points, encoding='utf-8-sig')))
    log_fp = os.path.join(opt.outdir, 'fetch_log.json')
    log = json.load(open(log_fp, encoding='utf-8')) if os.path.exists(log_fp) else {}
    todo = [r for r in rows if opt.force or not os.path.exists(
        os.path.join(opt.outdir, 'chips', str(r[opt.id_col]) + '.png'))]
    emit('点位 %d，待抓 %d（zoom=%d，源=%s，workers=%d）' % (len(rows), len(todo), opt.zoom, opt.source, opt.workers))
    cache, n, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(max_workers=opt.workers) as ex:
        for res in ex.map(lambda r: one_point(r, opt, cache), todo):
            n += 1
            log[res['id']] = res
            if res.get('error'):
                emit('  FAIL %s %s' % (res['id'], res['error']))
            if n % 25 == 0:
                emit('  进度 %d/%d（%.0fs）' % (n, len(todo), time.time() - t0))
                json.dump(log, open(log_fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(log, open(log_fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    ok = sum(1 for r in rows if os.path.exists(os.path.join(opt.outdir, 'chips', str(r[opt.id_col]) + '.png')))
    miss = [str(r[opt.id_col]) for r in rows
            if not os.path.exists(os.path.join(opt.outdir, 'chips', str(r[opt.id_col]) + '.png'))]
    emit('完成：%d/%d 切片就绪，用时 %.0fs%s' % (ok, len(rows), time.time() - t0,
                                          '' if not miss else '；缺：' + ','.join(miss[:10])))
    return 0 if not miss else 1


if __name__ == '__main__':
    raise SystemExit(main())

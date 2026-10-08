# -*- coding: utf-8 -*-
"""m5j_m3b_imagery.py — M3-B 复核点高分辨率影像切片（z18 瓦片拼图）与判读用联系表

* 输入：F:/lc_work/v31_exp/data/m3/m3b_最小复核清单.csv（210 点）
* 规则源：Esri World Imagery z18 为主源（直连实测 0.5–0.7 s/瓦片），Google 卫星为备源；
  每个点取 3×3 瓦片（768×768 px）拼图后按点裁 512×512；叠加十字丝（中心留空 48 px）
  与 10 m 目标框（按纬度换算米/像素）；左上角标 point_id（判读时不含组别/AI 标，防锚定）。
* 门槛：每点 9 瓦片，失败重试 2 次；若主源整块为空（std<1）或失败 → 备源；
  逐点记录 source 与 tile 完整性；缺图点列出待补。
* 输出：F:/lc_work/v31_exp/data/m3/m3b_imagery/
        chips/<point_id>.png            512×512 单点切片
        sheets/sheet_%02d.png           3×3 联系表（1536×1536）
        fetch_log.json                  每点 source/mpp/时间
* 用法：python m5j_m3b_imagery.py fetch [--workers 8]
        python m5j_m3b_imagery.py sheet
幂等：切片已存在且 fetch_log 有记录则跳过（--force 重写）。
"""
import io
import json
import math
import os
import sys
import time
import csv
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')
from PIL import Image, ImageDraw, ImageFont

M3D = r'F:/lc_work/v31_exp/data/m3'
SRC = os.path.join(M3D, 'm3b_最小复核清单.csv')
OUTD = os.path.join(M3D, 'm3b_imagery')
CHIPD = os.path.join(OUTD, 'chips')
SHEETD = os.path.join(OUTD, 'sheets')
LOG = os.path.join(OUTD, 'fetch_log.json')
Z = 18
CROP = 512
GAP = 24            # 十字丝中心留空（半径）
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/%d/%d/%d'
GOOG = 'http://mt0.google.com/vt/lyrs=s&hl=en&x=%d&y=%d&z=%d'


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def lonlat_to_px(lon, lat, z):
    n = 256.0 * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z, x, y, src):
    url = (ESRI % (z, y, x)) if src == 'esri' else (GOOG % (x, y, z))
    req = urllib.request.Request(url, headers=UA)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open(req, timeout=25) as r:
        return r.read()


def one_point(row, force=False, cache=None):
    pid, lon, lat = row['point_id'], float(row['lon']), float(row['lat'])
    fp = os.path.join(CHIPD, pid + '.png')
    if os.path.exists(fp) and not force:
        return {'tile': pid, 'cache': True}
    px, py = lonlat_to_px(lon, lat, Z)
    tx, ty = int(px // 256), int(py // 256)
    mos = Image.new('RGB', (768, 768))
    used = set()
    for i in range(-1, 2):
        for j in range(-1, 2):
            key = (Z, tx + i, ty + j)
            b = (cache or {}).get(key)
            src = 'esri'
            if b is None:
                for attempt, s in ((0, 'esri'), (1, 'google'), (2, 'esri')):
                    try:
                        b = fetch_tile(Z, tx + i, ty + j, s)
                        src = s
                        break
                    except Exception as e:
                        if attempt == 2:
                            return {'tile': pid, 'err': '%s@%d/%d/%d %s' % (s, Z, tx + i, ty + j, e)}
                        time.sleep(0.6)
                if cache is not None:
                    cache[key] = b
            try:
                im = Image.open(io.BytesIO(b)).convert('RGB')
            except Exception as e:
                return {'tile': pid, 'err': 'decode %s' % e}
            used.add(src)
            mos.paste(im, ((i + 1) * 256, (j + 1) * 256))
    ox = int(round(px - (tx - 1) * 256)) - CROP // 2
    oy = int(round(py - (ty - 1) * 256)) - CROP // 2
    ox = max(0, min(768 - CROP, ox))
    oy = max(0, min(768 - CROP, oy))
    im = mos.crop((ox, oy, ox + CROP, oy + CROP))
    d = ImageDraw.Draw(im)
    c = CROP // 2
    # 十字丝（中心留空 GAP）
    for x0, y0, x1, y1 in ((0, c, c - GAP, c), (c + GAP, c, CROP, c),
                           (c, 0, c, c - GAP), (c, c + GAP, c, CROP)):
        d.line((x0, y0, x1, y1), fill=(255, 220, 0), width=2)
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** Z)
    half = max(6, int(round(10.0 / mpp / 2)))
    d.rectangle((c - half, c - half, c + half, c + half), outline=(0, 255, 255), width=2)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 22)
    except Exception:
        font = ImageFont.load_default()
    txt = pid
    tb = d.textbbox((8, 6), txt, font=font)
    d.rectangle((tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3), fill=(0, 0, 0))
    d.text((8, 6), txt, font=font, fill=(255, 255, 0))
    im.save(fp)
    return {'tile': pid, 'src': '/'.join(sorted(used)), 'mpp': round(mpp, 3),
            'half_px': half, 'n_tiles': 9}


def cmd_fetch(force=False, workers=8):
    os.makedirs(CHIPD, exist_ok=True)
    rows = list(csv.DictReader(open(SRC, encoding='utf-8-sig')))
    log = json.load(open(LOG, encoding='utf-8')) if os.path.exists(LOG) else {}
    cache = {}
    todo = [r for r in rows if force or not os.path.exists(os.path.join(CHIPD, r['point_id'] + '.png'))]
    emit('待抓取 %d / %d 点（workers=%d）' % (len(todo), len(rows), workers))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(lambda r: one_point(r, force, cache), todo):
            done += 1
            log[res['tile']] = res
            if 'err' in res:
                emit('  FAIL %s %s' % (res['tile'], res['err']))
            if done % 20 == 0:
                emit('  进度 %d/%d' % (done, len(todo)))
                json.dump(log, open(LOG, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    json.dump(log, open(LOG, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    n_ok = sum(1 for r in rows if os.path.exists(os.path.join(CHIPD, r['point_id'] + '.png')))
    miss = [r['point_id'] for r in rows if not os.path.exists(os.path.join(CHIPD, r['point_id'] + '.png'))]
    emit('完成：%d/%d 切片就绪%s' % (n_ok, len(rows), '' if not miss else '；缺：' + ','.join(miss[:10])))
    srcs = {}
    for k, v in log.items():
        srcs[v.get('src', '?')] = srcs.get(v.get('src', '?'), 0) + 1
    emit('瓦片源分布：%s' % srcs)


def cmd_sheet(cols=2, rows_per=2):
    os.makedirs(SHEETD, exist_ok=True)
    pts = sorted(f[:-4] for f in os.listdir(CHIPD) if f.endswith('.png'))
    per = cols * rows_per
    n = 0
    for k in range(0, len(pts), per):
        grp = pts[k:k + per]
        W = cols * (CROP + 6) + 6
        H = rows_per * (CROP + 6) + 6
        sh = Image.new('RGB', (W, H), (40, 40, 40))
        for i, pid in enumerate(grp):
            im = Image.open(os.path.join(CHIPD, pid + '.png'))
            r, c = divmod(i, cols)
            sh.paste(im, (6 + c * (CROP + 6), 6 + r * (CROP + 6)))
        fp = os.path.join(SHEETD, 'sheet_%02d.png' % (k // per))
        sh.save(fp)
        n += 1
    emit('联系表 %d 张（每张 %d 点，格 %d×%d）→ %s' % (n, per, cols, rows_per, SHEETD))


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a or a[0] == 'fetch':
        w = int(a[a.index('--workers') + 1]) if '--workers' in a else 8
        cmd_fetch(force='--force' in a, workers=w)
    elif a[0] == 'sheet':
        cmd_sheet()
    else:
        raise SystemExit(__doc__)

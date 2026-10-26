# -*- coding: utf-8 -*-
"""m6_mangrove_validate.py — 红树林扩充 验证：① 双路证据复查 ② z18 影像切片与联系表

* ① --what spot：随机抽 keep 点，用**独立最小构建**重取关键波段（不复用 m4 代码），与 m4 产物比对（max|Δ| 期望 0）
* ② --what chips：对 keep 抽样 N 点 + 原 652 抽样 M 点（对照）拉 z18（Esri 主 / Google 备）512² 切片，
   画十字丝 + 10 m 青框 + 编号；拼联系表（10 列），供人工/AI 判读。
* 用法：python m6_mangrove_validate.py --what spot --n 40 --acct <acct>
        python m6_mangrove_validate.py --what chips --n 150 --ctrl 50
"""
import argparse, glob, io, json, math, os, sys, time, urllib.request
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
WORK = r'F:/lc_work/v31_exp/data/mangrove'
DEST = r'F:/地理所/论文/中国土地覆盖数据_2017-2024/数据/单类样本的单独处理_2026.10.12/红树林'
VD = os.path.join(WORK, 'validate')
Z, CROP, GAP = 18, 512, 24
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/%d/%d/%d'
GOOG = 'http://mt0.google.com/vt/lyrs=s&hl=en&x=%d&y=%d&z=%d'


def lonlat_to_px(lon, lat, z):
    n = 256.0 * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def load(stage):
    fs = sorted(glob.glob(os.path.join(WORK, 'out', 'feats', stage, '*.parquet')))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d['cid'] = d.cid.astype(str)
    return d.drop_duplicates('cid')


def do_spot(n, acct):
    import shapefile
    keep = pd.read_parquet(os.path.join(WORK, 'out', 'qc', 'keep_v1.parquet'))
    s = keep.sample(min(n, len(keep)), random_state=11).reset_index(drop=True)
    import ee, v31_common as VC
    VC.ensure_ctx(acct)
    wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    giri = ee.ImageCollection('LANDSAT/MANGROVE_FORESTS').first().select('1')
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    dem = ee.Image('USGS/SRTMGL1_003').select('elevation').rename('dem')
    dw = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2024-01-01')
          .select(['water', 'trees']).mean())
    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    def m(im):
        s_ = im.select('SCL')
        return im.updateMask(s_.neq(3).And(s_.neq(8)).And(s_.neq(9)))
    c = col.filterDate('2023-06-01', '2023-09-01').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60)).map(m).median()
    ndvi = c.normalizedDifference(['B8', 'B4']).multiply(1000).rename('s2_ndvi_xia')
    img = ee.Image.cat([wc.eq(95).unmask(0).rename('wc95'), giri.eq(1).unmask(0).rename('giri'),
                        gsw.select('max_extent').unmask(-1), dw.select('trees').unmask(-1).rename('trees'),
                        dem.unmask(-9999), ndvi.unmask(-9999)]).toFloat()
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]), {'cid': c_})
                               for c_, x, y in zip(s.cid, s.lon, s.lat)])
    r = img.sampleRegions(collection=fc, scale=10, geometries=False, tileScale=4).getInfo()['features']
    got = {f['properties']['cid']: f['properties'] for f in r}
    COLS = ['wc95', 'giri', 'max_extent', 'trees', 'dem', 's2_ndvi_xia']
    b = pd.DataFrame([dict(cid=c_, **{cc: got.get(c_, {}).get(cc) for cc in COLS}) for c_ in s.cid])
    for cc in COLS:
        b[cc] = pd.to_numeric(b[cc], errors='coerce')
    a = load('ev')
    for cc in ('max_extent',):
        b[cc] = b[cc].where(lambda x: x >= 0)
    m2 = s[['cid']].merge(a[['cid'] + COLS], on='cid', suffixes=('', '_m4'))
    dif = {}
    for cc in COLS:
        x, y = b[cc], m2[cc + '_m4'] if cc + '_m4' in m2 else m2[cc]
        both = x.notna() & y.notna()
        dif[cc] = round(float((x[both] - y[both]).abs().max()) if both.any() else -1, 6)
    os.makedirs(VD, exist_ok=True)
    json.dump(dict(n=len(s), diffs=dif, acct=acct), open(os.path.join(VD, 'spot_check.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(dif, ensure_ascii=False))
    print('MISMATCH:', [k for k, v in dif.items() if v > 1e-6] or 'NONE（全 0.0）')


def fetch_tile(z, x, y, src='esri'):
    url = (ESRI % (z, y, x)) if src == 'esri' else (GOOG % (x, y, z))
    req = urllib.request.Request(url, headers=UA)
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open(req, timeout=25) as r:
        return r.read()


def one_chip(tag, lon, lat, outd):
    from PIL import Image, ImageDraw, ImageFont
    fp = os.path.join(outd, 'chips', tag + '.png')
    if os.path.exists(fp):
        return True
    px, py = lonlat_to_px(lon, lat, Z)
    tx, ty = int(px // 256), int(py // 256)
    mos = Image.new('RGB', (768, 768)); cache = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            b = cache.get((tx + i, ty + j))
            if b is None:
                for att, s in ((0, 'esri'), (1, 'google'), (2, 'esri')):
                    try:
                        b = fetch_tile(Z, tx + i, ty + j, s); break
                    except Exception:
                        if att == 2:
                            return False
                        time.sleep(0.6)
                cache[(tx + i, ty + j)] = b
            mos.paste(Image.open(io.BytesIO(b)).convert('RGB'), ((i + 1) * 256, (j + 1) * 256))
    ox = int(round(px - (tx - 1) * 256)) - CROP // 2
    oy = int(round(py - (ty - 1) * 256)) - CROP // 2
    ox = max(0, min(768 - CROP, ox)); oy = max(0, min(768 - CROP, oy))
    im = mos.crop((ox, oy, ox + CROP, oy + CROP))
    d = ImageDraw.Draw(im); c = CROP // 2
    for x0, y0, x1, y1 in ((0, c, c - GAP, c), (c + GAP, c, CROP, c), (c, 0, c, c - GAP), (c, c + GAP, c, CROP)):
        d.line((x0, y0, x1, y1), fill=(255, 220, 0), width=2)
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** Z)
    half = max(6, int(round(10.0 / mpp / 2)))
    d.rectangle((c - half, c - half, c + half, c + half), outline=(0, 255, 255), width=2)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 22)
    except Exception:
        font = ImageFont.load_default()
    tb = d.textbbox((8, 6), tag, font=font)
    d.rectangle((tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3), fill=(0, 0, 0))
    d.text((8, 6), tag, font=font, fill=(255, 255, 0))
    im.save(fp)
    return True


def do_chips(n, ctrl):
    import shapefile
    from PIL import Image, ImageDraw, ImageFont
    keep = pd.read_parquet(os.path.join(WORK, 'out', 'qc', 'keep_v1.parquet'))
    ks = keep.sample(min(n, len(keep)), random_state=11).reset_index(drop=True)
    r = shapefile.Reader(os.path.join(DEST, '2023', '红树林_2023_WGS84.shp'), encoding='utf-8')
    o = pd.DataFrame([{'tag': 'O%03d' % i, 'lon': p.points[0][0], 'lat': p.points[0][1]}
                      for i, p in enumerate(r.iterShapes())]).sample(min(ctrl, 652), random_state=7).reset_index(drop=True)
    r.close()
    ks = ks.reset_index(drop=True)
    ks['tag'] = ['N%03d' % i for i in range(len(ks))]
    outd = VD
    os.makedirs(os.path.join(outd, 'chips'), exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(t_, x, y) for t_, x, y in zip(ks.tag, ks.lon, ks.lat)] +            [(t_, x, y) for t_, x, y in zip(o.tag, o.lon, o.lat)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda a: one_chip(a[0], a[1], a[2], outd), jobs))
    okN = sum(res[:len(ks)]); okO = sum(res[len(ks):])
    # 联系表（10 列，缩到 224²）
    def sheet(df, name):
        from PIL import Image as I
        cols = 10
        rows = (len(df) + cols - 1) // cols
        TH = 224
        sh = I.new('RGB', (cols * TH, rows * (TH + 18)), (20, 20, 20))
        dr = ImageDraw.Draw(sh)
        try:
            f = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 14)
        except Exception:
            f = ImageFont.load_default()
        for k, (t, x, y) in enumerate(zip(df.tag, df.lon, df.lat)):
            fp = os.path.join(outd, 'chips', t + '.png')
            if not os.path.exists(fp):
                continue
            im = I.open(fp).resize((TH, TH))
            cx, cy = (k % cols) * TH, (k // cols) * (TH + 18)
            sh.paste(im, (cx, cy))
            dr.text((cx + 4, cy + TH + 2), '%s %.4f,%.4f' % (t, x, y), font=f, fill=(255, 255, 0))
        sh.save(os.path.join(outd, name))
        return os.path.join(outd, name)
    p1 = sheet(ks, 'contact_new.png')
    p2 = sheet(o, 'contact_orig.png')
    json.dump(dict(n_new=int(okN), n_orig=int(okO), sheet_new=p1, sheet_orig=p2),
              open(os.path.join(VD, 'chips_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('联系表：%s ｜ %s' % (p1, p2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', required=True, choices=['spot', 'chips'])
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--ctrl', type=int, default=50)
    ap.add_argument('--acct', default='vidalmvtpronet18')
    a = ap.parse_args()
    if a.what == 'spot':
        do_spot(a.n, a.acct)
    else:
        do_chips(a.n, a.ctrl)


if __name__ == '__main__':
    main()

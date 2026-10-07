# -*- coding: utf-8 -*-
"""d_tiles.py — 把瓦片试产成果下载到本地并生成可直接查看的图
产物（F:/lc_work/v31_exp/results/tiles/）：
  <tile>_60m.tif    60m 概览 GeoTIFF（QGIS 可直接打开，带地理坐标）
  <tile>_预览.png   24 类彩色预览图（含图例，无需 GIS 软件）
  <tile>_面积.csv   逐类像元数与面积（km²）
用法：python d_tiles.py [--tile qinling] [--scale 60] [--full10m]
"""
import os, sys, io, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import requests
import prod_conf as PC

OUTD = os.path.join(VC.RES, 'tiles')
TILES = {
    'qinling':   dict(box=[107.0, 33.0, 109.0, 35.0], note='森林（秦岭）'),
    'qinling_e': dict(box=[109.0, 33.0, 111.0, 35.0], note='森林（秦岭东）'),
    'songnen':   dict(box=[123.0, 45.0, 125.0, 47.0], note='农林（松嫩）'),
    'sanjiang':  dict(box=[133.0, 47.0, 135.0, 49.0], note='湿地（三江）'),
}
# 24 类配色（陆表常用近似色，仅用于预览）
COLORS = {1: (0.93, 0.87, 0.55), 2: (0.55, 0.80, 0.35), 3: (0.35, 0.65, 0.95),
          4: (0.05, 0.35, 0.10), 5: (0.30, 0.62, 0.25), 6: (0.10, 0.45, 0.35),
          7: (0.45, 0.72, 0.55), 8: (0.55, 0.62, 0.20), 9: (0.75, 0.72, 0.30),
          10: (0.85, 0.78, 0.40), 11: (0.90, 0.92, 0.65), 12: (0.80, 0.85, 0.60),
          13: (0.95, 0.90, 0.75), 14: (0.30, 0.55, 0.55), 15: (0.45, 0.70, 0.70),
          16: (0.85, 0.80, 0.65), 17: (0.70, 0.60, 0.75), 18: (0.90, 0.20, 0.35),
          19: (0.70, 0.45, 0.65), 20: (0.80, 0.75, 0.55), 21: (0.85, 0.20, 0.20),
          22: (0.75, 0.70, 0.65), 23: (0.15, 0.35, 0.85), 24: (0.95, 0.98, 1.00)}


def fetch(url, fp):
    r = requests.get(url, proxies=PC.PROXY, timeout=900, stream=True)
    r.raise_for_status()
    with open(fp, 'wb') as f:
        for ch in r.iter_content(1 << 20):
            f.write(ch)
    return os.path.getsize(fp)


def get_image(acct, t):
    ee, _ = VC.ctx(acct)
    reg = VC.jload(os.path.join(VC.DATA, 'asset_registry.json'))
    aid = reg.get('tile_raster_%s' % t)
    if not isinstance(aid, dict):
        return None, None
    return ee, ee.Image(aid['asset'])


def preview(t, fp_tif, box):
    """读 GeoTIFF → PNG 预览 + 面积统计。"""
    try:
        import rasterio
    except ImportError:
        VC.emit('缺 rasterio，跳过预览'); return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    with rasterio.open(fp_tif) as ds:
        arr = ds.read(1)
        res_x = abs(ds.transform.a); res_y = abs(ds.transform.e)
    h, w = arr.shape
    rgb = np.zeros((h, w, 3), dtype='float32')
    for k, c in COLORS.items():
        rgb[arr == k] = c
    rgb[arr == 0] = (0.9, 0.9, 0.9)
    names = VC.V31_NAMES()
    # 面积（按像元面积算，10m 分辨率下 CSV 用 100 m²；预览图用实际采样分辨率）
    px = {int(k): int(v) for k, v in zip(*np.unique(arr, return_counts=True))}
    lat_mid = (box[1] + box[3]) / 2
    m_per_deg_y = 111320.0
    m_per_deg_x = 111320.0 * np.cos(np.deg2rad(lat_mid))
    cell_m2 = (res_x * m_per_deg_x) * (res_y * m_per_deg_y)
    rows = []
    for k, v in sorted(px.items()):
        if k == 0:
            continue
        rows.append(dict(code=k, name=names.get(str(k), '?'), pixels=v,
                         area_km2=round(v * cell_m2 / 1e6, 2)))
    import csv
    with open(os.path.join(OUTD, '%s_面积.csv' % t), 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=['code', 'name', 'pixels', 'area_km2'])
        wr.writeheader(); wr.writerows(rows)
    # 出图
    fig, ax = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw={'width_ratios': [3, 1]})
    ax[0].imshow(rgb, interpolation='nearest')
    ax[0].set_title('%s · %s · 24 类分类（%d m 概览）' % (t, TILES[t]['note'], int(res_x * m_per_deg_x)), fontsize=13)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].axis('off')
    txt = ['%s' % TILES[t]['note'], '', '逐类面积 top12（km²）：']
    for r in sorted(rows, key=lambda x: -x['area_km2'])[:12]:
        txt.append('%2d %s  %.0f' % (r['code'], r['name'], r['area_km2']))
    txt += ['', '总覆盖 %.0f km²' % sum(r['area_km2'] for r in rows)]
    ax[1].text(0, 1, '\n'.join(txt), va='top', fontsize=11, family='Microsoft YaHei')
    patches = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in sorted(COLORS)]
    labels = ['%d %s' % (k, names.get(str(k), '?')) for k in sorted(COLORS)]
    fig.legend(patches, labels, loc='lower center', ncol=8, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    plt.subplots_adjust(bottom=0.12)
    fp_png = os.path.join(OUTD, '%s_预览.png' % t)
    plt.tight_layout(); plt.savefig(fp_png, dpi=130); plt.close()
    VC.emit('  预览 %s（%d 类，总覆盖 %.0f km²）' % (fp_png, len(rows), sum(r['area_km2'] for r in rows)))


def main():
    os.makedirs(OUTD, exist_ok=True)
    only = sys.argv[sys.argv.index('--tile') + 1] if '--tile' in sys.argv else None
    scale = int(sys.argv[sys.argv.index('--scale') + 1]) if '--scale' in sys.argv else 60
    reg = VC.jload(os.path.join(VC.DATA, 'asset_registry.json'))
    for t, meta in TILES.items():
        if only and t != only:
            continue
        info = reg.get('tile_raster_%s' % t)
        if not isinstance(info, dict):
            VC.emit('%s 未提交' % t); continue
        fp_tif = os.path.join(OUTD, '%s_%dm.tif' % (t, scale))
        if os.path.exists(fp_tif) and os.path.getsize(fp_tif) > 1000:
            VC.emit('%s 已下载，跳过' % t)
        else:
            try:
                ee, img = get_image(info['host'], t)
                if img is None:
                    VC.emit('%s 无资产' % t); continue
                tif = img.rename('class').toByte()
                url = tif.getDownloadURL(dict(scale=scale, region=meta['box'], crs='EPSG:4326',
                                              format='GEO_TIFF', filePerBand=False))
                sz = fetch(url, fp_tif)
                VC.emit('%s 下载 %s（%.1f MB）' % (t, fp_tif, sz / 1e6))
            except Exception as e:
                VC.emit('%s 下载失败（可能仍在导出）：%s' % (t, str(e)[:90]))
                continue
        try:
            preview(t, fp_tif, meta['box'])
        except Exception as e:
            VC.emit('%s 预览失败：%s' % (t, str(e)[:80]))
    VC.emit('目录：%s' % OUTD)


if __name__ == '__main__':
    main()

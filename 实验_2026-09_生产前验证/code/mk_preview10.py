# -*- coding: utf-8 -*-
"""mk_preview10.py — 用 10m 成品重出高清预览（含图例）+ 面积表"""
import os, sys, csv
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, 'code')
from d_tiles import COLORS

OUTD = os.path.join(VC.RES, 'tiles')
TILES = {'qinling': '森林（秦岭西）', 'qinling_e': '森林（秦岭东）',
         'songnen': '农林（松嫩）', 'sanjiang': '湿地（三江）'}


def main():
    names = VC.V31_NAMES()
    for t, note in TILES.items():
        fp = os.path.join(OUTD, '%s_10m.tif' % t)
        if not os.path.exists(fp):
            VC.emit('%s 缺 10m 文件' % t); continue
        with rasterio.open(fp) as ds:
            step = max(1, int(ds.width / 3000))
            arr = ds.read(1, out_shape=(ds.height // step, ds.width // step))
            res = ds.res
            # 全分辨率按块累计（内存安全）+ 纬度余弦加权算真实面积
            from collections import Counter
            cnt = Counter()
            step_rows = 2048
            for r0 in range(0, ds.height, step_rows):
                hh = min(step_rows, ds.height - r0)
                blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
                lat0 = ds.xy(r0, 0, offset='ul')[1]
                lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
                latm = (lat0 + lat1) / 2
                cell_km2 = (res[0] * 111.32 * np.cos(np.deg2rad(latm))) * (abs(res[1]) * 110.57)
                v, c = np.unique(blk[(blk > 0) & (blk < 200)], return_counts=True)
                for a, b in zip(v, c):
                    cnt[int(a)] += int(b) * cell_km2
        h, w = arr.shape
        rgb = np.full((h, w, 3), 0.92, dtype='float32')
        for k, c in COLORS.items():
            rgb[arr == k] = c
        rows = [dict(code=k2, name=names.get(str(k2), '?'), pixels=int(v2 / 100.0),
                     area_km2=round(v2, 2)) for k2, v2 in sorted(cnt.items())]
        with open(os.path.join(OUTD, '%s_面积.csv' % t), 'w', newline='', encoding='utf-8-sig') as f:
            wr = csv.DictWriter(f, fieldnames=['code', 'name', 'pixels', 'area_km2'])
            wr.writeheader(); wr.writerows(rows)
        fig, ax = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [3, 1]})
        ax[0].imshow(rgb, interpolation='nearest')
        ax[0].set_title('%s · %s · 24 类地表覆盖（2023，10m）' % (t, note), fontsize=14)
        ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[1].axis('off')
        txt = [note, '', '逐类面积 top12（km²）：']
        for r in sorted(rows, key=lambda x: -x['area_km2'])[:12]:
            txt.append('%2d %-6s %8.0f' % (r['code'], r['name'], r['area_km2']))
        txt += ['', '总覆盖 %.0f km²' % sum(r['area_km2'] for r in rows), '', '类数 %d / 24' % len(rows)]
        ax[1].text(0, 1, '\n'.join(txt), va='top', fontsize=11)
        patches = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k2]) for k2 in sorted(COLORS)]
        labels = ['%d %s' % (k2, names.get(str(k2), '?')) for k2 in sorted(COLORS)]
        fig.legend(patches, labels, loc='lower center', ncol=8, fontsize=8, frameon=False)
        plt.subplots_adjust(bottom=0.11)
        png = os.path.join(OUTD, '%s_预览10m.png' % t)
        plt.tight_layout(); plt.savefig(png, dpi=120); plt.close()
        VC.emit('%s 预览 → %s（%d 类，总覆盖 %.0f km²）' % (t, os.path.basename(png), len(rows),
                                                          sum(r['area_km2'] for r in rows)))


if __name__ == '__main__':
    main()

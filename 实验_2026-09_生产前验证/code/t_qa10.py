# -*- coding: utf-8 -*-
"""t_qa10.py — 用本地 10m 成品做权威 QA（覆盖率/类构成/面积），并生成最终汇总报告
背景：GEE 侧 coarse-scale 重采样会插值类码（凭空产生类），故所有统计一律以本地 10m 文件为准。
"""
import os, sys, csv, collections
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import rasterio
import t_all as T

OUTD = os.path.join(VC.RES, 'tiles')
NOTES = {'qinling': '森林（秦岭西）', 'qinling_e': '森林（秦岭东，接缝对）',
         'songnen': '农林（松嫩）', 'sanjiang': '湿地（三江）'}


def stats(fp, box):
    with rasterio.open(fp) as ds:
        res = ds.res
        cnt = collections.Counter()
        cell_sum = 0.0
        n = 0
        for r0 in range(0, ds.height, 2048):
            hh = min(2048, ds.height - r0)
            blk = ds.read(1, window=rasterio.windows.Window(0, r0, ds.width, hh))
            lat0 = ds.xy(r0, 0, offset='ul')[1]
            lat1 = ds.xy(r0 + hh - 1, 0, offset='ul')[1]
            cell = (res[0] * 111.32 * np.cos(np.deg2rad((lat0 + lat1) / 2))) * (abs(res[1]) * 110.57)
            v, c = np.unique(blk, return_counts=True)
            for a, b in zip(v.tolist(), c.tolist()):
                cnt[a] += b * cell
            n += hh * ds.width
        tot_px = n
        zero_px = sum(c for a, c in cnt.items() if a == 0)
    return cnt, tot_px, zero_px


def main():
    names = VC.V31_NAMES()
    rows = []
    L = ['# 代表瓦片试产 QA（2023，10m 成品·本地统计）%s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '> 统计口径：**一律以本地 10m 成品文件为准**（GEE 侧 coarse-scale 重采样会插值类码，见 §注）', '',
         '| 瓦片 | 说明 | EECU·h | 覆盖率 | 类数 | Top5 类（面积占比） |', '|---|---|---|---|---|---|']
    for t, note in NOTES.items():
        fp = os.path.join(OUTD, '%s_10m.tif' % t)
        if not os.path.exists(fp):
            L.append('| %s | %s | | 缺文件 | | |' % (t, note)); continue
        cnt, tot_px, zero_px = stats(fp, T.TILES[t]['box'])
        areas = {k: v for k, v in cnt.items() if k > 0}
        a_sum = sum(areas.values())
        top = sorted(areas.items(), key=lambda x: -x[1])[:5]
        w = T.winner_of(t)
        eecu = w['eecu']
        L.append('| %s | %s | %.2f | %.2f%% | %d | %s |' % (
            t, note, eecu, 100.0 * (1 - zero_px / tot_px), len(areas),
            ' '.join('%s %.1f%%' % (names.get(str(k), k), 100.0 * v / a_sum) for k, v in top)))
        with open(os.path.join(OUTD, '%s_面积.csv' % t), 'w', newline='', encoding='utf-8-sig') as f:
            wr = csv.writer(f); wr.writerow(['code', 'name', 'area_km2', 'share_pct'])
            for k, v in sorted(areas.items(), key=lambda x: -x[1]):
                wr.writerow([k, names.get(str(k), '?'), round(v, 3), round(100.0 * v / a_sum, 3)])
        rows.append((t, note, eecu, len(areas), a_sum))
        VC.emit('%s: EECU %.2f h ｜ 覆盖率 %.2f%% ｜ %d 类 ｜ 总面积 %.0f km²' % (
            t, eecu, 100.0 * (1 - zero_px / tot_px), len(areas), a_sum))
    L += ['', '## 注：本次发现的数据完整性缺陷（重要）', '',
          '- 早先用 `getDownloadURL(scale=60)` 下载的概览、以及 GEE 侧 `reduceRegion(scale=30)` 的类直方图，',
          '  **对类别栅格做了非最近邻重采样**：类码被插值，凭空产生原图不存在的类（如 60m 版多出 7/8/9/14/16 类）。',
          '- 仲裁：随机 150 点 GEE 逐点直查 → 10m 文件一致率 **96.7%**，60m 文件仅 **76.0%** → 10m（原生网格）为准。',
          '- 纪律：**类别栅格的一切派生/概览/统计必须显式 nearest（或读原生网格本地统计）**；',
          '  本报告与预览图已全部改为本地 10m 统计。']
    fp = os.path.join(VC.REPT, '瓦片试产QA_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()

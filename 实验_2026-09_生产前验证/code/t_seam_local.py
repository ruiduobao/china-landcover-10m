# -*- coding: utf-8 -*-
"""t_seam_local.py — 用本地 10m 成品做跨瓦片接缝比较 + 重出高清预览
接缝线 109.0°E（秦岭西块 vs 东块），对照线 108.4°E / 109.6°E（各瓦片内部）
"""
import os, sys, csv
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import rasterio
from rasterio.windows import from_bounds

OUTD = os.path.join(VC.RES, 'tiles')
TILES = {
    'qinling':   dict(box=[107.0, 33.0, 109.0, 35.0], note='森林（秦岭西）'),
    'qinling_e': dict(box=[109.0, 33.0, 111.0, 35.0], note='森林（秦岭东）'),
    'songnen':   dict(box=[123.0, 45.0, 125.0, 47.0], note='农林（松嫩）'),
    'sanjiang':  dict(box=[133.0, 47.0, 135.0, 49.0], note='湿地（三江）'),
}


def comp(fp, box):
    """返回 {类: 占比}（0 值不计）。"""
    with rasterio.open(fp) as ds:
        win = from_bounds(*box, transform=ds.transform).round_offsets().round_lengths()
        a = ds.read(1, window=win).ravel()
    a = a[(a > 0) & (a < 200)]
    if a.size == 0:
        return {}
    k, c = np.unique(a, return_counts=True)
    return {int(x): float(y) / a.size for x, y in zip(k, c)}


def tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def main():
    fA = os.path.join(OUTD, 'qinling_10m.tif')
    fB = os.path.join(OUTD, 'qinling_e_10m.tif')
    y0, y1 = 33.2, 34.8      # 避开南北边带，只看中部
    D = 0.1
    cw = comp(fA, [109.0 - D, y0, 109.0, y1])
    ce = comp(fB, [109.0, y0, 109.0 + D, y1])
    seam = tvd(cw, ce)
    inA = tvd(comp(fA, [108.4 - D, y0, 108.4, y1]), comp(fA, [108.4, y0, 108.4 + D, y1]))
    inB = tvd(comp(fB, [109.6 - D, y0, 109.6, y1]), comp(fB, [109.6, y0, 109.6 + D, y1]))
    base = float(np.mean([inA, inB]))
    names = VC.V31_NAMES()
    top = sorted(((k, cw.get(k, 0) - ce.get(k, 0)) for k in set(cw) | set(ce)), key=lambda x: -abs(x[1]))[:8]
    L = ['# 秦岭/秦岭东 接缝比较（109°E，10m 成品，纬度 33.2–34.8）%s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '| 指标 | TVD | 含义 |', '|---|---|---|',
         '| **接缝线 109.0°E**（跨瓦片，两侧各 0.1°） | **%.4f** | 不同局部模型产出的交界跳变 |' % seam,
         '| 对照 108.4°E（秦岭西块内部） | %.4f | 同一模型内部的自然跳变 |' % inA,
         '| 对照 109.6°E（秦岭东块内部） | %.4f | 同上 |' % inB,
         '| 内部对照均值 | %.4f | 基准 |' % base,
         '| **接缝 / 内部** | **%.2f** | ≈1 无接缝伪影；≫1 需核查 |' % (seam / base if base else float('nan')), '',
         '接缝两侧类占比差异 top8（西 − 东，正=西侧更多）：', '']
    for k, d in top:
        L.append('- %02d %s：%+.1fpp（西 %.1f%% / 东 %.1f%%）' % (
            k, names.get(str(k), '?'), d * 100, cw.get(k, 0) * 100, ce.get(k, 0) * 100))
    fp = os.path.join(VC.REPT, '接缝比较_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('接缝 TVD=%.4f | 内部对照=%.4f | 比值=%.2f → %s' % (seam, base, seam / base if base else 0, fp))


if __name__ == '__main__':
    main()

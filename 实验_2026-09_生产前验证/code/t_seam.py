# -*- coding: utf-8 -*-
"""t_seam.py — 跨瓦片接缝比较（109°E，秦岭西/东两块）
设计：接缝处的类构成跳变 vs 瓦片内部的自然跳变（对照线）
  · 接缝线 109.0°E：西侧带(108.9–109.0, 取自秦岭块) vs 东侧带(109.0–109.1, 取自秦岭东块) → TVD_seam
  · 对照线 108.4°E（秦岭块内部）与 109.6°E（秦岭东块内部）→ TVD_intra
判读：TVD_seam 与 TVD_intra 同量级 → 无显著接缝伪影；远大于 → 存在接缝
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np
import t_all as T

D = 0.1


def asset_of(t):
    reg = VC.jload(os.path.join(VC.DATA, 'asset_registry.json'))
    win = T.winner_of(t)
    assert win['state'] == 'COMPLETED', '%s 未完成' % t
    return win['info']['host'], win['info']['asset']


def comp(ee, asset, box):
    h = ee.Image(asset).reduceRegion(reducer=ee.Reducer.frequencyHistogram(),
                                     geometry=ee.Geometry.Rectangle(box), scale=30,
                                     maxPixels=10 ** 10, tileScale=4).getInfo()
    h = list(h.values())[0] if h else {}
    tot = sum(float(v) for k, v in h.items() if float(k) > 0)
    return ({int(float(k)): float(v) / tot for k, v in h.items() if float(k) > 0} if tot else {})


def tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def main():
    import ee
    hA, aA = asset_of('qinling')
    hB, aB = asset_of('qinling_e')
    eeA, _ = VC.ctx(hA)
    y0, y1 = 33.0, 35.0
    # 接缝
    cw = comp(eeA, aA, [109.0 - D, y0, 109.0, y1])
    eeB, _ = VC.ctx(hB)
    ce = comp(eeB, aB, [109.0, y0, 109.0 + D, y1])
    seam = tvd(cw, ce)
    # 对照（瓦片内部）
    inA = tvd(comp(eeA, aA, [108.4 - D, y0, 108.4, y1]), comp(eeA, aA, [108.4, y0, 108.4 + D, y1]))
    inB = tvd(comp(eeB, aB, [109.6 - D, y0, 109.6, y1]), comp(eeB, aB, [109.6, y0, 109.6 + D, y1]))
    base = float(np.mean([inA, inB]))
    ratio = seam / base if base > 0 else float('nan')
    names = VC.V31_NAMES()
    top = sorted(((k, cw.get(k, 0) - ce.get(k, 0)) for k in set(cw) | set(ce)),
                 key=lambda x: -abs(x[1]))[:6]
    L = ['# 秦岭/秦岭东 接缝比较（109°E）%s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '| 指标 | TVD |', '|---|---|',
         '| 接缝线 109.0°E（跨瓦片，西带 vs 东带） | **%.4f** |' % seam,
         '| 对照 108.4°E（秦岭块内部） | %.4f |' % inA,
         '| 对照 109.6°E（秦岭东块内部） | %.4f |' % inB,
         '| 内部对照均值 | %.4f |' % base,
         '| **接缝/内部 比** | **%.2f** |' % ratio, '',
         '判读：比值 ≈1 表示接缝处的类构成跳变与瓦片内部自然跳变相当（无可辨识接缝伪影）；',
         '比值 ≫1（例如 >2）表示接缝处存在额外跳变，需要核查训练邻域/边界的处理。', '',
         '接缝两侧类占比差异 top6（西减东，正=西侧更多）：', '']
    for k, d in top:
        L.append('- %02d %s：%+.1fpp（西 %.1f%% / 东 %.1f%%）' % (
            k, names.get(str(k), '?'), d * 100, cw.get(k, 0) * 100, ce.get(k, 0) * 100))
    fp = os.path.join(VC.REPT, '接缝比较_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('接缝 TVD=%.4f ｜ 内部对照均值=%.4f ｜ 比值=%.2f → %s' % (
        seam, base, ratio, fp))


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""t_edge.py — 瓦片边缘效应检查（接缝风险的先行指标，不依赖相邻瓦片）
做法：对已完成瓦片，比较四条边缘带（外侧 0.1°）与其向内的参照带（0.1–0.2°）的类构成差异
  · 差异小 → 该边界处"局部模型"未产生明显edge效应，接缝风险低
  · 差异大 → 该侧边界不稳定，接缝需重点核查
指标：逐带类构成向量（按类面积占比）之间的**总变差距离 TVD**（0=完全相同，1=完全不相交）
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v31_common as VC
import numpy as np

TILES = {
    'qinling':   dict(box=[107.0, 33.0, 109.0, 35.0], note='森林（秦岭）'),
    'songnen':   dict(box=[123.0, 45.0, 125.0, 47.0], note='农林（松嫩）'),
    'sanjiang':  dict(box=[133.0, 47.0, 135.0, 49.0], note='湿地（三江）'),
}
REG = os.path.join(VC.DATA, 'asset_registry.json')
D = 0.1


def comp(ee, asset, box):
    img = ee.Image(asset)
    h = img.reduceRegion(reducer=ee.Reducer.frequencyHistogram(),
                         geometry=ee.Geometry.Rectangle(box), scale=30,
                         maxPixels=10 ** 10, tileScale=4).getInfo()
    h = list(h.values())[0] if h else {}
    tot = sum(float(v) for k, v in h.items() if float(k) > 0)
    if tot <= 0:
        return {}
    return {int(float(k)): float(v) / tot for k, v in h.items() if float(k) > 0}


def tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def main():
    L = ['# 瓦片边缘效应检查 %s' % VC.time.strftime('%Y-%m-%d %H:%M'), '',
         '> 边缘带 = 距边界 0.1° 内；参照带 = 向内 0.1–0.2°。TVD 为两条带类构成的总变差距离（0 最好）', '',
         '| 瓦片 | 西 | 东 | 南 | 北 | 平均 | 判读 |', '|---|---|---|---|---|---|---|']
    for t, meta in TILES.items():
        x0, y0, x1, y1 = meta['box']
        reg = VC.jload(REG)
        info = None
        for k in ('tile_raster_%s' % t, 'tile_race_%s' % t):
            v = reg.get(k)
            if isinstance(v, dict):
                try:
                    e, _ = VC.ctx(v['host'])
                    if e.data.getTaskStatus(v['task'])[0]['state'] == 'COMPLETED':
                        info = v; break
                except Exception:
                    pass
        if not info:
            L.append('| %s | | | | | | 未完成 |' % t); continue
        ee, _ = VC.ctx(info['host'])
        asset = info['asset']
        bands = {
            '西': ([x0, y0, x0 + D, y1], [x0 + D, y0, x0 + 2 * D, y1]),
            '东': ([x1 - D, y0, x1, y1], [x1 - 2 * D, y0, x1 - D, y1]),
            '南': ([x0, y0, x1, y0 + D], [x0, y0 + D, x1, y0 + 2 * D]),
            '北': ([x0, y1 - D, x1, y1], [x0, y1 - 2 * D, x1, y1 - D]),
        }
        vals = {}
        for k, (eb, rb) in bands.items():
            try:
                ce = comp(ee, asset, eb); cr = comp(ee, asset, rb)
                vals[k] = round(tvd(ce, cr), 4)
            except Exception as ex:
                vals[k] = None
        good = [v for v in vals.values() if v is not None]
        avg = round(float(np.mean(good)), 4) if good else None
        judge = '✅ 低（<0.10）' if avg is not None and avg < 0.10 else (
            '⚠️ 偏高（0.10–0.20）' if avg is not None and avg < 0.20 else '❌ 高（≥0.20）')
        L.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            t, *['%.3f' % vals[k] if vals.get(k) is not None else '—' for k in ['西', '东', '南', '北']],
            '%.3f' % avg if avg is not None else '—', judge))
        VC.emit('%s 边缘 TVD：西 %s 东 %s 南 %s 北 %s → 平均 %s %s' % (
            t, vals.get('西'), vals.get('东'), vals.get('南'), vals.get('北'), avg, judge))
    fp = os.path.join(VC.REPT, '瓦片边缘效应_%s.md' % VC.time.strftime('%Y%m%d'))
    open(fp, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    VC.emit('→ %s' % fp)


if __name__ == '__main__':
    main()

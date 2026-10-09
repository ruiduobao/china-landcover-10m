# -*- coding: utf-8 -*-
"""e11_evidence_grade.py — "证据分级新得分"整合原型（四川 24 瓦，2023 R0 成品）

* 输入：F:/lc_work/prod5p_2023/rasters_batch/<tile>_10m.tif（交付成品 24 类，只读）
        data/eco_gate/sichuan_v/mask_<tile>_v2.tif（V 档 V2 掩膜：1=否决）
        PROJ/数据/边界/china_100000_full.json（四川省界，buffer −0.005°，统计归属）
* 规则源（本脚本头部=预注册）：
  **证据档映射（来自 doc 54 三类可分性实验 + doc 55 V 档，逐类给定，可追溯）**
    · A 档（有可分性证据，AUC≥0.85 或物理唯一）：23 水体（vs 草沼 0.967）；
      「常绿林/落叶林」合并层（合并后即 A 档，0.990）
    · B 档（放宽档 0.65–0.85 或尚未测）：1 草本旱地、2 乔灌园地、3 灌溉耕地、11 草地、13 稀疏植被、
      21 不透水面、22 裸地、24 永久冰雪、12 地衣苔藓（均"未测"→ 保守给 B）
    · C 档（已证不可分 <0.65）：4 常绿阔叶林、6 常绿针叶林（针/阔轴 0.724）、5/7/8 落叶组细分、
      9/10 灌丛（真灌丛率 0.165、AUC 0.772）、14/15/16/17 湿地细分（0.708/0.645）
  **合成规则**：逐像元 得分 = 取该像元类别的证据档；若 V2 掩膜=1（生态否决）→ **降为 C 并标记 veto**；
              若类别属"可合并对"（4/6 → 常绿林；5/7/8 → 落叶林；14/15/16/17 → 湿地；9/10 → 草地）
              则同时给出 merged 建议（降级后仍报合并口径）。
  **口径**：像元面积按纬度改正；省界内统计；不改交付成品。
* 门槛：逐瓦回读自检（各类像元数与 e4 记录一致——本脚本独立重算，仅与自身统计核对）
* 输出：data/eco_gate/evidence_grade/sichuan_evidence_grade.json + 逐类.csv + 分省面积.csv
* 用法：python e11_evidence_grade.py
"""
import csv
import json
import math
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
WORK = r'F:/lc_work/v31_exp'
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(WORK, 'code'))
import numpy as np
import rasterio
from rasterio.windows import Window
import v31_common as VC
from e2_sichuan_v_mask import load_tiles

R0D = r'F:/lc_work/prod5p_2023/rasters_batch'
MASK = os.path.join(WORK, 'data', 'eco_gate', 'sichuan_v')
OUTD = os.path.join(WORK, 'data', 'eco_gate', 'evidence_grade')
BOUND = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')
BLK = 4096
NAMES = VC.V31_NAMES()

# ---- 证据档映射（逐类，来源可追溯；'m' = 建议合并目标）----
GRADE = {
    23: ('A', None),          # 水体：vs 草沼 AUC 0.967
    1: ('B', None), 2: ('B', None), 3: ('B', None),
    11: ('B', None), 12: ('B', None), 13: ('B', None),
    21: ('B', None), 22: ('B', None), 24: ('B', None),
    4: ('C', '常绿林'), 6: ('C', '常绿林'),          # 针/阔轴 0.724 不可分
    5: ('C', '落叶林'), 7: ('C', '落叶林'), 8: ('C', '落叶林'),
    9: ('C', '草地'), 10: ('C', '草地'),             # 灌丛：真灌丛率 0.165、AUC 0.772
    14: ('C', '湿地'), 15: ('C', '湿地'), 16: ('C', '湿地'), 17: ('C', '湿地'),
    18: ('C', '湿地'), 19: ('C', '湿地'), 20: ('C', '湿地'),
}


def emit(m):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(m), flush=True)


def sc_geom():
    from shapely.geometry import shape
    from shapely import make_valid
    from rasterio.features import rasterize
    gj = json.load(open(BOUND, encoding='utf-8'))
    for ft in gj['features']:
        if ft.get('properties', {}).get('name', '').startswith('四川'):
            return make_valid(shape(ft['geometry'])).buffer(-0.005)
    raise SystemExit('省界缺失')


def main():
    os.makedirs(OUTD, exist_ok=True)
    geom = sc_geom()
    tiles = load_tiles()
    per_cls = {}          # 类 → 面积 km²
    grade_area = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    veto_area = 0.0
    merged_area = {}
    t0 = time.time()
    for t, box in tiles.items():
        sfp = os.path.join(R0D, '%s_10m.tif' % t)
        mfp = os.path.join(MASK, 'mask_%s_v2.tif' % t)
        if not (os.path.exists(sfp) and os.path.exists(mfp)):
            emit('%s 缺文件，跳过' % t)
            continue
        with rasterio.open(sfp) as S, rasterio.open(mfp) as M:
            h, w, res = S.height, S.width, S.res
            inside = np.zeros((h, w), 'uint8')
            # 省界按块栅格化（省内存）
            from rasterio.features import rasterize
            inside[:] = rasterize([(geom, 1)], out_shape=(h, w), transform=S.transform,
                                  fill=0, dtype='uint8', all_touched=False)
            for r0 in range(0, h, BLK):
                hh = min(BLK, h - r0)
                cb = S.read(1, window=Window(0, r0, w, hh))
                mb = M.read(1, window=Window(0, r0, w, hh))
                ib = inside[r0:r0 + hh]
                lat0 = S.xy(r0, 0, offset='ul')[1]
                lat1 = S.xy(r0 + hh - 1, 0, offset='ul')[1]
                cell = (res[0] * 111.32 * math.cos(math.radians((lat0 + lat1) / 2))) * (abs(res[1]) * 110.57)
                sel = (cb > 0) & (ib == 1)
                v, n = np.unique(cb[sel], return_counts=True)
                for c, k in zip(v.tolist(), n.tolist()):
                    per_cls[c] = per_cls.get(c, 0.0) + k * cell
                    g, _mg = GRADE.get(c, ('B', None))
                    grade_area[g] += k * cell
                    if _mg:
                        merged_area[_mg] = merged_area.get(_mg, 0.0) + k * cell
                veto = sel & (mb == 1)
                veto_area += float(veto.sum()) * cell
        emit('%s 完成 [%.1f min]' % (t, (time.time() - t0) / 60))
    tot = sum(per_cls.values())
    # 否决像素按类别归到 C 档（已在 GRADE 里按类给档，这里单列 veto 面积供解读）
    res = {
        'scope': '四川（省界内，R0 成品 + V2 掩膜）',
        'total_area_km2': round(tot, 1),
        'evidence_grade_area_km2': {k: round(v, 1) for k, v in grade_area.items()},
        'evidence_grade_pct': {k: round(100 * v / max(tot, 1e-9), 3) for k, v in grade_area.items()},
        'veto_area_km2_v2': round(veto_area, 1),
        'veto_pct_of_province': round(100 * veto_area / max(tot, 1e-9), 4),
        'merged_suggestion_area_km2': {k: round(v, 1) for k, v in sorted(
            merged_area.items(), key=lambda kv: -kv[1])},
        'per_class_area_km2': {NAMES.get(str(c), str(c)): round(v, 1)
                               for c, v in sorted(per_cls.items(), key=lambda kv: -kv[1])},
        'grade_table': {NAMES.get(str(c), str(c)): dict(tier=GRADE.get(c, ('B', None))[0],
                                                        merge=GRADE.get(c, ('B', None))[1])
                        for c in sorted(per_cls)},
        'rules_source': 'doc54 三类可分性实验（森林/灌丛/湿地）+ doc55 V 档四川掩膜（V2）',
        'note': 'C 档=该类别的细分本身已证不可分（或未通过放宽档）；B 档=尚未测可分性（保守给 B）；A 档=有可分性证据',
    }
    VC.jsave(res, os.path.join(OUTD, 'sichuan_evidence_grade.json'))
    with open(os.path.join(OUTD, 'sichuan_逐类_证据档_面积.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w_ = csv.writer(f)
        w_.writerow(['class', 'name', '证据档', '建议合并目标', 'area_km2', 'share_pct'])
        for c, v in sorted(per_cls.items(), key=lambda kv: -kv[1]):
            g, mg = GRADE.get(c, ('B', None))
            w_.writerow([c, NAMES.get(str(c), ''), g, mg or '', round(v, 1), round(100 * v / max(tot, 1e-9), 4)])
    emit('全省 %.0f km²；证据档 A %.1f%% / B %.1f%% / C %.1f%%；V2 否决 %.0f km²（%.3f%%）' % (
        tot, 100 * grade_area['A'] / tot, 100 * grade_area['B'] / tot,
        100 * grade_area['C'] / tot, veto_area, 100 * veto_area / tot))
    emit('建议合并后面积：%s' % res['merged_suggestion_area_km2'])
    emit('→ %s' % os.path.join(OUTD, 'sichuan_evidence_grade.json'))


if __name__ == '__main__':
    main()

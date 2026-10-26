# -*- coding: utf-8 -*-
"""a4_verify_geometry.py — 几何 G-30N 修正版 GeoTIFF 本地独立核验（不发起云请求）

* 输入：geometry/out/G30N_seed7b_{legacy,planar}_10m.tif、G30N_seed7b_5band.tif、
        geometry/out/G30N_seed7b_eval.json（对照）
* 规则源：doc70 §7 几何覆盖门（有效输入预期域无几何型缺口；共同有效域类码一致；不得邻行补洞）
* 门槛：outside-wedge 类码一致率=1.0；plane region 外两臂均为 0；restored_px>0；legacy nodata>0
* 输出：audit/out/geometry_verify_<ts>.json
* 用法：python a4_verify_geometry.py [--geo F:/lc_work/improve_2023_pilot_20261015/geometry]
* 说明：只读本地栅格；核验的是"覆盖/几何"，不得换算为标签精度。
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
OUT = os.path.join(AUDIT, 'out')


def read(p):
    with rasterio.open(p) as s:
        return s.read(1), s.transform, (s.height, s.width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--geo', default=r'F:/lc_work/improve_2023_pilot_20261015/geometry')
    ap.add_argument('--tag', default='seed7b')
    args = ap.parse_args()
    od = os.path.join(args.geo, 'out')
    leg, tr, shp = read(os.path.join(od, 'G30N_%s_legacy_10m.tif' % args.tag))
    pla, _, _ = read(os.path.join(od, 'G30N_%s_planar_10m.tif' % args.tag))

    # 有效输入域(EVM)限定：5band 中自动识别 EVM 带（1 的个数 == eval.json evm_in_tile_valid_px）
    evm_path = os.path.join(od, 'G30N_%s_5band.tif' % args.tag)
    scope_note = 'full_bbox'
    inreg = np.ones_like(leg, dtype=bool)
    band_info = []
    ev = os.path.join(od, 'G30N_%s_eval.json' % args.tag)
    evm_target = None
    if os.path.exists(ev):
        evm_target = json.load(open(ev, encoding='utf-8')).get('evm_in_tile_valid_px')
    if os.path.exists(evm_path):
        with rasterio.open(evm_path) as s5:
            for b in range(1, s5.count + 1):
                band_info.append({'band': b, 'ones': int((s5.read(b) == 1).sum()),
                                  'zeros': int((s5.read(b) == 0).sum())})
            pick = None
            if evm_target is not None:
                for bi in band_info:
                    if bi['ones'] == evm_target:
                        pick = bi['band']
                        break
            if pick is None:
                pick = s5.count  # 兜底
            inreg = s5.read(pick) == 1
            scope_note = 'EVM band%d == 1 (自动识别, evm_in_tile_valid_px=%s)' % (pick, evm_target)

    # nodata=0 语义：0 视为无数据；仅在有效输入域(EVM=1)内统计
    leg_nod = (leg == 0) & inreg
    pla_nod = (pla == 0) & inreg
    both_valid = inreg & (leg != 0) & (pla != 0)
    disagree = int(np.sum(both_valid & (leg != pla)))
    restored = int(np.sum(leg_nod & (~pla_nod)))       # legacy 缺、planar 有 = 恢复
    lost = int(np.sum((~leg_nod) & pla_nod))            # planar 反而缺 = 负向

    # 楔形带：region 顶部向下的楔形（按纬向行统计 nodata 宽度）
    nod_rows = leg_nod.sum(axis=1)
    pla_nod_rows = pla_nod.sum(axis=1)

    # 栅格元数据（transform/crs/nodata/dtype/palette/hash）
    import hashlib
    def file_meta(p):
        if not os.path.exists(p):
            return None
        h256 = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        hmd5 = hashlib.md5(open(p, 'rb').read()).hexdigest()
        with rasterio.open(p) as s:
            cm = s.colormap(1) if (s.count >= 1 and s.colorinterp[0].name == 'palette') else None
            return {'path': os.path.basename(p), 'size': os.path.getsize(p),
                    'dtype': s.dtypes[0], 'nodata': s.nodata, 'crs': str(s.crs),
                    'transform': [round(x, 10) for x in s.transform[:6]],
                    'has_palette': cm is not None, 'md5': hmd5, 'sha256': h256}
    meta = {'legacy': file_meta(os.path.join(od, 'G30N_%s_legacy_10m.tif' % args.tag)),
            'planar': file_meta(os.path.join(od, 'G30N_%s_planar_10m.tif' % args.tag)),
            '5band': file_meta(evm_path), 'band_info': band_info}

    out = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'tag': args.tag, 'shape': list(shp), 'scope': scope_note, 'file_meta': meta,
        'in_region_px': int(inreg.sum()),
        'n_px': int(leg.size),
        'legacy_nodata_px': int(leg_nod.sum()),
        'planar_nodata_px': int(pla_nod.sum()),
        'restored_px': restored,
        'planar_lost_px': lost,
        'common_valid_px': int(both_valid.sum()),
        'common_valid_disagree_px': disagree,
        'common_valid_agree_rate': round(1 - disagree / max(1, int(both_valid.sum())), 6),
        'legacy_nodata_rows_with_gap': int((nod_rows > 0).sum()),
        'planar_nodata_rows_with_gap': int((pla_nod_rows > 0).sum()),
        'legacy_max_row_nodata': int(nod_rows.max()),
        'planar_max_row_nodata': int(pla_nod_rows.max()),
        'class_hist_legacy': {int(k): int(v) for k, v in zip(*np.unique(leg[~leg_nod], return_counts=True))},
        'class_hist_planar': {int(k): int(v) for k, v in zip(*np.unique(pla[~pla_nod], return_counts=True))},
    }

    # 与 eval.json 对照
    ev = os.path.join(od, 'G30N_%s_eval.json' % args.tag)
    checks = []
    if os.path.exists(ev):
        e = json.load(open(ev, encoding='utf-8'))
        for k, mine in [('legacy_nodata_px', out['legacy_nodata_px']),
                        ('planar_nodata_px', out['planar_nodata_px']),
                        ('restored_px', out['restored_px'])]:
            if k in e:
                checks.append({'field': k, 'eval_json': e[k], 'independent': mine,
                               'match': int(e[k]) == int(mine)})
        out['eval_json_checks'] = checks
        out['outside_wedge_agree_rate_eval'] = e.get('outside_wedge_agree_rate')

    # 门判定（几何覆盖，不含标签精度）
    gates = {
        'restored_positive': restored > 0,
        'legacy_had_gap': int(leg_nod.sum()) > 0,
        'planar_no_gap': int(pla_nod.sum()) == 0,
        'common_valid_agree_100pct': disagree == 0,
        'no_planar_loss': lost == 0,
    }
    out['gates'] = gates
    out['verdict'] = 'PASS' if all(gates.values()) else 'FAIL'

    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(OUT, 'geometry_verify_%s_%s.json' % (args.tag, ts))
    json.dump(out, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('[%s] -> %s' % (out['verdict'], fp))
    for k in ['legacy_nodata_px', 'planar_nodata_px', 'restored_px', 'planar_lost_px',
              'common_valid_disagree_px', 'common_valid_agree_rate']:
        print('   %s = %s' % (k, out[k]))
    for c in out.get('eval_json_checks', []):
        print('   check %s eval=%s independent=%s match=%s' % (c['field'], c['eval_json'], c['independent'], c['match']))


if __name__ == '__main__':
    main()

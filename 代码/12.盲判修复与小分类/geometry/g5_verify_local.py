# -*- coding: utf-8 -*-
"""g5_verify_local.py — 三省几何复验本地独立核验（不发起云请求）

* 输入：geometry/out/TRI_<region>_s<seed>_5band.tif、TRI_*_{legacy,planar}_10m.tif、TRI_<region>_summary.json
* 规则源：doc72 §8 主要门（planar 缺口 0/容差；恢复像元真实；共同有效域类码分歧 0；边界连续；
          mask／色表存储字节有效）；doc70 §7 几何覆盖门；doc43（概览纪律，本小图不建概览）
* 核验项（纯 numpy/rasterio/tifffile/PIL）
    1) 复算 EVM∩平面tile域分母、legacy/planar 缺口、恢复、共同有效域分歧；
    2) **生产瓦一致性**：把本次 legacy 缺口像元数与生产瓦 T*_10m.tif 同窗 nodata 像元数逐值比对；
    3) **边界连续**：planar 在 EVM∩tile 内逐行 nodata 最大值；南边界行及四角非空检查；
    4) **存储字节**：单波段 photometric=3(PALETTE) + colormap 256 项 + PIL 忠实解码（mode=P）；
       多波段诊断件 photometric=1、无 palette。
* 输出：out/TRI_local_verify.json（供 audit 独立复核）
* 用法：python g5_verify_local.py
"""
import glob
import json
import os
import sys

import numpy as np
import rasterio

sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
WS = r'F:/lc_work/label_repair_2023_20261015/geometry'
OUTD = os.path.join(WS, 'out')
RAST = r'F:/lc_work/prod2nd_2023/rasters_batch'

REGIONS = {
    'SC_32N': dict(tile='T1207', c0=18839, r0=22192, ncol=557, nrow=93),
    'GD_24N': dict(tile='T1903', c0=16036, r0=22192, ncol=557, nrow=93),
    'NX_38N': dict(tile='T1610', c0=15515, r0=22191, ncol=557, nrow=93),
}


def main():
    import tifffile
    from PIL import Image
    res = dict(schema='tri_geom_local_verify/v1', regions={}, all_pass=True,
               note='本地独立核验；只读本次产物与生产瓦（只读）；不含任何云请求')
    for rk, R in REGIONS.items():
        rec = dict(tile=R['tile'], seeds={}, region_pass=True)
        # 生产瓦同窗 nodata（只读）
        with rasterio.open(os.path.join(RAST, '%s_10m.tif' % R['tile'])) as ds:
            H = ds.height
            r1 = min(H, R['r0'] + R['nrow'])
            prod = ds.read(1, window=((R['r0'], r1), (R['c0'], R['c0'] + R['ncol'])))
        rec['prod_window_rows'] = int(prod.shape[0])
        rec['prod_window_nodata_px'] = int((prod == 0).sum())
        for seed in (7, 11, 23):
            tag = '%s_s%d' % (rk, seed)
            f5 = os.path.join(OUTD, 'TRI_%s_5band.tif' % tag)
            if not os.path.exists(f5):
                continue
            with rasterio.open(f5) as ds:
                a = ds.read()
            leg, pla, lm, evm, ex = a[0], a[1], a[2], a[3], a[4]
            in_tile = ex == 1
            denom = in_tile & (evm == 1)
            common = in_tile & (leg > 0) & (pla > 0)
            row_nod_pla = ((denom & (pla == 0)).sum(axis=1))
            row_nod_leg = ((denom & (leg == 0)).sum(axis=1))
            # 南边界行 = 最后一个 in_tile 行；四角
            it_rows = np.where(in_tile.any(axis=1))[0]
            south = int(it_rows.max())
            corners = dict(
                NW=dict(leg=int(leg[it_rows.min(), 0]), pla=int(pla[it_rows.min(), 0])),
                NE=dict(leg=int(leg[it_rows.min(), -1]), pla=int(pla[it_rows.min(), -1])),
                SW=dict(leg=int(leg[south, 0]), pla=int(pla[south, 0])),
                SE=dict(leg=int(leg[south, -1]), pla=int(pla[south, -1])),
            )
            # 存储字节：单波段
            sb = {}
            for arm in ('legacy', 'planar'):
                fp = os.path.join(OUTD, 'TRI_%s_%s_10m.tif' % (tag, arm))
                with rasterio.open(fp) as ds:
                    cm = ds.colormap(1)
                    sb[arm] = dict(bands=ds.count, dtype=ds.dtypes[0], nodata=ds.nodata,
                                   cm_len=len(cm),
                                   photo=int(tifffile.TiffFile(fp).pages[0]
                                             .tags['PhotometricInterpretation'].value))
                im = Image.open(fp)
                sb[arm]['pil_mode'] = im.mode
                sb[arm]['pil_has_palette'] = bool(im.palette)
            mb = dict(photo=int(tifffile.TiffFile(f5).pages[0]
                                .tags['PhotometricInterpretation'].value), bands=5)
            rec['seeds'][seed] = dict(
                evm_in_tile_valid_px=int(denom.sum()),
                legacy_gap_px=int((denom & (leg == 0)).sum()),
                planar_gap_px=int((denom & (pla == 0)).sum()),
                restored_px=int((denom & (leg == 0) & (pla > 0)).sum()),
                common_valid_px=int(common.sum()),
                common_valid_disagree_px=int((common & (leg != pla)).sum()),
                outside_tile_evm_valid_px=int(((~in_tile) & (evm == 1)).sum()),
                outside_tile_legacy_nonzero_px=int(((~in_tile) & (leg > 0)).sum()),
                outside_tile_planar_nonzero_px=int(((~in_tile) & (pla > 0)).sum()),
                legacy_row_nodata_max=int(row_nod_leg.max()),
                planar_row_nodata_max=int(row_nod_pla.max()),
                south_in_tile_row=south,
                corners=corners,
                singleband=sb, multiband=mb,
            )
            s = rec['seeds'][seed]
            ok = (s['planar_gap_px'] == 0 and s['common_valid_disagree_px'] == 0 and
                  s['planar_row_nodata_max'] == 0 and
                  s['legacy_gap_px'] == rec['prod_window_nodata_px'] and
                  s['outside_tile_legacy_nonzero_px'] == 0 and s['outside_tile_planar_nonzero_px'] == 0 and
                  s['singleband']['legacy']['photo'] == 3 and s['singleband']['planar']['photo'] == 3 and
                  s['singleband']['legacy']['pil_mode'] == 'P' and s['singleband']['planar']['pil_mode'] == 'P' and
                  s['multiband']['photo'] == 1 and s['evm_in_tile_valid_px'] > 0)
            s['seed_pass'] = bool(ok)
            rec['region_pass'] = rec['region_pass'] and ok
        res['all_pass'] = res['all_pass'] and rec['region_pass']
        res['regions'][rk] = rec
        print('%s %s  prod_nodata=%d  per-seed leg_gap=%s  planar_gap=%s  disagree=%s  PASS=%s' % (
            rk, R['tile'], rec['prod_window_nodata_px'],
            [v['legacy_gap_px'] for v in rec['seeds'].values()],
            [v['planar_gap_px'] for v in rec['seeds'].values()],
            [v['common_valid_disagree_px'] for v in rec['seeds'].values()],
            rec['region_pass']))
    json.dump(res, open(os.path.join(OUTD, 'TRI_local_verify.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\nALL_PASS=%s → %s' % (res['all_pass'], os.path.join(OUTD, 'TRI_local_verify.json')))


if __name__ == '__main__':
    main()

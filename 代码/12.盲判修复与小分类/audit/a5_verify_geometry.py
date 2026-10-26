# -*- coding: utf-8 -*-
"""a5_verify_geometry.py — 三省几何三窗（TRI_*）小图独立核验（不发起任何云请求）

* 输入：--geom <ws>/geometry（默认 F:/lc_work/label_repair_2023_20261015/geometry）
        自动发现 out/TRI_*_eval.json、out/TRI_*_{legacy,planar}_10m.tif、out/TRI_*_5band.tif、
        reg/tri_*.json、delivery/*_palette.tif
* 规则源：doc72 §8（主要门：planar 缺口 0、恢复像元真实、共同有效域类码分歧 0、边界连续、
          单band photo=PALETTE、mask/色表存储字节有效）；doc43/AGENTS §13（调色板只用 PIL 忠实解码）
* 门槛（逐 region×seed）：tile 内 planar 空洞=0；tile 内 legacy 空洞>0；恢复像元>0；
        两臂 tile 内共同有效像元类码分歧=0；tile 外两臂均为 0；网格/crsTransform 与注册一致；
        eval.json 计数算术闭合；tile 内计数与本地复算一致；类直方图复算一致；
        单band 交付件 PIL photo==3(PALETTE) 且含 ColorMap。任一 FAIL → 该单元不通过。
* 关键实现：tile 内/外域由注册的 tile_box 南界 + 注册网格独立推导（不依赖 5band 波段语义），
        并与 eval 的 outside_tile_px 对照自证。
* 输出：<ws>/audit/out/geometry_verify3_<ts>.json + geometry_manifest3_<ts>.csv
* 用法：python a5_verify_geometry.py [--geom ...] [--seeds 7 11 23]
* 说明：只读本地小图（<10 KB/件）；核验的是覆盖与几何，不得换算为标签精度。
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

WS_DEFAULT = r'F:/lc_work/label_repair_2023_20261015'
REGIONS = ['SC_32N', 'GD_24N', 'NX_38N']
PALETTE_TAG = 3
COLORMAP_TAG = 320


def fh(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def pil_tags(path):
    """用 PIL 读存储字节里的 PhotometricInterpretation / ColorMap（不靠现算路径自证）。"""
    try:
        from PIL import Image
        im = Image.open(path)
        t = im.tag_v2
        return {'photo': int(t.get(262, -1)), 'has_colormap': COLORMAP_TAG in t,
                'palette_ok': int(t.get(262, -1)) == PALETTE_TAG and COLORMAP_TAG in t,
                'mode': im.mode}
    except Exception as e:
        return {'photo': None, 'has_colormap': None, 'palette_ok': None, 'error': str(e)}


def read_tif(path):
    with rasterio.open(path) as d:
        return {'arr': d.read(1), 'w': d.width, 'h': d.height, 'crs': str(d.crs),
                'transform': list(d.transform)[:6], 'nodata': d.nodata, 'dtype': d.dtypes[0]}


def read_bands(path):
    with rasterio.open(path) as d:
        return [d.read(i) for i in range(1, d.count + 1)]


def intile_rows(transform, h, tile_south, mode='top'):
    """由注册网格独立推导 tile 内行。
    mode='top'   ：像元北缘 ≥ tile 南界（与 ee 区域-格网求交、即 eval 口径一致）
    mode='center'：像元中心 ≥ tile 南界（仅差最南一行，用于对照展示边界约定）
    """
    px, _, x0, _, py, y_top = transform
    off = 0.0 if mode == 'top' else 0.5
    lat = y_top + (np.arange(h) + off) * py
    return np.nonzero(lat >= tile_south)[0]


def main():
    ap = argparse.ArgumentParser(description='三省几何三窗小图独立核验')
    ap.add_argument('--ws', default=WS_DEFAULT)
    ap.add_argument('--geom', default=None)
    ap.add_argument('--seeds', nargs='*', type=int, default=[7, 11, 23])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ws = a.ws.replace('\\', '/')
    geom = (a.geom or os.path.join(ws, 'geometry')).replace('\\', '/')
    outd = os.path.join(ws, 'audit', 'out')
    os.makedirs(outd, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    units, manifest = [], []
    for f in sorted(glob.glob(os.path.join(geom, '**', '*.tif'), recursive=True) +
                    glob.glob(os.path.join(geom, '**', '*.json'), recursive=True)):
        manifest.append({'path': f.replace('\\', '/'), 'size': os.path.getsize(f), 'sha256': fh(f)})

    for region in REGIONS:
        for s in a.seeds:
            uid = 'TRI_%s_s%d' % (region, s)
            rec = {'unit': uid, 'checks': []}

            def add(level, code, detail):
                rec['checks'].append({'level': level, 'code': code, 'detail': detail})

            regp = os.path.join(geom, 'reg', 'tri_%s_s%d.json' % (region, s))
            evp = os.path.join(geom, 'out', 'TRI_%s_s%d_eval.json' % (region, s))
            lp = os.path.join(geom, 'out', 'TRI_%s_s%d_legacy_10m.tif' % (region, s))
            pp = os.path.join(geom, 'out', 'TRI_%s_s%d_planar_10m.tif' % (region, s))
            bp = os.path.join(geom, 'out', 'TRI_%s_s%d_5band.tif' % (region, s))
            miss = [os.path.basename(x) for x in (regp, evp, lp, pp) if not os.path.exists(x)]
            if miss:
                add('FAIL', 'A0_artifacts', '缺件：%s（该单元未完成，不计入通过）' % miss)
                rec['gate'] = 'FAIL'
                units.append(rec)
                continue
            regd = json.load(open(regp, encoding='utf-8'))
            ev = json.load(open(evp, encoding='utf-8'))
            rec.update({'province': regd.get('province'), 'tile_id': regd.get('tile_id'),
                        'code_hash': regd.get('code_hash'), 'account_id': regd.get('account_id')})

            L, P = read_tif(lp), read_tif(pp)
            w, h = regd['input_identity']['width'], regd['input_identity']['height']
            ct = [float(x) for x in regd['input_identity']['crsTransform']]
            ok_grid = (L['w'] == P['w'] == w and L['h'] == P['h'] == h
                       and L['crs'] == P['crs'] == 'EPSG:4326'
                       and L['dtype'] == P['dtype'] == 'uint8'
                       and all(abs(u - v) < 1e-12 for u, v in zip(L['transform'], ct))
                       and all(abs(u - v) < 1e-12 for u, v in zip(P['transform'], ct)))
            add('PASS' if ok_grid else 'FAIL', 'A1_grid_matches_registration',
                '%dx%d %s %s transform==注册' % (L['w'], L['h'], L['crs'], L['dtype']) if ok_grid
                else '网格/CRS/transform 与注册不一致：L=%s P=%s reg=%s' % (L['transform'], P['transform'], ct))
            add('PASS' if L['nodata'] == P['nodata'] == 0 else 'FAIL', 'A1b_nodata',
                'nodata=%s/%s（应为 0）' % (L['nodata'], P['nodata']))

            # tile 内域：由注册 tile 南界 + 网格独立推导（北缘口径为主，与 eval 一致；
            # 另给中心口径对照，二者仅差最南一行，边界约定差异显式记录而非隐藏）
            tile_south = float(regd['tile_box'][1])
            ri = intile_rows(L['transform'], h, tile_south, 'top')
            ri_c = intile_rows(L['transform'], h, tile_south, 'center')
            inside = np.zeros((h, w), dtype=bool)
            inside[ri, :] = True
            n_out = int((~inside).sum())
            ev_out = ev.get('outside_tile_px')
            add('PASS' if ev_out == n_out else 'WARN', 'A1c_tile_domain_derivation',
                '本地推导 tile 外 %d == eval outside_tile_px %s（北缘口径自证；中心口径 %d，差 %d px = %d 行，属边界约定）'
                % (n_out, ev_out, int((h - len(ri_c)) * w),
                   int((h - len(ri_c)) * w - n_out), int(round(((h - len(ri_c)) * w - n_out) / w)))
                if ev_out == n_out else
                '本地推导 tile 外 %d ≠ eval %s（北缘口径；中心口径 %d）→ 先核 tile_box/网格'
                % (n_out, ev_out, int((h - len(ri_c)) * w)))

            ln0_in = int(((L['arr'] == 0) & inside).sum())
            pn0_in = int(((P['arr'] == 0) & inside).sum())
            add('PASS' if pn0_in == 0 else 'FAIL', 'A2_planar_no_gap_intile',
                'tile 内 planar 空洞 %d px（门=0）' % pn0_in)
            add('PASS' if ln0_in > 0 else 'FAIL', 'A2b_legacy_gap_intile',
                'tile 内 legacy 空洞 %d px（应>0）' % ln0_in)
            restored = int(((L['arr'] == 0) & (P['arr'] != 0) & inside).sum())
            add('PASS' if restored > 0 else 'FAIL', 'A2c_restored',
                'tile 内恢复像元 %d px' % restored)
            outnz_l = int(((L['arr'] != 0) & ~inside).sum())
            outnz_p = int(((P['arr'] != 0) & ~inside).sum())
            add('PASS' if outnz_l == 0 and outnz_p == 0 else 'FAIL', 'A2d_outside_tile_zero',
                'tile 外非零：legacy %d / planar %d（应为 0；eval 声明 %s/%s）'
                % (outnz_l, outnz_p, ev.get('outside_tile_legacy_nonzero_px'),
                   ev.get('outside_tile_planar_nonzero_px')))

            both = inside & (L['arr'] != 0) & (P['arr'] != 0)
            dis = int((L['arr'][both] != P['arr'][both]).sum())
            add('PASS' if dis == 0 else 'FAIL', 'A3_common_valid_agree',
                'tile 内两臂共同有效 %d px，类码分歧 %d px（门=0）' % (int(both.sum()), dis))

            arith = [('evm_in_tile=common+legacy_gap', ev.get('common_valid_px', 0) + ev.get('legacy_gap_px', 0),
                      ev.get('evm_in_tile_valid_px')),
                     ('evm_valid=in_tile+outside', ev.get('evm_in_tile_valid_px', 0) + ev.get('outside_tile_evm_valid_px', 0),
                      ev.get('evm_valid_px')),
                     ('n_px=w*h', L['w'] * L['h'], ev.get('n_px')),
                     ('restored=legacy_gap', ev.get('legacy_gap_px'), ev.get('restored_px'))]
            bad = [(k, x, y) for k, x, y in arith if x != y]
            add('PASS' if not bad else 'FAIL', 'A4_eval_arithmetic',
                '计数闭合（4 式）' if not bad else '算术不闭合：%s' % bad)

            cmp_pairs = [('legacy_nodata_px', ev.get('legacy_nodata_px'), ln0_in),
                         ('planar_nodata_px', ev.get('planar_nodata_px'), pn0_in),
                         ('legacy_gap_px', ev.get('legacy_gap_px'), ln0_in),
                         ('restored_px', ev.get('restored_px'), restored),
                         ('common_valid_disagree_px', ev.get('common_valid_disagree_px'), dis),
                         ('common_valid_px', ev.get('common_valid_px'), int(both.sum()))]
            badc = [(k, x, y) for k, x, y in cmp_pairs if x != y]
            add('PASS' if not badc else 'FAIL', 'A5_eval_vs_local',
                'eval 声明与本地复算一致（%d 项）' % len(cmp_pairs) if not badc
                else '不一致（字段,eval,本地）：%s' % badc)
            add('INFO', 'A5b_evm_full_intile',
                'eval evm_in_tile_valid_px=%s，tile 内像元总数=%d（相等即 EVM 在本窗 tile 内全有效）'
                % (ev.get('evm_in_tile_valid_px'), int(inside.sum())))

            def hist(arr):
                v, c = np.unique(arr[arr != 0], return_counts=True)
                return {str(int(k)): int(n) for k, n in zip(v, c)}
            hl, hp = hist(L['arr']), hist(P['arr'])
            ok_hist = (hl == {str(k): v for k, v in (ev.get('legacy_class_hist') or {}).items()}
                       and hp == {str(k): v for k, v in (ev.get('planar_class_hist') or {}).items()})
            add('PASS' if ok_hist else 'FAIL', 'A6_class_hist',
                '类直方图复算一致（legacy %d 类 / planar %d 类）' % (len(hl), len(hp)) if ok_hist
                else '直方图不一致：legacy 本地 %s vs eval %s；planar 本地 %s vs eval %s'
                     % (hl, ev.get('legacy_class_hist'), hp, ev.get('planar_class_hist')))
            rec['class_codes_observed'] = sorted(int(k) for k in set(hl) | set(hp))

            # 楔形签名：tile 内 nodata 逐行应为"连续带"，且随行加深单调嵌套（集合非减）。
            # 窗口是否含瓦片中央经线决定该带是"中央对称"还是"单侧锚定"，因此不硬性要求对称。
            nod = (L['arr'] == 0)
            row_in = nod[ri]
            prof = row_in.sum(axis=1)
            nz = np.nonzero(prof)[0]
            colsets = [np.nonzero(row_in[rl_])[0] for rl_ in nz]
            noncontig = [int(ri[nz[i]]) for i, c in enumerate(colsets)
                         if len(c) and (int(c.max()) - int(c.min()) + 1) != len(c)]
            notnested = [int(ri[nz[i + 1]]) for i in range(len(colsets) - 1)
                         if not set(colsets[i].tolist()) <= set(colsets[i + 1].tolist())]
            px = float(regd['input_identity']['crsTransform'][0])
            tile_c0 = int(regd['region_tile_rc']['c0'])
            tile_w = int(round(2.0 / px))
            apex_in = tile_c0 <= (tile_w // 2) <= (tile_c0 + w - 1)
            rp = (P['arr'] == 0).sum(axis=1)
            add('PASS' if int(rp[ri].max(initial=0)) == 0 else 'FAIL', 'A7_planar_row_gap_intile',
                'tile 内 planar 逐行空洞最大 %d（门=0）' % int(rp[ri].max(initial=0)))
            add('INFO', 'A7b_legacy_wedge_profile',
                'tile 内 legacy 空洞 max=%d 列/行、非空行 %d 行（窗口行 %s–%s）；满行 %d 行；'
                '窗口含瓦片中央经线=%s（决定带形为中央对称/单侧锚定）；解析楔高 %.2f 行'
                '（窗口仅含 tile 南缘，被 tile 下界截断属预期）；eval legacy_row_nodata_max=%s'
                % (int(prof.max(initial=0)), len(nz), int(ri[nz.min()]) if len(nz) else '-',
                   int(ri[nz.max()]) if len(nz) else '-', int((prof == w).sum()), apex_in,
                   float(ev.get('wedge_h_rows') or 0), ev.get('legacy_row_nodata_max')))
            if noncontig:
                add('FAIL', 'A7c_wedge_signature',
                    'legacy 空洞存在不连续行（行 %s）：与楔形连续带签名不符' % noncontig[:8])
            elif notnested:
                add('FAIL', 'A7c_wedge_signature',
                    'legacy 空洞行集合非单调嵌套（行 %s）：与测地楔形签名不符' % notnested[:8])
            else:
                add('PASS', 'A7c_wedge_signature',
                    'tile 内空洞逐行连续且随行加深单调嵌套（%s带，楔形签名成立）'
                    % ('中央对称' if apex_in else '单侧锚定'))

            if os.path.exists(bp):
                bands = read_bands(bp)
                info = [{'band': i + 1, 'n_zero': int((b == 0).sum()),
                         'uniq': [int(x) for x in np.unique(b)[:8]]} for i, b in enumerate(bands)]
                rec['fiveband'] = info
                add('INFO', 'A8_fiveband', '5band 零值计数 %s（诊断件不要求调色板；语义不假设）'
                    % [b['n_zero'] for b in info])
            else:
                add('WARN', 'A8_fiveband', '缺 5band 诊断件')

            # A9 调色板 + 数据同一性：逐 seed 逐臂核对交付件（PIL 存储字节 + 与 out/ 原件逐像元一致）
            dl_info, dl_miss, dl_bad, dl_ident = [], [], [], []
            for arm, orig in (('legacy', L), ('planar', P)):
                dp = os.path.join(geom, 'delivery', 'TRI_%s_s%d_%s_10m_palette.tif' % (region, s, arm))
                if not os.path.exists(dp):
                    dl_miss.append(arm)
                    continue
                t = pil_tags(dp)
                dl_info.append({'arm': arm, 'file': os.path.basename(dp), **t})
                if not t.get('palette_ok'):
                    dl_bad.append(arm)
                try:
                    with rasterio.open(dp) as dd:
                        same_grid = (dd.width == orig['w'] and dd.height == orig['h']
                                     and abs(dd.transform[5] - orig['transform'][5]) < 1e-12
                                     and abs(dd.transform[2] - orig['transform'][2]) < 1e-12)
                        ident = same_grid and bool(np.array_equal(dd.read(1), orig['arr']))
                except Exception as e:
                    ident, same_grid = False, False
                    dl_info[-1]['read_error'] = str(e)
                if not ident:
                    dl_ident.append(arm)
            rec['delivery'] = dl_info
            if dl_miss:
                add('WARN', 'A9_delivery_palette',
                    '交付件缺 %s（按你的口径缺件记 WARN 不阻断）' % dl_miss)
            elif dl_bad:
                add('FAIL', 'A9_delivery_palette',
                    '交付件 PIL 存储字节不合格（photo!=3 或缺 ColorMap）：%s' % dl_bad)
            else:
                add('PASS', 'A9_delivery_palette',
                    '交付件 2 臂均 PIL photo=3 且含 ColorMap：%s'
                    % [d['file'] for d in dl_info])
            if dl_miss:
                pass
            elif dl_ident:
                add('FAIL', 'A9b_delivery_data_identity',
                    '交付件与 out/ 原件逐像元不一致或网格不符：%s' % dl_ident)
            else:
                add('PASS', 'A9b_delivery_data_identity',
                    '交付件与 out/ 原件逐像元一致、网格相同（%d 臂 × 3 seed 口径）' % len(dl_info))

            el = float(ev.get('elapsed_s') or 0)
            dls = float(regd.get('deadline_s') or 1200)
            add('PASS' if 0 < el <= dls else 'FAIL', 'A10_timing',
                'elapsed_s=%.1f（门≤%.0f）；req_s=%s' % (el, dls, ev.get('req_s')))
            same_hash = regd.get('code_hash') == ev.get('code_hash')
            add('PASS' if same_hash else 'FAIL', 'A10b_code_hash_consistent',
                'reg/eval code_hash 一致：%s' % regd.get('code_hash') if same_hash
                else 'reg %s vs eval %s' % (regd.get('code_hash'), ev.get('code_hash')))
            add('PASS' if ev.get('boundary_continuous') else 'WARN', 'A10c_boundary',
                'boundary_continuous=%s；outside_wedge_agree_rate=%s；common_valid_agree_rate=%s'
                % (ev.get('boundary_continuous'), ev.get('outside_wedge_agree_rate'),
                   ev.get('common_valid_agree_rate')))

            rec['gate'] = 'FAIL' if any(c['level'] == 'FAIL' for c in rec['checks']) else (
                'WARN' if any(c['level'] == 'WARN' for c in rec['checks']) else 'PASS')
            units.append(rec)

    missing_units = [u['unit'] for u in units if any(c['code'] == 'A0_artifacts' for c in u['checks'])]
    summary = {
        'schema': 'geometry_verify3/v1',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'geom_dir': geom, 'seeds': a.seeds,
        'units_expected': len(REGIONS) * len(a.seeds), 'units_checked': len(units),
        'missing_units': missing_units,
        'by_gate': {g: sum(1 for u in units if u['gate'] == g) for g in ('PASS', 'WARN', 'FAIL')},
        'gate': ('INCOMPLETE' if missing_units else
                 ('PASS' if units and all(u['gate'] == 'PASS' for u in units) else 'FAIL')),
        'scope_note': '仅覆盖/几何核验：不得换算为地类标签精度；未发起云请求；未读大幅栅格',
        'units': units,
    }
    fp = a.out or os.path.join(outd, 'geometry_verify3_%s.json' % ts)
    json.dump(summary, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    mp = os.path.join(outd, 'geometry_manifest3_%s.csv' % ts)
    with open(mp, 'w', encoding='utf-8-sig', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=['path', 'size', 'sha256'])
        wtr.writeheader()
        wtr.writerows(manifest)
    for u in units:
        fails = [c['code'] for c in u['checks'] if c['level'] == 'FAIL']
        print('%-16s %s %s' % (u['unit'], u['gate'], ('FAIL: %s' % fails) if fails else ''))
    print('总门=%s（%s，缺件单元 %d）｜产物 %d 件 → %s'
          % (summary['gate'], summary['by_gate'], len(missing_units), len(manifest), fp))


if __name__ == '__main__':
    main()

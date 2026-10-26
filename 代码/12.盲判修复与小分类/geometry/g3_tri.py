# -*- coding: utf-8 -*-
"""g3_tri.py — 三省南缘小幅几何复验：legacy(测地) vs planar(平面) 同表达式小框真实分类对照

* 输入（只读）
    - 训练池资产 projects/leafy-environs-509005-n1/assets/prod2nd/smp2023v2_merged（第二轮冻结池，cl 24 码）
    - AEF 2023 GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL（64 维）
    - 参考网格：各 region 的 ref_tif_for_grid（生产资产 affine，只读表头）
* 规则源
    - doc72 §8（三省各 1 新南缘；同 v2 原训练表达式、同模型两种 clip；平面 region + crsTransform + 显式 mask 物化；
      EVM=AEF64 共同 mask 不先错误裁剪；tile 外单列；每 region 3 seed ≤1200s 不 batch）
    - doc70 §4.4/§7；doc69 §6.4（边缘确定性扫描）；geom_planar.py（平面/测地几何、网格吸附、EVM、楔形解析）
* 门槛
    - **单请求 5 波段**（legacy/planar/legacy_mask/evm/expmask）→ 同一次训练选取，两臂严格同模型；
    - 每请求 deadline ≤60s 且按剩余时间裁剪；不发 batch；父进程 1200s 硬截止（见 run_g3.py）；
    - EVM 分母限定 **平面 tile box 内**（expmask==1）；瓦外对照行另单列，不计缺口。
* 输出（F:/lc_work/label_repair_2023_20261015/geometry/）
    - out/TRI_<region>_s<seed>_5band.tif（uint8×5，无调色板）
    - out/TRI_<region>_s<seed>_{legacy,planar}_10m.tif（单波段 + PALETTE + nodata=0）
    - out/TRI_<region>_s<seed>_eval.json（逐项计数与一致率）
    - reg/tri_<region>_s<seed>.json（运行前登记）
* 用法
    python g3_tri.py --selfcheck                        # 配置→几何/网格自检（无云端计算）
    python g3_tri.py --region SC_32N --register-only     # 只登记（零 GEE 计算）
    python g3_tri.py --region SC_32N --seeds 7,11,23     # 真实两臂对照（逐 seed 单请求）
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
sys.stdout.reconfigure(encoding='utf-8')
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
WS = r'F:/lc_work/label_repair_2023_20261015/geometry'
OUTD = os.path.join(WS, 'out')
REGD = os.path.join(WS, 'reg')
sys.path.insert(0, HERE)
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产')

import numpy as np
import geom_planar as GP

CONFIG_FP = os.path.join(HERE, 'config', 'planar_tri3.json')
ACCT = None                 # 由 --acct 或配置/健康检查确定（ID 记录，不含密钥）
SRC = 'projects/leafy-environs-509005-n1/assets/prod2nd/smp2023v2_merged'
AEF_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
YEAR = 2023
N_LIMIT = 20000
RF = dict(numberOfTrees=100, minLeafPopulation=2, maxNodes=5000)
SEEDS = [7, 11, 23]
DEADLINE_S = 1200.0
REQ_TIMEOUT_S = 60.0
REGIONS = []
IMPL_VERSION = 'v2_unmask0_photo3'
COLORS = {1: (0.93, 0.87, 0.55), 2: (0.55, 0.80, 0.35), 3: (0.35, 0.65, 0.95),
          4: (0.05, 0.35, 0.10), 5: (0.30, 0.62, 0.25), 6: (0.10, 0.45, 0.35),
          7: (0.45, 0.72, 0.55), 8: (0.55, 0.62, 0.20), 9: (0.75, 0.72, 0.30),
          10: (0.85, 0.78, 0.40), 11: (0.90, 0.92, 0.65), 12: (0.80, 0.85, 0.60),
          13: (0.95, 0.90, 0.75), 14: (0.30, 0.55, 0.55), 15: (0.45, 0.70, 0.70),
          16: (0.85, 0.80, 0.65), 17: (0.70, 0.60, 0.75), 18: (0.90, 0.20, 0.35),
          19: (0.70, 0.45, 0.65), 20: (0.80, 0.75, 0.55), 21: (0.85, 0.20, 0.20),
          22: (0.75, 0.70, 0.65), 23: (0.15, 0.35, 0.85), 24: (0.95, 0.98, 1.00)}


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def load_config(fp):
    """读取可执行工程配置（planar_geom_config/v1）→ 覆盖模块级登记值。"""
    global SRC, AEF_ID, YEAR, N_LIMIT, RF, SEEDS, DEADLINE_S, REGIONS, ACCT
    cfg = json.load(open(fp, encoding='utf-8'))
    assert cfg.get('schema') == 'planar_geom_config/v1', 'config schema 不符'
    ne = cfg.get('NOT_ENABLED', {})
    for k in ('O_CAP_orchard_cap', 'F_VIEW_forest_filter', 'T_STRATA_terrain_strata'):
        assert ne.get(k) is False, '配置不得启用未过门规则 %s' % k
    g = cfg['geometry']
    assert g['plane_clip'] is True and g['plane_region'] is True, 'plane_clip/plane_region 必须为 true'
    assert g['region_never_four_number_list'] is True, 'region 不得传 4 数字列表'
    assert cfg['materialize_mask'] is True, 'materialize_mask 必须为 true'
    assert cfg['grid_rule']['use_explicit_crsTransform'] is True, '必须用显式 crsTransform'
    assert cfg['grid_rule']['scale_must_be_null_when_crsTransform_set'] is True, 'scale 必须置空'
    assert g['train_filter_bounds_geometry_unchanged'] is True, '本配置不得改训练窗几何'
    SRC = cfg['model']['pool_asset']
    AEF_ID = cfg['model']['aef_collection']
    YEAR = int(cfg['model']['year'])
    N_LIMIT = int(cfg['model']['n_train_limit'])
    RF = dict(cfg['model']['rf'])
    SEEDS = [int(s) for s in cfg['seeds']]
    DEADLINE_S = float(cfg['deadline_s'])
    REGIONS = cfg['regions']
    ACCT = ACCT or cfg.get('account_id')
    return cfg


def region_of(key):
    for r in REGIONS:
        if r['region'] == key:
            return r
    raise KeyError('未知 region %s（可选 %s）' % (key, [r['region'] for r in REGIONS]))


def grid_of(reg):
    px, py, x0, ytop, W, H = GP.grid_of_tif(reg['ref_tif_for_grid'])
    return dict(px=px, py=py, x0=x0, ytop=ytop, W=W, H=H)


def window_bbox(g, reg):
    xl = g['x0'] + reg['c0'] * g['px']
    xr = g['x0'] + (reg['c0'] + reg['ncol']) * g['px']
    yt = g['ytop'] - reg['r0'] * g['py']
    yb = g['ytop'] - (reg['r0'] + reg['nrow']) * g['py']
    return [xl, yb, xr, yt]


def code_hash(cfg_fp=CONFIG_FP):
    h = hashlib.md5()
    for fp in [os.path.join(HERE, 'geom_planar.py'), os.path.abspath(__file__), cfg_fp]:
        h.update(open(fp, 'rb').read())
    return h.hexdigest()


def selfcheck(fp=CONFIG_FP):
    """配置→几何/网格自检：无云端计算。"""
    cfg = load_config(fp)
    out = dict(config=fp, n_regions=len(REGIONS), regions={}, seeds=SEEDS, deadline_s=DEADLINE_S,
               materialize_mask=True, code_hash=code_hash(fp))
    ok = True
    for reg in REGIONS:
        g = grid_of(reg)
        tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'],
                                  reg['c0'], reg['r0'], reg['ncol'], reg['nrow'])
        wb = window_bbox(g, reg)
        dx = max(abs(a - b) for a, b in zip(wb, reg['region_bbox_deg']))
        in_tile = max(0, min(g['H'], reg['r0'] + reg['nrow']) - reg['r0'])
        rec = dict(tile=reg['tile_id'], crsTransform=tr, width=w, height=h,
                   region_bbox=wb, bbox_delta_vs_config=dx, rows_in_tile=in_tile,
                   rows_outside_tile=reg['nrow'] - in_tile,
                   wedge_h_deg=GP.wedge_height_deg(reg['tile_box']),
                   wedge_h_rows=GP.wedge_height_deg(reg['tile_box']) / g['py'],
                   xor_ok=GP.assert_scale_xor_transform(scale=None, crs_transform=tr))
        rec['bbox_ok'] = bool(dx < 1e-7)
        rec['rows_ok'] = bool(reg['nrow'] <= 128 and 256 <= reg['ncol'] <= 600)
        ok = ok and rec['bbox_ok'] and rec['rows_ok']
        out['regions'][reg['region']] = rec
    out['PASS'] = bool(ok)
    return out


def registration(reg, seed):
    g = grid_of(reg)
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'],
                              reg['c0'], reg['r0'], reg['ncol'], reg['nrow'])
    wh = GP.wedge_height_deg(reg['tile_box'])
    return dict(
        schema='label_repair_geom_spec/v1',
        exp_id='TRI_%s_s%d' % (reg['region'], seed),
        impl_version=IMPL_VERSION,
        owner='geometry',
        stage='doc72-8',
        hypothesis='H6 几何修复仅恢复覆盖、不改变模型（planar clip 消除测地楔形 nodata；共同有效域类码一致）',
        province=reg['prov'],
        region_bbox=window_bbox(g, reg),
        region_tile_rc=dict(c0=reg['c0'], r0=reg['r0'], ncol=reg['ncol'], nrow=reg['nrow']),
        tile_id=reg['tile_id'],
        tile_box=reg['tile_box'],
        training_bbox=reg['train_window_box'],
        input_identity=dict(pool=SRC, aef=AEF_ID, year=YEAR,
                            grid=dict(px=g['px'], py=g['py'], x0=g['x0'], y_top=g['ytop']),
                            crsTransform=tr, width=w, height=h),
        class_map_hash=hashlib.md5(json.dumps({str(k): v for k, v in COLORS.items()},
                                              sort_keys=True).encode()).hexdigest(),
        candidate_rule='clip 几何：legacy=ee.Geometry.Rectangle(box)（测地默认）vs planar=ee.Geometry.BBox(box)（平面）',
        arms=['legacy', 'planar'],
        seeds=[seed],
        seed_set_registered=SEEDS,
        n_train_limit=N_LIMIT,
        rf_params=dict(RF, seed=seed),
        deadline_s=DEADLINE_S,
        request_timeout_s=REQ_TIMEOUT_S,
        target_metrics=['evm_in_tile_valid_px', 'legacy_gap_px', 'planar_gap_px',
                        'restored_px', 'common_valid_px', 'common_valid_disagree_px',
                        'outside_wedge_agree_rate', 'outside_tile_evm_valid_px',
                        'planar_row_nodata_max', 'boundary_continuous'],
        expected_effect=('楔内(planar box 内、测地 arc 外) EVM==1 而 legacy==0 的像元数 >0（工程缺口）；'
                         'planar 恢复该区；共同有效域类码分歧=0；楔高解析预测 %.4f° (%.1f 行)'
                         % (wh, wh / g['py'])),
        stop_rules=['单请求 >60s 无响应即按剩余时间重试，剩余 <30s 放弃',
                    '父进程 1200s 硬截止 → 记 TIMEOUT 并终止子进程',
                    '若 20k 训练交互失败 → 写 FAIL，另注册 2k 工程对照（不伪称原生产精度）'],
        code_hash=code_hash(),
        account_id=ACCT,
        registered_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
    )


def build_image(ee, reg, seed):
    """5 波段：legacy / planar / legacy_mask / evm / expmask（同一分类表达式 → 两臂同模型、同训练选取）。

    掩膜物化（G-30N 探针结论，必须）：computePixels(GEO_TIFF) 不传递图像掩膜 → 类别波段
    一律 .unmask(0, False) 显式填 0（0=nodata），并另出 legacy_mask 波段交叉核对。
    """
    BOX = reg['tile_box']
    BOXP = reg['train_window_box']
    boxp = GP.legacy_rect(BOXP, ee=ee)                       # 训练窗：固定 legacy（生产同式）
    fc = (ee.FeatureCollection(SRC).filterBounds(boxp)
          .randomColumn('rc', seed).sort('rc').limit(N_LIMIT))
    feats = ['A%02d' % i for i in range(64)]
    clf = ee.Classifier.smileRandomForest(**RF).train(fc, 'cl', feats)
    aef = (ee.ImageCollection(AEF_ID)
           .filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
           .filterBounds(GP.planar_rect(BOX, ee=ee)).mosaic()
           .select(feats))
    cls = aef.classify(clf).rename('class').uint8()
    legacy = cls.clip(GP.legacy_rect(BOX, ee=ee)).unmask(0, False).rename('legacy')
    planar = cls.clip(GP.planar_rect(BOX, ee=ee)).unmask(0, False).rename('planar')
    lmask = (cls.clip(GP.legacy_rect(BOX, ee=ee)).mask().unmask(0, False)
             .rename('legacy_mask').uint8())
    evm = GP.common_valid_mask(aef, ee=ee).unmask(0, False)
    expmask = GP.expected_region_mask(BOX, ee=ee).unmask(0, False)
    return legacy.addBands(planar).addBands(lmask).addBands(evm).addBands(expmask)


def fetch(ee, img, tr, w, h, deadline_t, tag=''):
    """computePixels 单请求（≤60s deadline，按剩余时间裁剪，自带重试）。"""
    import ee as _ee
    _ee.data.setMaxRetries(0)
    last = ''
    for attempt in range(4):
        rem = deadline_t - time.time()
        if rem < 30:
            raise TimeoutError('剩余 %.0fs < 30s，放弃 %s' % (rem, tag))
        _ee.data.setDeadline(min(REQ_TIMEOUT_S, rem - 5) * 1000.0)
        t0 = time.time()
        try:
            params = {
                'expression': img,
                'fileFormat': 'GEO_TIFF',
                'grid': {'dimensions': {'width': int(w), 'height': int(h)},
                         'affineTransform': {'scaleX': tr[0], 'shearX': tr[1], 'translateX': tr[2],
                                             'shearY': tr[3], 'scaleY': tr[4], 'translateY': tr[5]},
                         'crsCode': 'EPSG:4326'},
            }
            b = _ee.data.computePixels(params)
            log('%s 请求成功 %.1fs（%.0f KB，第 %d 次）' % (tag, time.time() - t0, len(b) / 1e3, attempt + 1))
            return b, time.time() - t0
        except Exception as e:
            last = str(e)[:150]
            log('%s 第 %d 次失败 %.1fs：%s' % (tag, attempt + 1, time.time() - t0, last))
            time.sleep(min(8.0, 2.0 * (attempt + 1)))
    raise RuntimeError('computePixels 4 次均失败：%s' % last)


def write_tif(path, arr, tr, w, h, count, dtype='uint8', palette=None, nodata=0):
    import rasterio
    prof = dict(driver='GTiff', dtype=dtype, count=count, nodata=nodata, crs='EPSG:4326',
                transform=rasterio.Affine(*tr), width=w, height=h,
                compress='deflate', predictor=2, tiled=True, blockxsize=256, blockysize=256)
    if palette and count == 1:
        prof['photometric'] = 'PALETTE'      # 单波段交付件：photometric=3，读取器可识别
    with rasterio.open(path, 'w', **prof) as ds:
        if count == 1:
            ds.write(arr.astype(dtype), 1)
            if palette:
                ds.write_colormap(1, palette)
        else:
            ds.write(arr.astype(dtype))


def palette_map():
    pal = {i: (230, 230, 230, 0) for i in range(256)}
    for k, c in COLORS.items():
        pal[k] = (int(round(c[0] * 255)), int(round(c[1] * 255)), int(round(c[2] * 255)), 255)
    for k in range(25, 256):
        pal[k] = (200, 200, 200, 255)
    return pal


def run_one(ee, reg, g, tr, w, h, tag, seed, deadline_t):
    """单个种子：构建 5 波段图 → 单请求 → 落 GeoTIFF → 评估。"""
    log('构建 5 波段图（legacy/planar/legacy_mask/evm/expmask），region=%s tag=%s seed=%d …'
        % (reg['region'], tag, seed))
    t_build = time.time()
    img = build_image(ee, reg, seed)
    log('图构建完成 %.1fs（惰性对象图，未触发计算）' % (time.time() - t_build))
    b, dt = fetch(ee, img, tr, w, h, deadline_t, tag='TRI-%s-5band' % reg['region'])
    fp5 = os.path.join(OUTD, 'TRI_%s_5band.tif' % tag)
    open(fp5 + '.raw', 'wb').write(b)
    import rasterio
    with rasterio.open(fp5 + '.raw') as ds:
        arr = ds.read()
    legacy, planar, lmask, evm, expmask = arr[0], arr[1], arr[2], arr[3], arr[4]
    write_tif(fp5, arr, tr, w, h, 5)                     # 多波段诊断件：不含调色板
    os.remove(fp5 + '.raw')
    pal = palette_map()
    write_tif(os.path.join(OUTD, 'TRI_%s_legacy_10m.tif' % tag), legacy, tr, w, h, 1, palette=pal)
    write_tif(os.path.join(OUTD, 'TRI_%s_planar_10m.tif' % tag), planar, tr, w, h, 1, palette=pal)
    log('GeoTIFF 落盘：5band + legacy + planar（单波段 photo=PALETTE）')

    # ---- 评估 ----
    NROW, NCOL, R0, C0 = reg['nrow'], reg['ncol'], reg['r0'], reg['c0']
    rows = np.arange(R0, R0 + NROW)
    cols = np.arange(C0, C0 + NCOL)
    lon = (g['x0'] + (cols + 0.5) * g['px'])[None, :].repeat(NROW, 0)
    lat = (g['ytop'] - (rows + 0.5) * g['py'])[:, None].repeat(NCOL, 1)
    lens = GP.wedge_lens_mask(lon, lat, reg['tile_box'])
    in_tile = expmask == 1
    denom = in_tile & (evm == 1)
    common = in_tile & (legacy > 0) & (planar > 0)
    leg_row_nod = ((denom & (legacy == 0)).sum(axis=1))
    pla_row_nod = ((denom & (planar == 0)).sum(axis=1))
    ev = dict(
        exp_id='TRI_%s_s%d' % (reg['region'], seed), impl_version=IMPL_VERSION, seed=seed,
        account_id=ACCT, region=reg['region'], prov=reg['prov'], tile_id=reg['tile_id'],
        region_bbox=window_bbox(g, reg), n_px=int(NROW * NCOL),
        crsTransform=tr, width=w, height=h,
        wedge_h_deg=GP.wedge_height_deg(reg['tile_box']),
        wedge_h_rows=GP.wedge_height_deg(reg['tile_box']) / g['py'],
        wedge_lens_px=int(lens.sum()),
        wedge_lens_in_tile_px=int((lens & in_tile).sum()),
        evm_valid_px=int((evm == 1).sum()),
        evm_in_tile_valid_px=int(denom.sum()),
        legacy_mask0_px=int((lmask == 0).sum()),
        legacy_mask_eq_unmask0_px=int(((lmask == 0) == (legacy == 0)).sum()),
        legacy_gap_px=int((denom & (legacy == 0)).sum()),
        planar_gap_px=int((denom & (planar == 0)).sum()),
        restored_px=int((denom & (legacy == 0) & (planar > 0)).sum()),
        legacy_nodata_px=int((in_tile & (legacy == 0)).sum()),
        planar_nodata_px=int((in_tile & (planar == 0)).sum()),
        common_valid_px=int(common.sum()),
        common_valid_disagree_px=int((common & (legacy != planar)).sum()),
        outside_wedge_both_px=int((in_tile & ~lens & (legacy > 0) & (planar > 0)).sum()),
        outside_wedge_disagree_px=int((in_tile & ~lens & (legacy > 0) & (planar > 0) & (legacy != planar)).sum()),
        outside_tile_px=int((~in_tile).sum()),
        outside_tile_evm_valid_px=int(((~in_tile) & (evm == 1)).sum()),
        outside_tile_legacy_nonzero_px=int(((~in_tile) & (legacy > 0)).sum()),
        outside_tile_planar_nonzero_px=int(((~in_tile) & (planar > 0)).sum()),
        legacy_row_nodata_max=int(leg_row_nod.max()),
        planar_row_nodata_max=int(pla_row_nod.max()),
        legacy_class_hist={int(k): int(v) for k, v in zip(*np.unique(legacy[in_tile & (legacy > 0)], return_counts=True))},
        planar_class_hist={int(k): int(v) for k, v in zip(*np.unique(planar[in_tile & (planar > 0)], return_counts=True))},
        req_s=round(dt, 2), elapsed_s=round(time.time() - deadline_t + DEADLINE_S, 1),
        code_hash=code_hash(),
    )
    ev['outside_wedge_agree_rate'] = (1.0 - ev['outside_wedge_disagree_px'] / max(1, ev['outside_wedge_both_px']))
    ev['common_valid_agree_rate'] = (1.0 - ev['common_valid_disagree_px'] / max(1, ev['common_valid_px']))
    ev['wedge_restore_rate'] = ev['restored_px'] / max(1, ev['legacy_gap_px'])
    ev['boundary_continuous'] = bool(ev['planar_gap_px'] == 0)
    json.dump(ev, open(os.path.join(OUTD, 'TRI_%s_eval.json' % tag), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('评估 → TRI_%s_eval.json' % tag)
    print(json.dumps({k: ev[k] for k in ['evm_in_tile_valid_px', 'legacy_gap_px', 'planar_gap_px',
                                         'restored_px', 'common_valid_px', 'common_valid_disagree_px',
                                         'outside_wedge_agree_rate', 'outside_tile_evm_valid_px',
                                         'planar_row_nodata_max', 'boundary_continuous', 'req_s', 'elapsed_s']},
                     ensure_ascii=False, indent=1))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', required=False, help='SC_32N / GD_24N / NX_38N')
    ap.add_argument('--register-only', action='store_true')
    ap.add_argument('--seeds', default=None, help='逗号分隔种子集合（默认配置 {7,11,23}）')
    ap.add_argument('--tag', default=None, help='实验标签；默认 <region>_s<seed>')
    ap.add_argument('--deadline', type=float, default=None)
    ap.add_argument('--acct', default=None, help='GEE 账号（须先 h0_health 通过）')
    ap.add_argument('--config', default=CONFIG_FP)
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    global ACCT, DEADLINE_S
    if a.acct:
        ACCT = a.acct
    cfg = load_config(a.config)
    if a.selfcheck:
        print(json.dumps(selfcheck(a.config), ensure_ascii=False, indent=1))
        return
    if a.deadline is not None:
        DEADLINE_S = float(a.deadline)
    if not a.region:
        raise SystemExit('--region 必填（%s）' % [r['region'] for r in REGIONS])
    reg = region_of(a.region)
    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(REGD, exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(',')] if a.seeds else list(SEEDS)
    deadline_t = time.time() + DEADLINE_S
    log('region=%s tile=%s seeds=%s deadline=%.0fs' % (reg['region'], reg['tile_id'], seeds, DEADLINE_S))

    g = grid_of(reg)
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], reg['c0'], reg['r0'], reg['ncol'], reg['nrow'])
    GP.assert_scale_xor_transform(scale=None, crs_transform=tr)

    if a.register_only:
        for sd in seeds:
            tg = a.tag or ('%s_s%d' % (reg['region'], sd))
            reg_rec = registration(reg, sd)
            json.dump(reg_rec, open(os.path.join(REGD, 'tri_%s.json' % tg), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)
            log('登记 → reg/tri_%s.json' % tg)
        return

    sys.path.insert(0, r'F:/lc_work/v31_exp/code')
    import v31_common as VC
    if not ACCT:
        raise SystemExit('--acct 必填（真实请求）；先跑 h0_health.py')
    ee, pid = VC.ctx(ACCT)
    log('账号 %s / 项目 %s 就绪；region %s；种子集 %s' % (ACCT, pid, reg['region'], seeds))

    summary = []
    for sd in seeds:
        tag = a.tag or ('%s_s%d' % (reg['region'], sd))
        reg_rec = registration(reg, sd)
        json.dump(reg_rec, open(os.path.join(REGD, 'tri_%s.json' % tag), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        log('=== region %s 种子 %d / tag %s（登记已写；剩余 %.0fs）==='
            % (reg['region'], sd, tag, deadline_t - time.time()))
        ev = run_one(ee, reg, g, tr, w, h, tag, sd, deadline_t)
        summary.append({k: ev[k] for k in ['exp_id', 'seed', 'evm_in_tile_valid_px', 'legacy_gap_px',
                                           'planar_gap_px', 'restored_px', 'common_valid_px',
                                           'common_valid_disagree_px', 'outside_wedge_agree_rate',
                                           'outside_tile_evm_valid_px', 'planar_row_nodata_max',
                                           'boundary_continuous', 'req_s', 'elapsed_s']})
    json.dump(dict(region=reg['region'], tile_id=reg['tile_id'], seed_set=seeds, limit_s=DEADLINE_S,
                   total_elapsed_s=round(time.time() - deadline_t + DEADLINE_S, 1), results=summary),
              open(os.path.join(OUTD, 'TRI_%s_summary.json' % reg['region']), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()

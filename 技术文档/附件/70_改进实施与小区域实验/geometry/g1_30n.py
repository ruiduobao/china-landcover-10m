# -*- coding: utf-8 -*-
"""g1_30n.py — G-30N 30°N 南缘楔形：legacy(测地) vs planar(平面) 同表达式小框真实分类对照

* 输入（只读）
    - 训练池资产 SRC = projects/leafy-environs-509005-n1/assets/prod2nd/smp2023v2_merged（第二轮冻结池，cl 24 码）
    - AEF 2023 GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL（64 维）
    - 参考网格   F:/lc_work/prod2nd_2023/rasters_batch/T1206_10m.tif（生产资产 affine，只读表头）
* 规则源
    - doc70 §4.4（平面 helper／实验驱动；legacy 与 planar 同模型）、§5 G-30N、§7 几何覆盖门
    - doc69 §6.4 + seams 报告 §8.1（EVM 金标准：AEF 全 64 波段共同有效掩膜，不先 clip，平面 region，固定 grid）
    - 生产表达式固定：c3_driver_prod2.py L211-224（filterBounds(box±2°) / randomColumn(seed).sort.limit(20000)
      / smileRandomForest(100, minLeaf=2, maxNodes=5000, seed=7) / aef.classify(clf).uint8().clip(box)）
* 门槛
    - **单请求 4 波段**（legacy, planar, evm, expmask）→ 同一次训练选取，两臂严格同模型；
    - 每请求 deadline ≤60s 且按剩余时间裁剪；不发 batch；父进程 1200s 硬截止（见 run_g1.py）；
    - EVM 分母限定 **平面 tile box 内**（expmask==1）；南侧 box 外行另单列。
* 输出（F:/lc_work/improve_2023_pilot_20261015/geometry/out/）
    - G30N_seed<seed>_4band.tif（uint8×4：legacy/planar/evm/expmask）
    - G30N_seed<seed>_legacy_10m.tif / _planar_10m.tif（单波段 + 调色板 + nodata=0）
    - G30N_seed<seed>_eval.json（逐项计数与一致率）
    - G30N_seed<seed>_train.tif（可选，--tiny 时的小窗连通测试）
* 用法
    python g1_30n.py --register-only            # 只登记（零 GEE）
    python g1_30n.py --tiny                     # 极小 computePixels 连通/时延测试（8x8 A00）
    python g1_30n.py --seed 7                   # 真实两臂对照（单请求 4 波段）
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
OUTD = os.path.join(HERE, 'out')
REGD = os.path.join(HERE, 'reg')
sys.path.insert(0, HERE)
sys.path.insert(0, r'F:/lc_work/v31_exp/code')
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/8.全国生产')
sys.path.insert(0, r'F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/9.第二次分类川粤宁')

import numpy as np
import geom_planar as GP

ACCT = 'k3r6ncvd'                       # 独占账号（h0_health.py 选定；ID 记录，不含密钥）
SRC = 'projects/leafy-environs-509005-n1/assets/prod2nd/smp2023v2_merged'
AEF_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
YEAR = 2023
REF_TIF = r'F:/lc_work/prod2nd_2023/rasters_batch/T1206_10m.tif'
TILE = 'T1206'
BOX = [97.0, 30.0, 99.0, 32.0]          # 生产瓦 box（clip 几何）
BOXP = [95.0, 28.0, 101.0, 34.0]        # 训练外扩框（box±2°，生产同式）
N_LIMIT = 20000
RF = dict(numberOfTrees=100, minLeafPopulation=2, maxNodes=5000)
SEEDS = [7, 11, 23]
# 小框（登记值）：南缘楔形中心列 × 含 30.0 接缝南侧 20 行
C0, R0, NCOL, NROW = 10854, 22192, 557, 93
DEADLINE_S = 1200.0
REQ_TIMEOUT_S = 60.0
# 24 类调色板（d_tiles.COLORS 同源）
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


CONFIG_FP = os.path.join(HERE, 'config', 'planar_g30n.json')
IMPL_VERSION = 'v2_unmask0'   # 掩膜物化实现；首件 G30N_seed7 为 v1_rawmask（VALIDITY_FAIL，原件保留）


def load_config(fp):
    """读取可执行工程配置（planar_geom_config/v1）并覆盖模块级登记值。

    配置是**已验收的工程配置**（geometry/config/planar_g30n.json）：
    平面 clip+region、materialize_mask=true、模型固定 v2 训练原几何不改、seeds[7,11,23]、deadline 1200。
    明确拒绝把未过门的 O-CAP / F-VIEW / T-STRATA 规则开进本配置。
    """
    global BOX, BOXP, C0, R0, NCOL, NROW, RF, N_LIMIT, SEEDS, DEADLINE_S, REF_TIF, SRC, AEF_ID, YEAR
    cfg = json.load(open(fp, encoding='utf-8'))
    assert cfg.get('schema') == 'planar_geom_config/v1', 'config schema 不符'
    ne = cfg.get('NOT_ENABLED', {})
    for k in ('O_CAP_orchard_cap', 'F_VIEW_forest_filter', 'T_STRATA_terrain_strata'):
        assert ne.get(k) is False, '配置不得启用未过门规则 %s' % k
    g = cfg['geometry']
    assert g['plane_clip'] is True and g['plane_region'] is True, 'plane_clip/plane_region 必须为 true'
    assert g['region_never_four_number_list'] is True, 'region 不得传 4 数字列表'
    assert cfg['materialize_mask'] is True, 'materialize_mask 必须为 true'
    assert cfg['grid']['use_explicit_crsTransform'] is True, '必须用显式 crsTransform'
    assert cfg['grid']['scale_must_be_null_when_crsTransform_set'] is True, 'scale 必须置空'
    assert g['train_filter_bounds_geometry_unchanged'] is True, '本配置不得改训练窗几何'
    REF_TIF = cfg['source_tile']['ref_tif_for_grid']
    BOX = [float(v) for v in cfg['source_tile']['tile_box']]
    BOXP = [float(v) for v in cfg['model']['train_window_box']]
    w = cfg['window']
    C0, R0, NCOL, NROW = int(w['tile_col0']), int(w['tile_row0']), int(w['ncol']), int(w['nrow'])
    SRC = cfg['model']['pool_asset']
    AEF_ID = cfg['model']['aef_collection']
    YEAR = int(cfg['model']['year'])
    N_LIMIT = int(cfg['model']['n_train_limit'])
    RF = dict(cfg['model']['rf'])
    SEEDS = [int(s) for s in cfg['seeds']]
    DEADLINE_S = float(cfg['deadline_s'])
    return cfg


def selfcheck(fp=CONFIG_FP):
    """配置→几何/网格自检：**无云端计算**（仅可选的 ee 客户端序列化）。

    校验：① 平面几何序列化 geodesic=false；② crsTransform 与 scale 互斥；③ 网格吸附对齐生产 affine；
          ④ 配置未启用未过门规则。
    """
    cfg = load_config(fp)
    g = grid()
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], C0, R0, NCOL, NROW)
    out = dict(config=fp, window_bbox=window_bbox(g), crsTransform=tr, width=w, height=h,
               wedge_h_deg=GP.wedge_height_deg(BOX), wedge_h_rows=GP.wedge_height_deg(BOX) / g['py'],
               seeds=SEEDS, deadline_s=DEADLINE_S, materialize_mask=True,
               xor_ok=GP.assert_scale_xor_transform(scale=None, crs_transform=tr),
               region_bbox_matches_config=(
                   max(abs(a - b) for a, b in zip(window_bbox(g), cfg['window']['region_bbox_deg'])) < 1e-9))
    try:
        import ee
        if not getattr(ee.data, '_credentials', None):
            # 仅初始化（元数据级，无计算请求）；失败则跳过序列化检查
            sys.path.insert(0, r'F:/lc_work/v31_exp/code')
            import v31_common as VC
            VC.ctx(ACCT)
        ser_planar = ee.serializer.encode(GP.planar_rect(BOX, ee=ee))
        ser_legacy = ee.serializer.encode(GP.legacy_rect(BOX, ee=ee))
        out['planar_serialized'] = json.dumps(ser_planar)[:220]
        out['planar_geodesic_false'] = ('"geodesic": {"constantValue": false}' in json.dumps(ser_planar))
        out['legacy_fn'] = 'GeometryConstructors.Polygon' if 'GeometryConstructors.Polygon' in json.dumps(ser_legacy) else '?'
        out['legacy_has_geodesic_key'] = ('geodesic' in json.dumps(ser_legacy))
        out['serialize_check'] = 'OK' if out['planar_geodesic_false'] else 'FAIL'
    except Exception as e:
        out['serialize_check'] = 'SKIPPED(ee 未初始化: %s)' % str(e)[:60]
    out['PASS'] = bool(out['xor_ok'] and out['region_bbox_matches_config'] and
                       out.get('serialize_check') in ('OK',) or out['serialize_check'].startswith('SKIPPED'))
    return out


def code_hash():
    h = hashlib.md5()
    for fp in [os.path.join(HERE, 'geom_planar.py'), os.path.abspath(__file__)]:
        h.update(open(fp, 'rb').read())
    return h.hexdigest()


def grid():
    px, py, x0, ytop, W, H = GP.grid_of_tif(REF_TIF)
    return dict(px=px, py=py, x0=x0, ytop=ytop, W=W, H=H)


def window_bbox(g):
    """小框经纬度（像元中心外扩半像元 → 覆盖完整像元）。"""
    xl = g['x0'] + C0 * g['px']
    xr = g['x0'] + (C0 + NCOL) * g['px']
    yt = g['ytop'] - R0 * g['py']
    yb = g['ytop'] - (R0 + NROW) * g['py']
    return [xl, yb, xr, yt]


def registration(tag, seed):
    g = grid()
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], C0, R0, NCOL, NROW)
    wh = GP.wedge_height_deg(BOX)
    return dict(
        schema='improve_pilot_spec/v1',
        exp_id='G30N_%s' % tag,
        impl_version=IMPL_VERSION,   # 当前实现恒为 v2_unmask0（掩膜物化）；v1_rawmask 仅历史首件 G30N_seed7
        owner='geometry',
        stage='S4',
        hypothesis='H6 几何修复仅恢复覆盖、不改变模型（planar clip 消除测地楔形 nodata，楔外类码一致）',
        province='川',
        region_bbox=window_bbox(g),
        region_tile_rc=dict(c0=C0, r0=R0, ncol=NCOL, nrow=NROW),
        tile_id=TILE,
        tile_box=BOX,
        training_bbox=BOXP,
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
                        'restored_px', 'outside_tile_consistent', 'wedge_outside_class_agree_rate'],
        expected_effect=('楔内(planar box 内、测地 arc 外) EVM==1 而 legacy==0 的像元数 >0（工程缺口）；'
                         'planar 恢复该区；楔外(非楔区) legacy==planar 类码一致率=1.0；'
                         '楔高解析预测 %.4f° (%.1f 行)' % (wh, wh / g['py'])),
        stop_rules=['单请求 >60s 无响应即按剩余时间重试，剩余 <30s 放弃',
                    '父进程 1200s 硬截止 → 记 TIMEOUT 并终止子进程',
                    '若 20k 训练交互失败 → 写 FAIL，另注册 2k 工程对照（不伪称原生产精度）'],
        code_hash=code_hash(),
        account_id=ACCT,
        registered_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
    )


def build_image(ee, seed):
    """5 波段：legacy / planar / legacy_mask / evm / expmask。同一分类表达式 → 两臂同模型、同训练选取。

    ⚠️ 掩膜物化（2026-10-15 探针 g0c_probe_aef 结论，**必须**）：
      `ee.data.computePixels`（GEO_TIFF）**不传递图像掩膜** —— 被 clip 掩掉的像元返回的是
      **底层数值**（probe_aef：clip_rectNum == noclip，逐值相同；`unmask(0)` 后 34,537 px 显 0）。
      因此类别波段一律 `.unmask(0, False)` 显式填 0（0=nodata），并另出 `legacy_mask` 波段交叉核对。
    """
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
    return (legacy.addBands(planar).addBands(lmask).addBands(evm).addBands(expmask))


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
        # photometric=PALETTE：否则部分读取器/QGIS 报告“无调色板”（ColorMap 标签在但 photo=1 MinIsBlack）
        prof['photometric'] = 'PALETTE'
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--register-only', action='store_true')
    ap.add_argument('--tiny', action='store_true')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--seeds', default=None, help='逗号分隔种子集合（doc70 预注册 {7,11,23}）；一次 harness 内串行')
    ap.add_argument('--tag', default=None, help='实验标签；默认 seed<seed>')
    ap.add_argument('--deadline', type=float, default=None)
    ap.add_argument('--config', default=CONFIG_FP, help='可执行工程配置（默认 config/planar_g30n.json）')
    ap.add_argument('--no-config', action='store_true', help='忽略配置，用脚本内登记值')
    ap.add_argument('--selfcheck', action='store_true', help='只做配置→几何/网格自检（无云端计算）')
    a = ap.parse_args()
    cfg = None
    if not a.no_config:
        cfg = load_config(a.config)
        log('已读配置 %s（plane_clip=%s materialize_mask=%s seeds=%s deadline=%.0fs）'
            % (os.path.basename(a.config), cfg['geometry']['plane_clip'], cfg['materialize_mask'],
               SEEDS, DEADLINE_S))
    if a.selfcheck:
        print(json.dumps(selfcheck(a.config if not a.no_config else CONFIG_FP),
                         ensure_ascii=False, indent=1))
        return
    if a.deadline is not None:
        globals()['DEADLINE_S'] = float(a.deadline)
    os.makedirs(OUTD, exist_ok=True)
    os.makedirs(REGD, exist_ok=True)
    deadline_t = time.time() + DEADLINE_S
    seeds = [int(x) for x in a.seeds.split(',')] if a.seeds else list(SEEDS)

    sys.path.insert(0, r'F:/lc_work/v31_exp/code')
    import v31_common as VC
    ee, pid = VC.ctx(ACCT)
    log('账号 %s / 项目 %s 就绪；种子集 %s' % (ACCT, pid, seeds))
    g = grid()
    tr, w, h = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], C0, R0, NCOL, NROW)
    GP.assert_scale_xor_transform(scale=None, crs_transform=tr)

    if a.register_only:
        for sd in seeds:
            tg = a.tag or ('seed%d' % sd)
            reg = registration(tg, sd)
            json.dump(reg, open(os.path.join(REGD, 'g30n_%s.json' % tg), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)
            log('登记 → reg/g30n_%s.json' % tg)
        return

    if a.tiny:
        # 极小连通/时延测试：AEF 单波段 A00，8x8 像元
        img = (ee.ImageCollection(AEF_ID).filterDate('%d-01-01' % YEAR, '%d-01-01' % (YEAR + 1))
               .mosaic().select(['A00']))
        tr8, w8, h8 = GP.snap_window(g['px'], g['py'], g['x0'], g['ytop'], C0 + 200, R0 - 20, 8, 8)
        t0 = time.time()
        b, dt = fetch(ee, img, tr8, w8, h8, deadline_t, tag='tiny-A00-8x8')
        p = os.path.join(OUTD, 'tiny_A00_8x8.tif')
        open(p, 'wb').write(b)
        import rasterio
        with rasterio.open(p) as ds:
            arr = ds.read()
        log('tiny 落盘 %s shape=%s min=%.3f max=%.3f' % (p, arr.shape, float(arr.min()), float(arr.max())))
        json.dump(dict(ok=True, dt_s=round(dt, 2), shape=list(arr.shape),
                       min=float(arr.min()), max=float(arr.max()),
                       total_s=round(time.time() - t0, 2), acct=ACCT, proj=pid),
                  open(os.path.join(OUTD, 'tiny_result.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        return

    summary = []
    for sd in seeds:
        tag = ('%s_s%d' % (a.tag, sd)) if a.tag else ('seed%d' % sd)
        reg = registration(tag, sd)
        json.dump(reg, open(os.path.join(REGD, 'g30n_%s.json' % tag), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        log('=== 种子 %d / tag %s（登记已写；剩余 %.0fs）===' % (sd, tag, deadline_t - time.time()))
        ev = run_one(ee, a, g, tr, w, h, reg, tag, sd, deadline_t)
        summary.append({k: ev[k] for k in ['exp_id', 'seed', 'evm_in_tile_valid_px', 'legacy_gap_px',
                                           'planar_gap_px', 'restored_px', 'wedge_restore_rate',
                                           'outside_wedge_agree_rate', 'outside_tile_evm_valid_px',
                                           'req_s', 'elapsed_s']})
    json.dump(dict(seed_set=seeds, limit_s=DEADLINE_S, total_elapsed_s=round(time.time() - deadline_t + DEADLINE_S, 1),
                   results=summary),
              open(os.path.join(OUTD, 'G30N_%s_summary.json' % (a.tag or 'seedset')), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def run_one(ee, a, g, tr, w, h, reg, tag, seed, deadline_t):
    """单个种子：构建 5 波段图 → 单请求 → 落 GeoTIFF → 评估。"""
    # ---- 真实两臂对照：单请求 5 波段（legacy/planar/legacy_mask/evm/expmask） ----
    log('构建 5 波段图（legacy/planar/legacy_mask/evm/expmask），tag=%s seed=%d …' % (tag, seed))
    t_build = time.time()
    img = build_image(ee, seed)
    log('图构建完成 %.1fs（惰性对象图，未触发计算）' % (time.time() - t_build))
    b, dt = fetch(ee, img, tr, w, h, deadline_t, tag='G30N-5band')
    fp4 = os.path.join(OUTD, 'G30N_%s_5band.tif' % tag)
    open(fp4 + '.raw', 'wb').write(b)
    import rasterio
    with rasterio.open(fp4 + '.raw') as ds:
        arr = ds.read()
        band_names = ds.descriptions
    log('5 波段回读 shape=%s dtype=%s names=%s' % (arr.shape, arr.dtype, band_names))
    legacy, planar, lmask, evm, expmask = arr[0], arr[1], arr[2], arr[3], arr[4]
    write_tif(fp4, arr, tr, w, h, 5)
    os.remove(fp4 + '.raw')
    pal = palette_map()
    write_tif(os.path.join(OUTD, 'G30N_%s_legacy_10m.tif' % tag), legacy, tr, w, h, 1, palette=pal)
    write_tif(os.path.join(OUTD, 'G30N_%s_planar_10m.tif' % tag), planar, tr, w, h, 1, palette=pal)
    log('GeoTIFF 落盘：5band + legacy + planar')

    # ---- 评估 ----
    rows = np.arange(R0, R0 + NROW)
    cols = np.arange(C0, C0 + NCOL)
    lon = (g['x0'] + (cols + 0.5) * g['px'])[None, :].repeat(NROW, 0)
    lat = (g['ytop'] - (rows + 0.5) * g['py'])[:, None].repeat(NCOL, 1)
    lens = GP.wedge_lens_mask(lon, lat, BOX)
    in_tile = expmask == 1
    denom = in_tile & (evm == 1)
    ev = dict(
        exp_id=reg['exp_id'], impl_version=reg['impl_version'], seed=seed, account_id=ACCT,
        region_bbox=reg['region_bbox'], n_px=int(NROW * NCOL),
        crsTransform=tr, width=w, height=h,
        wedge_h_deg=GP.wedge_height_deg(BOX),
        wedge_h_rows=GP.wedge_height_deg(BOX) / g['py'],
        wedge_lens_px=int(lens.sum()),
        wedge_lens_in_tile_px=int((lens & in_tile).sum()),
        evm_valid_px=int((evm == 1).sum()),
        evm_in_tile_valid_px=int(denom.sum()),
        legacy_mask0_px=int((lmask == 0).sum()),
        legacy_mask_eq_unmask0_px=int(((lmask == 0) == (legacy == 0)).sum()),
        # 工程缺口（分母=EVm∩平面 tile box）
        legacy_gap_px=int((denom & (legacy == 0)).sum()),
        planar_gap_px=int((denom & (planar == 0)).sum()),
        restored_px=int((denom & (legacy == 0) & (planar > 0)).sum()),
        legacy_nodata_px=int((in_tile & (legacy == 0)).sum()),
        planar_nodata_px=int((in_tile & (planar == 0)).sum()),
        # 楔外一致率（分母=平面 tile box 内、非解析楔区、两臂均有值）
        outside_wedge_both_px=int((in_tile & ~lens & (legacy > 0) & (planar > 0)).sum()),
        outside_wedge_disagree_px=int((in_tile & ~lens & (legacy > 0) & (planar > 0) & (legacy != planar)).sum()),
        # tile box 外（南侧对照行）：两臂应一致为 nodata；EVM 可能有值
        outside_tile_px=int((~in_tile).sum()),
        outside_tile_evm_valid_px=int(((~in_tile) & (evm == 1)).sum()),
        outside_tile_legacy_nonzero_px=int(((~in_tile) & (legacy > 0)).sum()),
        outside_tile_planar_nonzero_px=int(((~in_tile) & (planar > 0)).sum()),
        legacy_class_hist={int(k): int(v) for k, v in zip(*np.unique(legacy[in_tile & (legacy > 0)], return_counts=True))},
        planar_class_hist={int(k): int(v) for k, v in zip(*np.unique(planar[in_tile & (planar > 0)], return_counts=True))},
        req_s=round(dt, 2), elapsed_s=round(time.time() - deadline_t + DEADLINE_S, 1),
        code_hash=reg['code_hash'],
    )
    ev['outside_wedge_agree_rate'] = (1.0 - ev['outside_wedge_disagree_px'] /
                                      max(1, ev['outside_wedge_both_px']))
    ev['wedge_restore_rate'] = ev['restored_px'] / max(1, ev['legacy_gap_px'])
    json.dump(ev, open(os.path.join(OUTD, 'G30N_%s_eval.json' % tag), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('评估 → G30N_%s_eval.json' % tag)
    print(json.dumps({k: ev[k] for k in ['evm_in_tile_valid_px', 'legacy_gap_px', 'planar_gap_px',
                                         'restored_px', 'wedge_restore_rate', 'outside_wedge_agree_rate',
                                         'outside_tile_evm_valid_px', 'req_s', 'elapsed_s']},
                     ensure_ascii=False, indent=1))
    return ev


if __name__ == '__main__':
    main()

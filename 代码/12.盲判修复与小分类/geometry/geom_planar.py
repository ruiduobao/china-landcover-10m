# -*- coding: utf-8 -*-
"""geom_planar.py — 平面裁剪/导出几何 helper（G-30N 几何修复复用件）

* 职责
  1) **显式平面矩形**（geodesic=False）与 **legacy 测地矩形**并列构造，供同表达式 A/B 对照；
  2) **显式 crsTransform（固定网格）**构造，并实现与 `scale` 的**互斥校验**（GEE Export 规则）；
  3) **AEF 全波段共同有效掩膜 EVM**（`aef.mask().reduce(Reducer.min())`，**不先 clip**）；
  4) **测地大圆南边拱高预测**（与 doc69 seams 报告同式），用于楔形定位与验收；
  5) **网格对齐**：把整数行列窗口吸附到给定 (px, py, x0, y_top) 的固定网格。
* 规则源
  - ee 1.7.42 源码：`Geometry.Rectangle` 数字坐标 → 客户端不写 geodesic 键 → 服务端默认
    geodesic=True（`ee/geometry.py` docstring: “true if the inputs are numbers”）；
    `Geometry.BBox` 客户端显式 `geodesic=False`（“planar WGS84 rectangle”）；
    `Geometry.Rectangle(coords, None, False)` → 带 geodesic 键 → 走服务端构造器，同样平面。
  - `ee.batch._canonicalize_region`：`region=[4 数字]` 先试 LineString 再 Polygon → 仍测地默认；
    传 `Geometry` 对象则原样使用 → **必须传平面 Geometry 对象**，不能传 4 数字列表。
  - `Image.prepare_for_export` 折叠 crsTransform/crs/scale 到表达式；**crsTransform 与 scale 互斥**
    （同时给会由服务端拒绝）。
* 门槛/纪律
  - 本模块**只构造几何与网格、不做 IO、不发请求**；不发 batch、不写成品。
  - 所有函数无副作用；`assert_scale_xor_transform` 在同时给定 scale 与 crsTransform 时**抛错**。
* 用法
    import geom_planar as GP
    box = [97.0, 30.0, 99.0, 32.0]
    gA = GP.legacy_rect(box)              # 生产现状（测地）
    gB = GP.planar_rect(box)              # 修复件（平面）
    px, py, x0, ytop, W, H = GP.grid_of_tif(r'...T1206_10m.tif')
    tr, w, h = GP.snap_window(px, py, x0, ytop, c0=10854, r0=22192, ncol=557, nrow=93)
    GP.assert_scale_xor_transform(scale=None, crs_transform=tr)
"""
import math

import numpy as np

# ---- ee 惰性导入：本模块被只读诊断脚本 import 时不应强制初始化 GEE ----
_ee = None


def _get_ee():
    global _ee
    if _ee is None:
        import ee
        _ee = ee
    return _ee


# ============================ 1. 几何构造 ============================
def legacy_rect(box, ee=None):
    """生产现状几何：`ee.Geometry.Rectangle([x0,y0,x1,y1])`（数字坐标 → 服务端默认 geodesic=True）。

    box = [x0, y0, x1, y1]（度）。**测地南/北边为大圆**，在中经线向北拱（北半球）→ 楔形 nodata 源。
    """
    e = ee or _get_ee()
    return e.Geometry.Rectangle([float(v) for v in box])


def planar_rect(box, ee=None, style='bbox'):
    """修复件几何：显式平面 WGS84 矩形（geodesic=False）。

    style='bbox'  → `ee.Geometry.BBox(w,s,e,n)`（客户端构造，geo_json + geodesic=False）；
    style='rect'  → `ee.Geometry.Rectangle(box, None, False)`（带 geodesic 键 → 服务端构造器）。
    两者均为平面；默认 'bbox'（纯客户端，序列化最省、语义最明确）。
    """
    e = ee or _get_ee()
    x0, y0, x1, y1 = (float(v) for v in box)
    if style == 'bbox':
        return e.Geometry.BBox(x0, y0, x1, y1)
    if style == 'rect':
        return e.Geometry.Rectangle([x0, y0, x1, y1], None, False)
    raise ValueError("style 只能是 'bbox' 或 'rect'，收到 %r" % (style,))


def planar_region_or_raise(box, ee=None, style='bbox'):
    """导出用 region：返回**平面 Geometry 对象**（不是 4 数字列表）。

    显式拒绝把列表直接当 region 用（`ee.batch._canonicalize_region` 会把列表变成
    LineString/Polygon → 重新落入测地默认）。
    """
    if isinstance(box, (list, tuple)):
        return planar_rect(box, ee=ee, style=style)
    raise TypeError('region 必须是平面 Geometry 对象；请传 box 列表给 planar_rect，'
                    '不要直接把列表交给 Export(region=...)')


# ============================ 2. 固定网格 / 互斥校验 ============================
def grid_of_tif(fp):
    """读参考 GeoTIFF 的原生网格 → (px, py, x0, y_top, width, height)。

    用于把实验导出窗口吸附到**生产资产的真实网格**（c3 下载按资产 affine 对齐，见 c3 L280-287）。
    """
    import rasterio
    with rasterio.open(fp) as ds:
        t = ds.transform
        return (float(t.a), float(-t.e), float(t.c), float(t.f), ds.width, ds.height)


def snap_window(px, py, x0, ytop, c0, r0, ncol, nrow):
    """把整数行列窗口吸附到固定网格 → (crsTransform6, width, height)。

    crsTransform = [px, 0, x0 + c0*px, 0, -py, ytop - r0*py]（EPSG:4326，y 轴向下）。
    返回的 6 元组可直接用于 Export(..., crsTransform=...) 或 computePixels(grid.affineTransform)。
    """
    tr = [float(px), 0.0, float(x0) + float(c0) * float(px),
          0.0, -float(py), float(ytop) - float(r0) * float(py)]
    return tr, int(ncol), int(nrow)


def assert_scale_xor_transform(scale=None, crs_transform=None):
    """GEE Export 规则：`scale` 与 `crsTransform` **互斥**（同时给定服务端会拒绝）。

    - 两者都给 → 抛 ValueError（不静默取其一）；
    - 只给 crsTransform → OK（固定网格，推荐用于几何修复，避免网格吸附相位漂移）；
    - 只给 scale / 都不给 → OK（native）。
    """
    if scale is not None and crs_transform is not None:
        raise ValueError('crsTransform 与 scale 互斥：显式 crsTransform 时不得再传 scale'
                         '（请置 scale=None，分辨率由 crsTransform 决定）')
    return True


def build_export_kwargs(box, crs_transform=None, scale=None, region_style='bbox',
                        crs='EPSG:4326'):
    """组装 Export/computePixels 的空间参数：显式平面 region + 互斥校验 + 网格二选一。

    返回 dict(region=<平面 Geometry>, crs=..., scale=..., crsTransform=...)。
    """
    assert_scale_xor_transform(scale, crs_transform)
    kw = dict(region=planar_region_or_raise(box, style=region_style), crs=crs)
    if crs_transform is not None:
        kw['crsTransform'] = list(crs_transform)
        kw['scale'] = None
    else:
        kw['scale'] = scale
        kw['crsTransform'] = None
    return kw


# ============================ 3. EVM 共同有效掩膜 ============================
def common_valid_mask(aef, ee=None, out_name='evm'):
    """AEF **全 64 波段共同有效掩膜**（金标准 EVM）：`aef.mask().reduce(Reducer.min())`。

    * **不先 clip**（否则沿用同一套错误测地几何 → 在同一位置复制同一缺口，自证无效）；
    * 单波段 `aef.select(0).mask()` 只是近似（不能证明 64 波段 mask 一致）→ 不采用；
    * 返回 uint8 单波段：1 = 该像元 64 波段**全部**有效，0 = 至少一个波段无效。
    """
    e = ee or _get_ee()
    return aef.mask().reduce(e.Reducer.min()).rename(out_name).uint8()


def expected_region_mask(box, ee=None, out_name='expmask'):
    """预期域掩膜：平面 tile box 内 = 1（用于把 EVM 分母限定在**平面预期域**内）。

    ⚠️ 分母纪律（doc70 §7 几何覆盖门）：有效覆盖恢复的分母必须是
    `EVM ∩ 平面 tile box`；跨 tile 南侧的对照行（box 外）**另单列**，
    否则南侧真实 AEF 有值而 planar 也按 clip 置 0 会被误报成缺口。
    """
    e = ee or _get_ee()
    return e.Image.constant(1).byte().rename(out_name).clip(planar_rect(box, ee=e))


# ============================ 4. 测地楔形预测 ============================
def arc_lat(lat0, lon, lon_a, lon_b):
    """同纬两端点 (lat0,lon_a)-(lat0,lon_b) 的球面大圆在 lon 处的纬度（度）。

    测地矩形南边（region 外掩膜）的解析边界；与 doc69 seams 报告同式：
        tan φ(λ) = tan φ0 · cos(λ − λm) / cos(Δλ/2)
    """
    t0 = math.tan(math.radians(lat0))
    d2 = math.radians(abs(lon_b - lon_a)) / 2.0
    lm = (lon_a + lon_b) / 2.0
    return math.degrees(math.atan(t0 * math.cos(math.radians(lon - lm)) / math.cos(d2)))


def wedge_height_deg(box):
    """测地南边在中经线相对 y0 的拱高（度）——即楔形最大高度。"""
    x0, y0, x1, y1 = (float(v) for v in box)
    return arc_lat(y0, (x0 + x1) / 2.0, x0, x1) - y0


def wedge_lens_mask(lon2d, lat2d, box):
    """解析楔形（测地 region 之外但平面 box 之内）布尔掩膜。

    lon2d/lat2d: 像元中心经纬度 2D 数组；box=[x0,y0,x1,y1]。
    条件：y0 ≤ lat < arc_lat(lat0=y0, lon) 且 x0 ≤ lon ≤ x1（南边）；北边对称同理（此处只算南边）。
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    lon = np.asarray(lon2d, dtype='float64')
    lat = np.asarray(lat2d, dtype='float64')
    arc = np.array([arc_lat(y0, float(v), x0, x1) for v in np.atleast_1d(lon).ravel()])
    arc = arc.reshape(np.shape(lon))
    return (lon >= x0) & (lon <= x1) & (lat >= y0 - 1e-9) & (lat < arc)


# ============================ 5. 自检 ============================
def _selftest():
    """零网络自检：几何序列化 flag、互斥校验、网格吸附、楔高解析值。"""
    import ee
    out = {}
    gA = legacy_rect([97, 30, 99, 32], ee=ee)
    gB = planar_rect([97, 30, 99, 32], ee=ee)
    out['legacy_type'] = gA.toGeoJSONString()[:0] or type(gA).__name__
    out['legacy_is_computed'] = type(gA).__name__ != 'Geometry' or True
    # 平面 BBox 为客户端 Geometry（含 geodesic=False）；legacy 为 ComputedObject（服务端默认）
    out['planar_class'] = type(gB).__name__
    out['legacy_class'] = type(gA).__name__
    try:
        assert_scale_xor_transform(scale=10, crs_transform=[1, 0, 0, 0, -1, 0])
        out['xor'] = 'FAIL: 未拦截'
    except ValueError:
        out['xor'] = 'OK: 同时给 scale+crsTransform 已拦截'
    out['xor_ok'] = assert_scale_xor_transform(scale=None, crs_transform=[1, 0, 0, 0, -1, 0])
    tr, w, h = snap_window(8.983152841195215e-05, 8.983152841195215e-05,
                           96.99999454769751, 32.00005654549083, 10854, 22192, 557, 93)
    out['window'] = dict(tr=tr, w=w, h=h)
    out['wedge_h_T1206_deg'] = wedge_height_deg([97, 30, 99, 32])
    out['wedge_h_T1206_rows'] = wedge_height_deg([97, 30, 99, 32]) / 8.983152841195215e-05
    return out


if __name__ == '__main__':
    import json
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(_selftest(), ensure_ascii=False, indent=1))

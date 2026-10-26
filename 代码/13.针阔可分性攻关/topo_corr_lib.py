# -*- coding: utf-8 -*-
"""topo_corr_lib.py — 地形辐射校正模块（GEE 端 + 本地端对称实现）
* 目的  : 抑制"针叶林按阴阳坡分裂"伪影——同一地物在阴/阳坡的辐亮度差被模型误读为类别差。
* 输入  : S2 SR 反射率（B2/B3/B4/B8/B11/B12）+ DEM 派生的坡度/坡向/入射角
* 规则源: AGENTS.md 硬约束（只读数据 / 不改生产配置）；
          坡向一律用 **ESRI 分支**（历史曾因公式错误导致阴坡配对为 0）
* 门槛  : ① 只对**物理反射率**做校正，**禁止对 AEF64 嵌入逐维做 C/Minnaert**（语义不对）
          ② 坡向 ESRI：a = 90 − aspect  （aspect>90 时 a = 450 − aspect），再取余 360
          ③ cos(i) 需 clamp（≥0.05）避免除零与爆炸
* 输出  : 校正后波段 / 指数；C 校正参数 c、Minnaert k 落 JSON 供复现
* 用法  : from topo_corr_lib import (gee_terrain, gee_c_correction, gee_scsc,
                                     gee_minnaert, gee_indices,
                                     local_terrain, local_C, local_SCS_C, local_Minnaert)

═══════════════════════════════════════════════════════════════════════════
 【坡向 ESRI 分支 — 强制约定，勿改】
   GEE ee.Terrain.aspect() 与 GDAL DEMProcessing(aspect) 默认均为 ArcGIS(ESRI) 约定：
       a = 90 − aspect_raw ；若 aspect_raw > 90 则 a = 450 − aspect_raw
   即 0°=北, 90°=东, 180°=南, 270°=西，**顺时针**。
   本地端用 numpy 的梯度算法算出的 aspect 是**数学约定（逆时针，0°=东）**，
   必须经 raw_esri = (450 − 360 − math_aspect_deg) % 360 转换后才能与 GEE 对齐。
   本模块本地端已内置该转换（ESRI_BRANCH=True 时）。
═══════════════════════════════════════════════════════════════════════════

【四种校正的物理语义】
 1) C 校正 (Teillet 1982)：统计经验法。对每个波段，用 "反射率 L vs cos(i)" 的
    线性回归 L = a + b·cos(i)，取 c = a/b，则
        Lc = L · (cos(z) + c) / (cos(i) + c)
    → 消除了随 cos(i) 线性变化的乘性+加性地形效应；是唯一能同时去掉加性偏置的。
 2) SCS+C (Soenen 2005)：把 C 校正的 cos(i) 换成 **地表法线方向的入射角余弦**
    cos(i_scs) = cos(s)*cos(z) + sin(s)*sin(z)*cos(φ_a − φ_z)
    并保留 C 的分母项；对**陡坡森林冠层**物理上更正确（冠层是竖直结构，
    应按"冠层表面"而非"水平地面"受光）。
 3) Minnaert：L_m = L · (cos(z) / cos(i))^k，k 由回归
    ln(L) = ln(L0) + k·ln(cos(i)) + (k−1)·ln(cos(z)) 定出。k=1 退化为余弦校正。
 4) 余弦（除 cos i）：L_cos = L · cos(z)/cos(i)。作为基线对照，已知会过校正暗区。

【"好校正"的判据（本轮预注册）】
   (a) 同一林型内，阴坡 vs 阳坡的指数差异**缩小**（|Δmean|↓、方差↓、Cohen's d↓）
   (b) 林型间（ENF vs EBF、常绿 vs 落叶）可分性**不降低**（AUC 不降，最好升）
"""
import numpy as np

ESRI_BRANCH = True          # 坡向约定：True = ESRI/ArcGIS（与 ee.Terrain.aspect 一致）

# ──────────────────────────────── 常量 ────────────────────────────────
COS_CLAMP = 0.05            # cos(i) 下限，防除零/爆炸
S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
S2_SCALE = 10000.0          # S2_SR_HARMONIZED 反射率定标


# ══════════════════════════════════════════════════════════════════════
#                           GEE 端（ee 惰性图）
# ══════════════════════════════════════════════════════════════════════
def gee_terrain(dem, scale=30, esri=True):
    """DEM → slope(deg) / aspect(ESRI, deg, 0=北顺时针) / hillshade。返回 Image。"""
    import ee
    terr = ee.Terrain.products(dem)          # aspect 已按 ESRI 约定
    out = terr.select(['slope', 'aspect'])
    if not esri:
        # 转成数学约定（0=东，逆时针）：math = (450 − esri) % 360
        asp = terr.select('aspect')
        out = out.addBands(asp.expression('(450 - a) % 360', {'a': asp}).rename('aspect_math')) \
                 .select(['slope', 'aspect_math']).rename(['slope', 'aspect'])
    return out


def gee_illumination(dem, solar_z, solar_az, scale=30):
    """坡度/坡向 → cos(i)。solar_z/az 为太阳天顶角(deg)/方位角(deg, ESRI=从北顺时针)。
    cos(i) = cos(s)cos(z) + sin(s)sin(z)cos(az − a)    —— 注意这里 a 是**坡向**，
    与 SCS 的 cos(i_scs) 形式相同（水平地面假设下它就是标准入射角余弦）。"""
    import ee
    terr = gee_terrain(dem, scale=scale, esri=True)
    rad = lambda x: x.multiply(np.pi).divide(180.0)
    s = rad(terr.select('slope'))
    z = rad(ee.Image.constant(solar_z))
    a = rad(terr.select('aspect'))
    az = rad(ee.Image.constant(solar_az))
    cosi = (s.cos().multiply(z.cos())
            .add(s.sin().multiply(z.sin()).multiply(az.subtract(a).cos())))
    return terr.addBands(cosi.rename('cosi').clamp(COS_CLAMP, 1.0))


def _gee_fit(feat_img, band, solar_z, scale, region, tileScale=2):
    """在 region 内对 (band, cosi) 做一元线性回归 band = a + b·cosi，返回 (a, b)。"""
    import ee
    fit = feat_img.select(['cosi', band]).reduceRegion(
        ee.Reducer.linearFit(), region, scale, bestEffort=True, tileScale=tileScale)
    return ee.Number(fit.get('offset')), ee.Number(fit.get('scale'))


def gee_c_correction(img, terr, solar_z, region, bands=S2_BANDS, scale=30,
                     tileScale=2, return_params=False):
    """C 校正（Teillet）。img: 反射率 Image（0-1 或 0-10000 均可，线性缩放不影响 c）。
    terr: gee_illumination 的输出（含 cosi）。逐波段拟合 c=a/b。"""
    import ee
    cosz = np.cos(np.deg2rad(solar_z))
    out = img
    params = {}
    for b in bands:
        a, bb = _gee_fit(terr.addBands(img.select(b)), b, solar_z, scale, region, tileScale)
        c = a.divide(bb)                      # c = a / b
        params[b] = dict(a=None, b=None, c=None)
        corrected = (img.select(b).multiply(cosz + c)).divide(terr.select('cosi').add(c))
        out = out.addBands(corrected.rename(b + '_C').float())
        if return_params:
            params[b] = dict(a=a, b=bb, c=c)
    out = out.select([b + '_C' for b in bands])
    return (out, params) if return_params else out


def gee_scsc(img, terr, solar_z, region, bands=S2_BANDS, scale=30,
             tileScale=2, return_params=False):
    """SCS+C（Soenen et al. 2005）。
    cos(i_scs) = cos(s)·cos(z) + sin(s)·sin(z)·cos(φ_a − φ_z)  —— 与水平地面 cos(i) 同式，
    但 C 项分母用 (cos(i_scs) + c)，且**乘性项保留 cos(s)·cos(z)**（SCS 的关键：
    不对水平面归一化，而是对冠层朝向归一化）。"""
    import ee
    rad = lambda x: x.multiply(np.pi).divide(180.0)
    s = rad(terr.select('slope'))
    z = rad(ee.Image.constant(solar_z))
    cosz = np.cos(np.deg2rad(solar_z))
    # SCS 归一化项：与水平 cos(z) 的比值上再乘 cos(s)
    scs_norm = s.cos().multiply(z.cos())      # cos(s)·cos(z)
    out = img
    params = {}
    for b in bands:
        a, bb = _gee_fit(terr.addBands(img.select(b)), b, solar_z, scale, region, tileScale)
        c = a.divide(bb)
        corrected = (img.select(b).multiply(scs_norm.add(c))).divide(
            terr.select('cosi').add(c))
        out = out.addBands(corrected.rename(b + '_SCSC').float())
        if return_params:
            params[b] = dict(a=a, b=bb, c=c)
    out = out.select([b + '_SCSC' for b in bands])
    return (out, params) if return_params else out


def gee_minnaert(img, terr, solar_z, region, bands=S2_BANDS, scale=30,
                 tileScale=2, return_params=False):
    """Minnaert 校正。对 ln(L) = ln(L0) + k·ln(cosi) 回归定 k（cos(z) 在区内近似常数，
    并入截距）。L_m = L·(cos(z)/cos(i))^k。"""
    import ee
    cosz = np.cos(np.deg2rad(solar_z))
    logimg = img.select(bands).log().rename([b + '_ln' for b in bands])
    logcosi = terr.select('cosi').log().rename('lncosi')
    feat = logcosi.addBands(logimg)
    out = img
    params = {}
    for b in bands:
        a, bb = _gee_fit(feat, b + '_ln', solar_z, scale, region, tileScale)
        # bb 即 k
        k = bb
        corrected = (img.select(b).multiply(
            terr.select('cosi').divide(cosz).pow(k.multiply(-1))))
        out = out.addBands(corrected.rename(b + '_MIN').float())
        if return_params:
            params[b] = dict(k=k, lnL0=a)
    out = out.select([b + '_MIN' for b in bands])
    return (out, params) if return_params else out


def gee_cosine(img, terr, solar_z, bands=S2_BANDS):
    """余弦校正（基线对照，已知在暗区过校正）。"""
    import ee
    cosz = np.cos(np.deg2rad(solar_z))
    out = img.select(bands).divide(terr.select('cosi')).multiply(cosz)
    return out.rename([b + '_COS' for b in bands])


def gee_indices(stack, suf, bands=S2_BANDS):
    """从（校正后）反射率重建光学指数。stack 的波段名形如 B4{suf}。
    返回带 {suf} 后缀的 NDVI / NBR / NDMI / EVI / NDVIre。
    EVI 用 S2 无蓝波段修正版（这里 B2 可用，故用标准 3 波段 EVI 形式）。"""
    import ee
    g = lambda b: stack.select(b + suf)
    ndvi = g('B8').subtract(g('B4')).divide(g('B8').add(g('B4'))).rename('NDVI' + suf)
    nbr = g('B8').subtract(g('B12')).divide(g('B8').add(g('B12'))).rename('NBR' + suf)
    ndmi = g('B8').subtract(g('B11')).divide(g('B8').add(g('B11'))).rename('NDMI' + suf)
    # EVI = 2.5*(NIR-R)/(NIR + 6R - 7.5B + 1)
    evi = (g('B8').subtract(g('B4')).multiply(2.5)).divide(
        g('B8').add(g('B4').multiply(6)).subtract(g('B2').multiply(7.5)).add(1.0)
    ).rename('EVI' + suf)
    # NDVIre（红边替代：S2 无 B5 时用 B8A 不可得，改用 (B8-B4)/(B8+B4) 的变体跳过）
    return ee.Image.cat([ndvi, nbr, ndmi, evi])


# ══════════════════════════════════════════════════════════════════════
#                          本地端（numpy，与 GEE 对称）
# ══════════════════════════════════════════════════════════════════════
def local_terrain(dem, res_m, esri=ESRI_BRANCH):
    """DEM (2D array, m) + 像元尺寸(m) → slope(deg), aspect(deg)。
    **坡向 ESRI 分支**：先算数学约定（0=东逆时针），再 raw_esri=(450−360−math)%360。
    边缘用中心差分丢失 → 返回 NaN。"""
    dzdy, dzdx = np.gradient(dem.astype('float64'), res_m)   # axis0=y(北), axis1=x(东)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # 数学约定（ESRI/ArcGIS 文档中的 "aspect" 原始公式）：0=东，逆时针
    math_asp = np.degrees(np.arctan2(dzdy, -dzdx)) % 360.0
    if esri:
        # ESRI: a = 90 − math 的等价转换；历史错法 = 直接 90−math（会在 math>90 时出错）
        # 正确分支（与 ee.Terrain.aspect 一致）：
        a = np.where(math_asp > 90.0, 450.0 - math_asp, 90.0 - math_asp) % 360.0
        # 注意：math>90 分支下 450−math ∈ (90,360)；math≤90 时 90−math ∈ [0,90)。与 ESRI 一致。
    else:
        a = math_asp
    out_slope = slope.copy()
    # 平坦处坡向无意义 → NaN
    a = np.where(slope < 1e-6, np.nan, a)
    return out_slope, a


def cos_incidence(slope_deg, aspect_esri_deg, solar_z_deg, solar_az_deg):
    """cos(i)：水平地面入射角余弦。aspect/solar_az 均为 ESRI 约定（0=北，顺时针）。"""
    s = np.deg2rad(slope_deg); z = np.deg2rad(solar_z_deg)
    a = np.deg2rad(aspect_esri_deg); az = np.deg2rad(solar_az_deg)
    ci = np.cos(s) * np.cos(z) + np.sin(s) * np.sin(z) * np.cos(az - a)
    return np.clip(ci, COS_CLAMP, 1.0)


def _ols(x, y):
    """一元最小二乘 y = a + b·x，返回 (a, b)。忽略 NaN。"""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return 0.0, 1.0, False
    xm, ym = x[m], y[m]
    xc = xm - xm.mean(); yc = ym - ym.mean()
    den = (xc * xc).sum()
    if den <= 1e-12:
        return float(ym.mean()), 0.0, False
    b = float((xc * yc).sum() / den)
    a = float(ym.mean() - b * xm.mean())
    return a, b, True


def fit_c(L, cosi, mask=None):
    """C 校正参数：回归 L = a + b·cos(i) → c = a/b。返回 dict(a,b,c,n,ok)。"""
    x, y = np.asarray(cosi, float), np.asarray(L, float)
    if mask is not None:
        x, y = x[mask], y[mask]
    a, b, ok = _ols(x, y)
    c = a / b if abs(b) > 1e-12 else np.nan
    return dict(a=a, b=b, c=c, ok=ok, n=int(np.isfinite(x).sum()))


def local_C(L, cosi, c, cosz):
    """C 校正：Lc = L·(cos z + c)/(cos i + c)"""
    return L * (cosz + c) / (np.asarray(cosi, float) + c)


def local_SCS_C(L, cosi, c, cosz, slope_deg):
    """SCS+C：L_scs = L·(cos s·cos z + c)/(cos i + c)"""
    return L * (np.cos(np.deg2rad(slope_deg)) * cosz + c) / (np.asarray(cosi, float) + c)


def local_Minnaert(L, cosi, k, cosz):
    """Minnaert：L_m = L·(cos z / cos i)^k"""
    return L * np.power(np.asarray(cosi, float) / cosz, -k)


def local_cosine(L, cosi, cosz):
    """余弦校正（基线）"""
    return L * cosz / np.asarray(cosi, float)


def fit_minnaert_k(L, cosi, mask=None, eps=1e-4):
    """Minnaert k：回归 ln(L) = ln(L0) + k·ln(cos i)。要求 L>0。"""
    x, y = np.asarray(cosi, float), np.asarray(L, float)
    if mask is not None:
        x, y = x[mask], y[mask]
    m = np.isfinite(x) & np.isfinite(y) & (y > eps) & (x > eps)
    if m.sum() < 30:
        return dict(k=1.0, lnL0=np.nan, ok=False, n=0)
    a, b, ok = _ols(np.log(x[m]), np.log(y[m]))
    return dict(k=float(b), lnL0=a, ok=ok, n=int(m.sum()))


def local_indices(B, suffix=''):
    """B: dict(B2/B3/B4/B8/B11/B12 数组) → dict(NDVI/NBR/NDMI/EVI)"""
    b2, b4, b8, b11, b12 = (B['B2'], B['B4'], B['B8'], B['B11'], B['B12'])
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (b8 - b4) / (b8 + b4)
        nbr = (b8 - b12) / (b8 + b12)
        ndmi = (b8 - b11) / (b8 + b11)
        evi = 2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b2 + 1.0)
    return {'NDVI' + suffix: ndvi, 'NBR' + suffix: nbr,
            'NDMI' + suffix: ndmi, 'EVI' + suffix: evi}


# ───────────────────────── 阴/阳坡划分（ESRI 坡向） ─────────────────────────
def aspect_class(aspect_esri_deg, slope_deg, slope_min=5.0):
    """按 ESRI 坡向（0=北,90=东,180=南,270=西,顺时针）划分阴阳坡。
    北半球：阳坡 = 偏南（含 SE/S/SW）→ aspect ∈ [90, 270)；阴坡 = 偏北 → [270,360)∪[0,90)
    返回: 1=阳坡(南向), 0=阴坡(北向), NaN=平坦(<slope_min) 或无效。
    注：南半球（lat<0）相反——但本项目样点全在中国（北半球），暂不分支。"""
    a = np.asarray(aspect_esri_deg, float)
    s = np.asarray(slope_deg, float)
    cls = np.where((a >= 90) & (a < 270), 1.0, 0.0)
    cls = np.where(np.isfinite(a) & (s >= slope_min), cls, np.nan)
    return cls


def aspect_index(aspect_esri_deg):
    """连续"朝南度" = cos(aspect − 180°)，∈[−1,1]。1=正南, −1=正北。供回归/协变量用。"""
    return np.cos(np.deg2rad(np.asarray(aspect_esri_deg, float) - 180.0))


def cohens_d(x, y):
    """Cohen's d（合并标准差）。自动忽略 NaN。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp < 1e-15:
        return np.nan
    return float((x.mean() - y.mean()) / sp)

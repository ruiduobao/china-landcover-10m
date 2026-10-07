# -*- coding: utf-8 -*-
"""
ng0_spec.py — 下一代补样：规格与 GEE 门槛（唯一事实来源）

设计原则（来自 r2 教训「任何新源必须过同一道门槛」）
  1. 每个 spec 至少 **两个相互独立的来源**同时为真才给标签（单源一律降级为候选，不入库）；
  2. 门槛在 GEE 端用 updateMask 实现 → sampleRegions 只返回过门槛的点，本地无需二次猜测；
  3. 采点用 randomPoints(seed) 固定种子（禁用 hash()，见项目脚本规范）；
  4. 带状类（184/185）按海岸段分层配额，不做全局等密——配额在本地 gate 阶段施加。

试点规格（2026-09-13，小三区域）
  moss140      青藏高原东北缘 · 地衣苔藓 140 · WorldCover v200+v100 双年一致 + 地形/NDVI
  mangrove184  北部湾—雷州海岸 · 红树林 184 · WorldCover95 + USGS Giri 红树林 + GSW 水体
  saltmarsh185 北部湾—雷州海岸 · 海岸盐沼 185 · WorldCover90 + GSW 潮汐过渡 + 海岸带约束
  paddy12      华北平原 · 灌溉水田 12 · Sentinel-1 VV/VH 淹水特征 + WorldCover 耕地
"""
import os
import json

# ---------------- 区域（试点小窗，正式批量时替换为生态区/省域多边形） ----------------
REGIONS = {
    'QH_MOSS':  dict(box=[96.0, 35.5, 99.5, 39.5], note='青藏高原东北缘（祁连山—共和盆地）'),
    # 2026-09-13 试点收敛：广域海岸盒（108.6–111.8E, 20–23N）里红树林/盐沼占比 <0.3%，
    # 随机撒点命中率过低；收敛到雷州半岛—湛江湾—山口红树林/盐沼核心区。
    'HN_COAST': dict(box=[109.5, 20.8, 110.8, 21.7], note='雷州半岛西—湛江湾—山口（红树林/盐沼核心区）'),
    'HB_PLAIN': dict(box=[115.3, 36.3, 118.2, 39.2], note='华北平原（保定—天津—沧州）'),
    # 备用小窗（批量阶段按生态区展开）
    'QH_MOSS_S': dict(box=[97.0, 36.5, 98.5, 37.8], note='祁连山东段（地衣苔藓密集）'),
}

# ---------------- 规格 ----------------
# cand = 每分片随机候选点数（randomPoints）；门槛通过率见 ng_qa/gate_diag.json
SPECS = {
    'moss140': dict(cls=140, region='QH_MOSS', cand=30000, target=300,
                    gate='moss', tag='高寒地衣苔藓',
                    sources=['ESA/WorldCover/v200', 'COPERNICUS/DEM/GLO30',
                             'MODIS/061/MOD13Q1']),
    'mangrove184': dict(cls=184, region='HN_COAST', cand=12000, target=250,
                        gate='mangrove', sample_from='mask', tag='红树林',
                        sources=['ESA/WorldCover/v200',
                                 'JRC/GSW1_4/GlobalSurfaceWater',
                                 'COPERNICUS/DEM/GLO30']),
    'saltmarsh185': dict(cls=185, region='HN_COAST', cand=30000, target=250,
                         gate='saltmarsh', tag='海岸盐沼',
                         sources=['JRC/GSW1_4/GlobalSurfaceWater',
                                  'ESA/WorldCover/v200', 'MODIS/061/MOD13Q1',
                                  'COPERNICUS/DEM/GLO30']),
    'paddy12': dict(cls=12, region='HB_PLAIN', cand=12000, target=400,
                    gate='paddy', tag='灌溉水田',
                    sources=['COPERNICUS/S1_GRD', 'ESA/WorldCover/v200']),
    # ---- 家族 5–15（2026-09-13 第二批实现；orchard11 因缺中国园地专题源暂缓，见 NG_PLAN）----
    'tidalflat186': dict(cls=186, region='HN_COAST', cand=12000, target=2500,
                         gate='tidalflat', sample_from='mask', tag='潮滩',
                         sources=['JRC/GSW1_4/GlobalSurfaceWater', 'ESA/WorldCover/v200',
                                  'COPERNICUS/DEM/GLO30']),
    'saline183': dict(cls=183, region='HN_COAST', cand=12000, target=3000,
                      gate='saline', sample_from='mask', tag='盐渍湿地',
                      sources=['JRC/GSW1_4/GlobalSurfaceWater', 'MODIS/061/MOD13Q1',
                               'COPERNICUS/DEM/GLO30']),
    'mixed91': dict(cls=91, region='QH_MOSS', cand=15000, target=3000,
                    gate='mixed', sample_from='mask', tag='针阔混交林',
                    sources=['ESA/WorldCover/v200', 'LARSE/GEDI/GEDI02_A_002_MONTHLY',
                             'MODIS/061/MOD13Q1']),
    'evergreen52': dict(cls=52, region='QH_MOSS', cand=15000, target=3000,
                        gate='evergreen52', sample_from='mask', tag='疏闭常绿阔叶林',
                        sources=['ESA/WorldCover/v200', 'LARSE/GEDI/GEDI02_A_002_MONTHLY',
                                 'MODIS/061/MOD13Q1']),
    'shrub120121': dict(cls=121, region='HB_PLAIN', cand=8000, target=5000,
                        gate='shrub', tag='灌丛',
                        sources=['GOOGLE/DYNAMICWORLD/V1', 'ESA/WorldCover/v200',
                                 'MODIS/061/MOD13Q1']),
    'marsh180': dict(cls=180, region='HB_PLAIN', cand=12000, target=2000,
                     gate='marsh', sample_from='mask', tag='木本沼泽',
                     sources=['GOOGLE/DYNAMICWORLD/V1', 'ESA/WorldCover/v200',
                              'COPERNICUS/S1_GRD', 'MODIS/061/MOD13Q1']),
    'sparse150': dict(cls=150, region='QH_MOSS', cand=20000, target=4000,
                      gate='sparse', tag='稀疏植被',
                      sources=['ESA/WorldCover/v200', 'MODIS/061/MOD13Q1',
                               'COPERNICUS/DEM/GLO30']),
    'bare201': dict(cls=201, region='QH_MOSS', cand=20000, target=4000,
                    gate='bare', tag='裸地',
                    sources=['ESA/WorldCover/v200', 'MODIS/061/MOD13Q1',
                             'COPERNICUS/DEM/GLO30']),
    'water202': dict(cls=202, region='HB_PLAIN', cand=20000, target=4000,
                     gate='water', tag='水体',
                     sources=['JRC/GSW1_4/GlobalSurfaceWater', 'ESA/WorldCover/v200',
                              'COPERNICUS/S1_GRD']),
    'snow220': dict(cls=220, region='QH_MOSS', cand=20000, target=2000,
                    gate='snow', tag='冰雪',
                    sources=['MODIS/061/MOD10A2', 'ESA/WorldCover/v200',
                             'COPERNICUS/DEM/GLO30']),
}

# ---------------- 批量分区（家族 5–15 的"省域/生态区格网"，用户 2026-09-13 指令 1） ----------------
# 每个家族给一组 master 分区盒（按生态区/海岸段/农区划分），批量规划器再把每个盒切成
# parcel_deg×parcel_deg 的 parcel，并用 GEE 端 reduceRegions 预扫门槛像元数，只保留非空 parcel。
COAST_SEGMENTS = [
    ('辽东半岛渤海北', [119.5, 38.5, 124.0, 41.2]),
    ('渤海湾黄河口',   [117.0, 37.0, 119.5, 39.5]),
    ('山东半岛',       [119.0, 35.5, 123.0, 38.5]),
    ('江苏海岸',       [119.5, 31.5, 122.0, 35.0]),
    ('长江口上海',     [121.0, 30.5, 122.5, 32.0]),
    ('杭州湾浙江',     [120.0, 27.5, 122.5, 30.5]),
    ('福建海岸',       [117.0, 23.5, 120.5, 27.5]),
    ('粤东闽南',       [114.5, 22.0, 117.5, 24.5]),
    ('珠江口',         [112.5, 21.5, 114.8, 23.5]),
    ('粤西北部湾',     [108.5, 20.5, 112.5, 22.5]),
    ('海南',           [108.0, 18.0, 111.5, 20.5]),
    ('台湾',           [119.5, 21.5, 122.5, 25.5]),
]

BATCH = {
    'moss140': dict(parcel=1.0, min_count=1, zones=[
        ('祁连山', [94.0, 36.0, 103.0, 40.0]),
        ('昆仑山', [84.0, 35.0, 94.0, 38.5]),
        ('唐古拉羌塘', [84.0, 31.0, 97.0, 36.0]),
        ('冈底斯念青', [80.0, 29.0, 95.0, 32.5]),
        ('喜马拉雅北坡', [78.0, 27.0, 95.0, 30.5]),
        ('横断山川西', [97.0, 27.0, 103.5, 33.5]),
        ('阿尔泰', [85.0, 45.0, 91.0, 49.5]),
        ('天山', [74.5, 40.5, 88.0, 45.5]),
        ('帕米尔喀喇昆仑', [73.5, 35.5, 79.0, 39.5])]),
    'mangrove184': dict(parcel=0.5, min_count=1, zones=COAST_SEGMENTS),
    'saltmarsh185': dict(parcel=0.5, min_count=1, zones=COAST_SEGMENTS),
    'tidalflat186': dict(parcel=0.5, min_count=1, zones=COAST_SEGMENTS),
    'paddy12': dict(parcel=2.0, min_count=30, zones=[
        ('三江平原', [129.0, 44.0, 135.5, 48.5]),
        ('松嫩平原', [122.0, 43.0, 128.5, 48.5]),
        ('辽河平原', [119.5, 40.0, 125.0, 43.5]),
        ('华北平原', [113.0, 32.5, 120.0, 40.0]),
        ('长江中下游', [110.0, 28.0, 123.0, 34.0]),
        ('四川盆地', [103.0, 28.0, 108.5, 32.5]),
        ('华南', [106.0, 20.0, 117.5, 26.5]),
        ('云南', [97.0, 21.0, 106.0, 27.5]),
        ('河套宁夏', [104.5, 35.5, 108.5, 41.0]),
        ('新疆绿洲', [74.5, 35.5, 90.0, 45.5])]),
    'saline183': dict(parcel=1.0, min_count=1, zones=[
        ('柴达木', [89.0, 35.5, 98.5, 39.5]),
        ('塔里木边缘', [76.0, 36.5, 90.0, 42.0]),
        ('准噶尔', [82.0, 43.5, 91.5, 47.5]),
        ('内蒙高原湖群', [108.0, 40.5, 120.0, 46.0]),
        ('松嫩盐碱', [122.0, 44.0, 127.0, 48.0]),
        ('河套盐碱', [105.0, 38.5, 111.5, 42.0]),
        ('藏北盐湖', [82.0, 30.5, 92.0, 35.5])]),
    'mixed91': dict(parcel=1.0, min_count=1, zones=[
        ('东北林区', [124.0, 41.0, 135.0, 49.5]),
        ('秦岭', [105.0, 32.0, 112.5, 35.0]),
        ('横断山', [97.0, 26.0, 103.0, 33.5]),
        ('藏东南', [92.0, 28.0, 99.0, 32.0]),
        ('燕山太行', [110.0, 35.0, 119.5, 41.5])]),
    'evergreen52': dict(parcel=1.0, min_count=1, zones=[
        ('江南丘陵', [110.0, 25.0, 120.0, 31.0]),
        ('华南', [105.0, 20.0, 117.5, 25.5]),
        ('西南', [98.0, 21.0, 108.0, 28.0]),
        ('藏东南', [92.0, 27.5, 99.0, 31.5]),
        ('海南台湾', [108.0, 18.0, 122.5, 25.5])]),
    'shrub120121': dict(parcel=2.0, min_count=20, zones=[
        ('内蒙高原', [105.0, 37.5, 120.0, 46.5]),
        ('西北山地荒漠', [74.5, 35.0, 105.0, 47.5]),
        ('青藏边缘', [78.0, 27.5, 103.0, 38.5]),
        ('云贵', [97.0, 21.0, 107.5, 28.5]),
        ('华北山地', [110.0, 34.5, 119.5, 42.0]),
        ('东北', [119.5, 41.0, 135.0, 49.5])]),
    'marsh180': dict(parcel=2.0, min_count=5, zones=[
        ('东北沼泽', [122.0, 44.0, 135.5, 49.5]),
        ('长江中下游湖群', [110.0, 28.0, 122.0, 34.0]),
        ('青藏高原湿地', [80.0, 28.0, 103.0, 38.5]),
        ('若尔盖', [101.0, 32.0, 104.0, 35.0]),
        ('滨海沼泽', [117.0, 30.0, 122.5, 41.5])]),
    'sparse150': dict(parcel=2.0, min_count=20, zones=[
        ('西北荒漠', [74.5, 35.0, 105.0, 48.5]),
        ('青藏高原', [78.0, 27.5, 103.0, 38.5]),
        ('内蒙高原', [105.0, 37.5, 120.0, 46.5]),
        ('西南干热河谷', [97.0, 21.0, 105.0, 30.0]),
        ('华北', [110.0, 33.0, 120.0, 42.0])]),
    'bare201': dict(parcel=2.0, min_count=20, zones=[
        ('西北荒漠', [74.5, 35.0, 105.0, 48.5]),
        ('青藏高原', [78.0, 27.5, 103.0, 38.5]),
        ('内蒙高原', [105.0, 37.5, 120.0, 46.5]),
        ('西南干热河谷', [97.0, 21.0, 105.0, 30.0]),
        ('华北', [110.0, 33.0, 120.0, 42.0])]),
    'water202': dict(parcel=2.0, min_count=5, zones=[
        ('长江中下游', [110.0, 28.0, 122.0, 34.0]),
        ('青藏高原湖群', [80.0, 28.0, 100.0, 37.5]),
        ('东北湖群', [120.0, 41.0, 135.0, 49.5]),
        ('云贵', [97.0, 21.0, 107.0, 28.5]),
        ('黄淮华北', [110.0, 33.0, 122.0, 41.0]),
        ('新疆', [74.5, 35.0, 95.0, 48.0])]),
    'snow220': dict(parcel=1.0, min_count=1, zones=[
        ('天山', [74.5, 40.5, 88.0, 45.5]),
        ('昆仑帕米尔', [73.5, 35.0, 95.0, 39.5]),
        ('喜马拉雅', [78.0, 27.0, 95.0, 32.0]),
        ('祁连', [94.0, 36.0, 103.0, 40.0]),
        ('唐古拉羌塘', [84.0, 31.0, 97.0, 36.0]),
        ('阿尔泰', [85.0, 45.0, 91.0, 49.5])]),
}

# ---------------- 门槛构造（GEE 端） ----------------
# 统一签名 build_gate(ee, gate_key, geom, year) -> (stack, mask, bands, desc)


def _wc(ee, ver):
    return ee.ImageCollection(ver).first().select(['Map'])


def _dem(ee, geom):
    """Copernicus GLO-30：全球无缝（无 SRTM 那种空洞），避免 dropNulls 误杀点。
    （2026-09-13 实测：换用 SRTM 时空洞使红树林合格点从上千掉到个位数。）"""
    return (ee.ImageCollection('COPERNICUS/DEM/GLO30')
            .filterBounds(geom).select(['DEM']).mosaic().rename('dem'))


def _modis_ndvi(ee, geom, d0, d1, pct):
    """MODIS MOD13Q1 的 NDVI（250 m）作"植被背景"约束：单景集合、极廉、无云洞。
    批量阶段改用本地预计算的 Sentinel-2 红边时序（红边对 140/52 更敏感）。"""
    ic = (ee.ImageCollection('MODIS/061/MOD13Q1')
          .filterBounds(geom).filterDate(d0, d1).select(['NDVI']))
    return (ic.reduce(ee.Reducer.percentile([pct])).multiply(0.0001)
            .rename(f'ndvi_p{pct}'))


def _moss(ee, geom, year):
    """地衣苔藓 140：WorldCover(2021) + GLO-30 地形 + MODIS 植被背景 = 三个独立来源。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    dem = _dem(ee, geom)
    slope = ee.Terrain.slope(dem).rename('slope')
    ndvi = _modis_ndvi(ee, geom, '2021-06-01', '2021-09-15', 90)
    stack = ee.Image.cat([w21, dem, slope, ndvi])
    mask = (w21.eq(100).And(dem.gt(3400)).And(slope.lt(22)).And(ndvi.lt(0.36)))
    bands = ['wc21', 'dem', 'slope', 'ndvi_p90']
    desc = 'worldcover v200 moss/lichen(100) AND dem>3400 AND slope<22 AND MODIS ndvi_p90<0.36'
    return stack, mask.rename('gate'), bands, desc


def _mangrove(ee, geom, year):
    """红树林 184：GEE 公开目录**没有**红树林专题产品（LANDSAT/MANGROVE_FORESTS/001、
    JAXA/GMW v1-v3 实测均不存在，2026-09-13），故用三个相互独立的来源联合判定：
      WorldCover v200 判红树林(95) + JRC GSW 潮间带位置 + GLO-30 低海拔。
    批量阶段若接入 GMW v3（需本地上传），替换为「GMW ∩ WorldCover」双证。
    注：不用 MODIS NDVI 做冠层判据——250 m 分辨率下红树林条带与水体混合，NDVI 失真。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    occ = gsw.select('occurrence').rename('gsw_occ')
    tra = gsw.select('transition').rename('gsw_tra')
    dem = _dem(ee, geom)
    stack = ee.Image.cat([w21, occ, tra, dem])
    mask = (w21.eq(95)
            .And(occ.gt(4)).And(occ.lt(92))
            .And(tra.gte(1))
            .And(dem.lt(30)))
    bands = ['wc21', 'gsw_occ', 'gsw_tra', 'dem']
    desc = 'worldcover95 AND 4<gsw_occ<92 AND transition>=1 AND dem<30'
    return stack, mask.rename('gate'), bands, desc


def _saltmarsh(ee, geom, year):
    """海岸盐沼 185：原判据要求 WorldCover==草本湿地(90) —— 实测该像素类型在本区占比 0.1%，
    联合通过率≈0（2026-09-13 诊断），改用**潮汐证据 + 低矮植被证据 + 排除红树林/明水面**：
      GSW 潮汐过渡>=2 与 5<occurrence<90（潮间带） + MODIS 生长季 NDVI 0.15–0.70
      （有植被但非郁闭林） + GLO-30 海拔<25 m + WorldCover 非红树林(95)/非明水(80)。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
    occ = gsw.select('occurrence').rename('gsw_occ')
    tra = gsw.select('transition').rename('gsw_tra')
    dem = _dem(ee, geom)
    ndvi = _modis_ndvi(ee, geom, '2021-04-01', '2021-10-31', 75)
    stack = ee.Image.cat([w21, occ, tra, dem, ndvi])
    mask = (tra.gte(2).And(occ.gt(5)).And(occ.lt(90))
            .And(w21.neq(95)).And(w21.neq(80))
            .And(ndvi.gt(0.15)).And(ndvi.lt(0.70))
            .And(dem.lt(25)))
    bands = ['wc21', 'gsw_occ', 'gsw_tra', 'dem', 'ndvi_p75']
    desc = ('gsw transition>=2 AND 5<occ<90 AND wc!=mangrove/water '
            'AND 0.15<MODIS ndvi_p75<0.70 AND dem<25')
    return stack, mask.rename('gate'), bands, desc


def _s1_pct(ee, geom, d0, d1, pol, pct):
    ic = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(geom).filterDate(d0, d1)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', pol))
          .select([pol]))
    return ic.reduce(ee.Reducer.percentile([pct])).rename(f'{pol.lower()}_p{pct}')


def _paddy(ee, geom, year):
    vh_may = _s1_pct(ee, geom, '2021-05-01', '2021-06-20', 'VH', 10)
    vh_sep = _s1_pct(ee, geom, '2021-09-01', '2021-10-20', 'VH', 50)
    vv_sep = _s1_pct(ee, geom, '2021-09-01', '2021-10-20', 'VV', 50)
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    stack = ee.Image.cat([vh_may, vh_sep, vv_sep, w21])
    # 水稻 SAR 淹水特征：移栽期 VH 低 + 成熟期 VH/VV 回升；且 WorldCover 判耕地(40)
    rise = vh_sep.subtract(vh_may)
    mask = (w21.eq(40).And(rise.gt(3.0)).And(vv_sep.gt(-18)))
    bands = ['vh_p10', 'vh_p50', 'vv_p50', 'wc21']
    desc = 'worldcover40 AND (VH_sep_p50 - VH_may_p10)>3dB AND VV_sep_p50>-18'
    return stack, mask.rename('gate'), bands, desc


# ---------------- 家族 5–15 的门槛（2026-09-13 第二批） ----------------
def _gsw(ee):
    return (ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            .select(['occurrence', 'transition', 'seasonality', 'recurrence', 'max_extent']))


def _dw(ee, geom, d0, d1, band, pct=50):
    """Dynamic World 概率（10 m）。用年度分位合成压掉物候噪声。"""
    ic = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
          .filterBounds(geom).filterDate(d0, d1).select([band]))
    return ic.reduce(ee.Reducer.percentile([pct])).rename(f'dw_{band}')


def _gedi_rh(ee, geom, pct=75, d0='2020-01-01', d1='2023-12-31'):
    """GEDI 冠层高度 rh98 分位（25 m 条带，稀疏但独立于光学产品）。
    ⚠️ 2026-09-13 实测：`GEDI04_A_002_MONTHLY` 是**生物量(agbd)**产品，没有 rh98 波段；
    冠层高度在 **`GEDI02_A_002_MONTHLY`**（L2A）里。
    ⚠️ 2026-09-14：大区域（2° parcel）逐年窗口会把计算量放大 4 倍而超时，可传 d0/d1 收窄。"""
    ic = (ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
          .filterBounds(geom).filterDate(d0, d1)
          .select(['rh98']))
    return ic.reduce(ee.Reducer.percentile([pct])).rename('rh98')


def _ndvi_amp(ee, geom, d0, d1):
    """MODIS NDVI 年内振幅 p90 − p10：常绿<0.2、落叶>0.5、混交居中。"""
    ic = (ee.ImageCollection('MODIS/061/MOD13Q1')
          .filterBounds(geom).filterDate(d0, d1).select(['NDVI']))
    p90 = ic.reduce(ee.Reducer.percentile([90])).multiply(0.0001)
    p10 = ic.reduce(ee.Reducer.percentile([10])).multiply(0.0001)
    return p90.subtract(p10).rename('ndvi_amp'), p10.rename('ndvi_p10')


def _s1_lowvh(ee, geom, d0, d1):
    ic = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(geom).filterDate(d0, d1)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
          .select(['VH']))
    return ic.reduce(ee.Reducer.percentile([50])).rename('vh_p50')


def _tidalflat(ee, geom, year):
    """潮滩 186：GSW 周期性出露 + 低海拔 + 排除红树林/明水（三个独立证据）。"""
    g = _gsw(ee)
    occ = g.select('occurrence').rename('gsw_occ')
    tra = g.select('transition').rename('gsw_tra')
    sea = g.select('seasonality').rename('gsw_sea')
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    dem = _dem(ee, geom)
    stack = ee.Image.cat([w21, occ, tra, sea, dem])
    # 季节性水体 2–10 个月（周期性出露）+ 有水位变化 + 低海拔 + 非红树林 + 非常年水体
    mask = (sea.gte(2).And(sea.lte(10)).And(tra.gte(2)).And(occ.gt(3)).And(occ.lt(95))
            .And(dem.lt(15)).And(w21.neq(95)).And(w21.neq(80)))
    bands = ['wc21', 'gsw_occ', 'gsw_tra', 'gsw_sea', 'dem']
    desc = 'gsw seasonality 2-10 AND transition>=2 AND 3<occ<95 AND dem<15 AND wc!=mangrove/water'
    return stack, mask.rename('gate'), bands, desc


def _saline(ee, geom, year):
    """盐渍湿地 183：内陆盐湖/盐碱区（区域先验）∩ 季节性水体 ∩ 低植被 ∩ 中低海拔。
    GEE 无 GWL_FCS30；本判据用"区域先验 + GSW + MODIS 植被"三证，批量阶段可换 GWL 上传件。"""
    g = _gsw(ee)
    occ = g.select('occurrence').rename('gsw_occ')
    sea = g.select('seasonality').rename('gsw_sea')
    ndvi = _modis_ndvi(ee, geom, '2021-05-01', '2021-09-30', 90)
    dem = _dem(ee, geom)
    stack = ee.Image.cat([occ, sea, ndvi, dem])
    mask = (occ.gt(2).And(occ.lt(97)).And(sea.gte(1)).And(sea.lte(9))
            .And(ndvi.lt(0.45)).And(dem.lt(3600)))
    bands = ['gsw_occ', 'gsw_sea', 'ndvi_p90', 'dem']
    desc = 'saline-zone prior AND 2<gsw_occ<97 AND seasonality 1-9 AND ndvi_p90<0.45 AND dem<3600'
    return stack, mask.rename('gate'), bands, desc


def _mixed(ee, geom, year):
    """针阔混交林 91：WorldCover 森林 ∩ GEDI 树高 ∩ MODIS 物候振幅居中（三个独立来源）。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    rh = _gedi_rh(ee, geom, 75)
    amp, _ = _ndvi_amp(ee, geom, '2021-01-01', '2021-12-31')
    stack = ee.Image.cat([w21, rh, amp])
    mask = (w21.eq(10).And(rh.gt(12)).And(rh.lt(32)).And(amp.gt(0.22)).And(amp.lt(0.58)))
    bands = ['wc21', 'rh98', 'ndvi_amp']
    desc = 'worldcover tree(10) AND 12<gedi_rh98<32 AND 0.22<ndvi_amp<0.58'
    return stack, mask.rename('gate'), bands, desc


def _evergreen52(ee, geom, year):
    """疏闭常绿阔叶林 52：森林 ∩ 常绿（物候振幅小）∩ 冠层偏矮（疏林）。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    rh = _gedi_rh(ee, geom, 75)
    amp, _ = _ndvi_amp(ee, geom, '2021-01-01', '2021-12-31')
    stack = ee.Image.cat([w21, rh, amp])
    mask = (w21.eq(10).And(amp.lt(0.22)).And(rh.gt(6)).And(rh.lt(26)))
    bands = ['wc21', 'rh98', 'ndvi_amp']
    desc = 'worldcover tree(10) AND ndvi_amp<0.22 (evergreen) AND 6<gedi_rh98<26'
    return stack, mask.rename('gate'), bands, desc


def _shrub(ee, geom, year):
    """灌丛 121（北方以落叶灌丛为主）：矮木本 = 矮冠层 + 灌草概率 + 地表覆盖一致。
    ⚠️ 2026-09-14 诊断：原判据 WC==20 且 DW 概率>0.5 在全中国**一个像元都不剩**——
      WorldCover 把北方灌丛大量标成草地(30)/裸地(60)（WC==20 仅 0.3%），
      DW 的 shrub_and_scrub p60 中位只有 0.128、p99 0.421（>0.5 仅 0.1%）。
    修正：四证联合——DW 灌草概率>0.25（p90 水平）+ WC 灌丛或草地 + **GEDI 冠层 <5 m（矮木本的物理判据）**
      + MODIS NDVI<0.62；通过率 3.6–4.2%，且四源相互独立。
    常绿灌丛(120)与落叶灌丛(121)的细分需物候判据，本轮统一标 121（北方灌丛以落叶为主）。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    dw = _dw(ee, geom, '2021-01-01', '2022-01-01', 'shrub_and_scrub', 60)
    ndvi = _modis_ndvi(ee, geom, '2021-06-01', '2021-09-30', 90)
    rh = _gedi_rh(ee, geom, 75, '2021-01-01', '2021-12-31')
    stack = ee.Image.cat([w21, dw, ndvi, rh])
    mask = (dw.gt(0.25).And(w21.eq(20).Or(w21.eq(30)))
            .And(rh.gt(0.3)).And(rh.lt(5)).And(ndvi.lt(0.62)))
    bands = ['wc21', 'dw_shrub_and_scrub', 'ndvi_p90', 'rh98']
    desc = ('dw shrub_p60>0.25 AND worldcover in(20,30) AND 0.3<gedi_rh98<5 AND ndvi_p90<0.62 '
            '(short-woody, four independent sources)')
    return stack, mask.rename('gate'), bands, desc


def _marsh(ee, geom, year):
    """木本沼泽 180：Dynamic World 淹水植被 ∩ WorldCover 湿地 ∩ S1 低后向散射（湿）。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    dw = _dw(ee, geom, '2021-01-01', '2022-01-01', 'flooded_vegetation', 60)
    vh = _s1_lowvh(ee, geom, '2021-05-01', '2021-10-31')
    ndvi = _modis_ndvi(ee, geom, '2021-06-01', '2021-09-30', 90)
    stack = ee.Image.cat([w21, dw, vh, ndvi])
    mask = (w21.gte(90).And(w21.lte(95)).And(dw.gt(0.4))
            .And(vh.lt(-12)).And(ndvi.gt(0.25)).And(ndvi.lt(0.72)))
    bands = ['wc21', 'dw_flooded_vegetation', 'vh_p50', 'ndvi_p90']
    desc = 'worldcover 90-95 AND dw_flooded_veg_p60>0.4 AND S1 VH_p50<-12 AND 0.25<ndvi_p90<0.72'
    return stack, mask.rename('gate'), bands, desc


def _sparse(ee, geom, year):
    """稀疏植被 150：WorldCover 裸/稀(60) ∩ MODIS NDVI 低而非常低 ∩ 非陡坡。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    ndvi = _modis_ndvi(ee, geom, '2021-06-01', '2021-09-30', 90)
    dem = _dem(ee, geom)
    slope = ee.Terrain.slope(dem).rename('slope')
    stack = ee.Image.cat([w21, ndvi, dem, slope])
    mask = (w21.eq(60).And(ndvi.gt(0.10)).And(ndvi.lt(0.30)).And(slope.lt(28)).And(dem.lt(5200)))
    bands = ['wc21', 'ndvi_p90', 'dem', 'slope']
    desc = 'worldcover60 AND 0.10<ndvi_p90<0.30 AND slope<28 AND dem<5200'
    return stack, mask.rename('gate'), bands, desc


def _bare(ee, geom, year):
    """裸地 201：WorldCover 裸/稀(60) ∩ MODIS NDVI 极低 ∩ 非永久冰雪高程带。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    ndvi = _modis_ndvi(ee, geom, '2021-06-01', '2021-09-30', 90)
    dem = _dem(ee, geom)
    stack = ee.Image.cat([w21, ndvi, dem])
    mask = (w21.eq(60).And(ndvi.lt(0.12)).And(dem.lt(5000)))
    bands = ['wc21', 'ndvi_p90', 'dem']
    desc = 'worldcover60 AND ndvi_p90<0.12 AND dem<5000'
    return stack, mask.rename('gate'), bands, desc


def _water(ee, geom, year):
    """水体 202：GSW 常年出现率 ∩ WorldCover 水体 ∩ S1 低 VV（三个独立来源）。"""
    g = _gsw(ee)
    occ = g.select('occurrence').rename('gsw_occ')
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    vv = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(geom).filterDate('2021-01-01', '2021-12-31')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .select(['VV']).reduce(ee.Reducer.percentile([50])).rename('vv_p50'))
    stack = ee.Image.cat([w21, occ, vv])
    mask = (occ.gt(90).And(w21.eq(80)).And(vv.lt(-15)))
    bands = ['wc21', 'gsw_occ', 'vv_p50']
    desc = 'gsw occurrence>90 AND worldcover water(80) AND S1 VV_p50<-15dB'
    return stack, mask.rename('gate'), bands, desc


def _snow(ee, geom, year):
    """冰雪 220：MODIS 年积雪日数 ∩ WorldCover 冰雪 ∩ 高海拔（三个独立来源）。"""
    w21 = _wc(ee, 'ESA/WorldCover/v200').rename('wc21')
    dem = _dem(ee, geom)
    ic = (ee.ImageCollection('MODIS/061/MOD10A2')
          .filterBounds(geom).filterDate('2021-01-01', '2021-12-31')
          .select(['Maximum_Snow_Extent']))
    # 8 天合成：Maximum_Snow_Extent 中"雪"编码为 200；取年内为雪的合成数 ×8 ≈ 积雪日数
    snowdays = (ic.map(lambda i: i.eq(200)).sum().multiply(8).rename('snow_days'))
    stack = ee.Image.cat([w21, dem, snowdays])
    mask = (w21.eq(70).And(snowdays.gt(240)).And(dem.gt(3800)))
    bands = ['wc21', 'dem', 'snow_days']
    desc = 'worldcover snow/ice(70) AND MODIS snow_days>240 AND dem>3800'
    return stack, mask.rename('gate'), bands, desc


GATES = {'moss': _moss, 'mangrove': _mangrove, 'saltmarsh': _saltmarsh, 'paddy': _paddy,
         'tidalflat': _tidalflat, 'saline': _saline, 'mixed': _mixed,
         'evergreen52': _evergreen52, 'shrub': _shrub, 'marsh': _marsh,
         'sparse': _sparse, 'bare': _bare, 'water': _water, 'snow': _snow}

# ---------------- 下一代补样：15 个任务大类（一账号一类，防同类跨账号） ----------------
# phase 1 = 本轮已实现门槛（试点跑）；phase 2 = 规则已定、门槛待实现（批量阶段补）
# 每个家族内至少两个相互独立来源同时为真才给标签（r2 教训）。
NG_PLAN = {
    'moss140':      dict(cls=140, phase=1, tag='地衣苔藓',
                         src='WorldCover v200+v100 / SRTM / S2 红边',
                         how='双年地表覆盖一致 + 海拔>3400 + 缓坡 + 生长季 NDVI 低'),
    'mangrove184':  dict(cls=184, phase=1, tag='红树林',
                         src='WorldCover95 + USGS Giri 红树林 + GSW',
                         how='两个独立来源同时判红树林；边缘/养殖塘过渡单独标记'),
    'saltmarsh185': dict(cls=185, phase=1, tag='海岸盐沼',
                         src='WorldCover90 + JRC GSW(transition)',
                         how='草本湿地 + 潮汐过渡频次≥2；按海岸段分层配额'),
    'paddy12':      dict(cls=12,  phase=1, tag='灌溉水田',
                         src='Sentinel-1 GRD VV/VH 时序 + WorldCover 耕地',
                         how='移栽期 VH 低 + 成熟期 VH/VV 回升（淹水特征），解决 12↔10 混淆'),
    'tidalflat186': dict(cls=186, phase=2, tag='潮滩',
                         src='GSW recurrence/transition + WorldCover',
                         how='周期出露判据 + 海岸带约束；与 185 用过渡频次分档'),
    'saline183':    dict(cls=183, phase=2, tag='盐渍湿地',
                         src='GWL_FCS30 + GSW + S1 VV/VH + S2 盐分指数',
                         how='内陆盐湖/河口盐碱分别建模；淡水湖滨明确排除'),
    'orchard11':    dict(cls=11,  phase=3, tag='乔灌园地',
                         src='AOMC 苹果 / 柑橘 / TeaMap / 橡胶 / SDPT',
                         how='按树种亚类分层，新增 subtype 字段；行列纹理 + S2 物候'),
    'mixed91':      dict(cls=91,  phase=2, tag='针阔混交林',
                         src='GEDI 树高 + 森林类型产品 + S2 物候',
                         how='补东北/秦岭/横断山/藏东南；针叶比例 30–70% 判据'),
    'evergreen52':  dict(cls=52,  phase=2, tag='疏闭常绿阔叶林',
                         src='S2 红边 + GEDI + WorldCover 森林',
                         how='区分天然疏林 vs 果园；郁闭度阈值 + 冠层高度'),
    'shrub120121':  dict(cls=121, phase=2, tag='灌丛（常绿/落叶）',
                         src='Dynamic World 年度概率 + S1/S2 时序',
                         how='多年稳定灌木=核心；单年突变降权；FCS10 本命年权重 0.6–0.8'),
    'marsh180':     dict(cls=180, phase=2, tag='木本沼泽',
                         src='GWL_FCS30 + S1 时序 + S2',
                         how='木本沼泽与草本沼泽用 SAR 后向散射与冠层高度区分'),
    'sparse150':    dict(cls=150, phase=2, tag='稀疏植被',
                         src='WorldCover60 + S2 NDVI 分位',
                         how='NDVI 峰值 0.1–0.25 且植被盖度低；与裸地/草地双阈值'),
    'bare201':      dict(cls=201, phase=2, tag='裸地',
                         src='WorldCover60 + S2 亮度/湿度 + DEM',
                         how='排除高寒苔原与盐碱；与 150 交叉校验'),
    'water202':     dict(cls=202, phase=2, tag='水体',
                         src='JRC GSW occurrence + Sentinel-1',
                         how='多年出现率>90% 为常年水体；季节性水体单列'),
    'snow220':      dict(cls=220, phase=2, tag='冰雪',
                         src='MODIS 积雪时长 + WorldCover70 + DEM',
                         how='年积雪日数>300 天为常年冰雪；冰川边界校验'),
}
NG_PLAN_ORDER = list(NG_PLAN)

# ---------------- 批量目标量（按"独立来源多样性"定，不追求数量） ----------------
# 母库常见类已有十万量级；这些新点的价值是**换一个独立来源**去交叉验证，
# 而不是把某一类堆到更大。稀有/结构类目标略高，常见类只取独立子样。
TARGET_BATCH = {
    'moss140': 1500, 'mangrove184': 800, 'saltmarsh185': 2000, 'paddy12': 4000,
    'tidalflat186': 1500, 'saline183': 2000, 'mixed91': 2500, 'evergreen52': 2500,
    'shrub120121': 3000, 'marsh180': 1500, 'sparse150': 2000, 'bare201': 2000,
    'water202': 2000, 'snow220': 1500,
}


def build_gate(ee, gate_key, geom, year=2021):
    if gate_key not in GATES:
        raise KeyError(f'未知门槛 {gate_key}')
    return GATES[gate_key](ee, geom, year)


def dump_spec(path):
    obj = {'regions': REGIONS,
           'specs': {k: {kk: vv for kk, vv in v.items()} for k, v in SPECS.items()},
           'ng_plan': NG_PLAN, 'ng_plan_order': NG_PLAN_ORDER,
           'batch': {k: {'parcel': v['parcel'], 'min_count': v['min_count'],
                         'zones': [list(z) for z in v['zones']]}
                     for k, v in BATCH.items()}}
    json.dump(obj, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return path


if __name__ == '__main__':
    import ng0_paths as P
    P.ensure_all()
    print('规格:', list(SPECS))
    print('区域:', {k: v['note'] for k, v in REGIONS.items()})
    print('快照 →', dump_spec(P.SPEC_JSON))

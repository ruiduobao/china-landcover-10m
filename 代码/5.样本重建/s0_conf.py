# -*- coding: utf-8 -*-
"""
s0_conf.py — r1 样本重建流水线唯一配置源（单一事实来源）
* 类别体系/码表 import lc_conf（旧权威），本文件只补：路径总表、产品投票查表、
  统一生态规则表、清洗参数。任何 s 系列脚本不得本地重定义码表/边界/生态规则。
* 本流水线全程不发起 GEE 请求。
"""
import os, sys
import numpy as np

PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
sys.path.insert(0, os.path.join(PROJ, '代码', '0.本地流水线'))
from lc_conf import CLASSES, LEVEL0, LEVEL1, CODE_MAP, GAP_CLASSES, class_name  # noqa 权威码表

# ---------------- 路径总表 ----------------
GRID_DIR   = os.path.join(PROJ, '数据/样本数据/GLC_Samples_Grid_2x2')
BOUND_JSON = os.path.join(PROJ, '数据/边界/china_100000_full.json')
GUB_SHP    = r'E:\data\非洲城市发展和驱动力分析\数据\GUB_全球城市边界数据\GUB_Global_2020\GUB_Global_2020.shp'
EXT        = os.path.join(PROJ, '数据/外部样本源')
RARE_DIR   = os.path.join(EXT, 'rare_class')
WORK       = os.path.join(PROJ, '数据/本地处理/样本重建')
DELIV      = os.path.join(PROJ, '数据/样本交付')
LOG_DIR    = os.path.join(PROJ, '数据/本地处理/日志')
os.makedirs(WORK, exist_ok=True)

# 教师产品路径（与 lc14 一致）
E = 'E:/gisrsdata.org/还未整理/地理资源/遥感地物分类'
WC_DIR      = E + '/ESA-WorldCover全球10m分辨率土地覆被'
ESRI_DIR    = E + '/ESRI10m土地利用数据(未裁剪)'
TH_DIR      = E + '/2017年10m土地利用数据_分省裁剪_清华大学数据集'
CN30_ZIP    = E + '/中国30米精细地表覆盖2020-2010-2000/2020年30米地表覆盖.zip'
CLCD30_DIR  = r'E:/2.1_CLCD土地覆盖数据/30米'
CLCD100_DIR = r'E:/2.1_CLCD土地覆盖数据/100米'
FCS10_DIR   = os.path.join(EXT, 'glc_fcs10_2023')              # 7 个 5° zip（Z 盘副本）
# n18_resample 固定种子重采成品（国界+生态内、面积加权、可复现）
FCS10_SAMPLES = os.path.join(EXT, 'glc_fcs10_samples.parquet')

# 旧库（只读复用：投票列 join 与 s2 校验，不参与新链逻辑）
V1 = os.path.join(PROJ, '数据/本地处理/样本底座/cn_samples_v1.parquet')
V2 = os.path.join(PROJ, '数据/本地处理/样本底座/cn_samples_v2.parquet')

# r1 中间产物
R1_BASE   = os.path.join(WORK, 'r1_base.parquet')
R1_EXT    = os.path.join(WORK, 'r1_ext.parquet')
R1_TOPUP  = os.path.join(WORK, 'r1_topup.parquet')
R1_THEMATIC = os.path.join(WORK, 'r1_thematic.parquet')
R1_POOL   = os.path.join(WORK, 'r1_pool.parquet')
R1_TRAIN  = os.path.join(WORK, 'r1_train.parquet')
R1_VALID  = os.path.join(WORK, 'r1_validation.parquet')

# ---------------- 五产品投票查表 ----------------
# WC/ESRI/TH/CLCD 逐字复制自 lc14_sample_products.XW（旧权威，已验证）
# CN30 已重写：该产品实为 GlobeLand30 码系（2026-09-06 已知地点实测破译：
#   10耕/20林/30草/40灌/50湿地/60水/70苔原/80建成/90裸/100冰雪）。
# 旧 lc14.XW 用 GLC 码系查此产品 = 错表（森林投耕地票、其余丢票），
# v2 的 cn30_l0 已污染，s5 用 v2_parts/F_cn30 的 raw 值按本表重算。
XW = {
 'WC':   {10:2, 20:3, 30:4, 40:1, 50:6, 60:7, 70:10, 80:9, 90:5, 95:5, 100:7},
 'ESRI': {0:-1, 1:9, 2:2, 3:4, 4:5, 5:1, 6:3, 7:6, 8:7, 9:10, 10:4, 11:-1},
 'TH':   {0:-1, 10:1, 20:2, 30:4, 40:3, 50:5, 60:9, 70:7, 80:6, 90:7, 100:10},
 'CLCD': {0:-1, 1:1, 2:2, 3:3, 4:4, 5:9, 6:10, 7:7, 8:6, 9:5},
 'CN30': {0:-1, 255:-1, 10:1, 20:2, 40:3, 30:4, 50:5, 80:6, 90:7, 70:-1,
          60:9, 100:10},
}
PROD_LUT = {k: np.full(256, -1, dtype=np.int8) for k in XW}
for _k, _m in XW.items():
    for _a, _b in _m.items():
        if 0 <= _a < 256:
            PROD_LUT[_k][_a] = _b

# ---------------- GLC_FCS10 2023 raw 码附加映射（lc_conf.CODE_MAP 之外的覆盖，同 n18_resample） ----------------
FCS10_EXTRA = {191: 190, 192: 200, 210: 202, 200: 201, 201: 201, 202: 201}

# ---------------- 统一生态规则（合并旧链三套矛盾定义为唯一一套） ----------------
# 1) 每类合法纬度区间（硬规则，范围外剔除）——采纳 n18_final_clean.LAT_RANGE 原值
ECO_LAT = {
    10: (18, 54), 11: (18, 42), 12: (18, 52),
    51: (18, 34), 52: (18, 36), 61: (18, 54), 62: (18, 54),
    71: (18, 48), 72: (18, 50), 81: (30, 54), 82: (30, 54),
    91: (33, 54), 92: (33, 54),
    120: (18, 40), 121: (21, 54),
    130: (18, 54), 140: (26, 54), 150: (26, 54),
    180: (38, 54), 181: (18, 54), 182: (18, 54),
    183: (30, 50), 184: (17.5, 26), 185: (18, 42), 186: (18, 42),
    190: (18, 54), 200: (18, 54), 201: (18, 54),
    202: (18, 54), 220: (28, 54),
}
# 2) 附加经纬联合硬规则（n16 ECO_RULES 的 hard 部分，LAT_RANGE 覆盖不了的）
ECO_EXTRA_HARD = {
    120: [('y>42',            '常绿灌丛42°N以北无分布'),
          ('(x>80)&(x<95)&(y>32)', '藏北高原无常绿灌丛'),
          ('(x>80)&(x<90)&(y>36)', '塔里木腹地无常绿灌丛')],
    121: [('(x>80)&(x<92)&(y>37)', '塔里木/藏北核心落叶灌丛不可靠')],
}
# 注：n16 的 soft 规则（51北界/130南界等）已被 ECO_LAT 硬区间完全覆盖，不再单列。

# 3) 省级行政区白名单（2026-09-09 新增，错位分布事故后的硬规则，helper 见 s1_geom.province_violation）
#    背景：GLC_FCS10 2023 在长三角/杭州湾沿岸把养殖塘/潮间带大面积错标为 140（地衣苔藓），
#    在长江中下游淡水湖区把 raw 184 错标为 183（盐渍湿地）；仅纬度带过滤不住，必须叠加
#    省级行政区白名单。184/185/186/220/81/82 一并纳入防复发。判定函数消费省名
#    （DataV china_100000_full.json），与 s1_geom.province_of 完全一致。
COASTAL_PROV = {'辽宁省', '河北省', '天津市', '山东省', '江苏省', '上海市', '浙江省',
                '福建省', '广东省', '广西壮族自治区', '海南省', '台湾省',
                '香港特别行政区', '澳门特别行政区'}
ECO_PROVINCE = {
    # 地衣苔藓：仅青藏高原及西北/横断山高山带（东部湿润区一律剔除）
    140: {'西藏自治区', '青海省', '新疆维吾尔自治区', '四川省', '甘肃省', '云南省'},
    # 盐渍湿地：沿海省 ∪ 内陆盐湖区（新疆/青藏/蒙甘宁/吉黑盐碱地）；剔除长江中下游淡水湖区
    183: COASTAL_PROV | {'新疆维吾尔自治区', '青海省', '西藏自治区', '内蒙古自治区',
                         '甘肃省', '宁夏回族自治区', '吉林省', '黑龙江省'},
    # 红树林：仅华南暖水海岸
    184: COASTAL_PROV - {'河北省', '天津市', '山东省', '江苏省', '上海市'},
    185: COASTAL_PROV,                       # 海岸盐沼
    186: COASTAL_PROV,                       # 潮滩
    # 永久冰雪：仅高山（青藏/天山/祁连/横断山等）
    220: {'西藏自治区', '青海省', '新疆维吾尔自治区', '四川省', '甘肃省', '云南省'},
    # 落叶针叶林（落叶松）：东北 + 西南亚高山 + 华北山地；剔除华南与东南沿海
    81: {'黑龙江省', '吉林省', '辽宁省', '内蒙古自治区', '河北省', '山西省', '陕西省',
         '甘肃省', '新疆维吾尔自治区', '四川省', '云南省', '西藏自治区', '青海省',
         '宁夏回族自治区', '湖北省', '重庆市', '河南省', '北京市', '山东省'},
}
ECO_PROVINCE[82] = ECO_PROVINCE[81]          # 疏闭落叶针叶林同 81

# ---------------- 清洗参数（唯一一处定义） ----------------
MIN_DIST = {          # 每类同类最小间距（米）
    'default': 500,
    10: 500, 11: 1000, 12: 500,
    51: 500, 52: 1000, 61: 500, 62: 1000,
    71: 500, 72: 1000, 81: 500, 82: 500,
    91: 500, 92: 1000,
    120: 1000, 121: 1000,
    130: 500, 140: 2000, 150: 500,
    180: 500, 181: 500, 182: 500, 183: 500, 184: 1000, 185: 500, 186: 500,
    190: 200, 200: 300, 201: 500, 202: 500, 220: 2000,
}
CELL_QUOTA = 200      # 0.25°格×类 最大点数
SEED       = 42
TIER_W     = {'gold': 1.0, 'silver': 0.8, 'external': 0.7, 'bronze': 0.6}  # 抽稀优先级

# ---------------- 投票分层规则（同 lc15） ----------------
# agree_n = 5 产品 level0 与教师标签一致的票数(0..5)
# gold=agree==5且非边界带; silver>=4; bronze==3; uncovered=有效票<=3且agree<3; reject=其余
VOTE_KEEP_EXTERNAL_MIN = 3   # external 点进入训练集所需最小一致票

# ---------------- 稀缺类补采 ----------------
RARE_CLASSES = [91, 92, 140, 11, 52, 62, 184, 185]   # 旧链恒少的地类
RARE_TARGET  = 5000      # 补采后每类目标点数（池内）
FCS10_TOPUP_PER_CELL = 15   # 每 1°格每类补采上限
FCS10_PURITY_WIN     = 3    # 3×3 纯净像元约束

BBOX = (73, 17, 136, 54)     # 全流水线统一粗框

# ---------------- GLC_FCS10 全覆盖重采参数（s4） ----------------
# E 盘已解压副本（Z 盘 E070-E075 zip 损坏，统一读 E 盘 tif）
FCS10_TILE_ROOT = r'E:\data\全球土地覆盖数据\数据\GLC_FCS10'
FCS10_MIN_FRAC = 0.02    # 类占比低于 2% 不采（非稀缺类）
FCS10_TOTAL_CAP = 80     # 每 1°格全类共享总配额（面积加权）
FCS10_SEED_BASE = 10000  # 固定种子 = cx*10000+cy
FCS10_SRC = 'glc_fcs10_2023'
FCS10_CONF = 0.85
FCS10_YEAR = 2023

def eco_violation(lon, lat, cls):
    """向量化生态规则判定：返回 hard 违规布尔数组（ECO_LAT + ECO_EXTRA_HARD）"""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    cls = np.asarray(cls, int)
    bad = np.zeros(len(lon), dtype=bool)
    for c, (lo, hi) in ECO_LAT.items():
        m = cls == c
        if m.any():
            bad[m] |= (lat[m] < lo) | (lat[m] > hi)
    for c, rules in ECO_EXTRA_HARD.items():
        m = cls == c
        if m.any():
            x, y = lon[m], lat[m]
            for expr, _ in rules:
                bad[m] |= eval(expr, {'x': x, 'y': y, '__builtins__': {}})
    return bad

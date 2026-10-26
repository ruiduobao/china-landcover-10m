# -*- coding: utf-8 -*-
"""rc_conf.py — 盲判修复与小分类：单一配置源（路径 / 角色 / 配额 / 码表 / 中性 ID）

* 依据：技术文档/72_盲判证据驱动样本修复与小分类实施协议.md（主 agent 冻结）
       技术文档/70 §4.2 命名空间安全、§6.1 空间块拆分；69 §5.3 坡向配对诊断
* 角色（主 agent 冻结，勿改）：
    R = 从**实际训练池**抽取的审计/修复点：可最终参与 candidate 训练，**绝不进入评价**；
    V = **独立空间块**的新评价点：不取自训练池同点、优先分歧/独立随机位置、与母库原点保持空间距离，
        只出预测不出标签，**V 所在 0.25° 块全部从训练排除**；
    Q = R/V 内部复判子集，不新选择、不改角色分母。
* 配额：总 200 点 = 林型 50 对（100 点，**整对同 role**）＋ 园地/森林/耕地争议 100 点；
        目标 R120 / V80（林型 R30 对=60 / V20 对=40；争议 R60 / V40）。
* 中性 ID：盲判面一律 `B####`（opaque，随机序），真实来源/先验/role/配对 只在 private/ 侧表。
* 输出：本文件只提供常量与工具；不读写数据。
* 用法：from rc_conf import *（脚本同目录）；或 python rc_conf.py 打印配置摘要
"""
import hashlib
import json
import os

# ---------- 路径（一律绝对 raw string） ----------
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'
CODE12 = os.path.join(PROJ, '代码', '12.盲判修复与小分类')
WORK = r'F:/lc_work'
WS = os.path.join(WORK, 'label_repair_2023_20261015')
MAT = os.path.join(WS, 'materials')     # 盲判材料（切片/联系表/表单骨架；无先验）
PRIV = os.path.join(WS, 'private')      # 侧表（先验/role/配对/来源）——判读代理不得读
BLIND = os.path.join(WS, 'blind')       # 盲判面（只有 point_id + lon/lat + 时序日期）
QC = os.path.join(WS, 'qc')             # 质检/泄漏审计/复判
EXP = os.path.join(WS, 'experiments')   # 小分类实验
AUD = os.path.join(WS, 'audit')         # 封存/哈希/注册
LOGS = os.path.join(WS, 'logs')
CACHE = os.path.join(WS, 'cache')       # 可重建缓存（chips 复用、S2 取数）

# ---------- 只读输入（原池 / 两版成品 / DEM，一律只读） ----------
POOL_V2 = os.path.join(WORK, 'prod2nd_2023', 'data', 'train2023_v2.parquet')   # 第二轮池（R 来源）
POOL_V1 = os.path.join(WORK, 'prod5p_2023', 'data', 'train2023_clean.parquet')  # 第一轮池（只读参考）
RAST2 = os.path.join(WORK, 'prod2nd_2023', 'rasters_batch')   # 第二轮瓦片（10 m uint8）
RAST1 = os.path.join(WORK, 'prod5p_2023', 'rasters_batch')    # 第一轮瓦片（同网格）
MOSAIC2 = os.path.join(WORK, 'prod2nd_2023', 'mosaic2')       # 第二轮合幅/省件
DEM = os.path.join(WORK, 'v31_exp', 'data', 'eco_gate', '_dem')  # SRTM 60 m 本地 DEM
MOTHER = os.path.join(PROJ, '数据', '本地处理', '样本重建', 'r7_train.parquet')  # 母库（只读，V 防近用）
TILE_PLAN2 = os.path.join(WORK, 'prod2nd_2023', 'plan', 'tiles_secondrun.csv')
BOUNDARY = os.path.join(PROJ, '数据', '边界', 'china_100000_full.json')

# ---------- 类别（24 类 v3 码，见 技术文档/37 §2.1） ----------
CL24 = {
    1: '草本旱地', 2: '乔灌园地', 3: '灌溉耕地', 4: '常绿阔叶林', 5: '落叶阔叶林',
    6: '常绿针叶林', 7: '落叶针叶林', 8: '针阔混交林', 9: '常绿灌丛', 10: '落叶灌丛',
    11: '草地', 12: '地衣苔藓', 13: '稀疏植被', 14: '木本沼泽', 15: '草本沼泽',
    16: '湖河滩地', 17: '盐渍湿地', 18: '红树林', 19: '海岸盐沼', 20: '潮滩',
    21: '不透水面', 22: '裸地', 23: '水体', 24: '永久冰雪',
}
FOREST_CL = [4, 5, 6, 7]            # 林型轴（针/阔 × 常绿/落叶）
DISPUTE_CL = [2, 1, 3, 4, 5, 6, 7]  # 园地 / 耕地(旱+灌溉) / 森林 —— 争议组
NEEDLE_CL = [6, 7]
BROAD_CL = [4, 5]

# ---------- 角色与配额（主 agent 冻结） ----------
ROLE_R = 'R'
ROLE_V = 'V'
ROLE_Q = 'Q'
QUOTA_TOTAL = 200
QUOTA_R = 120
QUOTA_V = 80
PAIRS_TOTAL = 50                    # 林型配对 50 对 = 100 点
PAIRS_R = 30                        # 整对同 role → R30 对(60 点) / V20 对(40 点)
PAIRS_V = 20
DISPUTE_TOTAL = 100                 # 争议点 100 点
DISPUTE_R = 60
DISPUTE_V = 40
BLOCK_DEG = 0.25                    # 角色/开发-确认空间块（doc70 §6.1）
PAIR_BLOCK_DEG = 0.25               # 配对要求同块（保证同 role 且同山体）
DISPUTE_MIX = dict(disagree=0.5, stable=0.5)   # 争议点：分歧区 50% + 稳定区 50%

# ---------- 坡向 / 地形口径（doc69 §5.1 同源） ----------
ASPECT_SUN = (135.0, 225.0)         # 阳坡
ASPECT_SHADE = [(315.0, 360.0), (0.0, 45.0)]   # 阴坡
MIN_SLOPE_DEG = 5.0                 # 平地不参与阴阳坡比较
PAIR_MAX_KM = 4.0                   # 同山体配对最大距离
PAIR_ELEV_TOL_M = 200.0             # 配对高程容差
DEM_NODATA = -32768

# ---------- 中性 ID / 随机 ----------
ID_PREFIX = 'B'
ID_WIDTH = 4
SEED = 20261015
S2_YEAR = 2023
S2_WINTER_YEARS = [2019, 2023]
S2_MONTHS = list(range(1, 13))

# ---------- GEE（专用账号：串行初始化；与 geometry 组账号互斥） ----------
# geometry 组 2026-10-15 独占 k3r6ncvd（备 p2vlkaia 不启用）→ materials 用 p2vlkaia 等其它健康号
GEE_AVOID = ['k3r6ncvd']
GEE_CANDIDATES = ['p2vlkaia', 'j8tv79n', 'es82gxq', 'q6hbzjk', 'ybzljyma']
S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
AEF_COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
S2_REQUEST_TIMEOUT_S = 90           # 单请求硬超时（主 agent 要求 ≤90 s）
S2_CHUNK = 40                       # 每子请求点数（可重试分块）

# ---------- 材料批次 ----------
BATCH_SIZE = 40                     # 每批 20–40 点出 blind sheets 供判读代理开始

# ---------- 局部训练框（R 集中，避免全国/全省散点；主 agent 2026-10-15 要求） ----------
# 每个框 = 一个当地模型训练上下文（doc70：训练用所属 2° 瓦片各向外扩 2°）
BOXES = {
    'F': dict(tile='T1707', bbox=[107.0, 32.0, 109.0, 34.0],      # 川东北山地（米仓山—大巴山）
              purpose='林型坡向配对（针/阔 × 阴阳坡）', note='林型对整对同 role；训练框=tile±2°'),
    'O': dict(tile='T1706', bbox=[107.0, 30.0, 109.0, 32.0],      # 川东平行岭谷（盆周东缘）
              purpose='园地/森林/耕地争议（含针阔争议）', note='与 F 框相邻，不同 2° 瓦片'),
}
BOX_TRAIN_EXPAND_DEG = 2.0          # 训练窗外扩（与生产/实验一致）
BOX_SEEDS = [7, 11, 23]             # 预注册配对种子（与 doc70 一致）
# 既有实验已看过的空间块（doc71 §2 表；V 块必须避开 → 保 V 的"未见"性）
SEEN_BOXES_2026_10_15 = [
    [106.0, 32.0, 107.0, 33.0],   # O-SC1 / T-F1
    [105.5, 32.0, 106.5, 33.0],   # T-F1
    [107.0, 32.0, 108.0, 33.0],   # O-SC2
    [112.0, 23.0, 113.0, 24.0],   # O-GD
    [106.0, 38.0, 107.0, 39.0],   # O-NX
    [107.0, 31.0, 108.0, 32.0],   # T-F2
    [106.0, 31.0, 107.0, 32.0],   # F-VIEW G2
]


def in_seen(lon, lat):
    for x0, y0, x1, y1 in SEEN_BOXES_2026_10_15:
        if x0 <= lon < x1 and y0 <= lat < y1:
            return True
    return False


def box_of(lon, lat):
    for k, v in BOXES.items():
        x0, y0, x1, y1 = v['bbox']
        if x0 <= lon < x1 and y0 <= lat < y1:
            return k
    return None


def ensure_dirs():
    for d in (WS, MAT, PRIV, BLIND, QC, EXP, AUD, LOGS, CACHE):
        os.makedirs(d, exist_ok=True)


def stable_uid(kind, key):
    """稳定可复算的 uid（namespace 安全）：kind 如 'R'/'V'，key 如 'v2:point_id'。"""
    h = hashlib.sha1(('%s|%s' % (kind, key)).encode('utf-8')).hexdigest()[:12]
    return '%s_%s' % (kind.lower(), h)


def block_id(lon, lat, deg=None):
    """空间块 id（默认 0.25°）。"""
    d = deg or BLOCK_DEG
    import math
    return '%s_%d_%d' % (('%g' % d).replace('.', 'p'), int(math.floor(lon / d)), int(math.floor(lat / d)))


def sha256_file(fp, chunk=1 << 20):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def jdump(fp, obj):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def jload(fp, default=None):
    try:
        with open(fp, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


if __name__ == '__main__':
    print(json.dumps({
        'WS': WS, 'POOL_V2': POOL_V2, 'RAST2': RAST2, 'DEM': DEM,
        'roles': {'R': QUOTA_R, 'V': QUOTA_V, 'total': QUOTA_TOTAL},
        'pairs': {'total': PAIRS_TOTAL, 'R': PAIRS_R, 'V': PAIRS_V},
        'dispute': {'total': DISPUTE_TOTAL, 'R': DISPUTE_R, 'V': DISPUTE_V},
        'block_deg': BLOCK_DEG, 'seed': SEED, 'id_prefix': ID_PREFIX,
    }, ensure_ascii=False, indent=1))

# -*- coding: utf-8 -*-
"""oe0_conf — OlmoEarth v1.3 嵌入接入的单一配置源
* 输入：无（常量）
* 规则源：2026-10-26 实测（原生 S3 桶 + Zarr v3 + 中国覆盖矩阵），见 技术文档/87
* 门槛：无
* 输出：常量集合，供 oe1–oe4 引用
* 用法：from oe0_conf import *
"""
import os

# ── 存储端点（2026-10-26 实测结论）────────────────────────────────────────
# ★ 首选：AWS 原生 path-style 桶。实测比 Cloudflare 代理快 2.5 倍（3.9 vs 1.6 MB/s 单流）
#   桶位于 us-west-2（已用 ?location 核实返回 us-west-2）
BUCKET = 'us-west-2.opendata.source.coop'
S3_ENDPOINT = 'https://s3.us-west-2.amazonaws.com'
AWS_REGION = 'us-west-2'

# 仅在自建 EC2（us-west-2 内）时用的内网写法：桶名 + region，不传 endpoint
# 实测：EC2 同区访问 S3 不计流量费（AWS 文档）；配 Gateway VPC Endpoint 走私有路由
S3_URL_NATIVE = f's3://{BUCKET}/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr'
# 本地开发用的 https 等价位（obstore 需 endpoint）
S3_URL_HTTP = f'{S3_ENDPOINT}/{BUCKET}/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr'

# ✗ 不推荐：Cloudflare 前置的官方教程端点。国内实测仅 1.4–1.6 MB/s，且多一跳 CDN
S3_URL_CF = 'https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr'

# ── 数据规格（从 zarr.json 读出的实测值，勿改）────────────────────────────
ZONE_PREFIX = 'utm'          # utm01..utm60，每带一个组
YEARS = list(range(2017, 2026))   # time 轴 9 个年份
N_DIM = 128                  # 完整嵌入维度
N_DIM_64 = 64                # Matryoshka：band 0:64 即完整 64 维嵌入
NODATA = -128                # int8 nodata（★不是"非陆地"，只是该年无可用影像）
GSD_M = 10.0                 # 10 m
SHARD_PX = 2048              # 外层 shard 2048×2048 像元
CHUNK_PX = 256               # 内层 chunk 256×256 像元 × 64 波段
COMPRESSION = 'zstd level 3'

# 反量化：int8 → float32。量化方式为 signed_power p=2
#   值 = (|q| / 127.5) ** 2.0 * sign(q)
# ★ 与 AEF（AlphaEarth 64 维 float32）不同：这里是 int8 有损量化，必须先反量化
QUANT_POWER = 2.0
QUANT_SCALE = 127.5

# ── 中国覆盖矩阵（2026-10-26 实测 12 点 × 9 年，决定性约束）─────────────
#   2017–2020：0/12 点有数据  → 中国境内完全无覆盖
#   2021     ：6/12 点有数据  → 部分覆盖，不可用于全国生产
#   2022–2025：12/12 点有数据 → 可用
COVERAGE_CN = {
    2017: 'none', 2018: 'none', 2019: 'none', 2020: 'none',
    2021: 'partial', 2022: 'full', 2023: 'full', 2024: 'full', 2025: 'full',
}
USABLE_YEARS_CN = [y for y, v in COVERAGE_CN.items() if v == 'full']   # [2022..2025]

# ── 中国涉及的 UTM 带（73°E–135°E → utm43..utm53）──────────────────────
ZONES_CN = [f'{ZONE_PREFIX}{z:02d}' for z in range(43, 54)]

# ── 成本基准（2026-10-26 实测/官方价目）─────────────────────────────────
# 中国全境单年：128 维 9.60 TB / 64 维 4.80 TB（陆地 9.6e6 km² 满算上限）
BYTES_PER_YEAR_128TB = 9.60
BYTES_PER_YEAR_64TB = 4.80
# 本机实测吞吐
MEASURED_SINGLE_MBPS = 31.0     # 单流 3.9 MB/s
MEASURED_8STREAM_MBPS = 50.0
MEASURED_SATURATED_MBPS = 71.0  # 64 流饱和
LOCAL_LINK_CEILING_MBPS = 106.0 # Cloudflare 16 并发实测本机下行上限
# AWS us-west-2 官方价（Price List API，2026-10-26）
EC2_PRICE = {'c6in.4xlarge': 0.9072, 'c7i.4xlarge': 0.7140,
             'c7g.4xlarge': 0.5800, 'm7i.2xlarge': 0.4032}
EGRESS_PER_GB = 0.09            # 前 10 TB/月
EGRESS_FREE_GB = 100.0          # 全球聚合每月免费 100 GB
INGRESS_PER_GB = 0.0            # 入站免费
SAME_REGION_S3_TO_EC2 = 0.0     # 同区 EC2↔S3 免费

PROJ = r'F:\地理所\论文\中国土地覆盖数据_2017-2024'
WORK = r'F:\lc_work\olmoearth'
CACHE = os.path.join(WORK, 'cache')      # 本地分片缓存
OUT = os.path.join(WORK, 'out')


def open_root(url=None, region=AWS_REGION, endpoint=S3_ENDPOINT):
    """打开 Zarr 根组。默认走原生 S3；在 EC2 内可传 url=S3_URL_NATIVE, endpoint=None。"""
    import zarr
    from obstore.store import S3Store
    zarr.config.set({'array.sharding_coalesce_max_gap_bytes': 0})
    url = url or S3_URL_HTTP
    kw = {'skip_signature': True}
    if endpoint:
        kw['endpoint'] = endpoint          # https 端点必须显式给 endpoint
    else:
        kw['region'] = region              # EC2 内网：只给 region，走 AWS 私有路由
    return zarr.open_group(zarr.storage.ObjectStore(S3Store.from_url(url, **kw),
                                                    read_only=True), mode='r')


def dequant(q):
    """int8 → float32 反量化（signed_power p=2）。nodata(-128) 返回 nan。"""
    import numpy as np
    q = np.asarray(q, dtype=np.float32)
    out = (np.abs(q) / QUANT_SCALE) ** QUANT_POWER * np.sign(q)
    out[q == NODATA] = np.nan
    return out


def lonlat_to_zone(lon):
    """经度 → UTM 带号（OlmoEarth 用北半球 EPSG:326NN，南半球靠 y<0 表达）。"""
    return int((lon + 180) // 6) + 1

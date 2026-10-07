# -*- coding: utf-8 -*-
"""
prod_conf.py — 全国生产链路的单一配置源（路径 / 账号编队 / 常量）
所有 p*.py 与生成的 worker 都 import 本文件，避免路径与账号信息多处漂移。
"""
import os

# ---------- 路径 ----------
PROJ = r'F:/地理所/论文/中国土地覆盖数据_2017-2024'          # 代码/文档/母库（权威）
KB = r'Z:/地理所/论文/中国土地覆盖数据_2017-2024'            # 重产物（权威）
MIRROR = r'K:/地理所/论文/中国土地覆盖数据_2017-2024'         # 镜像（可选）
WORK = r'F:/lc_work'                                        # SSD 工作区（可重建）
CRED = r'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'

PROD = os.path.join(PROJ, '代码', '8.全国生产')              # 本目录
FLEET = os.path.join(PROD, 'fleet')                         # 各账号差异化 worker
PLAN = os.path.join(PROD, 'plan')                           # 分区/分配（小文件，随代码走）
LEDGER = os.path.join(KB, '生产输出', '元数据')               # 台账
RASTER = os.path.join(KB, '生产输出', '栅格')                # 成品栅格
PLOG = os.path.join(KB, '生产输出', '日志')                   # 生产日志
# 下载暂存（大文件，走 F 的 SSD，拼接完即迁 Z）
STAGE = os.path.join(WORK, 'prod_stage')
# 母库与评估（复用既有事实来源）
MOTHER = os.path.join(PROJ, '数据/本地处理/样本重建/r7_train.parquet')
VALIDITY = os.path.join(PROJ, '数据/本地处理/样本重建/r7_train_validity.parquet')
HOLDOUT = os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2_holdout_blocks.json')
POOL = os.path.join(PROJ, '数据/本地处理/样本底座/validation_pool_v2.parquet')
BOUNDARY = os.path.join(PROJ, '数据/边界/china_100000_full.json')

# ---------- GEE ----------
AEF = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
FEATS = ['A%02d' % i for i in range(64)]
YEARS = list(range(2017, 2025))
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
PROXY_URL = 'socks5h://127.0.0.1:7890'

# ---------- 生产参数 ----------
TILE_DEG = 2.0                # 分区格网（度）
BBOX = (73.0, 18.0, 135.0, 54.0)   # 中国陆域外包（含少量海域，落格时按边界筛）
SUBTILE_DEG = 0.25            # 下载子块：10m 单波段 uint8 下 0.25°≈7.7MB，安全低于 48MB 上限
TILE_SCALE = 4                # classify 的 tileScale（各账号会再差异化）
MAX_TREE_MB = 38.0            # 内联树字符串体积预算（GEE 请求上限 48MB，留 20% 余量）

# ---------- 账号编队 ----------
# role: free = 当前无任务可直接用；busy = 有任务在跑，排队前先复核；spare = 备用不启用
# proj = 该账号**已验证可用**的项目号（来自补样舰队实测）；folder = 该账号的资产目录名（差异化）
ACCOUNTS = {
    'zixen8v8':     dict(proj='braided-horizon-508210-a5', role='free',  folder='lcf_a01'),
    'y30b63ye':     dict(proj='quick-cache-508211-s9',     role='free',  folder='lcf_a02'),
    'zitwwufh':     dict(proj='fast-drake-508210-f7',      role='free',  folder='lcf_a03'),
    'nhqz5uj':      dict(proj='sunlit-center-508411-g3',   role='free',  folder='lcf_a04'),
    'r45smj3u':     dict(proj='grounded-apogee-508406-r1', role='free',  folder='lcf_a05'),
    'seqsiu':       dict(proj='electric-orbit-508407-g4',  role='free',  folder='lcf_a06'),
    'iu5f4z':       dict(proj='ringed-sentinel-508412-d8', role='free',  folder='lcf_a07'),
    '1yxth5g':      dict(proj='round-concept-508309-h8',   role='free',  folder='lcf_a08'),   # 2026-09-15 换仓：vast-logic 累计 80h，按单仓 150h 纪律切换
    '11e0n00':      dict(proj='rock-sentinel-508409-s9',   role='free',  folder='lcf_c01'),   # 2026-09-15 换仓：N0501 在 lustrous-vial 僵死 8h（用户授权取消重提）
    'ief3nj':       dict(proj='cellular-client-508410-t6', role='free',  folder='lcf_c02'),   # 2026-09-15 换仓：temporal-web 累计 148h，触单仓 150h 纪律线
    'hte4021n':     dict(proj='turnkey-skill-508412-s8',   role='free',  folder='lcf_a09'),
    '5rqs3xw2':     dict(proj='tokyo-kingdom-508501-t0',   role='busy',  folder='lcf_b01'),
    '679i9zo111222': dict(proj='silicon-webbing-508500-t4', role='busy', folder='lcf_b02'),
    'als74akz':     dict(proj='the-method-508211-s4',      role='busy',  folder='lcf_b03'),
    'e5h08k':       dict(proj='tonal-baton-508403-m1',     role='busy',  folder='lcf_b04'),
    'gm9ufoo4':     dict(proj='xenon-momentum-508404-k3',  role='busy',  folder='lcf_b05'),
    'hqzub6':       dict(proj='copper-bot-508501-c4',      role='busy',  folder='lcf_b06'),
    # 备用（本轮不启用；需要扩容时把 role 改 free 并在对照表登记任务类型）
    'ppzynq':       dict(proj='', role='spare', folder='lcf_s02'),
    'pcrw36d':      dict(proj='', role='spare', folder='lcf_s03'),
    '3qb2r4':       dict(proj='', role='spare', folder='lcf_s04'),
}

# 换仓登记：账号 → [换仓前仍在跑任务的旧项目]。worker 提交前会一并检查，
# 否则新仓的 operations 看不到旧仓在跑的任务，会造成同瓦片重复提交（2026-09-15 实测踩中）
LEGACY_PROJS = {'ief3nj': ['temporal-web-508410-h0'],
                '11e0n00': ['lustrous-vial-508409-q1']}

# 同域账号对（同自定域=强关联信号）→ 组内两账号**绝不可派同类任务**
SAME_DOMAIN_PAIRS = [('s8xpcl1w', 'hte4021n'), ('ppzynq', 'r45smj3u')]


def usable_accounts(include_busy=False):
    ok = [a for a, v in ACCOUNTS.items() if v['role'] == 'free' and v['proj']]
    if include_busy:
        ok += [a for a, v in ACCOUNTS.items() if v['role'] == 'busy' and v['proj']]
    return sorted(ok)


def ensure_dirs():
    for d in (FLEET, PLAN, LEDGER, RASTER, PLOG, STAGE):
        os.makedirs(d, exist_ok=True)


def cred_home(acct):
    return os.path.join(CRED, acct)


def apply_account_env(acct):
    """凭证隔离：Windows 必须同时设 HOME 与 USERPROFILE，缺一会静默用错账号。"""
    home = cred_home(acct)
    os.environ['HOME'] = home
    os.environ['USERPROFILE'] = home
    os.environ.setdefault('HTTP_PROXY', PROXY_URL)
    os.environ.setdefault('HTTPS_PROXY', PROXY_URL)
    return home

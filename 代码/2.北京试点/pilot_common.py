# -*- coding: utf-8 -*-
"""
pilot_common.py — 北京试点公共模块
* 账号初始化（HOME+USERPROFILE 双设 + 代理 + ee.Initialize）
* 北京边界/常量
"""
import os, json
import numpy as np

GEE_ACC = r'F:\地理所\论文\博士论文\中期\实验区域\02_中间数据\gee_accounts'
PROXY = {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}

ACCOUNTS = {
    'zsi8emo': {
        'email': 'zsi8emo@gmail.com',
        'pids': ["inductive-seat-507709-s5", "oval-realm-507709-b0",
                 "prefab-mapper-507709-i7", "articulate-life-507709-d0",
                 "peak-elevator-507709-c4", "phonic-cinema-507709-q8",
                 "turnkey-energy-507709-g9", "swift-arcadia-507709-e1",
                 "helical-math-507709-c6", "polar-access-507709-b5",
                 "speedy-method-507709-b4", "the-flame-507708-t5"],
        'role': '样本嵌入提取 + 锚年2020',
    },
    's4ezbd': {
        'email': 's4ezbd@gmail.com',
        'pids': ["swift-yew-507709-e7", "fresh-ward-507709-m0",
                 "premium-hybrid-507709-q3", "sound-machine-507709-s5",
                 "alpine-talon-507709-n6", "logical-river-507709-p2",
                 "opportune-scope-507709-r1", "spartan-lacing-507709-d1",
                 "nifty-edge-507709-s7", "psyched-bruin-507709-v9",
                 "elegant-zodiac-507709-g8", "key-design-507709-g4"],
        'role': '年更新 2017-2019, 2021',
    },
    'w2qe4hiu': {
        'email': 'w2qe4hiu@gmail.com',
        'pids': ["pacific-torus-507710-v5", "norse-idea-507709-n5",
                 "symmetric-fold-507709-a8", "proud-archive-507709-i9",
                 "healthy-result-507709-i3", "deep-ground-507710-u8",
                 "fine-jetty-507709-s4", "fair-canto-507710-t2",
                 "starlit-sandbox-507710-b6", "fair-jigsaw-507709-p4",
                 "bamboo-basis-507710-c2", "white-device-507710-q2"],
        'role': '年更新 2022-2024 + 验证',
    },
}
ANCHOR_PID = {'zsi8emo': 'inductive-seat-507709-s5',
              's4ezbd': 'swift-yew-507709-e7',
              'w2qe4hiu': 'pacific-torus-507710-v5'}

# ---------- 生产配置（区域参数唯一入口；无配置文件时行为与旧版完全一致） ----------
# 全国生产时只需改 数据/本地处理/生产配置.json：bbox=[73,18,135,54]、region="中国"、
# out_dir/task_prefix 换名，p2-p8 无需再改代码。
import json as _json
_CFG_PATH = '数据/本地处理/生产配置.json'
CFG = {}
if os.path.exists(_CFG_PATH):
    CFG = _json.load(open(_CFG_PATH, encoding='utf-8'))

AREA_NAME = CFG.get('area_name', 'beijing')
YEARS = CFG.get('years', list(range(2017, 2025)))
ANCHOR_YEAR = CFG.get('anchor_year', 2020)
BBOX = tuple(CFG.get('bbox', (115.4, 39.4, 117.5, 41.1)))
OUT_DIR = CFG.get('out_dir', '数据/本地处理/北京试点')
TASK_PREFIX = CFG.get('task_prefix', 'CNLC10_BJ')
TRAIN_FILE = CFG.get('train_file', os.path.join(OUT_DIR, 'samples_bj_train.parquet'))
HOLDOUT_FILE = CFG.get('holdout_file', os.path.join(OUT_DIR, 'samples_bj_holdout.parquet'))
EMB_COL = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
FEATS_ALL = [f'A{i:02d}' for i in range(64)]

_REGION_CACHE = None
def area_geojson():
    """试验区边界 geometry：配置 region=行政区名（DataV），缺省北京市"""
    global _REGION_CACHE
    if _REGION_CACHE is None:
        name = CFG.get('region', '北京市')
        d = json.load(open('数据/边界/china_100000_full.json', encoding='utf-8'))
        for f in d['features']:
            if f['properties'].get('name') == name:
                _REGION_CACHE = f['geometry']
                break
        if _REGION_CACHE is None:
            lon0, lat0, lon1, lat1 = BBOX
            _REGION_CACHE = {'type': 'Polygon', 'coordinates': [[
                [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}
    return _REGION_CACHE

# 兼容旧名（旧脚本调用 PC.beijing_geojson()）
BJ_GEOJSON = None
def beijing_geojson():
    global BJ_GEOJSON
    if BJ_GEOJSON is None:
        BJ_GEOJSON = area_geojson()
    return BJ_GEOJSON

def load_account(acct, pid=None):
    """初始化指定账号；返回 (ee模块, pid)"""
    d = os.path.join(GEE_ACC, acct)
    os.environ['HOME'] = d
    os.environ['USERPROFILE'] = d
    set_proxy()
    import ee
    pid = pid or ANCHOR_PID[acct]
    import time
    for a in range(3):
        try:
            ee.Initialize(project=pid)
            break
        except Exception as e:
            if a == 2:
                raise
            print(f'  ee.Initialize 重试({a+1}): {str(e)[:80]}')
            time.sleep(20)
    print(f'[ee] {acct} @ {pid} OK')
    return ee, pid

def set_proxy():
    os.environ.setdefault('HTTP_PROXY', 'socks5h://127.0.0.1:7890')
    os.environ.setdefault('HTTPS_PROXY', 'socks5h://127.0.0.1:7890')

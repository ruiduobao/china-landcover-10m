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

YEARS = list(range(2017, 2025))
BBOX = (115.4, 39.4, 117.5, 41.1)
EMB_COL = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
FEATS_ALL = [f'A{i:02d}' for i in range(64)]

BJ_GEOJSON = None
def beijing_geojson():
    global BJ_GEOJSON
    if BJ_GEOJSON is None:
        d = json.load(open('数据/边界/china_100000_full.json', encoding='utf-8'))
        for f in d['features']:
            if f['properties'].get('name') == '北京市':
                BJ_GEOJSON = f['geometry']
                break
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


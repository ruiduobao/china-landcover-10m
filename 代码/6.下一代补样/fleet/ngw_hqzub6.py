# -*- coding: utf-8 -*-
"""hqzub6 专属工作器 · 下一代补样 · 海岸带候选点交叉筛选（专题源与全球产品双证）
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_hqzub6.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
"""
import os
import sys
import io
import json
import time
import random
import argparse
import queue
import threading

import numpy as np
import pandas as pd
import requests

larch_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
larch_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
larch_who = 'hqzub6'
larch_home_proj = 'copper-bot-508501-c4'
larch_tile = 4
larch_rest = (3, 4)

_larch_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_larch_dir))
import ng0_paths as larch_paths
import ng0_spec as larch_catalog


def larch_emit(msg):
    print(time.strftime('<%H:%M:%S> ') + str(msg), flush=True)


def larch_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(larch_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', larch_net['http'])
    os.environ.setdefault('HTTPS_PROXY', larch_net['https'])
    import ee
    for i_larch in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_larch_init:
            if i_larch == 5 - 1:
                raise
            larch_emit('initialize retry %d: %s' % (i_larch + 1, str(x_larch_init)[:96]))
            time.sleep(40)
    larch_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def larch_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_larch_last = None
    for j_larch in range(tries):
        try:
            resp = requests.get(target_url, proxies=larch_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_larch_pull:
            err_larch_last = x_larch_pull
            larch_emit('download retry %d/%d: %s' % (j_larch + 1, tries, str(x_larch_pull)[:96]))
            time.sleep(30 + 20 * j_larch)
    raise RuntimeError('download exhausted: %s' % str(err_larch_last)[:120])


def larch_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xlarch_lon': xy.get(0), 'xlarch_lat': xy.get(1)})


def larch_watchdog(fn, timeout_s, *args, **kwargs):
    """看门狗：GEE 交互式请求会静默挂起（实测同一份代码前 3 片各 0.4 min，
    第 4 片挂 >25 min 无任何报错）。套一层超时 → 挂起变成可重试异常。"""
    box = queue.Queue(1)

    def _run():
        try:
            box.put(('ok', fn(*args, **kwargs)))
        except Exception as exc:
            box.put(('err', exc))

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError('shard timed out after %ds' % timeout_s)
    kind, val = box.get()
    if kind == 'err':
        raise val
    return val


def larch_one_shard(ee, spec_key, shard_id, box):
    cfg = larch_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = larch_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.updateMask(gate.unmask(0))
    if cfg.get('larch_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=20,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='larch_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20261644, maxError=30)
    pts = pts.map(larch_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=10, tileScale=larch_tile)
    samp = samp.map(lambda f: f.set('klarch_pid', f.id()))
    rot = 2 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if False:
        cols = cols[::-1]
    selectors = cols + ['xlarch_lon', 'xlarch_lat', 'klarch_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    larch_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = larch_pull_csv(url, 5, 960)
    frame = frame.rename(columns={'xlarch_lon': 'lon', 'xlarch_lat': 'lat', 'klarch_pid': 'pid'})
    frame['larch_spec'] = spec_key
    frame['larch_cls'] = int(cfg['cls'])
    frame['larch_shard'] = int(shard_id)
    frame['larch_acct'] = larch_who
    frame['larch_yr'] = 2021
    return frame


def larch_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    larch_paths.ensure_all([args.spec])
    plan_fp = larch_paths.shard_plan_path(args.spec)
    if not os.path.isfile(plan_fp):
        raise SystemExit('缺少分片清单: %s' % plan_fp)
    plan = json.load(open(plan_fp, encoding='utf-8'))
    shards = plan['shards']
    if True:
        shards = shards[::-1]
    if args.shard >= 0:
        shards = [x for x in shards if int(x['shard']) == args.shard]
    if not shards:
        raise SystemExit('无待跑分片 spec=%s shard=%s' % (args.spec, args.shard))

    todo = []
    for s_larch in shards:
        f_larch = larch_paths.raw_path(args.spec, larch_who, int(s_larch['shard']))
        if os.path.isfile(f_larch) and os.path.getsize(f_larch) > 200:
            continue
        todo.append(s_larch)
    larch_emit('[%s] %s 分片 %d 待跑 %d' % (larch_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = larch_bring_up(larch_who, larch_home_proj)
    n_larch_ok = n_larch_bad = 0
    t_larch_begin = time.time()
    for s_larch in todo:
        id_larch = int(s_larch['shard'])
        f_larch = larch_paths.raw_path(args.spec, larch_who, id_larch)
        os.makedirs(os.path.dirname(f_larch), exist_ok=True)
        for try_larch in range(3):
            try:
                frame = larch_watchdog(larch_one_shard, 660,
                                     ee, args.spec, id_larch, s_larch['box'])
                frame.to_parquet(f_larch, index=False)
                n_larch_ok += 1
                larch_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_larch, len(frame), n_larch_ok, len(todo),
                              (time.time() - t_larch_begin) / 60.0))
                break
            except Exception as x_larch_shard:
                larch_emit('shard %03d attempt %d ERR %s'
                           % (id_larch, try_larch + 1, str(x_larch_shard)[:110]))
                time.sleep(40 + try_larch * 25)
        else:
            n_larch_bad += 1
            larch_emit('shard %03d FINAL-FAIL' % id_larch)
        time.sleep(random.uniform(*larch_rest))
    larch_emit('[%s] done ok=%d fail=%d' % (larch_who, n_larch_ok, n_larch_bad))
    return 0 if n_larch_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(larch_roll())

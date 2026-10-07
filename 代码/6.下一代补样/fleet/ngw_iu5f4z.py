# -*- coding: utf-8 -*-
"""iu5f4z 专属工作器 · 下一代补样 · 生态区候选点构建（多源一致性）
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_iu5f4z.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

elm_net = {'http': 'socks5h://127.0.0.1:7890',
             'https': 'socks5h://127.0.0.1:7890'}
elm_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
elm_who = 'iu5f4z'
elm_home_proj = 'ringed-sentinel-508412-d8'
elm_tile = 3
elm_rest = (1, 6)

_elm_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_elm_dir))
import ng0_paths as elm_paths
import ng0_spec as elm_catalog


def elm_emit(msg):
    print(time.strftime('[%H:%M] ') + str(msg), flush=True)


def elm_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(elm_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', elm_net['http'])
    os.environ.setdefault('HTTPS_PROXY', elm_net['https'])
    import ee
    for i_elm in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_elm_init:
            if i_elm == 5 - 1:
                raise
            elm_emit('initialize retry %d: %s' % (i_elm + 1, str(x_elm_init)[:96]))
            time.sleep(30)
    elm_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def elm_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_elm_last = None
    for j_elm in range(tries):
        try:
            resp = requests.get(target_url, proxies=elm_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_elm_pull:
            err_elm_last = x_elm_pull
            elm_emit('download retry %d/%d: %s' % (j_elm + 1, tries, str(x_elm_pull)[:96]))
            time.sleep(15 + 15 * j_elm)
    raise RuntimeError('download exhausted: %s' % str(err_elm_last)[:120])


def elm_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xelm_lon': xy.get(0), 'xelm_lat': xy.get(1)})


def elm_watchdog(fn, timeout_s, *args, **kwargs):
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


def elm_one_shard(ee, spec_key, shard_id, box):
    cfg = elm_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = elm_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.updateMask(gate)
    if cfg.get('elm_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=30,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='elm_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20260685, maxError=10)
    pts = pts.map(elm_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=20, tileScale=elm_tile)
    samp = samp.map(lambda f: f.set('kelm_pid', f.id()))
    rot = 5 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if False:
        cols = cols[::-1]
    selectors = cols + ['xelm_lon', 'xelm_lat', 'kelm_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    elm_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = elm_pull_csv(url, 4, 600)
    frame = frame.rename(columns={'xelm_lon': 'lon', 'xelm_lat': 'lat', 'kelm_pid': 'pid'})
    frame['elm_spec'] = spec_key
    frame['elm_cls'] = int(cfg['cls'])
    frame['elm_shard'] = int(shard_id)
    frame['elm_acct'] = elm_who
    frame['elm_yr'] = 2021
    return frame


def elm_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    elm_paths.ensure_all([args.spec])
    plan_fp = elm_paths.shard_plan_path(args.spec)
    if not os.path.isfile(plan_fp):
        raise SystemExit('缺少分片清单: %s' % plan_fp)
    plan = json.load(open(plan_fp, encoding='utf-8'))
    shards = plan['shards']
    if False:
        shards = shards[::-1]
    if args.shard >= 0:
        shards = [x for x in shards if int(x['shard']) == args.shard]
    if not shards:
        raise SystemExit('无待跑分片 spec=%s shard=%s' % (args.spec, args.shard))

    todo = []
    for s_elm in shards:
        f_elm = elm_paths.raw_path(args.spec, elm_who, int(s_elm['shard']))
        if os.path.isfile(f_elm) and os.path.getsize(f_elm) > 200:
            continue
        todo.append(s_elm)
    elm_emit('[%s] %s 分片 %d 待跑 %d' % (elm_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = elm_bring_up(elm_who, elm_home_proj)
    n_elm_ok = n_elm_bad = 0
    t_elm_begin = time.time()
    for s_elm in todo:
        id_elm = int(s_elm['shard'])
        f_elm = elm_paths.raw_path(args.spec, elm_who, id_elm)
        os.makedirs(os.path.dirname(f_elm), exist_ok=True)
        for try_elm in range(2):
            try:
                frame = elm_watchdog(elm_one_shard, 540,
                                     ee, args.spec, id_elm, s_elm['box'])
                frame.to_parquet(f_elm, index=False)
                n_elm_ok += 1
                elm_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_elm, len(frame), n_elm_ok, len(todo),
                              (time.time() - t_elm_begin) / 60.0))
                break
            except Exception as x_elm_shard:
                elm_emit('shard %03d attempt %d ERR %s'
                           % (id_elm, try_elm + 1, str(x_elm_shard)[:110]))
                time.sleep(30 + try_elm * 20)
        else:
            n_elm_bad += 1
            elm_emit('shard %03d FINAL-FAIL' % id_elm)
        time.sleep(random.uniform(*elm_rest))
    elm_emit('[%s] done ok=%d fail=%d' % (elm_who, n_elm_ok, n_elm_bad))
    return 0 if n_elm_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(elm_roll())

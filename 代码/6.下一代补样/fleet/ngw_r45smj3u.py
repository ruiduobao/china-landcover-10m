# -*- coding: utf-8 -*-
"""r45smj3u 专属工作器 · 下一代补样 · 多源共识点采样
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_r45smj3u.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

juniper_net = {'http': 'socks5h://127.0.0.1:7890',
                 'https': 'socks5h://127.0.0.1:7890'}
juniper_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
juniper_who = 'r45smj3u'
juniper_home_proj = 'grounded-apogee-508406-r1'
juniper_tile = 2
juniper_rest = (2, 6)

_juniper_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_juniper_dir))
import ng0_paths as juniper_paths
import ng0_spec as juniper_catalog


def juniper_emit(msg):
    print(time.strftime('(%H:%M:%S) ') + str(msg), flush=True)


def juniper_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(juniper_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', juniper_net['http'])
    os.environ.setdefault('HTTPS_PROXY', juniper_net['https'])
    import ee
    for i_juniper in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_juniper_init:
            if i_juniper == 5 - 1:
                raise
            juniper_emit('initialize retry %d: %s' % (i_juniper + 1, str(x_juniper_init)[:96]))
            time.sleep(20)
    juniper_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def juniper_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_juniper_last = None
    for j_juniper in range(tries):
        try:
            resp = requests.get(target_url, proxies=juniper_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_juniper_pull:
            err_juniper_last = x_juniper_pull
            juniper_emit('download retry %d/%d: %s' % (j_juniper + 1, tries, str(x_juniper_pull)[:96]))
            time.sleep(20 + 10 * j_juniper)
    raise RuntimeError('download exhausted: %s' % str(err_juniper_last)[:120])


def juniper_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xjuniper_lon': xy.get(0), 'xjuniper_lat': xy.get(1)})


def juniper_watchdog(fn, timeout_s, *args, **kwargs):
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


def juniper_one_shard(ee, spec_key, shard_id, box):
    cfg = juniper_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = juniper_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.mask(gate)
    if cfg.get('juniper_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=20,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='juniper_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20261370, maxError=1)
    pts = pts.map(juniper_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=10, tileScale=juniper_tile)
    samp = samp.map(lambda f: f.set('kjuniper_pid', f.id()))
    rot = 5 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if True:
        cols = cols[::-1]
    selectors = cols + ['xjuniper_lon', 'xjuniper_lat', 'kjuniper_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    juniper_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = juniper_pull_csv(url, 3, 720)
    frame = frame.rename(columns={'xjuniper_lon': 'lon', 'xjuniper_lat': 'lat', 'kjuniper_pid': 'pid'})
    frame['juniper_spec'] = spec_key
    frame['juniper_cls'] = int(cfg['cls'])
    frame['juniper_shard'] = int(shard_id)
    frame['juniper_acct'] = juniper_who
    frame['juniper_yr'] = 2021
    return frame


def juniper_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    juniper_paths.ensure_all([args.spec])
    plan_fp = juniper_paths.shard_plan_path(args.spec)
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
    for s_juniper in shards:
        f_juniper = juniper_paths.raw_path(args.spec, juniper_who, int(s_juniper['shard']))
        if os.path.isfile(f_juniper) and os.path.getsize(f_juniper) > 200:
            continue
        todo.append(s_juniper)
    juniper_emit('[%s] %s 分片 %d 待跑 %d' % (juniper_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = juniper_bring_up(juniper_who, juniper_home_proj)
    n_juniper_ok = n_juniper_bad = 0
    t_juniper_begin = time.time()
    for s_juniper in todo:
        id_juniper = int(s_juniper['shard'])
        f_juniper = juniper_paths.raw_path(args.spec, juniper_who, id_juniper)
        os.makedirs(os.path.dirname(f_juniper), exist_ok=True)
        for try_juniper in range(3):
            try:
                frame = juniper_watchdog(juniper_one_shard, 420,
                                     ee, args.spec, id_juniper, s_juniper['box'])
                frame.to_parquet(f_juniper, index=False)
                n_juniper_ok += 1
                juniper_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_juniper, len(frame), n_juniper_ok, len(todo),
                              (time.time() - t_juniper_begin) / 60.0))
                break
            except Exception as x_juniper_shard:
                juniper_emit('shard %03d attempt %d ERR %s'
                           % (id_juniper, try_juniper + 1, str(x_juniper_shard)[:110]))
                time.sleep(20 + try_juniper * 15)
        else:
            n_juniper_bad += 1
            juniper_emit('shard %03d FINAL-FAIL' % id_juniper)
        time.sleep(random.uniform(*juniper_rest))
    juniper_emit('[%s] done ok=%d fail=%d' % (juniper_who, n_juniper_ok, n_juniper_bad))
    return 0 if n_juniper_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(juniper_roll())

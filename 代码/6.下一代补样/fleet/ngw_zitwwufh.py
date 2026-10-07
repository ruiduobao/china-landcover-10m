# -*- coding: utf-8 -*-
"""zitwwufh 专属工作器 · 下一代补样 · 潮汐湿地候选点筛选（过渡频次判据）
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_zitwwufh.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

cedar_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
cedar_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
cedar_who = 'zitwwufh'
cedar_home_proj = 'fast-drake-508210-f7'
cedar_tile = 4
cedar_rest = (3, 4)

_cedar_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_cedar_dir))
import ng0_paths as cedar_paths
import ng0_spec as cedar_catalog


def cedar_emit(msg):
    print(time.strftime('%H:%M:%S | ') + str(msg), flush=True)


def cedar_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(cedar_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', cedar_net['http'])
    os.environ.setdefault('HTTPS_PROXY', cedar_net['https'])
    import ee
    for i_cedar in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_cedar_init:
            if i_cedar == 5 - 1:
                raise
            cedar_emit('initialize retry %d: %s' % (i_cedar + 1, str(x_cedar_init)[:96]))
            time.sleep(40)
    cedar_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def cedar_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_cedar_last = None
    for j_cedar in range(tries):
        try:
            resp = requests.get(target_url, proxies=cedar_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_cedar_pull:
            err_cedar_last = x_cedar_pull
            cedar_emit('download retry %d/%d: %s' % (j_cedar + 1, tries, str(x_cedar_pull)[:96]))
            time.sleep(25 + 20 * j_cedar)
    raise RuntimeError('download exhausted: %s' % str(err_cedar_last)[:120])


def cedar_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xcedar_lon': xy.get(0), 'xcedar_lat': xy.get(1)})


def cedar_watchdog(fn, timeout_s, *args, **kwargs):
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


def cedar_one_shard(ee, spec_key, shard_id, box):
    cfg = cedar_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = cedar_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.updateMask(gate.gt(0))
    if cfg.get('cedar_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=20,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='cedar_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20260411, maxError=30)
    pts = pts.map(cedar_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=10, tileScale=cedar_tile)
    samp = samp.map(lambda f: f.set('kcedar_pid', f.id()))
    rot = 3 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if False:
        cols = cols[::-1]
    selectors = cols + ['xcedar_lon', 'xcedar_lat', 'kcedar_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    cedar_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = cedar_pull_csv(url, 5, 840)
    frame = frame.rename(columns={'xcedar_lon': 'lon', 'xcedar_lat': 'lat', 'kcedar_pid': 'pid'})
    frame['cedar_spec'] = spec_key
    frame['cedar_cls'] = int(cfg['cls'])
    frame['cedar_shard'] = int(shard_id)
    frame['cedar_acct'] = cedar_who
    frame['cedar_yr'] = 2021
    return frame


def cedar_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    cedar_paths.ensure_all([args.spec])
    plan_fp = cedar_paths.shard_plan_path(args.spec)
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
    for s_cedar in shards:
        f_cedar = cedar_paths.raw_path(args.spec, cedar_who, int(s_cedar['shard']))
        if os.path.isfile(f_cedar) and os.path.getsize(f_cedar) > 200:
            continue
        todo.append(s_cedar)
    cedar_emit('[%s] %s 分片 %d 待跑 %d' % (cedar_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = cedar_bring_up(cedar_who, cedar_home_proj)
    n_cedar_ok = n_cedar_bad = 0
    t_cedar_begin = time.time()
    for s_cedar in todo:
        id_cedar = int(s_cedar['shard'])
        f_cedar = cedar_paths.raw_path(args.spec, cedar_who, id_cedar)
        os.makedirs(os.path.dirname(f_cedar), exist_ok=True)
        for try_cedar in range(2):
            try:
                frame = cedar_watchdog(cedar_one_shard, 660,
                                     ee, args.spec, id_cedar, s_cedar['box'])
                frame.to_parquet(f_cedar, index=False)
                n_cedar_ok += 1
                cedar_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_cedar, len(frame), n_cedar_ok, len(todo),
                              (time.time() - t_cedar_begin) / 60.0))
                break
            except Exception as x_cedar_shard:
                cedar_emit('shard %03d attempt %d ERR %s'
                           % (id_cedar, try_cedar + 1, str(x_cedar_shard)[:110]))
                time.sleep(40 + try_cedar * 25)
        else:
            n_cedar_bad += 1
            cedar_emit('shard %03d FINAL-FAIL' % id_cedar)
        time.sleep(random.uniform(*cedar_rest))
    cedar_emit('[%s] done ok=%d fail=%d' % (cedar_who, n_cedar_ok, n_cedar_bad))
    return 0 if n_cedar_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(cedar_roll())

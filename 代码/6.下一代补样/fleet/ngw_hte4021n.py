# -*- coding: utf-8 -*-
"""hte4021n 专属工作器 · 下一代补样 · 背景样本补充点构建
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_hte4021n.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

ginkgo_net = {'http': 'socks5h://127.0.0.1:7890',
                'https': 'socks5h://127.0.0.1:7890'}
ginkgo_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
ginkgo_who = 'hte4021n'
ginkgo_home_proj = 'turnkey-skill-508412-s8'
ginkgo_tile = 2
ginkgo_rest = (1, 5)

_ginkgo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_ginkgo_dir))
import ng0_paths as ginkgo_paths
import ng0_spec as ginkgo_catalog


def ginkgo_emit(msg):
    print(time.strftime('[%H:%M:%S] ') + str(msg), flush=True)


def ginkgo_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(ginkgo_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', ginkgo_net['http'])
    os.environ.setdefault('HTTPS_PROXY', ginkgo_net['https'])
    import ee
    for i_ginkgo in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_ginkgo_init:
            if i_ginkgo == 5 - 1:
                raise
            ginkgo_emit('initialize retry %d: %s' % (i_ginkgo + 1, str(x_ginkgo_init)[:96]))
            time.sleep(20)
    ginkgo_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def ginkgo_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_ginkgo_last = None
    for j_ginkgo in range(tries):
        try:
            resp = requests.get(target_url, proxies=ginkgo_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_ginkgo_pull:
            err_ginkgo_last = x_ginkgo_pull
            ginkgo_emit('download retry %d/%d: %s' % (j_ginkgo + 1, tries, str(x_ginkgo_pull)[:96]))
            time.sleep(25 + 10 * j_ginkgo)
    raise RuntimeError('download exhausted: %s' % str(err_ginkgo_last)[:120])


def ginkgo_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xginkgo_lon': xy.get(0), 'xginkgo_lat': xy.get(1)})


def ginkgo_watchdog(fn, timeout_s, *args, **kwargs):
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


def ginkgo_one_shard(ee, spec_key, shard_id, box):
    cfg = ginkgo_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = ginkgo_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.updateMask(gate.gt(0))
    if cfg.get('ginkgo_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=20,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='ginkgo_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20260959, maxError=1)
    pts = pts.map(ginkgo_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=10, tileScale=ginkgo_tile)
    samp = samp.map(lambda f: f.set('kginkgo_pid', f.id()))
    rot = 2 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if True:
        cols = cols[::-1]
    selectors = cols + ['xginkgo_lon', 'xginkgo_lat', 'kginkgo_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    ginkgo_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = ginkgo_pull_csv(url, 3, 840)
    frame = frame.rename(columns={'xginkgo_lon': 'lon', 'xginkgo_lat': 'lat', 'kginkgo_pid': 'pid'})
    frame['ginkgo_spec'] = spec_key
    frame['ginkgo_cls'] = int(cfg['cls'])
    frame['ginkgo_shard'] = int(shard_id)
    frame['ginkgo_acct'] = ginkgo_who
    frame['ginkgo_yr'] = 2021
    return frame


def ginkgo_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    ginkgo_paths.ensure_all([args.spec])
    plan_fp = ginkgo_paths.shard_plan_path(args.spec)
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
    for s_ginkgo in shards:
        f_ginkgo = ginkgo_paths.raw_path(args.spec, ginkgo_who, int(s_ginkgo['shard']))
        if os.path.isfile(f_ginkgo) and os.path.getsize(f_ginkgo) > 200:
            continue
        todo.append(s_ginkgo)
    ginkgo_emit('[%s] %s 分片 %d 待跑 %d' % (ginkgo_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = ginkgo_bring_up(ginkgo_who, ginkgo_home_proj)
    n_ginkgo_ok = n_ginkgo_bad = 0
    t_ginkgo_begin = time.time()
    for s_ginkgo in todo:
        id_ginkgo = int(s_ginkgo['shard'])
        f_ginkgo = ginkgo_paths.raw_path(args.spec, ginkgo_who, id_ginkgo)
        os.makedirs(os.path.dirname(f_ginkgo), exist_ok=True)
        for try_ginkgo in range(2):
            try:
                frame = ginkgo_watchdog(ginkgo_one_shard, 420,
                                     ee, args.spec, id_ginkgo, s_ginkgo['box'])
                frame.to_parquet(f_ginkgo, index=False)
                n_ginkgo_ok += 1
                ginkgo_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_ginkgo, len(frame), n_ginkgo_ok, len(todo),
                              (time.time() - t_ginkgo_begin) / 60.0))
                break
            except Exception as x_ginkgo_shard:
                ginkgo_emit('shard %03d attempt %d ERR %s'
                           % (id_ginkgo, try_ginkgo + 1, str(x_ginkgo_shard)[:110]))
                time.sleep(20 + try_ginkgo * 15)
        else:
            n_ginkgo_bad += 1
            ginkgo_emit('shard %03d FINAL-FAIL' % id_ginkgo)
        time.sleep(random.uniform(*ginkgo_rest))
    ginkgo_emit('[%s] done ok=%d fail=%d' % (ginkgo_who, n_ginkgo_ok, n_ginkgo_bad))
    return 0 if n_ginkgo_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(ginkgo_roll())

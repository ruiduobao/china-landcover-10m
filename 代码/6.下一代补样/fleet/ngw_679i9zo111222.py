# -*- coding: utf-8 -*-
"""679i9zo111222 专属工作器 · 下一代补样 · 生态区候选点构建（多源一致性）
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_679i9zo111222.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

oak_net = {'http': 'socks5h://127.0.0.1:7890',
             'https': 'socks5h://127.0.0.1:7890'}
oak_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
oak_who = '679i9zo111222'
oak_home_proj = 'silicon-webbing-508500-t4'
oak_tile = 4
oak_rest = (1, 6)

_oak_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_oak_dir))
import ng0_paths as oak_paths
import ng0_spec as oak_catalog


def oak_emit(msg):
    print(time.strftime('%H:%M:%S | ') + str(msg), flush=True)


def oak_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(oak_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', oak_net['http'])
    os.environ.setdefault('HTTPS_PROXY', oak_net['https'])
    import ee
    for i_oak in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_oak_init:
            if i_oak == 5 - 1:
                raise
            oak_emit('initialize retry %d: %s' % (i_oak + 1, str(x_oak_init)[:96]))
            time.sleep(40)
    oak_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def oak_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_oak_last = None
    for j_oak in range(tries):
        try:
            resp = requests.get(target_url, proxies=oak_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_oak_pull:
            err_oak_last = x_oak_pull
            oak_emit('download retry %d/%d: %s' % (j_oak + 1, tries, str(x_oak_pull)[:96]))
            time.sleep(25 + 20 * j_oak)
    raise RuntimeError('download exhausted: %s' % str(err_oak_last)[:120])


def oak_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xoak_lon': xy.get(0), 'xoak_lat': xy.get(1)})


def oak_watchdog(fn, timeout_s, *args, **kwargs):
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


def oak_one_shard(ee, spec_key, shard_id, box):
    cfg = oak_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = oak_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.updateMask(gate.gt(0))
    if cfg.get('oak_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=20,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='oak_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20262055, maxError=30)
    pts = pts.map(oak_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=10, tileScale=oak_tile)
    samp = samp.map(lambda f: f.set('koak_pid', f.id()))
    rot = 5 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if False:
        cols = cols[::-1]
    selectors = cols + ['xoak_lon', 'xoak_lat', 'koak_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    oak_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = oak_pull_csv(url, 5, 840)
    frame = frame.rename(columns={'xoak_lon': 'lon', 'xoak_lat': 'lat', 'koak_pid': 'pid'})
    frame['oak_spec'] = spec_key
    frame['oak_cls'] = int(cfg['cls'])
    frame['oak_shard'] = int(shard_id)
    frame['oak_acct'] = oak_who
    frame['oak_yr'] = 2021
    return frame


def oak_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    oak_paths.ensure_all([args.spec])
    plan_fp = oak_paths.shard_plan_path(args.spec)
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
    for s_oak in shards:
        f_oak = oak_paths.raw_path(args.spec, oak_who, int(s_oak['shard']))
        if os.path.isfile(f_oak) and os.path.getsize(f_oak) > 200:
            continue
        todo.append(s_oak)
    oak_emit('[%s] %s 分片 %d 待跑 %d' % (oak_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = oak_bring_up(oak_who, oak_home_proj)
    n_oak_ok = n_oak_bad = 0
    t_oak_begin = time.time()
    for s_oak in todo:
        id_oak = int(s_oak['shard'])
        f_oak = oak_paths.raw_path(args.spec, oak_who, id_oak)
        os.makedirs(os.path.dirname(f_oak), exist_ok=True)
        for try_oak in range(2):
            try:
                frame = oak_watchdog(oak_one_shard, 660,
                                     ee, args.spec, id_oak, s_oak['box'])
                frame.to_parquet(f_oak, index=False)
                n_oak_ok += 1
                oak_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_oak, len(frame), n_oak_ok, len(todo),
                              (time.time() - t_oak_begin) / 60.0))
                break
            except Exception as x_oak_shard:
                oak_emit('shard %03d attempt %d ERR %s'
                           % (id_oak, try_oak + 1, str(x_oak_shard)[:110]))
                time.sleep(40 + try_oak * 25)
        else:
            n_oak_bad += 1
            oak_emit('shard %03d FINAL-FAIL' % id_oak)
        time.sleep(random.uniform(*oak_rest))
    oak_emit('[%s] done ok=%d fail=%d' % (oak_who, n_oak_ok, n_oak_bad))
    return 0 if n_oak_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(oak_roll())

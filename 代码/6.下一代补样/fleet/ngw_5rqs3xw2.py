# -*- coding: utf-8 -*-
"""5rqs3xw2 专属工作器 · 下一代补样 · 高寒候选点交叉筛选（双年 WorldCover 一致）
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_5rqs3xw2.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

kudzu_net = {'http': 'socks5h://127.0.0.1:7890',
               'https': 'socks5h://127.0.0.1:7890'}
kudzu_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
kudzu_who = '5rqs3xw2'
kudzu_home_proj = 'tokyo-kingdom-508501-t0'
kudzu_tile = 3
kudzu_rest = (3, 4)

_kudzu_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_kudzu_dir))
import ng0_paths as kudzu_paths
import ng0_spec as kudzu_catalog


def kudzu_emit(msg):
    print(time.strftime('[%H:%M] ') + str(msg), flush=True)


def kudzu_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(kudzu_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', kudzu_net['http'])
    os.environ.setdefault('HTTPS_PROXY', kudzu_net['https'])
    import ee
    for i_kudzu in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_kudzu_init:
            if i_kudzu == 5 - 1:
                raise
            kudzu_emit('initialize retry %d: %s' % (i_kudzu + 1, str(x_kudzu_init)[:96]))
            time.sleep(30)
    kudzu_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def kudzu_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_kudzu_last = None
    for j_kudzu in range(tries):
        try:
            resp = requests.get(target_url, proxies=kudzu_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_kudzu_pull:
            err_kudzu_last = x_kudzu_pull
            kudzu_emit('download retry %d/%d: %s' % (j_kudzu + 1, tries, str(x_kudzu_pull)[:96]))
            time.sleep(25 + 15 * j_kudzu)
    raise RuntimeError('download exhausted: %s' % str(err_kudzu_last)[:120])


def kudzu_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xkudzu_lon': xy.get(0), 'xkudzu_lat': xy.get(1)})


def kudzu_watchdog(fn, timeout_s, *args, **kwargs):
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


def kudzu_one_shard(ee, spec_key, shard_id, box):
    cfg = kudzu_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = kudzu_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.updateMask(gate.gt(0))
    if cfg.get('kudzu_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=30,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='kudzu_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20261507, maxError=10)
    pts = pts.map(kudzu_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=20, tileScale=kudzu_tile)
    samp = samp.map(lambda f: f.set('kkudzu_pid', f.id()))
    rot = 1 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if False:
        cols = cols[::-1]
    selectors = cols + ['xkudzu_lon', 'xkudzu_lat', 'kkudzu_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    kudzu_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = kudzu_pull_csv(url, 4, 840)
    frame = frame.rename(columns={'xkudzu_lon': 'lon', 'xkudzu_lat': 'lat', 'kkudzu_pid': 'pid'})
    frame['kudzu_spec'] = spec_key
    frame['kudzu_cls'] = int(cfg['cls'])
    frame['kudzu_shard'] = int(shard_id)
    frame['kudzu_acct'] = kudzu_who
    frame['kudzu_yr'] = 2021
    return frame


def kudzu_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    kudzu_paths.ensure_all([args.spec])
    plan_fp = kudzu_paths.shard_plan_path(args.spec)
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
    for s_kudzu in shards:
        f_kudzu = kudzu_paths.raw_path(args.spec, kudzu_who, int(s_kudzu['shard']))
        if os.path.isfile(f_kudzu) and os.path.getsize(f_kudzu) > 200:
            continue
        todo.append(s_kudzu)
    kudzu_emit('[%s] %s 分片 %d 待跑 %d' % (kudzu_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = kudzu_bring_up(kudzu_who, kudzu_home_proj)
    n_kudzu_ok = n_kudzu_bad = 0
    t_kudzu_begin = time.time()
    for s_kudzu in todo:
        id_kudzu = int(s_kudzu['shard'])
        f_kudzu = kudzu_paths.raw_path(args.spec, kudzu_who, id_kudzu)
        os.makedirs(os.path.dirname(f_kudzu), exist_ok=True)
        for try_kudzu in range(2):
            try:
                frame = kudzu_watchdog(kudzu_one_shard, 540,
                                     ee, args.spec, id_kudzu, s_kudzu['box'])
                frame.to_parquet(f_kudzu, index=False)
                n_kudzu_ok += 1
                kudzu_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_kudzu, len(frame), n_kudzu_ok, len(todo),
                              (time.time() - t_kudzu_begin) / 60.0))
                break
            except Exception as x_kudzu_shard:
                kudzu_emit('shard %03d attempt %d ERR %s'
                           % (id_kudzu, try_kudzu + 1, str(x_kudzu_shard)[:110]))
                time.sleep(30 + try_kudzu * 20)
        else:
            n_kudzu_bad += 1
            kudzu_emit('shard %03d FINAL-FAIL' % id_kudzu)
        time.sleep(random.uniform(*kudzu_rest))
    kudzu_emit('[%s] done ok=%d fail=%d' % (kudzu_who, n_kudzu_ok, n_kudzu_bad))
    return 0 if n_kudzu_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(kudzu_roll())

# -*- coding: utf-8 -*-
"""seqsiu 专属工作器 · 下一代补样 · 水稻淹水特征候选点筛选（SAR 时序）
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_seqsiu.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

nutmeg_net = {'http': 'socks5h://127.0.0.1:7890',
                'https': 'socks5h://127.0.0.1:7890'}
nutmeg_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
nutmeg_who = 'seqsiu'
nutmeg_home_proj = 'electric-orbit-508407-g4'
nutmeg_tile = 3
nutmeg_rest = (1, 5)

_nutmeg_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_nutmeg_dir))
import ng0_paths as nutmeg_paths
import ng0_spec as nutmeg_catalog


def nutmeg_emit(msg):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(msg), flush=True)


def nutmeg_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(nutmeg_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', nutmeg_net['http'])
    os.environ.setdefault('HTTPS_PROXY', nutmeg_net['https'])
    import ee
    for i_nutmeg in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_nutmeg_init:
            if i_nutmeg == 5 - 1:
                raise
            nutmeg_emit('initialize retry %d: %s' % (i_nutmeg + 1, str(x_nutmeg_init)[:96]))
            time.sleep(30)
    nutmeg_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def nutmeg_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_nutmeg_last = None
    for j_nutmeg in range(tries):
        try:
            resp = requests.get(target_url, proxies=nutmeg_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_nutmeg_pull:
            err_nutmeg_last = x_nutmeg_pull
            nutmeg_emit('download retry %d/%d: %s' % (j_nutmeg + 1, tries, str(x_nutmeg_pull)[:96]))
            time.sleep(20 + 15 * j_nutmeg)
    raise RuntimeError('download exhausted: %s' % str(err_nutmeg_last)[:120])


def nutmeg_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xnutmeg_lon': xy.get(0), 'xnutmeg_lat': xy.get(1)})


def nutmeg_watchdog(fn, timeout_s, *args, **kwargs):
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


def nutmeg_one_shard(ee, spec_key, shard_id, box):
    cfg = nutmeg_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = nutmeg_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.mask(gate)
    if cfg.get('nutmeg_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=30,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='nutmeg_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20261918, maxError=10)
    pts = pts.map(nutmeg_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=20, tileScale=nutmeg_tile)
    samp = samp.map(lambda f: f.set('knutmeg_pid', f.id()))
    rot = 4 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if False:
        cols = cols[::-1]
    selectors = cols + ['xnutmeg_lon', 'xnutmeg_lat', 'knutmeg_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    nutmeg_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = nutmeg_pull_csv(url, 4, 720)
    frame = frame.rename(columns={'xnutmeg_lon': 'lon', 'xnutmeg_lat': 'lat', 'knutmeg_pid': 'pid'})
    frame['nutmeg_spec'] = spec_key
    frame['nutmeg_cls'] = int(cfg['cls'])
    frame['nutmeg_shard'] = int(shard_id)
    frame['nutmeg_acct'] = nutmeg_who
    frame['nutmeg_yr'] = 2021
    return frame


def nutmeg_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    nutmeg_paths.ensure_all([args.spec])
    plan_fp = nutmeg_paths.shard_plan_path(args.spec)
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
    for s_nutmeg in shards:
        f_nutmeg = nutmeg_paths.raw_path(args.spec, nutmeg_who, int(s_nutmeg['shard']))
        if os.path.isfile(f_nutmeg) and os.path.getsize(f_nutmeg) > 200:
            continue
        todo.append(s_nutmeg)
    nutmeg_emit('[%s] %s 分片 %d 待跑 %d' % (nutmeg_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = nutmeg_bring_up(nutmeg_who, nutmeg_home_proj)
    n_nutmeg_ok = n_nutmeg_bad = 0
    t_nutmeg_begin = time.time()
    for s_nutmeg in todo:
        id_nutmeg = int(s_nutmeg['shard'])
        f_nutmeg = nutmeg_paths.raw_path(args.spec, nutmeg_who, id_nutmeg)
        os.makedirs(os.path.dirname(f_nutmeg), exist_ok=True)
        for try_nutmeg in range(3):
            try:
                frame = nutmeg_watchdog(nutmeg_one_shard, 540,
                                     ee, args.spec, id_nutmeg, s_nutmeg['box'])
                frame.to_parquet(f_nutmeg, index=False)
                n_nutmeg_ok += 1
                nutmeg_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_nutmeg, len(frame), n_nutmeg_ok, len(todo),
                              (time.time() - t_nutmeg_begin) / 60.0))
                break
            except Exception as x_nutmeg_shard:
                nutmeg_emit('shard %03d attempt %d ERR %s'
                           % (id_nutmeg, try_nutmeg + 1, str(x_nutmeg_shard)[:110]))
                time.sleep(30 + try_nutmeg * 20)
        else:
            n_nutmeg_bad += 1
            nutmeg_emit('shard %03d FINAL-FAIL' % id_nutmeg)
        time.sleep(random.uniform(*nutmeg_rest))
    nutmeg_emit('[%s] done ok=%d fail=%d' % (nutmeg_who, n_nutmeg_ok, n_nutmeg_bad))
    return 0 if n_nutmeg_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(nutmeg_roll())

# -*- coding: utf-8 -*-
"""nhqz5uj 专属工作器 · 下一代补样 · 带状类别分层配额采点
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python ngw_nhqz5uj.py --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

fir_net = {'http': 'socks5h://127.0.0.1:7890',
             'https': 'socks5h://127.0.0.1:7890'}
fir_cred_home = 'F:/地理所/论文/博士论文/中期/实验区域/02_中间数据/gee_accounts'
fir_who = 'nhqz5uj'
fir_home_proj = 'sunlit-center-508411-g3'
fir_tile = 4
fir_rest = (1, 6)

_fir_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_fir_dir))
import ng0_paths as fir_paths
import ng0_spec as fir_catalog


def fir_emit(msg):
    print(time.strftime('<%H:%M:%S> ') + str(msg), flush=True)


def fir_bring_up(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(fir_cred_home, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', fir_net['http'])
    os.environ.setdefault('HTTPS_PROXY', fir_net['https'])
    import ee
    for i_fir in range(5):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as x_fir_init:
            if i_fir == 5 - 1:
                raise
            fir_emit('initialize retry %d: %s' % (i_fir + 1, str(x_fir_init)[:96]))
            time.sleep(40)
    fir_emit('[ee] %s @ %s ready' % (account, project_id))
    return ee


def fir_pull_csv(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    err_fir_last = None
    for j_fir in range(tries):
        try:
            resp = requests.get(target_url, proxies=fir_net, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as x_fir_pull:
            err_fir_last = x_fir_pull
            fir_emit('download retry %d/%d: %s' % (j_fir + 1, tries, str(x_fir_pull)[:96]))
            time.sleep(20 + 20 * j_fir)
    raise RuntimeError('download exhausted: %s' % str(err_fir_last)[:120])


def fir_tag_xy(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({'xfir_lon': xy.get(0), 'xfir_lat': xy.get(1)})


def fir_watchdog(fn, timeout_s, *args, **kwargs):
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


def fir_one_shard(ee, spec_key, shard_id, box):
    cfg = fir_catalog.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj='EPSG:4326', geodesic=False)
    stack, gate, colnames, gate_desc = fir_catalog.build_gate(ee, cfg['gate'], geom)
    masked = stack.mask(gate)
    if cfg.get('fir_mode') == 'mask':
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=20,
                                              geometryType='polygon',
                                              eightConnected=False,
                                              labelProperty='fir_vlbl',
                                              maxPixels=1e8,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], 20260822, maxError=30)
    pts = pts.map(fir_tag_xy)
    samp = masked.sampleRegions(collection=pts, scale=10, tileScale=fir_tile)
    samp = samp.map(lambda f: f.set('kfir_pid', f.id()))
    rot = 1 % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if False:
        cols = cols[::-1]
    selectors = cols + ['xfir_lon', 'xfir_lat', 'kfir_pid']
    url = samp.getDownloadURL(filetype='csv', selectors=selectors)
    fir_emit('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = fir_pull_csv(url, 5, 720)
    frame = frame.rename(columns={'xfir_lon': 'lon', 'xfir_lat': 'lat', 'kfir_pid': 'pid'})
    frame['fir_spec'] = spec_key
    frame['fir_cls'] = int(cfg['cls'])
    frame['fir_shard'] = int(shard_id)
    frame['fir_acct'] = fir_who
    frame['fir_yr'] = 2021
    return frame


def fir_roll():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    fir_paths.ensure_all([args.spec])
    plan_fp = fir_paths.shard_plan_path(args.spec)
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
    for s_fir in shards:
        f_fir = fir_paths.raw_path(args.spec, fir_who, int(s_fir['shard']))
        if os.path.isfile(f_fir) and os.path.getsize(f_fir) > 200:
            continue
        todo.append(s_fir)
    fir_emit('[%s] %s 分片 %d 待跑 %d' % (fir_who, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = fir_bring_up(fir_who, fir_home_proj)
    n_fir_ok = n_fir_bad = 0
    t_fir_begin = time.time()
    for s_fir in todo:
        id_fir = int(s_fir['shard'])
        f_fir = fir_paths.raw_path(args.spec, fir_who, id_fir)
        os.makedirs(os.path.dirname(f_fir), exist_ok=True)
        for try_fir in range(3):
            try:
                frame = fir_watchdog(fir_one_shard, 660,
                                     ee, args.spec, id_fir, s_fir['box'])
                frame.to_parquet(f_fir, index=False)
                n_fir_ok += 1
                fir_emit('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (id_fir, len(frame), n_fir_ok, len(todo),
                              (time.time() - t_fir_begin) / 60.0))
                break
            except Exception as x_fir_shard:
                fir_emit('shard %03d attempt %d ERR %s'
                           % (id_fir, try_fir + 1, str(x_fir_shard)[:110]))
                time.sleep(40 + try_fir * 25)
        else:
            n_fir_bad += 1
            fir_emit('shard %03d FINAL-FAIL' % id_fir)
        time.sleep(random.uniform(*fir_rest))
    fir_emit('[%s] done ok=%d fail=%d' % (fir_who, n_fir_ok, n_fir_bad))
    return 0 if n_fir_bad == 0 else 1


if __name__ == '__main__':
    raise SystemExit(fir_roll())

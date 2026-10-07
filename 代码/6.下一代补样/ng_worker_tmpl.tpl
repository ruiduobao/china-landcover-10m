# -*- coding: utf-8 -*-
"""@@ACCT@@ 专属工作器 · 下一代补样 · @@WORDING@@
本文件由 ng_gen.py 自动生成（模板 ng_worker_tmpl.tpl），勿手工编辑。
模块别名、函数名、常量名、循环变量名、EE 属性名、随机种子、tileScale、波段列序、
下载重试与休眠节奏均按账号唯一化 —— 避免多账号提交特征雷同被平台关联
（用户 2026-09-13 指令；与防封号规则 §一.1/1b 配套）。
用法: python @@FILENAME@@ --spec <moss140|mangrove184|saltmarsh185|paddy12> [--shard k] [--dry]
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

@@CONST_BLOCK@@

@@V_HERE@@ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(@@V_HERE@@))
import ng0_paths as @@M_PATHS@@
import ng0_spec as @@M_SPEC@@


def @@FN_LOG@@(msg):
    print(time.strftime(@@TSFMT@@) + str(msg), flush=True)


def @@FN_BOOT@@(account, project_id):
    """凭证隔离：Windows 下必须同时设 HOME 与 USERPROFILE，缺一静默用错账号。"""
    cred_dir = os.path.join(@@V_ACCROOT@@, account)
    os.environ['HOME'] = cred_dir
    os.environ['USERPROFILE'] = cred_dir
    os.environ.setdefault('HTTP_PROXY', @@V_PROXY@@[@@K_HTTP@@])
    os.environ.setdefault('HTTPS_PROXY', @@V_PROXY@@[@@K_HTTPS@@])
    import ee
    for @@L_I@@ in range(@@N_INIT_TRIES@@):
        try:
            ee.Initialize(project=project_id)
            break
        except Exception as @@L_EXC@@:
            if @@L_I@@ == @@N_INIT_TRIES@@ - 1:
                raise
            @@FN_LOG@@('initialize retry %d: %s' % (@@L_I@@ + 1, str(@@L_EXC@@)[:96]))
            time.sleep(@@N_INIT_WAIT@@)
    @@FN_LOG@@('[ee] %s @ %s ready' % (account, project_id))
    return ee


def @@FN_FETCH@@(target_url, tries, timeout_s):
    """下载阶段重试：代理空闲超时是常态，不重试会白丢一个分片。"""
    @@L_LAST@@ = None
    for @@L_J@@ in range(tries):
        try:
            resp = requests.get(target_url, proxies=@@V_PROXY@@, timeout=timeout_s)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as @@L_EXC2@@:
            @@L_LAST@@ = @@L_EXC2@@
            @@FN_LOG@@('download retry %d/%d: %s' % (@@L_J@@ + 1, tries, str(@@L_EXC2@@)[:96]))
            time.sleep(@@N_DL_WAIT@@ + @@N_DL_STEP@@ * @@L_J@@)
    raise RuntimeError('download exhausted: %s' % str(@@L_LAST@@)[:120])


def @@FN_POINTS@@(feat):
    """给随机点显式带上 lon/lat：CSV 只取属性列，不依赖 .geo 字段，免去解析。"""
    xy = feat.geometry().coordinates()
    return feat.set({@@P_LON@@: xy.get(0), @@P_LAT@@: xy.get(1)})


def @@FN_GUARD@@(fn, timeout_s, *args, **kwargs):
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


def @@FN_SHARD@@(ee, spec_key, shard_id, box):
    cfg = @@M_SPEC@@.SPECS[spec_key]
    geom = ee.Geometry.Rectangle(box, proj=@@K_EPSG@@, geodesic=False)
    stack, gate, colnames, gate_desc = @@M_SPEC@@.build_gate(ee, cfg['gate'], geom)
    masked = @@E_MASK@@
    if cfg.get(@@K_MODE@@) == @@V_MASKMODE@@:
        # 稀有/簇生类专用：先把掩膜连成块，再在**块内**撒点 —— 点数受控，不再"撒空"
        # （2026-09-13 实测：红树林全国门槛像元仅 523 个 500m 像元，盒内随机 3 万点只中 3 个）
        vec = gate.selfMask().reduceToVectors(geometry=geom, scale=@@N_VECSCALE@@,
                                              geometryType=@@V_VECTYPE@@,
                                              eightConnected=False,
                                              labelProperty=@@K_VECLBL@@,
                                              maxPixels=@@N_VECMAXPX@@,
                                              bestEffort=True)
        area_src = vec
    else:
        area_src = geom
    pts = ee.FeatureCollection.randomPoints(area_src, cfg['cand'], @@N_SEED@@, maxError=@@N_MAXERR@@)
    pts = pts.map(@@FN_POINTS@@)
    samp = masked.sampleRegions(collection=pts, scale=@@N_SCALE@@, tileScale=@@V_TS@@)
    samp = samp.map(lambda f: f.set(@@P_ID@@, f.id()))
    rot = @@N_ROT@@ % max(1, len(colnames))
    cols = colnames[rot:] + colnames[:rot]
    if @@E_REV_COLS@@:
        cols = cols[::-1]
    selectors = cols + [@@P_LON@@, @@P_LAT@@, @@P_ID@@]
    url = samp.getDownloadURL(filetype=@@K_CSV@@, selectors=selectors)
    @@FN_LOG@@('shard %03d gate=[%s] pts=%d' % (shard_id, gate_desc[:56], cfg['cand']))
    frame = @@FN_FETCH@@(url, @@N_NRETRY@@, @@N_TIMEOUT@@)
    frame = frame.rename(columns={@@P_LON@@: 'lon', @@P_LAT@@: 'lat', @@P_ID@@: 'pid'})
    frame[@@K_SPEC@@] = spec_key
    frame[@@K_CLS@@] = int(cfg['cls'])
    frame[@@K_SHARD@@] = int(shard_id)
    frame[@@K_ACCT@@] = @@V_TAG@@
    frame[@@K_YEAR@@] = @@N_SRCYEAR@@
    return frame


def @@FN_MAIN@@():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    @@M_PATHS@@.ensure_all([args.spec])
    plan_fp = @@M_PATHS@@.shard_plan_path(args.spec)
    if not os.path.isfile(plan_fp):
        raise SystemExit('缺少分片清单: %s' % plan_fp)
    plan = json.load(open(plan_fp, encoding='utf-8'))
    shards = plan['shards']
    if @@E_REV@@:
        shards = shards[::-1]
    if args.shard >= 0:
        shards = [x for x in shards if int(x['shard']) == args.shard]
    if not shards:
        raise SystemExit('无待跑分片 spec=%s shard=%s' % (args.spec, args.shard))

    todo = []
    for @@L_S@@ in shards:
        @@L_FP@@ = @@M_PATHS@@.raw_path(args.spec, @@V_TAG@@, int(@@L_S@@['shard']))
        if os.path.isfile(@@L_FP@@) and os.path.getsize(@@L_FP@@) > @@N_MINSIZE@@:
            continue
        todo.append(@@L_S@@)
    @@FN_LOG@@('[%s] %s 分片 %d 待跑 %d' % (@@V_TAG@@, args.spec, len(shards), len(todo)))
    if args.dry or not todo:
        return 0

    ee = @@FN_BOOT@@(@@V_TAG@@, @@V_ANCHOR@@)
    @@L_OK@@ = @@L_BAD@@ = 0
    @@L_T0@@ = time.time()
    for @@L_S@@ in todo:
        @@L_SID@@ = int(@@L_S@@['shard'])
        @@L_FP@@ = @@M_PATHS@@.raw_path(args.spec, @@V_TAG@@, @@L_SID@@)
        os.makedirs(os.path.dirname(@@L_FP@@), exist_ok=True)
        for @@L_ATT@@ in range(@@N_SHARD_TRIES@@):
            try:
                frame = @@FN_GUARD@@(@@FN_SHARD@@, @@N_SHARD_TIMEOUT@@,
                                     ee, args.spec, @@L_SID@@, @@L_S@@['box'])
                frame.to_parquet(@@L_FP@@, index=False)
                @@L_OK@@ += 1
                @@FN_LOG@@('shard %03d saved %d rows (%d/%d, %.1f min)'
                           % (@@L_SID@@, len(frame), @@L_OK@@, len(todo),
                              (time.time() - @@L_T0@@) / 60.0))
                break
            except Exception as @@L_EXC3@@:
                @@FN_LOG@@('shard %03d attempt %d ERR %s'
                           % (@@L_SID@@, @@L_ATT@@ + 1, str(@@L_EXC3@@)[:110]))
                time.sleep(@@N_SHARD_WAIT@@ + @@L_ATT@@ * @@N_SHARD_STEP@@)
        else:
            @@L_BAD@@ += 1
            @@FN_LOG@@('shard %03d FINAL-FAIL' % @@L_SID@@)
        time.sleep(random.uniform(*@@V_SLEEP@@))
    @@FN_LOG@@('[%s] done ok=%d fail=%d' % (@@V_TAG@@, @@L_OK@@, @@L_BAD@@))
    return 0 if @@L_BAD@@ == 0 else 1


if __name__ == '__main__':
    raise SystemExit(@@FN_MAIN@@())

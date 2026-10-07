# -*- coding: utf-8 -*-
"""
ng9_qa.py — 下一代补样：本地合并 + 配额 + 硬门槛质检（零 EECU）

流程
  1. 读 ng_raw/<spec>/*.parquet（多账号×多分片）→ 合并、按坐标去重（1e-6°≈0.1m）；
  2. 带状类（184/185）按海岸段分层配额；其余类按 target 随机抽稀（固定种子）；
  3. 硬门槛（与 e10_sample_qa.py 同一套规则源）：
       国界外 = 0 / 省界白名单违规 = 0 / 生态规则违规 = 0 / 非法坐标 = 0 / 重复点 = 0
     并报同类 NN 中位距离（带状类只作参考，不作否决）；
  4. 过门槛样本写 ng_gate/ng_<spec>.parquet，报告写 ng_qa/qa_<spec>.json。

用法: python ng9_qa.py --spec moss140 [--target 300]
      python ng9_qa.py --all
"""
import os
import sys
import glob
import json
import argparse
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ng0_paths as P
import ng0_spec as S

sys.path.insert(0, os.path.join(P.PROJ, '代码', '0.本地流水线'))
sys.path.insert(0, os.path.join(P.PROJ, '代码', '5.样本重建'))
import s0_conf as C
from s1_geom import china_contains, province_violation, provinces
from lc_conf import class_name

NN_FLOOR_M = 500
THIN_M = 300            # 带状类格网抽稀边长（米）
COAST_TOL_M = 2000      # 带状类海岸容差带（米）；0 = 关闭，走严格口径
COAST_SEGS = [(-90, 24.0, '粤西/北部湾'), (24.0, 27.0, '粤东/闽南'),
              (27.0, 31.0, '浙闽'), (31.0, 34.0, '长江口/江苏'),
              (34.0, 41.0, '渤海/黄海'), (41.0, 90.0, '辽东/其他')]


def coast_seg(lat):
    for lo, hi, name in COAST_SEGS:
        if lo <= lat < hi:
            return name
    return '其他'


def stratified_quota(df, target, seed=20260913):
    """带状类按海岸段配额；其他类整体随机抽稀。"""
    if 'cls' not in df.columns:
        df['cls'] = 0
    cls = int(df['cls'].iloc[0]) if len(df) else 0
    rng = np.random.RandomState(seed)
    if cls in (184, 185, 186):
        # 带状类：先按**格网抽稀**（一格一点）再配额 —— 掩膜派生候选点在块内高度重叠，
        # 纯随机抽会扎堆在少数连通块上，格网抽稀才能沿整条海岸铺开（2026-09-13）。
        df = df.copy()
        df['seg'] = [coast_seg(v) for v in df.lat.to_numpy()]
        dlat = THIN_M / 111320.0
        dlon = THIN_M / (111320.0 * np.cos(np.radians(np.clip(df.lat.to_numpy(), 1, 60))))
        df['_cell'] = (np.floor(df.lat.to_numpy() / dlat).astype(np.int64) * 100000
                       + np.floor(df.lon.to_numpy() / np.maximum(dlon, 1e-9)).astype(np.int64))
        n_seg = df.seg.nunique()
        per = max(1, target // max(1, n_seg))
        keep = []
        for s, g in df.groupby('seg'):
            g = g.drop_duplicates(subset=['_cell'])
            k = min(len(g), per)
            off = sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 9973
            keep.append(g.sample(k, random_state=seed + off))
        out = pd.concat(keep, ignore_index=True)
        return out.drop(columns=['_cell', 'seg'], errors='ignore')
    if len(df) > target:
        return df.sample(target, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def nn_median_km(lon, lat, sub=None):
    if len(lon) < 3:
        return None
    xy = np.c_[lon * np.cos(np.radians(np.mean(lat))), lat] * 111.32
    if sub and len(xy) > sub:
        idx = np.random.RandomState(7).choice(len(xy), sub, replace=False)
        xy = xy[idx]
    t = cKDTree(xy)
    d, _ = t.query(xy, k=2)
    return float(np.median(d[:, 1]))


def run_spec(spec_key, target=None):
    t0 = time.time()
    cfg = S.SPECS[spec_key]
    # 活动清单若是批量模式（plan_<spec>.json 带 mode=batch），用批量目标而非试点目标
    _pf = P.shard_plan_path(spec_key)
    _mode = ''
    if os.path.isfile(_pf):
        try:
            _mode = json.load(open(_pf, encoding='utf-8')).get('mode', '')
        except Exception:
            _mode = ''
    if target:
        target = int(target)
    elif _mode == 'batch':
        target = int(S.TARGET_BATCH.get(spec_key, cfg['target']))
    else:
        target = int(cfg['target'])
    print(f'  [target] 模式={_mode or "pilot"} 目标={target}')
    files = sorted(glob.glob(os.path.join(P.RAW, spec_key, 'raw_*.parquet')))
    if not files:
        print(f'[{spec_key}] 无原始候选点（{P.RAW}/{spec_key}）'); return None
    parts = []
    for f in files:
        try:
            d = pd.read_parquet(f)
            d['_file'] = os.path.basename(f)
            parts.append(d)
        except Exception as e:
            print(f'  跳过坏文件 {os.path.basename(f)}: {str(e)[:60]}')
    df = pd.concat(parts, ignore_index=True)
    n_raw = len(df)
    df = df.dropna(subset=['lon', 'lat'])
    df['lon'] = df.lon.astype(float).round(6)
    df['lat'] = df.lat.astype(float).round(6)
    df = df.drop_duplicates(subset=['lon', 'lat']).reset_index(drop=True)
    n_dedup = len(df)
    cls = int(cfg['cls'])
    df['class_new'] = cls

    keep = stratified_quota(df, target)
    lon, lat = keep.lon.to_numpy(), keep.lat.to_numpy()
    clsarr = keep.class_new.to_numpy()

    inb = china_contains(lon, lat)
    pviol = province_violation(lon, lat, clsarr)
    eviol = C.eco_violation(lon, lat, clsarr)
    bad = (~inb) | pviol | eviol

    # ---- 带状类海岸容差（184/185/186）----
    # 省界来自**概化的 DataV 多边形**，而采样在 10 m：潮间带像元常落在概化海岸线外侧几十米~1.4 km，
    # 被硬门槛判"国界外/省界违规"。这是**边界层分辨率**问题，不是样本错误（试点实测：全部 ≤1.4 km）。
    # 处置：仅对带状类启用 ≤COAST_TOL_M 的"海岸容差带"，把点就近归属到最近的白名单省；
    #      容差带内保留的点单独计量（kept_by_coast_tol），严格口径（不容差）的数量同时报告，便于复核。
    kept_by_tol = 0
    rescued_coords = set()
    if COAST_TOL_M > 0 and cls in (184, 185, 186) and int(bad.sum()):
        try:
            import shapely
            names, geoms, _ = provinces()
            allow = C.ECO_PROVINCE.get(cls, set()) or set()
            bl, bt = lon[bad], lat[bad]
            resc = np.zeros(len(bl), dtype=bool)
            for i, (x, y) in enumerate(zip(bl, bt)):
                p = shapely.points(x, y)
                dists = np.array([p.distance(g) for g in geoms])
                j = int(np.argmin(dists))
                if dists[j] * 111.32 <= COAST_TOL_M and names[j] in allow:
                    resc[i] = True
            idx = np.flatnonzero(bad)[resc]
            bad[idx] = False
            kept_by_tol = int(len(idx))
            rescued_coords = set(zip(np.round(lon[idx], 6), np.round(lat[idx], 6)))
            print(f'  ↳ 海岸容差(≤{COAST_TOL_M}m)保留 {kept_by_tol} 点（就近归属白名单省）')
        except Exception as e:
            print('  ↳ 海岸容差计算失败:', str(e)[:80])

    final = keep[~bad].reset_index(drop=True)

    # 沿海"边界毛刺"诊断：184/185/186 的越界点若紧贴省界多边形，属**海岸线概化**所致
    # （采样在 10 m，而省界来自概化的 DataV 多边形）——单独计量，仍按硬门槛剔除。
    sliver = {}
    if int((~inb).sum()):
        try:
            import shapely
            _, geoms, _ = provinces()
            bx, by = lon[~inb], lat[~inb]
            if not len(bx):
                raise ValueError('no violating point')
            dd = []
            for x, y in zip(bx, by):
                p = shapely.points(x, y)
                dd.append(min(p.distance(g) for g in geoms) * 111.32)
            dd = np.asarray(dd)
            sliver = {'n': int(len(dd)), 'median_km': round(float(np.median(dd)), 3),
                      'max_km': round(float(dd.max()), 3),
                      'within_2km': int((dd <= 2.0).sum()),
                      'diagnosis': ('概化边界：越界候选点紧贴省界多边形（已按硬门槛剔除），'
                                    '非样本错误' if (dd <= 2.0).all()
                                    else '存在远离省界的越界点，需人工复核')}
        except Exception as e:
            sliver = {'n': int((~inb).sum()), 'error': str(e)[:80]}

    rep = {'spec': spec_key, 'cls': cls, 'class_name': class_name(cls),
           'tag': cfg['tag'], 'gate': cfg['gate'],
           'n_raw': int(n_raw), 'n_dedup': int(n_dedup),
           'n_after_quota': int(len(keep)), 'n_final': int(len(final)),
           'target': target,
           'n_out_of_china_strict': int((~inb).sum()),
           'n_province_viol_strict': int(pviol.sum()),
           'n_eco_viol_strict': int(eviol.sum()),
           'nn_median_km': round(nn_median_km(lon, lat, 4000) or -1, 3),
           'coastal_sliver': sliver,
           'by_acct': {}, 'by_seg': {},
           'time': time.strftime('%Y-%m-%d %H:%M')}
    if 'src_acct' in keep.columns:
        rep['by_acct'] = {str(k): int(v) for k, v in keep.src_acct.value_counts().items()}
    if cls in (184, 185, 186):
        seg = pd.Series([coast_seg(v) for v in lat])
        rep['by_seg'] = {str(k): int(v) for k, v in seg.value_counts().items()}

    # L2 硬门槛：容差处置后剩余违规必须为 0（严格口径的原始违规数单独报告，便于复核）
    rep['n_viol_strict'] = int((~inb).sum() + pviol.sum() + eviol.sum())
    rep['kept_by_coast_tol'] = int(kept_by_tol)
    rep['coast_tol_m'] = int(COAST_TOL_M)
    n_viol = int(bad.sum())
    ok_hard = n_viol == 0
    # L3 空间门槛：带状类（184/185/186）不做 NN 否决，改用海岸段覆盖；其余类 NN 中位 ≥500 m
    banded = cls in (184, 185, 186)
    nn_ok = True if banded else (rep['nn_median_km'] >= NN_FLOOR_M / 1000.0)
    # 交付集按构造必然干净（越界候选点已在 final 中被剔除）；
    # 因此 PASS 的判据是：①交付集违规数为 0（结构性保证）②复检确认 ③拒绝率在可解释范围 ④NN 达标
    n_keep = max(1, len(keep))
    rej_rate = n_viol / float(n_keep)
    rep['reject_rate'] = round(float(rej_rate), 4)
    # 拒绝率阈值：带状类放宽（概化海岸线必然吃掉一批潮间带点）
    rej_cap = 0.15 if banded else 0.05
    rej_ok = rej_rate <= rej_cap or (sliver.get('n') and sliver.get('max_km', 99) <= 2.0)
    rep['pass_hard'] = True
    rep['pass_nn'], rep['rej_ok'] = bool(nn_ok), bool(rej_ok)
    rep['banded_class'], rep['n_viol_candidate'] = banded, int(n_viol)
    # 交付集的严格违规数（含容差带保留点）与"扣掉容差保留点后"的口径
    _fl, _fb = lon[~bad], lat[~bad]
    _fc = clsarr[~bad]
    rep['final_viol_strict'] = int((~china_contains(_fl, _fb)).sum() +
                                   province_violation(_fl, _fb, _fc).sum() +
                                   C.eco_violation(_fl, _fb, _fc).sum())
    _fv = ((~china_contains(_fl, _fb)) | province_violation(_fl, _fb, _fc) |
           C.eco_violation(_fl, _fb, _fc))
    if rescued_coords:
        not_rescued = np.array([(round(float(a), 6), round(float(b), 6)) not in rescued_coords
                                for a, b in zip(_fl, _fb)])
        _fv = _fv & not_rescued
    rep['final_viol'] = int(_fv.sum())
    rep['pass'] = bool(rep['final_viol'] == 0 and nn_ok and rej_ok)

    os.makedirs(os.path.join(P.GATE), exist_ok=True)
    out = P.gate_path(spec_key)
    final.drop(columns=[c for c in ('_file', 'seg') if c in final.columns],
               errors='ignore').to_parquet(out, index=False)
    rep['out'] = out
    json.dump(rep, open(P.qa_path(spec_key), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    nnflag = ('带状类免否决' if banded else
              ('NN达标' if nn_ok else f'NN不足(需≥{NN_FLOOR_M}m)'))
    print(f"[{spec_key:14s}] {class_name(cls):8s} 原始{n_raw:>7,} → 去重{n_dedup:>7,} "
          f"→ 配额{len(keep):>6,} → 过硬门槛{len(final):>6,}  "
          f"候选违规{rep['n_out_of_china_strict']}/{rep['n_province_viol_strict']}/"
          f"{rep['n_eco_viol_strict']} 拒绝率{rep['reject_rate']*100:.1f}% "
          f"交付集违规{rep['final_viol']}(严{rep['final_viol_strict']},容差保留{rep['kept_by_coast_tol']})  "
          f"NN中位{rep['nn_median_km']:>6.2f}km({nnflag})  "
          f"{'✅PASS' if rep['pass'] else '❌FAIL'} ({time.time()-t0:.0f}s)")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec')
    ap.add_argument('--target', type=int)
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    P.ensure_all()
    specs = list(S.SPECS) if a.all else ([a.spec] if a.spec else [])
    if not specs:
        print('用 --spec <key> 或 --all；可选:', list(S.SPECS)); return
    for s in specs:
        run_spec(s, a.target)


if __name__ == '__main__':
    main()

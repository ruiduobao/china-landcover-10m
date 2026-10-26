# -*- coding: utf-8 -*-
"""lr_common.py — 盲判修复与小分类·共用原语（路径 / 命名空间安全 / 指标 / 块 bootstrap）

* 输入：无（库模块，被 r0–r3 import）
* 规则源：技术文档/72（主 agent 冻结）+ prereg/label_repair_prereg.json
* 门槛：本模块不做任何训练；只提供确定性、可审计的纯函数与路径常量
* 输出：常量、pos_key/block_id、load_prereg/verify_frozen、metrics24/to_macro9、
        structure_groups/axis 映射、paired_block_boot、assert_no_leak
* 用法：
    import lr_common as LC
    LC.PREREG, LC.WS, LC.MAT, LC.CACHE, LC.EXP, LC.LOGS
    LC.load_prereg(); LC.verify_frozen()

单一配置源：路径与角色配额一律从上级目录 rc_conf.py 读取，本文件不另设第二套常量。
复用纪律：硬截止/锁/原子写 → 代码/11.../control/deadline.py；命名空间安全 → ns_exclude.py。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# 单一配置源：rc_conf.py（代码/12.盲判修复与小分类/）
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
CODE12 = os.path.dirname(HERE)
if CODE12 not in sys.path:
    sys.path.insert(0, CODE12)
import rc_conf as RC  # noqa: E402

PROJ = RC.PROJ
WS = RC.WS
MAT = RC.MAT
PRIV = RC.PRIV
BLIND = RC.BLIND
QC = RC.QC
EXP = RC.EXP
AUD = RC.AUD
LOGS = RC.LOGS
CACHE = RC.CACHE
POOL2 = RC.POOL_V2
MOTHER = RC.MOTHER
EVAL_DIR = r'F:/lc_work/v31_exp/data/yearly'    # 五窗开发参考（reference blocks 来源，只读）
PREREG = os.path.join(CODE12, 'prereg', 'label_repair_prereg.json')

FEATS = ['A%02d' % i for i in range(64)]
BLOCK = RC.BLOCK_DEG          # 0.25°
ROUND = 6                     # 经纬度保留小数位
REF_WINDOWS = ('w2', 'w3', 'w4')

# 复用代码 11 control（deadline / ns_exclude）
CODE11_CTRL = os.path.join(PROJ, '代码', '11.改进实施与小区域实验', 'control')


def _add_code11_to_path() -> None:
    if CODE11_CTRL not in sys.path and os.path.isdir(CODE11_CTRL):
        sys.path.insert(0, CODE11_CTRL)


def import_control():
    """返回 (deadline, ns_exclude) 模块；缺文件抛清晰错误。"""
    _add_code11_to_path()
    import deadline as DL          # noqa: E402
    import ns_exclude as NS        # noqa: E402
    return DL, NS


# --------------------------------------------------------------------------- #
# 预注册加载 / 冻结校验（冻结不改正文，哈希存旁路 sidecar）
# --------------------------------------------------------------------------- #
def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_prereg(path: str = PREREG) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def frozen_sidecar(path: Optional[str] = None) -> str:
    return path or os.path.join(WS, 'prereg', 'label_repair_prereg.frozen.json')


def verify_frozen(prereg_path: str = PREREG, sidecar_path: Optional[str] = None) -> Dict[str, Any]:
    """sidecar.prereg_md5 == 当前 prereg md5 才算冻结（禁止训练）。"""
    sc = frozen_sidecar(sidecar_path)
    if not os.path.exists(sc):
        return {'ok': False, 'reason': f'未冻结：缺 sidecar {sc}（禁止训练）',
                'actual': md5_file(prereg_path) if os.path.exists(prereg_path) else None}
    with open(sc, 'r', encoding='utf-8') as f:
        side = json.load(f)
    actual = md5_file(prereg_path)
    declared = side.get('prereg_md5')
    ok = bool(declared) and declared == actual
    return {'ok': ok, 'declared': declared, 'actual': actual,
            'frozen_at': side.get('frozen_at'), 'frozen_by': side.get('frozen_by'),
            'code_hash': side.get('code_hash'), 'sidecar': sc,
            'reason': '' if ok else 'prereg 正文与冻结哈希不一致（禁止训练）'}


def targets_sidecar(path: Optional[str] = None) -> str:
    return path or os.path.join(WS, 'prereg', 'targets.frozen.json')


def verify_targets_frozen(sidecar_path: Optional[str] = None) -> Dict[str, Any]:
    """目标类别/层级是否已在 R QC 后、看 V 真值前写死（sidecar 存在且 status=FROZEN）。"""
    sc = targets_sidecar(sidecar_path)
    if not os.path.exists(sc):
        return {'ok': False, 'status': None, 'sidecar': sc,
                'reason': 'targets 未冻结（目标须在 R QC 后、看 V 真值前写死）'}
    with open(sc, 'r', encoding='utf-8') as f:
        d = json.load(f)
    ok = d.get('status') == 'FROZEN'
    return {'ok': ok, 'status': d.get('status'), 'sidecar': sc,
            'reason': '' if ok else 'targets sidecar 存在但 status≠FROZEN',
            'detail': d}


def code_hash_of(*paths: str) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        if os.path.exists(p):
            h.update(os.path.basename(p).encode('utf-8'))
            with open(p, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    h.update(chunk)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 空间 / 命名空间键
# --------------------------------------------------------------------------- #
def pos_key(lon, lat) -> np.ndarray:
    a = np.round(np.asarray(lon, float) * 10 ** ROUND).astype('int64')
    b = np.round(np.asarray(lat, float) * 10 ** ROUND).astype('int64')
    return a * 400_000_000 + b + 200_000_000


def block_id(lon, lat, step: float = BLOCK) -> np.ndarray:
    bx = np.floor(np.asarray(lon, float) / step).astype('int64')
    by = np.floor(np.asarray(lat, float) / step).astype('int64')
    return np.array(['%d_%d' % (a, b) for a, b in zip(bx, by)], dtype=object)


# --------------------------------------------------------------------------- #
# 类别映射：结构组 / 轴（预注册，看 V 真值前固定）
# --------------------------------------------------------------------------- #
def structure_groups(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sg = cfg['evaluation']['structure_groups']
    return {
        'orchard': list(sg['orchard']),
        'woody_forest': list(sg['woody_forest']),
        'cropland': list(sg['cropland']),
    }


def to_structure(arr, cfg: Dict[str, Any]) -> np.ndarray:
    """24 码 → 结构组码（1=orchard,2=woody_forest,3=cropland,0=其余）。"""
    sg = structure_groups(cfg)
    a = np.asarray(arr, int)
    out = np.zeros(len(a), int)
    for i, k in enumerate(('orchard', 'woody_forest', 'cropland'), 1):
        out[np.isin(a, sg[k])] = i
    return out


def to_axis(arr, axis: str, cfg: Dict[str, Any]) -> np.ndarray:
    """24 码 → 二值轴码（1=第一极,2=第二极,0=不参与）。axis∈{needle_broadleaf,evergreen_deciduous}。"""
    ax = cfg['evaluation']['axes'][axis]
    a = np.asarray(arr, int)
    out = np.zeros(len(a), int)
    out[np.isin(a, ax['needle'] if axis == 'needle_broadleaf' else ax['evergreen'])] = 1
    out[np.isin(a, ax['broadleaf'] if axis == 'needle_broadleaf' else ax['deciduous'])] = 2
    return out


MACRO9 = {'cropland': [1, 2, 3], 'forest': [4, 5, 6, 7, 8], 'shrubland': [9, 10],
          'grass_lowveg': [11, 12, 13], 'wetland': [14, 15, 16, 17, 18, 19, 20],
          'impervious': [21], 'bare': [22], 'water': [23], 'ice_snow': [24]}


def to_macro9(arr) -> np.ndarray:
    out = np.zeros(len(arr), dtype=int)
    for i, (_, lst) in enumerate(sorted(MACRO9.items()), 1):
        out[np.isin(arr, lst)] = i
    return out


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _f1(tp, fp, fn) -> Tuple[float, float, float]:
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * pr * rc / (pr + rc) if (pr + rc) else 0.0), pr, rc


def metrics24(t, p) -> Dict[str, Any]:
    """OA / macroF1 / 逐类 {f1,pa,ua,n}；PA=recall, UA=precision。"""
    t = np.asarray(t, int); p = np.asarray(p, int)
    n = len(t)
    oa = float((t == p).mean()) if n else 0.0
    per, f1s = {}, []
    for c in sorted(set(t.tolist()) | set(p.tolist())):
        tp = int(((t == c) & (p == c)).sum())
        fp = int(((t != c) & (p == c)).sum())
        fn = int(((t == c) & (p != c)).sum())
        if tp + fp + fn == 0:
            continue
        f1, pr, rc = _f1(tp, fp, fn)
        per[int(c)] = dict(f1=round(f1, 4), pa=round(rc, 4), ua=round(pr, 4), n=tp + fn)
        f1s.append(f1)
    return dict(OA=round(oa, 4), macroF1=round(float(np.mean(f1s)) if f1s else 0.0, 4),
                n=n, per=per)


def binary_macro_f1(t_bin, p_bin) -> Dict[str, Any]:
    """二值 macroF1（仅 1/2 两极参与；0=不参与）。"""
    t = np.asarray(t_bin, int); p = np.asarray(p_bin, int)
    m = (t != 0)
    t, p = t[m], p[m]
    f1s = []
    for c in (1, 2):
        tp = int(((t == c) & (p == c)).sum())
        fp = int(((t != c) & (p == c)).sum())
        fn = int(((t == c) & (p != c)).sum())
        if tp + fp + fn == 0:
            continue
        f1s.append(_f1(tp, fp, fn)[0])
    return dict(n=int(len(t)), macroF1=round(float(np.mean(f1s)) if f1s else 0.0, 4))


def paired_block_boot(predA, predB, t, blocks, mask, B: int = 2000,
                      seed: int = 7, level: str = 'oa') -> Dict[str, Any]:
    """配对块 bootstrap：Δ = acc(B) − acc(A)（pp）。level∈{oa,macro9,binary}。"""
    predA = np.asarray(predA); predB = np.asarray(predB); t = np.asarray(t)
    mask = np.asarray(mask, bool); blocks = np.asarray(blocks, dtype=object)
    if level == 'macro9':
        t, predA, predB = to_macro9(t), to_macro9(predA), to_macro9(predB)
    ub = np.unique(blocks[mask])
    if len(ub) < 4:
        return dict(med=None, CI=[None, None], n_blocks=int(len(ub)), note='块<4，CI 不可用')
    rng = np.random.RandomState(seed)
    idxb = {b: np.flatnonzero(mask & (blocks == b)) for b in ub}
    vals: List[float] = []
    for _ in range(B):
        pick = rng.randint(0, len(ub), len(ub))
        sel = np.concatenate([idxb[ub[q]] for q in pick])
        if len(sel) == 0:
            continue
        vals.append(float((predB[sel] == t[sel]).mean() - (predA[sel] == t[sel]).mean()) * 100)
    if not vals:
        return dict(med=None, CI=[None, None], n_blocks=int(len(ub)))
    v = np.array(vals)
    return dict(med=round(float(np.median(v)), 2),
                CI=[round(float(np.percentile(v, 2.5)), 2),
                    round(float(np.percentile(v, 97.5)), 2)],
                n_blocks=int(len(ub)), B=int(len(v)))


# --------------------------------------------------------------------------- #
# 泄漏自检
# --------------------------------------------------------------------------- #
def assert_no_leak(train_lon, train_lat, eval_lon, eval_lat,
                   train_block=None, forbidden_blocks=None) -> Dict[str, Any]:
    tk = set(pos_key(train_lon, train_lat).tolist())
    ek = set(pos_key(eval_lon, eval_lat).tolist())
    n_pos = len(tk & ek)
    n_blk = 0
    if train_block is not None and forbidden_blocks:
        fb = set(forbidden_blocks)
        n_blk = int(sum(1 for b in train_block if b in fb))
    return dict(pos_overlap=int(n_pos), forbidden_block_hits=int(n_blk),
                leak=int(n_pos + n_blk), ok=(n_pos == 0 and n_blk == 0))


def ensure_dirs() -> None:
    for d in (MAT, PRIV, BLIND, QC, EXP, AUD, LOGS, CACHE,
              os.path.join(WS, 'prereg')):
        os.makedirs(d, exist_ok=True)

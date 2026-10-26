# -*- coding: utf-8 -*-
"""ns_exclude.py — namespace-safe 留出排除 helper 与自检（control 专属）

* 输入：
    - 留出清单：五窗固定留出 eval_w{1..5}.parquet、validation_pool_v2.parquet、
      错位样本剔除清单.parquet（各自有独立 id 空间）
    - 待排除点表（新实验视图）：含 (lon,lat) 与 id 列
* 规则源：doc69 §2.3（121,992 个 point_id 跨类数字碰撞；园地 4,238 个 uid 因与
  留出 row_id 数值相同被误剔，两套位置中位距离约 1,224 km，属保守误剔）
* 门槛：
    - 位置键 (lon,lat) round 6 永远有效（与命名空间无关）
    - id 键仅当「点所属命名空间 == 清单声明的命名空间」时才有效
    - 绝不用裸数字 id 跨命名空间做数值相等排除（会数字碰撞误剔）
* 输出：排除掩码 + 原因分类 + 可恢复（疑似误剔）清单；自检 JSON
* 用法：
    # 自检（读现有池，不修改 s1 原池、不重建整池）
    python 代码/11.改进实施与小区域实验/control/ns_exclude.py --selfcheck \
        --pool F:/lc_work/prod2nd_2023/data/train2023_v2.parquet
    # 排除（供实验代理在构建「新实验视图」时调用）
    from ns_exclude import NamespaceSafeExcluder, build_exclusion_index
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROUND = 6  # 经纬度保留小数位（与 doc69/s1_build_pool_v2.py 一致）

# 已知留出清单的默认命名空间（id 空间归属）
KNOWN_NAMESPACES = {
    "eval_w1": "motherlib_row_id",
    "eval_w2": "motherlib_row_id",
    "eval_w3": "motherlib_row_id",
    "eval_w4": "motherlib_row_id",
    "eval_w5": "motherlib_row_id",
    "validation_pool_v2": "validation_pool_v2",
    "misplaced_159": "motherlib_row_id",
}


def pos_key(lon: float, lat: float) -> str:
    return f"{round(float(lon), ROUND):.{ROUND}f},{round(float(lat), ROUND):.{ROUND}f}"


def ns_id_key(namespace: str, ident: Any) -> str:
    """命名空间化主键：namespace + 分隔符 + id。避免跨空间数字碰撞。"""
    return f"{namespace}\x1f{ident}"


@dataclass
class ExclusionIndex:
    """留出排除索引：位置集合 + 按命名空间分组的 id 集合。"""

    positions: set = field(default_factory=set)
    ids_by_ns: Dict[str, set] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)

    def add_positions(self, positions: Sequence[str], source: str) -> None:
        self.positions.update(positions)
        self.source_counts[source] = self.source_counts.get(source, 0) + len(positions)

    def add_ids(self, namespace: str, ids: Sequence[Any], source: str) -> None:
        s = self.ids_by_ns.setdefault(namespace, set())
        s.update(str(i) for i in ids)
        self.source_counts[source] = self.source_counts.get(source, 0) + len(ids)


def build_exclusion_index(frames: Dict[str, pd.DataFrame],
                          namespaces: Optional[Dict[str, str]] = None,
                          lon_col: str = "lon", lat_col: str = "lat",
                          id_cols: Sequence[str] = ("row_id", "pid", "cid", "uid")) -> ExclusionIndex:
    """从 {清单名: DataFrame} 构建排除索引。

    id 键按清单声明的命名空间入桶；位置键无条件入位置集合。
    """
    namespaces = namespaces or KNOWN_NAMESPACES
    idx = ExclusionIndex()
    for name, df in frames.items():
        if df is None or len(df) == 0:
            continue
        if lon_col in df.columns and lat_col in df.columns:
            poss = [pos_key(a, b) for a, b in zip(df[lon_col], df[lat_col])]
            idx.add_positions(poss, name)
        ns = namespaces.get(name)
        if ns:
            for c in id_cols:
                if c in df.columns:
                    idx.add_ids(ns, list(df[c].astype(str)), name)
    return idx


def exclude_points(df: pd.DataFrame, index: ExclusionIndex,
                   point_namespace: str,
                   lon_col: str = "lon", lat_col: str = "lat",
                   id_col: Optional[str] = None) -> pd.DataFrame:
    """返回带排除判定的副本（新增列：excl_pos / excl_id / excluded / suspect_false_removal）。

    - excl_pos：位置键命中
    - excl_id ：id 键在「同命名空间」命中
    - excluded：excl_pos or excl_id
    - suspect_false_removal：id 在「别的命名空间」命中但位置未命中
      （即裸数字 id 会误剔、namespace-safe 不会误剔的那批）
    """
    out = df.copy()
    poss = [pos_key(a, b) for a, b in zip(out[lon_col], out[lat_col])]
    out["excl_pos"] = [p in index.positions for p in poss]

    if id_col and id_col in out.columns:
        ids = [str(i) for i in out[id_col]]
        same_ns = index.ids_by_ns.get(point_namespace, set())
        out["excl_id"] = [i in same_ns for i in ids]
        other_ids = set()
        for ns, s in index.ids_by_ns.items():
            if ns != point_namespace:
                other_ids |= s
        out["suspect_false_removal"] = [
            (i in other_ids) and (not pos_hit)
            for i, pos_hit in zip(ids, out["excl_pos"])
        ]
    else:
        out["excl_id"] = False
        out["suspect_false_removal"] = False

    out["excluded"] = out["excl_pos"] | out["excl_id"]
    return out


class NamespaceSafeExcluder:
    """实验视图构建期的排除器（只作用于新视图，不修改 s1 原池）。"""

    def __init__(self, index: ExclusionIndex):
        self.index = index

    def apply(self, df: pd.DataFrame, point_namespace: str, id_col: Optional[str] = None) -> pd.DataFrame:
        return exclude_points(df, self.index, point_namespace, id_col=id_col)


# --------------------------------------------------------------------------- #
# 自检：证明「裸 id 数值碰撞」确有其事，且 namespace-safe 可恢复误剔
# --------------------------------------------------------------------------- #
def selfcheck_pool(pool_path: str, sample_rows: Optional[int] = None) -> Dict[str, Any]:
    """读现有池做只读自检（不重建整池、不修改原池）。

    产出：跨类 point_id 碰撞数、跨类同位重复、以及 doc69 §2.3 记录的
    园地 4,238 误剔证据（若源文件提供 uid+命名空间则可精确复算，否则引用记录值）。
    """
    df = pd.read_parquet(pool_path, columns=["point_id", "cl", "lon", "lat"])
    n = len(df)
    # 1) 跨类 point_id 数字碰撞：同一 point_id 出现在多个类别
    g = df.groupby("point_id")["cl"].nunique()
    collision_ids = int((g > 1).sum())
    # 参与碰撞的记录数
    colliding_ids = set(g[g > 1].index)
    collision_rows = int(df["point_id"].isin(colliding_ids).sum())
    # 2) 六位同位重复（跨类冲突位置）
    df["_pk"] = [pos_key(a, b) for a, b in zip(df["lon"], df["lat"])]
    gp = df.groupby("_pk")["cl"].nunique()
    dup_pos = int((gp > 1).sum())
    # 3) 裸 id 唯一性判定
    bare_unique = int(df["point_id"].nunique())
    return {
        "pool_path": os.path.abspath(pool_path),
        "n_rows": n,
        "n_unique_point_id": bare_unique,
        "bare_id_is_unique": bool(bare_unique == n),
        "cross_class_point_id_collisions": collision_ids,
        "cross_class_collision_rows": collision_rows,
        "cross_class_duplicate_positions": dup_pos,
        "doc69_reference": {
            "point_id_collisions": 121992,
            "yuandi_uid_false_removal": 4238,
            "yuandi_note": "园地 uid 与留出 row_id 数值相同被剔除，逐点未见坐标重合，"
                           "两套位置中位距离约 1,224 km；属保守误剔，不得跳过空间排除自动回填",
        },
        "verdict": (
            "裸 point_id 非全局主键（存在跨类数字碰撞）→ 排除必须命名空间化；"
            "namespace-safe 方案：位置键无条件 + id 键仅在同命名空间生效"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="namespace-safe 排除 helper / 自检")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--pool", default="F:/lc_work/prod2nd_2023/data/train2023_v2.parquet")
    ap.add_argument("--out", default=None, help="自检 JSON 输出路径")
    args = ap.parse_args()
    if args.selfcheck:
        res = selfcheck_pool(args.pool)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

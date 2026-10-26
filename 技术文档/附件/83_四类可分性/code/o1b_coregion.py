# -*- coding: utf-8 -*-
"""o1b_coregion.py — Task D 去混淆：在「落叶松真实分布区」内取 ENF 对照点，检验针叶内部可分性

* 动因  o1 的 Task D 里 ENF 纬度中位 30.4°、DNF 48.2°，最近邻中位 356 km —— 经纬度基线 AUC 0.977
        纯属地理分离，不能回答「同一片林区里落叶松 vs 常绿针叶」。本脚本只在 DNF 实际分布窗口
        （经度>118、纬度>40 的东北窗口，以及新疆阿尔泰窗口）补取 ENF 点，构造共现样本后重测。
* 输入  needle_broad_veg30_v1.parquet + OlmoEarth zarr（2023）
* 输出  _o1b_coregion.json（共现区 Task D 指标 + 读取统计），点并入 _o1_feats_partial.parquet
* 用法  python code/o1b_coregion.py [--n 60] [--deadline-min 8]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o1_olmo_probe as M  # noqa: E402

OPT = M.OPT
PART = M.PART
OUT = os.path.join(OPT, "_o1b_coregion.json")

# 落叶松真实分布窗口（与 DNF 实测中位一致）
WINDOWS = [
    ("东北", 118.0, 135.0, 40.0, 54.0),
    ("阿尔泰", 85.0, 91.0, 45.0, 50.0),
]


def in_windows(lon, lat):
    m = np.zeros(len(lon), dtype=bool)
    for _, x0, x1, y0, y1 in WINDOWS:
        m |= (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--deadline-min", "--deadline_min", dest="deadline_min", type=float, default=8.0)
    args = ap.parse_args()
    t0 = time.time()

    def log(m):
        print(m, flush=True)

    df = pd.read_parquet(M.SRC)
    lm, cls_of, ncls_of, note_of = M.build_labels(df)
    df["leaf4"] = df["gen"].map(cls_of)
    df["ncls"] = df["gen"].map(ncls_of).fillna(0).astype(int)
    df["win"] = in_windows(df.lon.to_numpy(), df.lat.to_numpy())

    have = pd.read_parquet(PART) if os.path.exists(PART) else pd.DataFrame(columns=["point_id"])
    have_ids = set(have.point_id.astype("int64")) if len(have) else set()

    # 窗口内 ENF 补点（0.3° 间隔），排除已取点
    rng = np.random.default_rng(M.BLOCK_SEED + 1)
    cand = df[(df.leaf4 == "ENF") & df.win & (~df.point_id.isin(have_ids))]
    log(f"[coreg] 窗口内 ENF 候选 {len(cand)}（已缓存 {int((df.leaf4=='ENF').sum())} 中 {int((df.leaf4=='ENF').sum())-len(cand)} 已在窗口外/已取）")
    pick = M.spaced_pick(cand, args.n, rng)
    log(f"[coreg] 补取 ENF {len(pick)} 点")

    if len(pick):
        pick = pick.copy()
        pick["_ord"] = 0
        feats, st = M.fetch(pick, time.time() + args.deadline_min * 60, log)
        log(f"[coreg] fetch ok={len(feats)} stats={ {k: v for k, v in st.items() if k != 'fail_detail'} }")

    # 合并全量已缓存特征
    F = pd.read_parquet(PART)
    # 回填窗口标记
    F = F.merge(df[["point_id", "win"]], on="point_id", how="left")
    F["win"] = F["win"].fillna(False)
    ocol = [f"o{k:02d}" for k in range(128)]
    F = F[F.nband == 128].reset_index(drop=True)
    log(f"[coreg] 全量 {len(F)}；窗口内 {F.leaf4.value_counts().to_dict()} | 窗口内分布 {F[F.win].leaf4.value_counts().to_dict()}")

    res = {"n_total": int(len(F)), "window_defs": WINDOWS,
           "window_class_counts": {k: int(v) for k, v in F[F.win].leaf4.value_counts().items()},
           "all_class_counts": {k: int(v) for k, v in F.leaf4.value_counts().items()}}

    # ---- 共现区 Task D：窗口内 ENF vs DNF ----
    W = F[F.win & F.leaf4.isin(["ENF", "DNF"])].reset_index(drop=True)
    log(f"[coreg] 窗口内 ENF vs DNF: {W.leaf4.value_counts().to_dict()}")
    if len(W) >= 60 and W.leaf4.nunique() == 2 and min(W.leaf4.value_counts()) >= 15:
        X = W[ocol].to_numpy(np.float32)
        Xb = W[["lon", "lat"]].to_numpy(np.float32)
        y = (W.leaf4 == "DNF").astype(int).to_numpy()
        g = M.blocks(W.lon, W.lat)
        gc = M.blocks(W.lon, W.lat, 200.0)
        r = M.run_task(X, Xb, y, g, "Dc_coregion_ENF_vs_DNF", [0, 1], log, gc)
        r.pop("_oof", None)
        res["D_coregion"] = r
        # 同点 AEF64 对照
        try:
            AFL = pd.read_parquet(os.path.join(OPT, "aef_all_points.parquet"))
            acol = [f"A{k:02d}" for k in range(64)]
            AFL["_k"] = [f"{a:.7f}_{b:.7f}" for a, b in zip(AFL.lon, AFL.lat)]
            W2 = W.copy()
            W2["_k"] = [f"{a:.7f}_{b:.7f}" for a, b in zip(W2.lon, W2.lat)]
            j = W2.merge(AFL[["_k"] + acol], on="_k", how="inner")
            Xa = j[acol].to_numpy(np.float32)
            ra = M.run_task(Xa, j[["lon", "lat"]].to_numpy(np.float32),
                            (j.leaf4 == "DNF").astype(int).to_numpy(), M.blocks(j.lon, j.lat),
                            "Dc_coregion_AEF64", [0, 1], log, M.blocks(j.lon, j.lat, 200.0))
            ra.pop("_oof", None)
            res["D_coregion_AEF64"] = ra
            Xcat = np.hstack([j[ocol].to_numpy(np.float32), Xa])
            rc = M.run_task(Xcat, j[["lon", "lat"]].to_numpy(np.float32),
                            (j.leaf4 == "DNF").astype(int).to_numpy(), M.blocks(j.lon, j.lat),
                            "Dc_coregion_CAT", [0, 1], log, M.blocks(j.lon, j.lat, 200.0))
            rc.pop("_oof", None)
            res["D_coregion_CAT"] = rc
        except Exception as e:  # noqa: BLE001
            log(f"[coreg] AEF64 对照失败：{e}")
            res["aef_error"] = str(e)
    else:
        log("[coreg] 共现区样本不足，跳过")
        res["D_coregion"] = {"skipped": True, "n": int(len(W))}

    res["elapsed_min"] = (time.time() - t0) / 60
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    log(f"[coreg] done -> {OUT}")


if __name__ == "__main__":
    main()

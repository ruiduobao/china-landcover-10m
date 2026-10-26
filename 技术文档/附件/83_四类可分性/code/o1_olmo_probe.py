# -*- coding: utf-8 -*-
"""o1_olmo_probe.py — OlmoEarth v1.3 嵌入对「常绿针叶/落叶针叶/常绿阔叶/落叶阔叶」四类的可分性探针

* 输入  F:/lc_work/nb_national_2026/opt/needle_broad_veg30_v1.parquet（138,175 点，gen=30m 植被群系）
        s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr（utm01..utm60，2023 年，128 维 int8）
* 规则源 任务书白名单关键词（本文件 KEYWORDS 写死，不临时改）：
        gen 群系名「含…的」前缀先剥离，再按关键词**最早出现位置**裁决类；
        同位置按关键词长度（长优先）→ 类优先级 DNF>ENF>EBF>DBF；"樟子松" 预置换为 "松" 防误配樟
        只保留唯一可判定的点，落不进任何类者丢弃并统计
* 门槛  每类 ≤400 点；同类点间空间间隔 ≥0.3°；每 shard(2048px=20.48km) 只读一次；
        读取失败重试 2 次（第 2 次降级 64 波段）；硬截止 34 min 后停止读取、用已取到的点分析
* 输出  o1_olmo.parquet（point_id, lon, lat, leaf4, gen, zone, nband, ncls, o00..o127）
        o1_label_map.csv（群系→四类映射明细，含歧义/丢弃裁决）
        o1_result.json（任务 A–E 指标 + 经纬度基线对照 + 样本数 + 读取统计）
* 用法  python code/o1_olmo_probe.py [--stage all|labels|fetch|analyze] [--quota 200]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

OPT = r"F:/lc_work/nb_national_2026/opt"
SRC = os.path.join(OPT, "needle_broad_veg30_v1.parquet")
OUT_PQ = os.path.join(OPT, "o1_olmo.parquet")
OUT_MAP = os.path.join(OPT, "o1_label_map.csv")
OUT_JSON = os.path.join(OPT, "o1_result.json")
PART = os.path.join(OPT, "_o1_feats_partial.parquet")

YEAR = 2023
ZARR_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
ENDPOINT = "https://data.source.coop"
SHARD = 2048      # 报告单位：一个 shard = 2048×2048 px = 20.48 km
CHUNK = 256       # 实际传输粒度：chunk=(1,64,256,256)，读 1×1 也要搬整个 256×256×64 = 4.2 MB
SPACING_DEG = 0.30
DEADLINE_MIN = 30.0
BLOCK_KM = 10.0
BLOCK_SEED = 20261025
LAT0 = 35.0

# ---------------- 关键词白名单（任务书原文，写死） ----------------
KEYWORDS = {
    "ENF": ["云杉", "冷杉", "铁杉", "柏", "松", "杉木", "柳杉", "黄山松", "华山松", "油松",
            "马尾松", "云南松", "黑松", "赤松", "樟子松", "白皮松", "巴山松", "台湾松",
            "乔松", "长苞铁杉", "银杉", "水松"],
    "DNF": ["落叶松", "红杉", "水杉", "池杉", "落羽杉", "金钱松"],
    "EBF": ["栲", "槠", "青冈", "木荷", "樟", "榕", "桉", "杜英", "润楠", "楠", "相思",
            "木麻黄", "红树林", "蒲桃", "山矾", "冬青", "含笑", "黄杞"],
    "DBF": ["栎", "桦", "杨", "柳", "枫香", "水青冈", "榛", "槭", "椴", "榆", "朴", "槐",
            "核桃", "楸", "泡桐", "化香", "檀", "樟木", "山杨", "白蜡", "臭椿", "黄连木",
            "榉", "珙桐", "漆树"],
}
# "樟子松" 里的 "樟" 是松属而非樟属：先置换为等长哨兵，避免被 EBF"樟"抢先
PROTECT = [("樟子松", "松\x00\x00")]  # 3 字 → 3 字符，保持位置次序
DEPTH_PRIORITY = ["DNF", "ENF", "EBF", "DBF"]
LABELS4 = ["ENF", "DNF", "EBF", "DBF"]


# ================= 1. 群系 → 四类映射 =================
def match_gen(gen: str):
    """返回 (label4|None, 命中关键词, 命中的类数, note)。规则：剥离『含…的』前缀后按最早位置裁决。"""
    if not isinstance(gen, str) or not gen.strip():
        return None, "", 0, "空名"
    name = gen.strip()
    note = []
    # 「含 A、B 的 C 林」→ 优势种在「的」之后
    if name.startswith("含") and "的" in name:
        name = name.split("的", 1)[1]
        note.append("剥离含X的")
    for src, dst in PROTECT:
        if src in name:
            name = name.replace(src, dst)
            note.append(f"{src}→松")
    raw = []
    for ci, cls in enumerate(LABELS4):
        for kw in KEYWORDS[cls]:
            p = name.find(kw)
            if p >= 0:
                raw.append((p, p + len(kw), ci, cls, kw))
    if not raw:
        return None, "", 0, "无关键词"
    # 子串消歧：某关键词的区间若被另一个更长关键词**完整包含**，则不计
    # （"松"在"落叶松"内 → 只算 DNF；"柳"在"柳杉"内 → 只算 ENF；"栎"在"石栎"内仍算 DBF）
    hits = []
    for p0, p1, ci, cls, kw in raw:
        contained = any((q0 <= p0 and p1 <= q1 and (q1 - q0) > (p1 - p0))
                        for q0, q1, _, _, _ in raw)
        if not contained:
            hits.append((p0, -(p1 - p0), ci, cls, kw))
    if not hits:
        return None, "", 0, "仅剩子串命中"
    hits.sort()
    p, neglen, _, cls, kw = hits[0]
    ncls = len({h[3] for h in hits})
    if ncls > 1:
        note.append("混交:" + "+".join(sorted({h[3] for h in hits})))
    if any(k in gen for k in ("混交", "疏林", "矮林", "矮曲林")):
        note.append("非郁闭/混交")
    if any(k in gen for k in ("石栎", "柯", "高山栎")):
        note.append("石栎/高山栎争议:关键词按栎计DBF")
    return cls, kw, ncls, ";".join(note)


def build_labels(df: pd.DataFrame):
    rows, cls_of, ncls_of, note_of = [], {}, {}, {}
    for gen, sub in df.groupby("gen", dropna=False):
        lab, kw, ncls, note = match_gen(gen)
        cls_of[gen], ncls_of[gen], note_of[gen] = lab, ncls, note
        rows.append({
            "gen": gen, "n_points": len(sub), "leaf_mode": sub["leaf"].mode().iat[0],
            "label4": lab or "", "matched_kw": kw, "n_classes_matched": ncls,
            "note": note, "prov_n": sub["prov"].nunique(),
        })
    lm = pd.DataFrame(rows).sort_values(["label4", "n_points"], ascending=[True, False])
    return lm, cls_of, ncls_of, note_of


# ================= 2. 空间间隔抽样 =================
def spaced_pick(sub: pd.DataFrame, quota: int, rng) -> pd.DataFrame:
    """同类内贪心 0.3° 间隔抽样：随机顺序遍历，与已选点球面距离 < SPACING_DEG 则跳过。
    用 0.3° 网格哈希做邻域检索（精确判定，O(n) 而非 O(n²)）。"""
    lon = sub.lon.to_numpy(np.float64)
    lat = sub.lat.to_numpy(np.float64)
    order = rng.permutation(len(sub))
    cell = SPACING_DEG
    # 每个点需检查 3x3 邻格（网格步长 = 阈值，故只需邻格）
    grid = defaultdict(list)
    picked = []
    for i in order:
        if len(picked) >= quota:
            break
        gi = int(np.floor(lon[i] / cell))
        gj = int(np.floor(lat[i] / cell))
        ok = True
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for j in grid.get((gi + di, gj + dj), ()):
                    dx = (lon[j] - lon[i]) * np.cos(np.radians(lat[i])) * 111.32
                    dy = (lat[j] - lat[i]) * 110.57
                    if np.hypot(dx, dy) < SPACING_DEG * 111.32:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid[(gi, gj)].append(i)
            picked.append(i)
    return sub.iloc[picked].copy()


# ================= 3. OlmoEarth 读取 =================
def utm_zone(lon):
    return int((lon + 180) // 6) + 1


def fetch(points: pd.DataFrame, deadline: float, log):
    from obstore.store import S3Store
    import zarr
    from pyproj import Transformer

    t0 = time.time()
    s3 = S3Store.from_url(ZARR_URL, endpoint=ENDPOINT, skip_signature=True)
    store = zarr.storage.ObjectStore(s3, read_only=True)
    root = zarr.open_group(store, mode="r")
    log(f"[fetch] zarr open {time.time()-t0:.1f}s")

    pts = points.copy()
    pts["zone"] = [utm_zone(v) for v in pts.lon]
    transformers, tidx, shapes, tf_cache = {}, {}, {}, {}
    for z in sorted(pts.zone.unique()):
        zz = root[f"utm{z:02d}"]
        tlist = [int(t) for t in zz["time"]]
        tidx[z] = tlist.index(YEAR)
        tf_cache[z] = zz.attrs["spatial:transform"]
        shapes[z] = zz.attrs["spatial:shape"]
        transformers[z] = Transformer.from_crs("EPSG:4326", zz.attrs["proj:code"], always_xy=True)

    # 逐点投影 + shard 分组
    rc = []
    for lat, lon, z in zip(pts.lat.to_numpy(), pts.lon.to_numpy(), pts.zone.to_numpy()):
        z = int(z)
        tf = tf_cache[z]
        x, y = transformers[z].transform(lon, lat)
        col = int((x - tf[2]) / tf[0])
        row = int((y - tf[5]) / tf[4])
        H, W = int(shapes[z][0]), int(shapes[z][1])
        rc.append((z, row, col, (row // SHARD, col // SHARD), 0 <= row < H and 0 <= col < W))
    pts["zone"] = [v[0] for v in rc]
    pts["row"] = [v[1] for v in rc]
    pts["col"] = [v[2] for v in rc]
    # 传输粒度是 chunk(256)，不是 shard(2048)：按 chunk 分组才能真正做到「读到就用」
    pts["chunk"] = [(v[0], v[1] // CHUNK, v[2] // CHUNK) for v in rc]
    pts["shard"] = [(v[0], v[3][0], v[3][1]) for v in rc]
    pts["inb"] = [v[4] for v in rc]
    log(f"[fetch] {len(pts)} 点 → {pts.chunk.apply(tuple).nunique()} chunk / "
        f"{pts.shard.apply(tuple).nunique()} shard；越界 {int((~pts.inb).sum())} 点")

    groups = defaultdict(list)
    for i in range(len(pts)):
        groups[pts.chunk.iat[i]].append(i)
    # 读取顺序 = 类轮转序号优先（pts 已按 _ord 轮转排好）→ 万一截止，四类仍均衡
    order = sorted(groups.items(),
                   key=lambda kv: (min(int(pts._ord.iat[i]) for i in kv[1]),
                                   kv[0][0], kv[0][1], kv[0][2]))

    feats = {}
    # 断点续跑：已有 PART 的点不再读（8s/chunk 的代价不允许因中断重来）
    if os.path.exists(PART):
        old = pd.read_parquet(PART)
        oc = [f"o{k:02d}" for k in range(128)]
        for _, r in old.iterrows():
            feats[int(r.point_id)] = (r[oc].to_numpy(np.float32), int(r.nband))
        log(f"[fetch] resume: 已缓存 {len(feats)} 点，跳过其 chunk")
    todo = [i for i in range(len(pts)) if int(pts.point_id.iat[i]) not in feats]
    fail, retry_n, read_times, chunk_times = [], 0, [], []
    n_done = len(feats)
    n_skip_chunk = 0
    for gi, (ch, idxs) in enumerate(order):
        idxs = [i for i in idxs if int(pts.point_id.iat[i]) not in feats]
        if not idxs:
            n_skip_chunk += 1
            continue
        if time.time() > deadline:
            log(f"[fetch] DEADLINE 到，剩余 {len(order)-gi} chunk 未读")
            for i in todo:
                if int(pts.point_id.iat[i]) not in feats:
                    fail.append({"point_id": int(pts.point_id.iat[i]), "reason": "deadline_skip"})
            break
        z = ch[0]
        zz = root[f"utm{z:02d}"]
        sub = pts.iloc[idxs]
        r0, r1 = int(sub.row.min()), int(sub.row.max())
        c0, c1 = int(sub.col.min()), int(sub.col.max())
        if not sub.inb.all():
            for i in idxs:
                fail.append({"point_id": int(pts.point_id.iat[i]), "reason": "out_of_bounds"})
            continue
        got = None
        nband = 128
        for attempt in range(3):  # 1 次正常 + 2 次重试，末次降级 64 波段
            nb = 128 if attempt < 2 else 64
            try:
                a = time.time()
                arr = zz["embeddings"][tidx[z], 0:nb, r0:r1 + 1, c0:c1 + 1]
                arr = np.asarray(arr)
                dt = time.time() - a
                read_times.append(dt)
                chunk_times.append(dt)
                got, nband = arr, nb
                break
            except Exception as e:  # noqa: BLE001
                retry_n += 1
                log(f"[fetch] retry {attempt+1} chunk={ch} nb={nb}: {type(e).__name__} {str(e)[:90]}")
                time.sleep(1.5)
        if got is None:
            for i in idxs:
                fail.append({"point_id": int(pts.point_id.iat[i]), "reason": "read_failed"})
            continue
        q = got.astype(np.float32)
        val = np.where(q != -128.0, (np.abs(q) / 127.5) ** 2 * np.sign(q), np.nan)
        for i in idxs:
            rr = int(pts.row.iat[i]) - r0
            cc = int(pts.col.iat[i]) - c0
            v = val[:, rr, cc]
            if np.isnan(v).any():
                fail.append({"point_id": int(pts.point_id.iat[i]), "reason": "nodata_-128"})
                continue
            f = np.zeros(128, dtype=np.float32)
            f[:nband] = v
            feats[int(pts.point_id.iat[i])] = (f, nband)
            n_done += 1
        if (gi + 1) % 25 == 0:
            log(f"[fetch] {gi+1}/{len(order)} chunk  ok={n_done} fail={len(fail)} "
                f"mean={np.mean(chunk_times):.2f}s elapsed={(time.time()-t0)/60:.1f}min "
                f"eta={(time.time()-t0)/60/(gi+1)*(len(order)-gi-1):.1f}min")
            _save_partial(pts, feats)

    _save_partial(pts, feats)
    all_t = np.array(read_times) if read_times else np.array([np.nan])
    stats = {"n_chunk": len(order),
             "n_chunk_read": len(chunk_times),
             "n_shard_uniq": int(pts.shard.apply(tuple).nunique()),
             "n_points_ok": len(feats), "n_fail": len(fail),
             "fail_by_reason": dict(Counter(f["reason"] for f in fail)),
             "fail_detail": fail[:200], "n_retry_events": retry_n,
             "read_s_mean": float(all_t.mean()), "read_s_median": float(np.median(all_t)),
             "read_s_min": float(all_t.min()), "read_s_max": float(all_t.max()),
             "read_s_p90": float(np.percentile(all_t, 90)),
             "nband_degraded": int(sum(1 for v in feats.values() if v[1] < 128)),
             "elapsed_min": (time.time() - t0) / 60}
    log(f"[fetch] DONE ok={len(feats)} fail={len(fail)} mean={stats['read_s_mean']:.2f}s "
        f"elapsed={stats['elapsed_min']:.1f}min")
    _merge_stats(stats)
    return feats, stats


def _merge_stats(stats):
    """多次 fetch 调用（断点续跑）的读取统计累加，写 _o1_read_stats.json"""
    p = os.path.join(OPT, "_o1_read_stats.json")
    prev = {}
    if os.path.exists(p):
        try:
            prev = json.load(open(p, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
    add = ("n_points_ok", "n_fail", "n_retry_events", "n_chunk_read", "nband_degraded")
    out = dict(stats)
    for k in add:
        out[k] = int(prev.get(k, 0)) + int(stats.get(k, 0) or 0)
    out["elapsed_min"] = round(float(prev.get("elapsed_min", 0)) + float(stats.get("elapsed_min", 0)), 2)
    if prev.get("fail_by_reason") or stats.get("fail_by_reason"):
        c = Counter()
        c.update(prev.get("fail_by_reason") or {})
        c.update(stats.get("fail_by_reason") or {})
        out["fail_by_reason"] = dict(c)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)


def _save_partial(pts, feats):
    """落盘 = 本次 pts 覆盖到的点 ∪ 磁盘已有行（跨脚本续跑时 feats 含外部点）"""
    if not feats:
        return
    pids = set(int(v) for v in pts.point_id)
    ids = [i for i in feats if i in pids]
    if not ids:
        return
    M = np.vstack([feats[i][0] for i in ids])
    nb = [feats[i][1] for i in ids]
    meta = pts.set_index("point_id").loc[ids, ["lon", "lat", "leaf4", "gen", "zone", "ncls"]]
    # 一次性拼宽表，避免 128 次 insert 造成的碎片化；point_id 必须保留为列
    fdf = pd.DataFrame(M, columns=[f"o{k:02d}" for k in range(128)])
    df = pd.concat([meta.reset_index(), pd.Series(nb, name="nband"), fdf], axis=1)
    if os.path.exists(PART):
        try:
            old = pd.read_parquet(PART)
            keep = old[~old.point_id.astype("int64").isin(df.point_id.astype("int64"))]
            df = pd.concat([keep, df], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    df.to_parquet(PART, index=False)


# ================= 4. 空间块 CV =================
def blocks(lon, lat, size_km=BLOCK_KM):
    kx = 111.32 * np.cos(np.radians(LAT0))
    ky = 110.57
    return (np.floor(np.asarray(lon) * kx / size_km).astype(np.int64) * 100000
            + np.floor(np.asarray(lat) * ky / size_km).astype(np.int64))


def run_task(X, Xb, y, groups, task, labels, log, groups_coarse=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_fscore_support,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = 5
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=BLOCK_SEED)
    folds = list(sgkf.split(X, y, groups))
    ncls = len(labels)
    res = {"task": task, "labels": labels, "n": int(len(y)),
           "class_counts": {str(c): int((y == c).sum()) for c in labels},
           "n_blocks": int(len(np.unique(groups)))}
    for tag, XX in (("feat", X), ("base_lonlat", Xb)):
        oof = np.zeros((len(y), ncls), dtype=np.float64)
        pi = np.zeros(len(y), dtype=int)
        f1s, aucs = [], []
        for tr, te in folds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       class_weight="balanced", n_jobs=8, random_state=BLOCK_SEED)
            m.fit(XX[tr], y[tr])
            p = m.predict_proba(XX[te])
            # 对齐列序到 labels
            cols = np.zeros((len(te), ncls))
            for j, c in enumerate(m.classes_):
                cols[:, labels.index(c)] = p[:, j]
            oof[te] = cols
            pi[te] = np.array(labels)[cols.argmax(1)]
            f1s.append(f1_score(y[te], pi[te], labels=list(range(ncls)), average="macro", zero_division=0))
            pr = cols if ncls > 2 else cols[:, 1]
            try:
                aucs.append(roc_auc_score(y[te], pr, multi_class="ovr", average="macro")
                            if ncls > 2 else roc_auc_score(y[te], pr))
            except ValueError:
                aucs.append(float("nan"))
        res[tag] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s)),
                    "macro_f1_folds": [float(v) for v in f1s],
                    "auc_mean": float(np.nanmean(aucs)), "auc_sd": float(np.nanstd(aucs)),
                    "auc_folds": [float(v) for v in aucs]}
        if tag == "feat":
            cm = confusion_matrix(y, pi, labels=list(range(ncls)))
            p_, r_, f_, s_ = precision_recall_fscore_support(y, pi, labels=list(range(ncls)),
                                                             zero_division=0)
            res["oof_per_class"] = {labels[i]: {"precision": float(p_[i]), "recall": float(r_[i]),
                                                "f1": float(f_[i]), "support": int(s_[i])}
                                    for i in range(ncls)}
            res["oof_confusion"] = {"labels": labels, "matrix": cm.tolist()}
            res["oof_macro_f1"] = float(f1_score(y, pi, average="macro", zero_division=0))
            res["_oof"] = (oof, pi)
    d = res["feat"]["macro_f1_folds"]
    b = res["base_lonlat"]["macro_f1_folds"]
    res["delta_macro_f1_mean"] = float(np.mean(np.array(d) - np.array(b)))
    res["delta_macro_f1_sd"] = float(np.std(np.array(d) - np.array(b)))
    res["delta_auc_mean"] = float(res["feat"]["auc_mean"] - res["base_lonlat"]["auc_mean"])
    log(f"[{task}] n={res['n']} blocks={res['n_blocks']} "
        f"F1 feat {res['feat']['macro_f1_mean']:.3f}±{res['feat']['macro_f1_sd']:.3f} | "
        f"base {res['base_lonlat']['macro_f1_mean']:.3f}±{res['base_lonlat']['macro_f1_sd']:.3f} | "
        f"Δ {res['delta_macro_f1_mean']:+.3f} || AUC feat {res['feat']['auc_mean']:.3f} "
        f"base {res['base_lonlat']['auc_mean']:.3f}")
    # 粗块敏感性：0.3° 间隔抽样下 10km 块几乎 1 点/块（≈普通 CV），
    # 故另做 200km 粗块留出，检验是否只是空间自相关
    if groups_coarse is not None and len(np.unique(groups_coarse)) >= 10:
        f2 = list(sgkf.split(X, y, groups_coarse))
        f1s2 = []
        for tr, te in f2:
            if len(np.unique(y[tr])) < 2:
                continue
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       class_weight="balanced", n_jobs=8, random_state=BLOCK_SEED)
            m.fit(X[tr], y[tr])
            pi2 = m.predict(X[te])
            f1s2.append(f1_score(y[te], pi2, labels=list(range(ncls)), average="macro", zero_division=0))
        if f1s2:
            res["feat_coarseblock"] = {"block_km": 200.0,
                                       "macro_f1_mean": float(np.mean(f1s2)),
                                       "macro_f1_sd": float(np.std(f1s2)),
                                       "n_blocks": int(len(np.unique(groups_coarse)))}
            log(f"[{task}] 粗块(200km) F1 {np.mean(f1s2):.3f}±{np.std(f1s2):.3f} "
                f"（{len(np.unique(groups_coarse))} 块）")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "labels", "fetch", "analyze"])
    ap.add_argument("--quota", type=int, default=175, help="每类上限（任务书 400；时限内降到 175）")
    ap.add_argument("--deadline-min", "--deadline_min", dest="deadline_min", type=float, default=DEADLINE_MIN)
    args = ap.parse_args()

    t_start = time.time()
    logf = open(os.path.join(OPT, "code", "_o1_run.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("=" * 100)
    log(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')} quota={args.quota} stage={args.stage}")

    df = pd.read_parquet(SRC)
    lm, cls_of, ncls_of, note_of = build_labels(df)
    lm.to_csv(OUT_MAP, index=False, encoding="utf-8-sig")
    df["leaf4"] = df["gen"].map(cls_of)
    df["ncls"] = df["gen"].map(ncls_of).fillna(0).astype(int)
    counts_all = df["leaf4"].value_counts(dropna=True).to_dict()
    dropped = df[df.leaf4.isna()]
    log(f"[labels] 映射后可用 {len(df)-len(dropped)} / {len(df)}；四类全量："
        + ", ".join(f"{k}={int(counts_all.get(k,0))}" for k in LABELS4)
        + f"；丢弃 {len(dropped)} 点 / {dropped['gen'].nunique()} 群系")

    if args.stage == "labels":
        return

    # 样本选择：稀缺类优先给足，四类等额；类内 0.3° 间隔
    rng = np.random.default_rng(BLOCK_SEED)
    sel = []
    for cls in ["DNF", "ENF", "EBF", "DBF"]:   # DNF 最稀缺，先挑
        sub = df[df.leaf4 == cls]
        if len(sub) == 0:
            continue
        pick = spaced_pick(sub, args.quota, rng)
        log(f"[sample] {cls}: 可用 {len(sub)} → 间隔抽样 {len(pick)}")
        sel.append(pick)
    pts = pd.concat(sel, ignore_index=True)
    # 类轮转排序：万一截止，四类仍均衡（_ord 保留，供 fetch 决定读取顺序）
    pts["_ord"] = pts.groupby("leaf4").cumcount()
    pts = pts.sort_values(["_ord", "leaf4"]).reset_index(drop=True)
    log(f"[sample] 合计 {len(pts)} 点；各类 {pts.leaf4.value_counts().to_dict()}；"
        f"预计读取 {len(pts)*0.9/60:.0f} min（稳态 0.9 s/点）")

    feats, rstats = None, {}
    if args.stage in ("all", "fetch"):
        feats, rstats = fetch(pts, t_start + args.deadline_min * 60, log)
        log(f"[fetch] stats {json.dumps({k: v for k, v in rstats.items() if k != 'fail_detail'}, ensure_ascii=False)}")
    rs_path = os.path.join(OPT, "_o1_read_stats.json")
    if os.path.exists(rs_path):
        try:
            rstats = json.load(open(rs_path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    # 载入特征
    if os.path.exists(PART):
        F = pd.read_parquet(PART)
        log(f"[load] 特征 {F.shape} ← {PART}")
    else:
        log("[load] 无特征文件，终止")
        return
    ocols = [f"o{k:02d}" for k in range(128)]
    F = F[F["nband"] == 128].reset_index(drop=True)
    # 只保留主抽样点（pts 的 point_id 集合），把 o1b 的定向补点排除在主任务外
    n_all = len(F)
    # ncls 用当前规则重算（缓存里的可能由旧规则写入）
    F["ncls"] = F["gen"].map(ncls_of).fillna(0).astype(int)
    main_ids = set(int(v) for v in pts.point_id)
    F = F[F.point_id.astype("int64").isin(main_ids)].reset_index(drop=True)
    log(f"[load] 磁盘 {n_all} 点 → 主任务用 {len(F)} 点（排除定向补点 {n_all-len(F)}）；"
        f"各类 {F.leaf4.value_counts().to_dict()}")

    # ---------------- 任务 A–E ----------------
    X = F[ocols].to_numpy(np.float32)
    Xb = F[["lon", "lat"]].to_numpy(np.float32)
    grp = blocks(F.lon, F.lat)                       # 10km（任务书口径）
    grp_coarse = blocks(F.lon, F.lat, 200.0)         # 200km 粗块敏感性
    lab = F.leaf4.map({c: i for i, c in enumerate(LABELS4)}).to_numpy()

    tasks = {}
    # A 常绿 vs 落叶：ENF+EBF = evergreen(0)，DNF+DBF = deciduous(1)
    mA = F.leaf4.isin(["ENF", "EBF", "DNF", "DBF"]).to_numpy()
    yA = F.leaf4.isin(["DNF", "DBF"]).astype(int).to_numpy()
    tasks["A_evergreen_vs_deciduous"] = run_task(X[mA], Xb[mA], yA[mA], grp[mA],
                                                 "A_evergreen_vs_deciduous", [0, 1], log,
                                                 grp_coarse[mA])
    # B 针叶 vs 阔叶
    yB = F.leaf4.isin(["EBF", "DBF"]).astype(int).to_numpy()
    tasks["B_needle_vs_broad"] = run_task(X, Xb, yB, grp, "B_needle_vs_broad", [0, 1], log, grp_coarse)
    # C 四分类
    tasks["C_four_class"] = run_task(X, Xb, lab, grp, "C_four_class", list(range(4)), log, grp_coarse)
    # D 针叶内部 ENF vs DNF（历史难题）
    mD = F.leaf4.isin(["ENF", "DNF"]).to_numpy()
    yD = (F.leaf4[mD] == "DNF").astype(int).to_numpy()
    if len(np.unique(yD)) == 2 and min(np.bincount(yD)) >= 20:
        tasks["D_needle_ENF_vs_DNF"] = run_task(X[mD], Xb[mD], yD, grp[mD],
                                                "D_needle_ENF_vs_DNF", [0, 1], log, grp_coarse[mD])
    else:
        log(f"[D] 样本不足，跳过：{F.leaf4[mD].value_counts().to_dict()}")
    # E 常绿内部 ENF vs EBF
    mE = F.leaf4.isin(["ENF", "EBF"]).to_numpy()
    yE = (F.leaf4[mE] == "EBF").astype(int).to_numpy()
    if len(np.unique(yE)) == 2 and min(np.bincount(yE)) >= 20:
        tasks["E_evergreen_ENF_vs_EBF"] = run_task(X[mE], Xb[mE], yE, grp[mE],
                                                   "E_evergreen_ENF_vs_EBF", [0, 1], log, grp_coarse[mE])
    else:
        log(f"[E] 样本不足，跳过：{F.leaf4[mE].value_counts().to_dict()}")

    # 补充：Task D 只看「纯群系」(ncls==1) 子集，排除落叶松×云杉/樟子松混交
    if "D_needle_ENF_vs_DNF" in tasks:
        mP = mD & (F.ncls.to_numpy() == 1)
        if mP.sum() > 40 and len(np.unique((F.leaf4[mP] == "DNF").astype(int))) == 2 \
                and min(np.bincount((F.leaf4[mP] == "DNF").astype(int))) >= 20:
            tasks["D2_needle_pure_only"] = run_task(
                X[mP], Xb[mP], (F.leaf4[mP] == "DNF").astype(int).to_numpy(), grp[mP],
                "D2_needle_pure_only", [0, 1], log, grp_coarse[mP])

    # ---------------- 汇总 ----------------
    # ---------------- AEF64 同点对照 ----------------
    aef_task = {}
    AEF_ALL = os.path.join(OPT, "aef_all_points.parquet")
    if os.path.exists(AEF_ALL):
        try:
            AFL = pd.read_parquet(AEF_ALL)
            acol = [f"A{k:02d}" for k in range(64)]
            # aef_all_points 与 veg30 点坐标逐位相同但 point_id 不同 → 用坐标键对齐
            # 用字符串键（float64 在 1e9 量级只剩 1e-7 精度，求和安全但字符串更稳）
            AFL["_k"] = [f"{a:.7f}_{b:.7f}" for a, b in zip(AFL.lon, AFL.lat)]
            F2 = F.copy()
            F2["_k"] = [f"{a:.7f}_{b:.7f}" for a, b in zip(F2.lon, F2.lat)]
            j = F2.merge(AFL[["_k"] + acol], on="_k", how="inner")
            log(f"[AEF64] 同点匹配 {len(j)} / {len(F)}")
            if len(j) >= 80:
                Xa = j[acol].to_numpy(np.float32)
                Xab = j[["lon", "lat"]].to_numpy(np.float32)
                ga = blocks(j.lon, j.lat)
                la = j.leaf4.map({c: i for i, c in enumerate(LABELS4)}).to_numpy()
                # 注意：Xb 必须与 X 同步子集，否则子集任务(D/E)的经纬度基线会错位
                def _t(Xa_, Xb_, y_, gl, nm, lb):
                    gco = blocks(j.lon, j.lat, 200.0) if len(gl) == len(j) else None
                    return run_task(Xa_, Xb_, y_, gl, nm, lb, log, gco)
                # 与 OlmoEarth 完全同点、同折：AEF64 跑同样任务
                ya = j.leaf4.isin(["DNF", "DBF"]).astype(int).to_numpy()
                aef_task["A_evergreen_vs_deciduous"] = _t(Xa, Xab, ya, ga, "AEF64_A", [0, 1])
                yb = j.leaf4.isin(["EBF", "DBF"]).astype(int).to_numpy()
                aef_task["B_needle_vs_broad"] = _t(Xa, Xab, yb, ga, "AEF64_B", [0, 1])
                aef_task["C_four_class"] = _t(Xa, Xab, la, ga, "AEF64_C", list(range(4)))
                mD2 = j.leaf4.isin(["ENF", "DNF"]).to_numpy()
                yD2 = (j.leaf4[mD2] == "DNF").astype(int).to_numpy()
                if len(np.unique(yD2)) == 2 and min(np.bincount(yD2)) >= 20:
                    aef_task["D_needle_ENF_vs_DNF"] = _t(Xa[mD2], Xab[mD2], yD2, ga[mD2], "AEF64_D", [0, 1])
                mE2 = j.leaf4.isin(["ENF", "EBF"]).to_numpy()
                yE2 = (j.leaf4[mE2] == "EBF").astype(int).to_numpy()
                if len(np.unique(yE2)) == 2 and min(np.bincount(yE2)) >= 20:
                    aef_task["E_evergreen_ENF_vs_EBF"] = _t(Xa[mE2], Xab[mE2], yE2, ga[mE2], "AEF64_E", [0, 1])
                # 拼接 OlmoEarth+AEF128 维，看两源信息是否互补
                jo = F2.set_index("_k").loc[j._k]
                Xcat = np.hstack([jo[ocols].to_numpy(np.float32), Xa])
                aef_task["concat_A_evergreen_vs_deciduous"] = _t(Xcat, Xab, ya, ga, "CAT_A", [0, 1])
                aef_task["concat_D_needle_ENF_vs_DNF"] = _t(Xcat[mD2], Xab[mD2], yD2, ga[mD2], "CAT_D", [0, 1])
                aef_task["concat_C_four_class"] = _t(Xcat, Xab, la, ga, "CAT_C", list(range(4)))
        except Exception as e:  # noqa: BLE001
            log(f"[AEF64] 对照失败（不阻断主结果）：{type(e).__name__} {e}")
            aef_task = {"error": f"{type(e).__name__}: {e}"}

    slim = {}
    for k, v in tasks.items():
        v = dict(v)
        v.pop("_oof", None)
        slim[k] = v
    aef_slim = {}
    for k, v in aef_task.items():
        if isinstance(v, dict) and "_oof" in v:
            v = dict(v)
            v.pop("_oof", None)
        aef_slim[k] = v
    out = {
        "meta": {"year": YEAR, "source": "OlmoEarth v1.3 128-d embeddings",
                 "zarr": ZARR_URL, "quota_per_class": args.quota,
                 "spacing_deg": SPACING_DEG, "cv": f"StratifiedGroupKFold 5-fold, {BLOCK_KM}km blocks",
                 "block_seed": BLOCK_SEED, "rf": "300 trees, min_samples_leaf=2, balanced, n_jobs=8",
                 "elapsed_min": (time.time() - t_start) / 60},
        "label_counts_full": {k: int(counts_all.get(k, 0)) for k in LABELS4},
        "label_points_dropped": {"n_points": int(len(dropped)),
                                "n_gen": int(dropped["gen"].nunique()),
                                "top_gens": dropped["gen"].value_counts().head(15).to_dict()},
        "sample_used": {k: int((F.leaf4 == k).sum()) for k in LABELS4},
        "read_stats": rstats,
        "tasks": slim,
        "aef64_same_point_comparison": aef_slim,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    F.to_parquet(OUT_PQ, index=False)
    log(f"[done] {OUT_JSON} / {OUT_PQ} / {OUT_MAP}；总耗时 {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()

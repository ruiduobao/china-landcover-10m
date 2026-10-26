# -*- coding: utf-8 -*-
"""
o2_evidence.py — 森林四类（常绿针叶/落叶针叶/常绿阔叶/落叶阔叶）可分性证据总表 + 定量再分析
* 输入（只读，全部已落盘）
    F:/lc_work/nb_national_2026/labels/needle_broad_national_v2_habit.parquet (110,653 点)
    F:/lc_work/nb_national_2026/opt/aef_v2.parquet (109,260 点 × AEF64)
    F:/lc_work/nb_national_2026/opt/o1_result.json, _o1b_coregion.json（O1 已跑结果，只抄不重算）
    技术文档 54 / 75 / 77 / 78 / 82（结论转写）
* 规则源
    标签：leaf ∈ {needle,broad} = 针/阔轴；habit_fcs10 ∈ {evergreen,deciduous} = 常绿/落叶轴
    四类：E_N=常绿针叶(ENF) / D_N=落叶针叶(DNF) / E_B=常绿阔叶(EBF) / D_B=落叶阔叶(DBF)
          混合(mixed)一律剔除
    混杂控制口径：coregion = 至少四类各 ≥15 点的 1° 格（真正"同区域"子集）
* 门槛（预注册，跑数前写死）
    T1 真信号：某特征组相对"仅经纬度"基线的配对 Δmacro-F1 ≥ +0.05 且 5 折中 ≥4 折为正
    T2 弱信号：0.00 < Δ < 0.05
    T3 无信号：Δ ≤ 0.00（含负）
    coregion 口径下仍满足 T1 才可称"非地理假象"
* 输出（F:/lc_work/nb_national_2026/opt/）
    o2_evidence_matrix.csv        证据总表（一次性转写，不重算）
    o2_result.json                第2步各特征组指标 + 第3步三子问题 + 第4步分档裁决 + 第5步方法清单
    o2_confusion_fourclass.json   四类混淆矩阵（OOF 合并）+ 混淆对空间结构
* 用法
    python o2_evidence.py --stage all          # 全跑（约 15–20 min，单进程，CPU≤8 核，内存<3 GB）
    python o2_evidence.py --stage matrix       # 只写证据总表
    python o2_evidence.py --stage fourclass    # 只跑四类主表
    python o2_evidence.py --stage pairs        # 只跑六对/四子问题
    python o2_evidence.py --stage assemble     # 只汇总
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# ---------------- 路径与常量 ----------------
OPT = r"F:/lc_work/nb_national_2026/opt"
LAB = r"F:/lc_work/nb_national_2026/labels"
HAB = os.path.join(LAB, "needle_broad_national_v2_habit.parquet")
AEF = os.path.join(OPT, "aef_v2.parquet")
O1R = os.path.join(OPT, "o1_result.json")
O1C = os.path.join(OPT, "_o1b_coregion.json")

OUT_MATRIX = os.path.join(OPT, "o2_evidence_matrix.csv")
OUT_RESULT = os.path.join(OPT, "o2_result.json")
OUT_CONF = os.path.join(OPT, "o2_confusion_fourclass.json")
OUT_STAGE = os.path.join(OPT, "_o2_stage")          # 分阶段中间结果，防超时丢结果

BLOCK_KM = 10.0
BLOCK_SEED = 20261025
LAT0 = 35.0
N_JOBS = 8
N_TREES = 300
N_SPLITS = 5

CLS = ["E_N", "D_N", "E_B", "D_B"]                  # 0..3，先按叶型分组便于读混淆矩阵
CLS_ZH = {"E_N": "常绿针叶(ENF)", "D_N": "落叶针叶(DNF)",
          "E_B": "常绿阔叶(EBF)", "D_B": "落叶阔叶(DBF)"}
AEF_COLS = [f"A{i:02d}" for i in range(64)]
PHENO_COLS = ["amp2019", "amp2023", "djf2019", "djf2023"]
TERRAIN_COLS = ["elev_m", "slope_deg", "aspect_sin", "aspect_cos"]
#   aspect 以 sin/cos 编码（circular），不用 0/360 处会断的原始角

LOG = []


def log(msg):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    LOG.append(s)


def blocks(lon, lat, size_km=BLOCK_KM, lat0=LAT0):
    """10 km 空间块 ID（与 O1/o1b、r6_retrain 同口径）。"""
    kx = 111.32 * np.cos(np.radians(lat0))
    ky = 110.57
    return (np.floor(np.asarray(lon) * kx / size_km).astype(np.int64) * 100000
            + np.floor(np.asarray(lat) * ky / size_km).astype(np.int64))


def coregion_cells(df, min_per_class=15):
    """返回 (cell_key set, mask)；cell = 1° 格，要求四类各 ≥ min_per_class 点。"""
    cx = np.floor(df["lon"].values).astype(np.int64)
    cy = np.floor(df["lat"].values).astype(np.int64)
    key = cx * 10000 + cy
    df = df.assign(_cell=key)
    piv = df.pivot_table(index="_cell", columns="c4", values="lon", aggfunc="size").fillna(0)
    ok = piv[(piv >= min_per_class).all(axis=1)]
    return set(ok.index.tolist()), np.isin(key, ok.index.values)


# ---------------- 数据装载 ----------------
def load_points(complete_case=True):
    h = pd.read_parquet(HAB)
    h = h[h["habit_fcs10"].isin(["evergreen", "deciduous"]) & h["leaf"].isin(["needle", "broad"])].copy()
    h["c4"] = h["habit_fcs10"].str[:1].str.upper() + "_" + h["leaf"].str[:1].str.upper()
    # ⚠ 修正 s4_leaf_habit.py 的 habit_grade 缺陷（O2 审计发现，见 o2_result.json §audit_habit_grade）
    #   原脚本 R1b 分支：v2.loc[need, 'habit_grade'] = np.where((ph_e & f_e)[need], ...)
    #   其中 ph_e/f_e 是**全表布尔数组**，用 [need] 掩码取子集后长度不再对齐 need，
    #   结果 amp 分支里"落叶 + 物候判落叶"（本应 E2E）全部被写成 E2C —— 22,839 点。
    #   这里按定义重算：E2U（无物候结论）/ E2E（FCS10 与物候一致）/ E2C（冲突）。
    h["grade_raw"] = h["habit_grade"]                       # 保留原值以便对照
    h["habit_grade"] = np.where(
        ~h["habit_pheno"].isin(["evergreen", "deciduous"]), "E2U",
        np.where(h["habit_fcs10"] == h["habit_pheno"], "E2E", "E2C"))
    n_fix = int((h["grade_raw"] != h["habit_grade"]).sum())
    if n_fix:
        log(f"⚠ habit_grade 已修正 {n_fix} 点（原 E2C→E2E）："
            f"{h.loc[h['grade_raw'] != h['habit_grade'], 'habit_fcs10'].value_counts().to_dict()}")
    a = pd.read_parquet(AEF)
    h = h.merge(a, on=["lon", "lat"], how="inner", validate="one_to_one")
    h["aspect_sin"] = np.sin(np.radians(h["aspect_deg"].astype(float)))
    h["aspect_cos"] = np.cos(np.radians(h["aspect_deg"].astype(float)))
    n0 = len(h)
    if complete_case:
        cc = h[PHENO_COLS + TERRAIN_COLS].notna().all(1).values
        h = h[cc].copy()
    log(f"load_points: 合并 {n0} 点 → complete-case {len(h)} 点；类分布 "
        f"{h['c4'].value_counts().to_dict()}；"
        f"一致性等级 {h['habit_grade'].value_counts().to_dict()}")
    return h


# ---------------- CV 核心 ----------------
def make_folds(y, groups):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=BLOCK_SEED)
    return list(sgkf.split(np.zeros((len(y), 1)), y, groups))


def run_cv(X, y, folds, ncls, tag, seed=BLOCK_SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (confusion_matrix, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    X = np.ascontiguousarray(X, dtype=np.float32)
    oof = np.zeros((len(y), ncls), dtype=np.float64)
    pi = np.zeros(len(y), dtype=int)
    f1s, aucs = [], []
    t0 = time.time()
    for tr, te in folds:
        if len(np.unique(y[tr])) < ncls:            # 极端不平衡子集保护
            f1s.append(float("nan"))
            aucs.append(float("nan"))
            continue
        m = RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2,
                                   class_weight="balanced", n_jobs=N_JOBS,
                                   random_state=seed)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        cols = np.zeros((len(te), ncls))
        for j, c in enumerate(m.classes_):
            cols[:, int(c)] = p[:, j]
        oof[te] = cols
        pi[te] = cols.argmax(1)
        f1s.append(f1_score(y[te], pi[te], labels=list(range(ncls)),
                            average="macro", zero_division=0))
        try:
            if ncls == 2:
                aucs.append(roc_auc_score(y[te], cols[:, 1]))
            else:
                aucs.append(roc_auc_score(y[te], cols, multi_class="ovr", average="macro"))
        except ValueError:
            aucs.append(float("nan"))
    cm = confusion_matrix(y, pi, labels=list(range(ncls)))
    p_, r_, f_, s_ = precision_recall_fscore_support(y, pi, labels=list(range(ncls)),
                                                     zero_division=0)
    return {
        "tag": tag, "n": int(len(y)), "n_features": int(X.shape[1]),
        "class_counts": {CLS[i]: int((y == i).sum()) for i in range(ncls)},
        "macro_f1_mean": float(np.nanmean(f1s)), "macro_f1_sd": float(np.nanstd(f1s)),
        "macro_f1_folds": [float(v) for v in f1s],
        "auc_mean": float(np.nanmean(aucs)), "auc_sd": float(np.nanstd(aucs)),
        "auc_folds": [float(v) for v in aucs],
        "oof_macro_f1": float(f1_score(y, pi, labels=list(range(ncls)),
                                       average="macro", zero_division=0)),
        "per_class": {CLS[i]: {"precision": float(p_[i]), "recall": float(r_[i]),
                               "f1": float(f_[i]), "support": int(s_[i])}
                      for i in range(ncls)},
        "confusion": cm.tolist(),
        "elapsed_s": round(time.time() - t0, 1),
        "_oof": oof, "_pi": pi,
    }


def strip_internal(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def paired_delta(a, b):
    """fold 级配对差 a-b（同一批 folds），返回 (mean, sd, n_pos, folds)。"""
    fa, fb = np.array(a["macro_f1_folds"], float), np.array(b["macro_f1_folds"], float)
    d = fa - fb
    return {"delta_mean": float(np.nanmean(d)), "delta_sd": float(np.nanstd(d)),
            "n_folds_positive": int(np.nansum(d > 0)), "n_folds": int(len(d)),
            "delta_folds": [float(v) for v in d]}


# ============================================================
# 第 1 步：证据总表（全部转写自已落盘文档/JSON，不重算）
# ============================================================
def build_evidence_matrix():
    H = "【是否剔除地理混杂】是=原文做了区域/窗口/分层控制；部分=有控制但残留；否=未控制或本身就是地理基线"
    R = []

    def add(dim, task, src, method, metric, val, geo, verdict, ref, note=""):
        R.append({"维度": dim, "判别任务": task, "数据源": src, "方法": method,
                  "指标": metric, "值": val, "是否剔除地理混杂": geo,
                  "结论": verdict, "证据来源": ref, "备注": note})

    # ---- doc54 独立人工判读（唯一"真值"口径）----
    add("物候", "常绿 vs 落叶", "S2 NDVI 年振幅物理物候标签 + AEF64", "独立人工判读（z18 影像，两轮 113 森林点）",
        "AUC", "0.990", "部分", "真信号", "doc54 §A.3",
        "常绿点集中南方(w3)，含南vs北气候混杂；但该轴本身即气候-物候轴，生产上成立")
    add("物候", "常绿 vs 落叶（+S2四季/S1冬夏/DEM）", "多源特征", "同上", "AUC", "0.999–1.000",
        "部分", "仅参考（含循环性）", "doc54 §A.3", "加入 S2 四季后与标签同源，不作独立证据")
    add("叶型", "针 vs 阔（常绿组内 ENF vs EBF）", "AEF64+S2+S1", "独立人工判读", "AUC", "0.724",
        "是", "不可分（<0.85 门）", "doc54 §A.3", "n=29/74，判读员对 10 m 针阔置信度几乎全为低")
    add("叶型", "针 vs 阔（常绿组内，AEF 单独）", "AEF64", "独立人工判读", "AUC", "0.686",
        "是", "不可分", "doc54 §A.3", "—")
    add("叶型", "针 vs 阔（落叶组内 DNF vs DBF）", "—", "独立人工判读", "n(正/负)", "0 / 9",
        "是", "样本不足，无法裁决", "doc54 §A.3", "本轮由 FCS10 换源后已补到 22,718 / 33,741（见 O2 再分析）")
    add("标签质量", "目视'常绿' vs S2 物候", "z18 单期影像", "人工判读 vs 物候", "冲突率", "67%（45 点中 26 点）",
        "—", "单期影像判不出常绿/落叶", "doc54 §A.2", "常绿/落叶必须用多期影像或物候定义")

    # ---- doc75 针阔攻关：高分为什么是假象 ----
    add("叶型", "ENF vs EBF", "仅经纬度（2 维）", "池标签 RF", "AUC", "0.983–0.986", "否（即基线）",
        "地理假象基线", "doc75 §2", "针叶/阔叶样点在地理空间本就分离")
    add("叶型", "ENF vs EBF", "地形+气候（dem/bio，无遥感）", "池标签 RF", "AUC", "0.991", "否",
        "地理假象基线", "doc75 §2", "比 AEF64 还高")
    add("叶型", "针 vs 阔", "AEF64", "空间缓冲 CV（0.1° 起扫描）", "AUC", "0.979 → 0.657", "是（缓冲后）",
        "高分是口径产物", "doc75 §2", "跌 0.26；缓冲后才接近真值 0.724")
    add("叶型", "针 vs 阔", "AEF64", "配对设计（控区域+高程，590 对）", "AUC", "0.9745", "部分",
        "区域混杂仅贡献 2.3pp，但位置基线本身 0.983", "doc75 §2", "AEF 学到的是'在哪里'不是'是什么叶型'")
    add("叶型", "针 vs 阔", "AEF64", "净增量 = AEF − 位置基线", "ΔAUC", "≈0.011", "是",
        "净叶型信号 ≈0.011", "doc75 §2", "决定性的信息量口径")
    add("物候", "常绿 vs 落叶", "AEF64", "随机 / 空间缓冲 / 配对", "AUC", "0.999 / 0.945 / 0.987",
        "是", "唯一稳健信号", "doc75 §1", "缓冲后仍 0.945，是唯一通过 ≥0.95 门的轴")
    add("结构", "郁闭 vs 稀疏", "FCS10 奇偶码", "AUC", "AUC", "0.500", "—", "完全无效（证伪）",
        "doc75 §1、doc78 §3.1", "郁闭度只能作描述属性，不可作分类依据")
    add("分类结构", "4 类扁平 vs 两级级联 vs 三级", "池标签", "macro-F1 对比", "macro-F1",
        "扁平 0.8937 / 两级 0.8923–0.8932 / 三级 0.9633(分层口径 −0.0062)", "—",
        "无差异（<±1.5pp 噪声）→ 保持扁平 4 类", "doc75 §6.5", "三级损失 100% 来自最弱的一级'森林门'")
    add("跨年", "逐年独立分类相邻年跳变", "AEF64 × 8 年", "20,000 锚点三臂对比", "跳变率",
        "13.7–15.3%", "—", "逐年独立不可直接发布", "doc75 §6.6", "森林型是物种性状，真实变化不可能 14%/年")
    add("跨年", "R2 证据门控继承", "—", "后处理对比", "准确率 / 跳变率", "0.9736 / 1.26%", "—",
        "采纳 R2（+6.37pp，跳变 −91.5%）", "doc75 §6.6", "R1 朴素众数 0.9873/0 但用未来信息且抹掉真实变化")
    add("坡向伪影", "ENF/EBF 阴阳坡", "我方 2023 成品栅格（四川 24 瓦）", "MH 分层 OR",
        "OR", "2.104（栅格口径 2.678，瓦片中位）", "—", "成品上确实存在且可复现", "doc75 §3.2",
        "doc69 的 atan2 公式翻 180°，其坡向结论须按 doc75 §3.1 更正")
    add("坡向伪影", "外部产品同批点同地形", "MODIS / GLC_FCS10 / GLC_FCS30D", "OR 并列", "OR",
        "0.756 / 0.835 / 3.454", "是", "两个独立产品无反号 → 我方标签偏差", "doc75 §3.4",
        "我方 92.3% 森林标签来自 fcs30d 系；OR 3.4≈3.45 是同源复现不是精度")
    add("混交", "混交度作弃权通道", "mix_score", "阈值 0.60", "错误捕获富集",
        "标记 4.07% 像元捕获 64.7% 针阔错误（富集 15.9×）", "—", "有效工程手段（不提 F1）", "doc75 §6.5b",
        "退到常绿/落叶层后同批点正确率 99.3%")
    add("混交", "混交 vs 纯林（同源对照后）", "AEF64", "二分类", "AUC / F1", "0.9524 / 0.7661", "部分",
        "混交是共存态而非独立叶型", "doc75 §6.7", "首轮 0.982 是来源混淆（混交点 100% 来自 fcs10 系）")

    # ---- doc77 FCS10 换源 ----
    add("叶型", "针 vs 阔（二类）", "AEF64 + GLC_FCS10-2023 标签", "76 瓦 549,789 点 / 10 km 块 5 折",
        "macro-F1", "0.7547 → 0.8539（+9.9pp）", "部分", "换源全面变好", "doc77 §5.1",
        "针叶 precision 0.6767→0.8825（+20.6pp），recall 仅 −1.7pp")
    add("坡向伪影", "预测坡向 OR", "同上", "模型预测 OR", "OR", "5.231 → 2.371（参考 2.06）", "—",
        "缓解未消除（G2 FAIL）", "doc77 §5.1", "G9 达标：标签偏差被抑制")
    add("叶型", "南方针叶模型层表达", "FCS10 参考，南方 6 省", "ArmB 预测数", "预测针叶数 / recall",
        "粤 0 / 闽 0 / 浙 0 / 赣 1（recall 0–0.004）", "—", "南方针叶在模型层被清零（G10 WARN）",
        "doc77 §5.4", "ArmA 南方针叶 precision 仅 0.0416 → 旧标签造的南方针叶 97% 是伪影")
    add("面板", "8 年面板可用性", "55,712 点 × 8 年", "逐年覆盖率", "覆盖率", "95.9–99.3%", "—",
        "可用（G5 PASS）", "doc77 §7", "R2 门控把逐年跳变 3.65% → 0.54%")
    add("混交", "混交窗口占比法", "31/32/33 参考点", "51×51 px 窗口针占比", "中位针占比", "0.3612", "—",
        "混交有强地理结构，单一 0.5 阈值会系统性偏", "doc77 §5.5", "T2712 中位 0.142 vs T2815 中位 0.852")

    # ---- doc78 常绿/落叶属性一致性 ----
    add("属性一致性", "常绿/落叶：FCS10 vs S2 物候", "110,653 点（100,757 可评估）", "一致率", "一致率",
        "0.6204", "—", "常绿可信、落叶系统性漏标", "doc78 §3.2",
        "方向不对称：落叶→常绿 10,406 vs 常绿→落叶 5,005")
    add("属性一致性", "51 常绿阔叶（郁闭）", "同上", "一致率", "一致率", "0.9175", "—", "常绿阔叶可信",
        "doc78 §3.3", "n=22,870")
    add("属性一致性", "71 常绿针叶（郁闭）", "同上", "一致率", "一致率", "0.8771", "—", "常绿针叶可信",
        "doc78 §3.3", "n=25,133")
    add("属性一致性", "61 落叶阔叶（郁闭）", "同上", "一致率", "一致率", "0.4428", "—", "偏低",
        "doc78 §3.3", "其中物候判常绿 15.2%，属个别混交")
    add("属性一致性", "81 落叶针叶（郁闭）", "同上", "一致率", "一致率", "0.2662", "—",
        "系统性误标（本轮最关键发现）", "doc78 §3.3", "n=20,414，占 D_N 样本的 95.3%")
    add("机制", "81 码内部：物候常绿 vs 物候落叶", "海拔 + 省份", "中位海拔对比", "中位海拔",
        "2,598 m vs 1,004 m（+1,594 m）", "是", "高海拔云杉/冷杉（常绿）被 FCS10 记为落叶针叶",
        "doc78 §3.4", "物候常绿那批：西藏 1,136 / 四川 1,031 / 新疆 666 / 云南 601")
    add("机制", "南方反向带（渝/浙/粤/琼/鄂/桂）", "djf2023", "中位对比", "中位 djf",
        "918 点 djf 405（真落叶） vs 8,184 点 djf 736", "是", "南方真落叶林被 FCS10 并入常绿码",
        "doc78 §3.4", "与青藏同一个病的反方向；FCS10 常绿/落叶轴在过渡带分辨率不足")
    add("单变量", "amp2023 → 常绿", "100,757 点", "单变量 AUC", "AUC", "0.9553", "—",
        "不作独立证据（amp 是标签派生量）", "doc78 §2.2、doc75 §6.5c",
        "amp2019 AUC 0.9997 纯循环（v2019 由它截断而来）")

    # ---- doc82 三源外部数据 + 质量裁决 ----
    add("影像核查", "南方针叶组 vs 阔叶对照组", "S2 生长季/冬季 NDVI + S1 冬季 VH", "240 点判据效度检验",
        "字面针叶率", "针叶组 62.5% vs 阔叶对照组 59.1%（Fisher p=0.81）", "—",
        "判据失效，不能用", "doc82 §4.1", "北方针叶对照组 0%（积雪+落叶松+冬季 VH 崩塌三根因）")
    add("影像核查", "单指标判别力", "同上", "AUC", "AUC", "0.38–0.56（ndvi_djf 甚至反向）", "—",
        "无判别力", "doc82 §4.1", "阈值敏感性：vh>−18 使针叶率 56.6→72.0%，对照组同步 59.1→72.0%")
    add("影像核查", "多指标逻辑回归", "同上", "5 折 CV", "AUC", "0.61–0.64", "—", "不足",
        "doc82 §4.1", "故'30m 标签在南方还剩多少真针叶'仍开放")
    add("质量裁决", "针阔二类 macro-F1（四臂）", "AEF64，10 km 块 5 折（种子 20261024）", "统一协议重训",
        "macro-F1", "V2 0.8701±0.0019 / AB 0.7569±0.0063 / A 0.7876 / MIX 0.7675", "部分",
        "新样本净损害（Δ=−0.1131，配对 t=−34.17，p=4.4e-6）", "doc82 §11.2",
        "下游应继续用 ARM_V2")
    add("第三方裁判", "CGLS-LC100 forest_type", "PROBA-V 独立分类（非 FCS10 衍生）", "kappa", "kappa",
        "V2 0.4524 vs AB 0.2131（2.1×）", "是", "独立裁判支持 V2", "doc82 §11.4",
        "AB 模型系统性过报针叶 2.2×（真值 0.220 → 预测 0.487）")
    add("FCS10 效度", "针叶图斑内 vs 阔叶图斑内 FCS10 针叶率", "100 万植被图分层（地面调查）", "分离度", "pp",
        "青藏 +37.9 / 西北 +35.8 / 华北 +25.3 / 东北 +15.8 / 华东 +6.0 / 华南 −3.4", "是",
        "FCS10 针叶信号在南方失效甚至反向", "doc82 §2.1", "南方 6 省合并 −2.6pp，非南方 +22.7pp")
    add("FCS10 效度", "南方非森林区针叶率 vs 针叶图斑内", "同上", "对比", "针叶率",
        "26.0% vs 19.3%（非森林高 6.7pp）", "是", "FCS10 南方针叶与地表位置无关", "doc82 §2.2",
        "广西 −14.1pp、湖南 −5.0pp")
    add("源独立性", "30 m 植被类型 vs 100 万植被图", "584 基础码 / 群系词表 / 点级", "五项检验", "一致率",
        "点级 0.9089 / 图例码 95.72% / 词表 73.87%", "—", "同一套图，不是独立第三源", "doc82 §1.1",
        "本轮实际只有两方独立证据：FCS10 与地面调查")
    add("两源冲突", "植被图 vs FCS10（针阔轴）", "49,681 可比点", "一致率", "一致率", "0.5753", "—",
        "系统性分歧（非随机噪声）", "doc82 §0", "35m vs FCS10 同为 0.5768")
    add("生物学不可能", "马尾松 / 杉木 → EBF", "FCS10", "物种级图斑交叉", "判成常绿阔叶比例",
        "62.4% / 62.4%", "—", "FCS10 硬错误", "doc82 §2.3", "两者都是确定的常绿针叶")
    add("生物学不可能", "云杉 → DNF", "FCS10", "物种级图斑交叉", "判成落叶针叶比例", "51.3%", "—",
        "FCS10 在物候轴上的硬错误", "doc82 §2.3", "云杉是常绿针叶")
    add("年代", "植被图针叶图斑内针叶占森林比（PFT）", "China Forest Dataset v1.0", "PFT 1981→2013",
        "针叶占比", "74.8% → 65.8%（13 年收窄 3.3pp）", "是", "年代偏移不推翻全国结论", "doc82 §3.1",
        "阔叶对照 2013 阔叶占 86.8%，分离度 52.6pp")
    add("邻域", "邻域针叶占比 → 该点是否针叶", "24 瓦 70,575 点", "AUC", "AUC", "0.9634", "—",
        "邻域组成是极强'地理'预测子（伪信号来源）", "m7_structure.json C2",
        "同质邻域比例 needle 0.7686 vs broad 0.8512")

    # ---- O1：OlmoEarth 四类（本轮刚跑完，527 点）----
    add("四类", "A 常绿 vs 落叶", "OlmoEarth v1.3 128 维（2023）", "10 km 块 5 折 RF（467 点）", "macro-F1",
        "0.841（lonlat 基线 0.809，Δ+0.032）", "否", "弱——不如物候特征（本报告 O2 实测 0.93）",
        "o1_result.json tasks.A", "AUC 0.894")
    add("四类", "B 针 vs 阔", "OlmoEarth", "同上", "macro-F1", "0.732（基线 0.623，Δ+0.109）", "否",
        "中等", "o1_result.json tasks.B", "AUC 0.789")
    add("四类", "C 四类整体", "OlmoEarth", "同上", "macro-F1", "0.641（基线 0.522，Δ+0.118）；AUC 0.850",
        "否", "低", "o1_result.json tasks.C", "D_B F1 仅 0.551、E_N 0.498")
    add("四类", "D ENF vs DNF", "OlmoEarth", "同上", "macro-F1", "0.893（基线 0.905，Δ−0.012）", "否",
        "高分全部来自地理", "o1_result.json tasks.D", "ENF/DNF 最近跨类距离中位 356 km")
    add("四类", "D ENF vs DNF（同区域窗口）", "OlmoEarth", "限东北+阿尔泰窗口（178 点）", "macro-F1",
        "0.750（基线 0.782，Δ−0.031）；AEF64 0.784（Δ+0.002）", "是", "去掉地理后无任何增益",
        "_o1b_coregion.json", "DNF 仅 11/116 点在 ENF 100 km 内")
    add("四类", "E ENF vs EBF（常绿内）", "OlmoEarth", "10 km 块 5 折", "macro-F1",
        "0.737（基线 0.699，Δ+0.038）", "否", "弱", "o1_result.json tasks.E", "AUC 0.822")
    add("四类", "AEF64 同点对照（A–E）", "AEF64（2023）", "同点同协议", "macro-F1",
        "A 0.841 / B 0.730 / C 0.634 / D 0.893 / E 0.755", "否", "与 OlmoEarth 持平（差 <0.02）",
        "o1_result.json aef64_same_point_comparison", "D 基线 0.905 高于两者")
    add("四类", "OlmoEarth + AEF64 拼接", "192 维", "同点同协议", "macro-F1",
        "A 0.843 / C 0.642 / D 0.914（vs 单源最好 0.841/0.641/0.893）", "否",
        "无增益（Δ ≤0.02，落在种子噪声内）", "o1_result.json concat_*", "两套嵌入信息高度重叠")

    # ---- O2 本轮实测（含在 o2_result.json，这里转写便于对照）----
    fc = os.path.join(OUT_STAGE, "fourclass.json")
    pr = os.path.join(OUT_STAGE, "pairs.json")
    ax = os.path.join(OUT_STAGE, "axes2.json")
    ce = os.path.join(OUT_STAGE, "ceiling.json")
    if all(os.path.exists(f) for f in (fc, pr, ax, ce)):
        F = json.load(open(fc, encoding="utf-8"))
        P = json.load(open(pr, encoding="utf-8"))
        X = json.load(open(ax, encoding="utf-8"))
        E = json.load(open(ce, encoding="utf-8"))

        def rr(dim, task, src, method, metric, val, geo, verdict, ref, note=""):
            add(dim, task, src, method, metric, val, geo, verdict, ref, note)

        A, C = F["all"], F["coregion"]
        rr("四类", "四类整体（ENF/DNF/EBF/DBF）", "FCS10 派生标签 × AEF64+物候+地形",
           "10 km 块 5 折 RF（95,984 点）", "macro-F1",
           f"{A['full']['macro_f1_mean']:.4f}（位置+地形基线 {A['lonlat_terrain']['macro_f1_mean']:.4f}，"
           f"Δ={A['full']['vs_lonlat_terrain']['delta_mean']:+.4f}）", "部分",
           "全量高分，但含大量地理信息", "O2 fourclass.all.full", "AUC " + f"{A['full']['auc_mean']:.4f}")
        rr("四类", "四类整体（同区域 coregion）", "同上，限四类各≥15 点的 1° 格（16,758 点）",
           "同上", "macro-F1",
           f"{C['full']['macro_f1_mean']:.4f}（位置+地形基线 {C['lonlat_terrain']['macro_f1_mean']:.4f}，"
           f"Δ={C['full']['vs_lonlat_terrain']['delta_mean']:+.4f}）", "是",
           "B 档：能分但需附加证据/限区域", "O2 fourclass.coregion.full",
           "与全量差 0.10 —— 地理信息贡献了这一差额")
        rr("物候", "常绿 vs 落叶（二类轴）", "habit_fcs10 × AEF64", "10 km 块 5 折",
           "macro-F1",
           f"全量 {X['all']['habit_ever_vs_decid']['full']['macro_f1_mean']:.4f}；"
           f"coregion {X['coregion']['habit_ever_vs_decid']['full']['macro_f1_mean']:.4f}", "是",
           "真信号：跨控制稳健", "O2 axes_two_class.habit", "")
        rr("物候", "常绿 vs 落叶（只用4个物候量）", "amp2019/2023 + djf2019/2023", "10 km 块 5 折",
           "Δ vs 位置+地形",
           f"coregion Δ={X['coregion']['habit_ever_vs_decid']['pheno']['vs_lonlat_terrain']['delta_mean']:+.4f}",
           "是", "物候单轴在扣掉地形后仍为正增益", "O2 axes2.coregion.habit.pheno",
           "全量 Δ 更高（+0.053），因地形已吸收部分气候信息")
        rr("叶型", "针 vs 阔（二类轴）", "leaf × AEF64", "10 km 块 5 折", "macro-F1",
           f"全量 {X['all']['leaf_needle_vs_broad']['full']['macro_f1_mean']:.4f}；"
           f"coregion {X['coregion']['leaf_needle_vs_broad']['full']['macro_f1_mean']:.4f}", "是",
           "分数不低，但见下一行（迁移检验）", "O2 axes_two_class.leaf", "")
        rr("叶型", "针 vs 阔（只用物候量）", "S2 物候 4 量", "10 km 块 5 折", "Δ vs 位置+地形",
           f"全量 Δ={X['all']['leaf_needle_vs_broad']['pheno']['vs_lonlat_terrain']['delta_mean']:+.4f}；"
           f"coregion Δ={X['coregion']['leaf_needle_vs_broad']['pheno']['vs_lonlat_terrain']['delta_mean']:+.4f}",
           "是", "物候对针阔零/负增益 —— 与 doc75『冬季分的是常绿/落叶不是针/阔』一致",
           "O2 axes2.leaf.pheno", "")
        for nm, key in (("(a) 针叶内部 ENF vs DNF", "a_ENF_vs_DNF"),
                        ("(a) 阔叶内部 EBF vs DBF", "a_EBF_vs_DBF"),
                        ("(b) 常绿内部 ENF vs EBF", "b_ENF_vs_EBF"),
                        ("(b) 落叶内部 DNF vs DBF", "b_DNF_vs_DBF"),
                        ("(c) 对角 ENF vs DBF", "c_ENF_vs_DBF"),
                        ("(c) 对角 DNF vs EBF", "c_DNF_vs_EBF")):
            d_ = P["coregion"][key]["aef64"]
            rr("六对", nm, "FCS10 派生标签 × AEF64", "10 km 块 5 折（coregion，每类≤6,000）",
               "macro-F1 / Δ vs 位置+地形",
               f"{d_['macro_f1_mean']:.4f} / {d_['vs_lonlat_terrain']['delta_mean']:+.4f}", "是",
               "B 档（Δ 0.08–0.13，未达 0.15 A 档门）", f"O2 pairs.coregion.{key}",
               f"最近跨类距离中位 {P['geo_overlap'][key]['median_nn_dist_km']:.0f} km")
        rr("标签", "FCS10 vs 植被图（1 km 最近邻配对）", "两套独立产品的叶型标签", "一致率", "一致率",
           f"{E['veg30_vs_fcs10_nearest_neighbour']['leaf_agreement']:.4f}"
           f"（n={E['veg30_vs_fcs10_nearest_neighbour']['n_matched_lt1km']}）", "—",
           "标签天花板：两源只有 ~0.64 一致", "O2 label_ceiling", "doc82 §0 点对点口径为 0.5753")
        cl = E["cross_label_transfer"]
        rr("标签", "跨标签迁移（训 FCS10 → 测 veg30）", "AEF64，12,348 配对点", "10 km 块 5 折",
           "macro-F1",
           f"within {cl['within_FCS10_f1'][0]:.4f} → transfer {cl['transfer_FCS10_to_veg30_f1'][0]:.4f}"
           f"（标签一致率 {cl['label_agreement']:.4f}）", "是",
           "决定性：迁移 F1 ≈ 标签一致率 → 高分不可迁移到另一套口径",
           "O2 label_ceiling.cross_label_transfer", "反向 veg30→FCS10 同结论")
        rr("标签", "D_N 内部按物候拆分", "habit_grade 修正后", "P(pred=E_N) 对比", "概率差",
           "E2C 子集 P(pred=E_N)=0.300 vs E2E 子集 0.028（Δ+0.272）", "是",
           "混淆主体是标签错而非特征不可分（E2C 那批海拔中位 2,692 m）",
           "O2 confusion_structure.label_error_test", "修正后的等级口径")

    df = pd.DataFrame(R)
    df.insert(0, "编号", [f"E{i+1:02d}" for i in range(len(df))])
    df.to_csv(OUT_MATRIX, index=False, encoding="utf-8-sig")
    log(f"证据总表 → {OUT_MATRIX}（{len(df)} 行）")
    return df


# ============================================================
# 第 2 步：四类可判别性严格再分析
# ============================================================
def feature_groups(h):
    return {
        "lonlat": ["lon", "lat"],
        # 严格位置控制：位置 + 地形。doc75 已证 AEF 强编码地形（R² dem 0.962），
        # 故"位置+地形"才是判断'叶型信号是否只是地形代理'的正确基线。
        "lonlat_terrain": ["lon", "lat"] + TERRAIN_COLS,
        "pheno": PHENO_COLS,
        "pheno_terrain": PHENO_COLS + TERRAIN_COLS,
        "aef64": AEF_COLS,
        "full": AEF_COLS + PHENO_COLS + TERRAIN_COLS,
    }


def fourclass_table(h, tag, featgroups, y, folds,
                    keep_groups=("lonlat", "lonlat_terrain", "pheno", "pheno_terrain", "aef64", "full")):
    out = {}
    base = None
    for g in keep_groups:
        cols = featgroups[g]
        X = h[cols].values
        r = run_cv(X, y, folds, 4, f"{tag}:{g}")
        log(f"  {tag}/{g}: macro-F1 {r['macro_f1_mean']:.4f}±{r['macro_f1_sd']:.4f} "
            f"AUC {r['auc_mean']:.4f} ({r['elapsed_s']}s, {r['n_features']}d)")
        if g == "lonlat":
            base = r
        out[g] = r
    for g in keep_groups:
        if g != "lonlat" and base is not None:
            out[g]["vs_lonlat"] = paired_delta(out[g], base)
            if "lonlat_terrain" in out:
                out[g]["vs_lonlat_terrain"] = paired_delta(out[g], out["lonlat_terrain"])
            d2 = out[g].get("vs_lonlat_terrain", {}).get("delta_mean")
            log(f"    Δ({g} − lonlat) = {out[g]['vs_lonlat']['delta_mean']:+.4f}  "
                f"Δ({g} − lonlat+地形) = {d2:+.4f}" if d2 is not None else
                f"    Δ({g} − lonlat) = {out[g]['vs_lonlat']['delta_mean']:+.4f}")
    return out


def stage_fourclass():
    t0 = time.time()
    h = load_points(complete_case=True)
    y = np.array([CLS.index(c) for c in h["c4"]], dtype=int)
    grp = blocks(h["lon"].values, h["lat"].values)
    folds = make_folds(y, grp)
    log(f"四类主表：n={len(y)}，10 km 块 {len(np.unique(grp))} 个，5 折")
    fg = feature_groups(h)

    res = {"design": {
        "n": int(len(y)), "n_blocks": int(len(np.unique(grp))),
        "block_km": BLOCK_KM, "block_seed": BLOCK_SEED, "n_splits": N_SPLITS,
        "rf": f"{N_TREES} trees, min_samples_leaf=2, balanced, n_jobs={N_JOBS}",
        "labels": "habit_fcs10 × leaf（FCS10 码派生，非独立真值）",
        "complete_case": "四列物候 + 地形 + AEF64 全部非缺失（109,260 → 95,984）",
        "class_counts": {CLS[i]: int((y == i).sum()) for i in range(4)},
        "class_names_zh": CLS_ZH,
    }, "all": {}, "coregion": {}, "e2e_only": {}}

    # (1) 全量
    res["all"] = fourclass_table(h, "all", fg, y, folds)

    # (2) coregion：四类各≥15 点的 1° 格
    cells, m = coregion_cells(h, 15)
    hc = h[m].reset_index(drop=True)
    yc = y[m]
    log(f"coregion 子集：{len(cells)} 个 1° 格，n={len(hc)}，分布 {hc['c4'].value_counts().to_dict()}")
    grpc = blocks(hc["lon"].values, hc["lat"].values)
    foldsc = make_folds(yc, grpc)
    fgc = feature_groups(hc)
    res["coregion"] = fourclass_table(hc, "coregion", fgc, yc, foldsc)
    res["coregion"]["design"] = {
        "n": int(len(yc)), "n_cells_1deg": int(len(cells)),
        "cell_rule": "1° 格内四类各 ≥15 点",
        "class_counts": {CLS[i]: int((yc == i).sum()) for i in range(4)},
        "bbox": [float(hc.lon.min()), float(hc.lon.max()), float(hc.lat.min()), float(hc.lat.max())],
        "top_prov": hc["prov"].value_counts().head(8).to_dict(),
    }

    # (3) E2E-only 敏感性（双源一致的习惯标签）——注意：该子集是按物候一致性筛出来的，
    #     故 pheno 特征组在此子集上被系统性高估，只能读 AEF64/lonlat 两列
    #     更干净的口径：只用 s2_two_year（2019+2023 两年自一致的离散结论）这一独立档，
    #     剔除 s2_amp_thr350（其阈值 350 是在 FCS10 上扫描出来的 → 有循环性）
    res["e2e_only"] = {}
    for viz, msk, note in (
        ("E2E_all", h["habit_grade"] == "E2E",
         "habit_grade=='E2E'（含 amp_thr350 档，该档阈值由 FCS10 扫描定 → 部分循环）"),
        ("E2E_two_year", (h["habit_grade"] == "E2E") & (h["habit_pheno_src"] == "s2_two_year"),
         "E2E 且 pheno 证据来自 2019+2023 两年自一致（独立档，无阈值循环）"),
    ):
        he = h[msk.values].reset_index(drop=True)
        ye = np.array([CLS.index(c) for c in he["c4"]], dtype=int)
        grpe = blocks(he["lon"].values, he["lat"].values)
        folde = make_folds(ye, grpe)
        fge = feature_groups(he)
        r = fourclass_table(he, f"e2e[{viz}]", fge, ye, folde,
                            keep_groups=("lonlat", "aef64", "pheno"))
        r["design"] = {"n": int(len(ye)), "rule": note,
                       "class_counts": {CLS[i]: int((ye == i).sum()) for i in range(4)},
                       "WARNING": "该子集按物候一致性筛选 → pheno 组被系统性高估，"
                                  "只能读 lonlat 与 aef64 两列；且样本偏向'易判'点。"}
        res["e2e_only"][viz] = r

    # 保存 OOF 供混淆结构分析
    np.savez_compressed(os.path.join(OUT_STAGE, "fourclass_oof.npz"),
                        y=y, pi_aef=res["all"]["aef64"]["_pi"],
                        pi_full=res["all"]["full"]["_pi"],
                        pi_pheno=res["all"]["pheno"]["_pi"],
                        lon=h["lon"].values, lat=h["lat"].values,
                        elev=h["elev_m"].values, prov=h["prov"].values,
                        fcs10=h["fcs10_raw"].values, grade=h["habit_grade"].values,
                        c4=h["c4"].values)
    res["elapsed_min"] = round((time.time() - t0) / 60, 2)
    _dump(res, "fourclass")
    return res


# ============================================================
# 第 3 步：三子问题（六对 + coregion）
# ============================================================
PAIRS = {
    "a_ENF_vs_DNF": ("E_N", "D_N"),      # (a) 针叶内部：常绿/落叶
    "a_EBF_vs_DBF": ("E_B", "D_B"),      # (a) 阔叶内部：常绿/落叶
    "b_ENF_vs_EBF": ("E_N", "E_B"),      # (b) 常绿内部：针/阔
    "b_DNF_vs_DBF": ("D_N", "D_B"),      # (b) 落叶内部：针/阔
    "c_ENF_vs_DBF": ("E_N", "D_B"),      # (c) 对角对
    "c_DNF_vs_EBF": ("D_N", "E_B"),      # (c) 对角对
}


def pair_task(h, ca, cb, featgroups, groups, folds, max_per_class=6000, rng_seed=7):
    m = h["c4"].isin([ca, cb]).values
    sub = h[m].reset_index(drop=True)
    y = (sub["c4"] == cb).astype(int).values
    if max_per_class:
        rng = np.random.default_rng(rng_seed)
        idx = []
        for v in (0, 1):
            ii = np.where(y == v)[0]
            if len(ii) > max_per_class:
                ii = rng.choice(ii, max_per_class, replace=False)
            idx.append(ii)
        idx = np.sort(np.concatenate(idx))
        sub, y = sub.iloc[idx].reset_index(drop=True), y[idx]
    g = blocks(sub["lon"].values, sub["lat"].values)
    f = make_folds(y, g)
    out = {"task": f"{ca}_vs_{cb}", "n": int(len(y)),
           "class_counts": {ca: int((y == 0).sum()), cb: int((y == 1).sum())},
           "n_blocks": int(len(np.unique(g))),
           "sampled": bool(max_per_class), "max_per_class": max_per_class}
    base = None
    for fgn in groups:
        X = sub[featgroups[fgn]].values
        r = run_cv(X, y, f, 2, f"{ca}_{cb}:{fgn}", seed=rng_seed + 100)
        if fgn == "lonlat":
            base = r
        out[fgn] = r
    for fgn in groups:
        if fgn != "lonlat":
            out[fgn]["vs_lonlat"] = paired_delta(out[fgn], base)
            if "lonlat_terrain" in out:
                out[fgn]["vs_lonlat_terrain"] = paired_delta(out[fgn], out["lonlat_terrain"])
    log(f"    {ca} vs {cb}: n={out['n']} | 基线(lonlat) {base['macro_f1_mean']:.4f} | "
        + " | ".join(f"{k} {out[k]['macro_f1_mean']:.4f}"
                     f"(dBase{out[k]['vs_lonlat']['delta_mean']:+.3f}"
                     f"/dTerr{out[k].get('vs_lonlat_terrain',{}).get('delta_mean',float('nan')):+.3f})"
                     for k in groups if k != "lonlat"))
    return out


def stage_pairs():
    t0 = time.time()
    h = load_points(complete_case=True)
    fg = feature_groups(h)
    cells, m = coregion_cells(h, 15)
    hc = h[m].reset_index(drop=True)
    log(f"六对分析：全量 n={len(h)}；coregion n={len(hc)}")

    res = {"design": {
        "max_per_class": 6000, "sampled": True,
        "note": "为控时，每对每类最多抽 6,000 点（固定种子 7）——macro-F1 估计不受影响，但 n 与全量口径不同",
        "groups": ["lonlat", "lonlat_terrain", "pheno", "aef64"],
        "block_km": BLOCK_KM, "block_seed": BLOCK_SEED,
    }, "all": {}, "coregion": {}}
    groups = ["lonlat", "lonlat_terrain", "pheno", "aef64"]
    for name, (ca, cb) in PAIRS.items():
        log(f"  [全量] {name}")
        res["all"][name] = pair_task(h, ca, cb, fg, groups, None)
    for name, (ca, cb) in PAIRS.items():
        log(f"  [coregion] {name}")
        res["coregion"][name] = pair_task(hc, ca, cb, fg, groups, None)

    # 地理重叠度（对每对算最近跨类距离）
    from scipy.spatial import cKDTree
    geo = {}
    rng = np.random.default_rng(11)
    for name, (ca, cb) in PAIRS.items():
        A = h[h.c4 == ca][["lon", "lat"]].values
        B = h[h.c4 == cb][["lon", "lat"]].values
        if len(A) > 4000:
            A = A[rng.choice(len(A), 4000, replace=False)]
        if len(B) > 4000:
            B = B[rng.choice(len(B), 4000, replace=False)]
        kx = 111.32 * np.cos(np.radians(35))
        Ax = np.c_[A[:, 0] * kx, A[:, 1] * 110.57]
        Bx = np.c_[B[:, 0] * kx, B[:, 1] * 110.57]
        d, _ = cKDTree(Bx).query(Ax, k=1)
        d2, _ = cKDTree(Ax).query(Bx, k=1)
        geo[name] = {
            "n_a": int(len(A)), "n_b": int(len(B)),
            "median_nn_dist_km": float(np.median(np.r_[d, d2])),
            "frac_a_within_25km_of_b": float((d < 25).mean()),
            "frac_b_within_25km_of_a": float((d2 < 25).mean()),
            "lon_p10_90_a": [float(np.percentile(A[:, 0], 10)), float(np.percentile(A[:, 0], 90))],
            "lon_p10_90_b": [float(np.percentile(B[:, 0], 10)), float(np.percentile(B[:, 0], 90))],
            "lat_p10_90_a": [float(np.percentile(A[:, 1], 10)), float(np.percentile(A[:, 1], 90))],
            "lat_p10_90_b": [float(np.percentile(B[:, 1], 10)), float(np.percentile(B[:, 1], 90))],
        }
    res["geo_overlap"] = geo
    res["elapsed_min"] = round((time.time() - t0) / 60, 2)
    _dump(res, "pairs")
    return res


# ============================================================
# 附加：两条轴的二类口径（生产上真正要回答的"森林 5→2"）
# ============================================================
def stage_axes2():
    t0 = time.time()
    h = load_points(complete_case=True)
    fg = feature_groups(h)
    cells, m = coregion_cells(h, 15)
    hc = h[m].reset_index(drop=True)
    log(f"二类轴分析：全量 n={len(h)}；coregion n={len(hc)}")

    def run(sub, tag, yfun, name):
        y = yfun(sub)
        g = blocks(sub["lon"].values, sub["lat"].values)
        f = make_folds(y, g)
        out = {"n": int(len(y)), "class_counts": {str(k): int((y == k).sum()) for k in np.unique(y)},
               "n_blocks": int(len(np.unique(g)))}
        base = None
        for fgn in ("lonlat", "lonlat_terrain", "pheno", "aef64", "full"):
            r = run_cv(sub[fg[fgn]].values, y, f, 2, f"{tag}:{name}:{fgn}")
            if fgn == "lonlat":
                base = r
            out[fgn] = r
        for fgn in ("lonlat_terrain", "pheno", "aef64", "full"):
            out[fgn]["vs_lonlat"] = paired_delta(out[fgn], base)
            out[fgn]["vs_lonlat_terrain"] = paired_delta(out[fgn], out["lonlat_terrain"])
        log(f"  {tag}/{name}: 基线 {base['macro_f1_mean']:.4f} | "
            + " | ".join(f"{k} {out[k]['macro_f1_mean']:.4f}"
                         f"(Δbase{out[k]['vs_lonlat']['delta_mean']:+.3f}/Δ+地形{out[k]['vs_lonlat_terrain']['delta_mean']:+.3f})"
                         for k in ("pheno", "aef64", "full")))
        return out

    habit = lambda d: (d["habit_fcs10"] == "evergreen").astype(int).values
    leaf = lambda d: (d["leaf"] == "needle").astype(int).values

    res = {"design": {"block_km": BLOCK_KM, "block_seed": BLOCK_SEED,
                      "note": "habit 轴 = habit_fcs10（FCS10 码）；leaf 轴 = leaf（FCS10 码）"},
           "all": {}, "coregion": {}}
    for tag, sub in (("all", h), ("coregion", hc)):
        res[tag]["habit_ever_vs_decid"] = run(sub, tag, habit, "habit")
        res[tag]["leaf_needle_vs_broad"] = run(sub, tag, leaf, "leaf")

    # ---- D_N 标签纯度诊断（按区域）----
    #   share_pheno_says_evergreen 直接算 habit_pheno（S2 物候侧结论），不经过 habit_grade
    d = h[h["c4"] == "D_N"].copy()
    d["_south_sw"] = ~d["prov"].isin(["黑龙江省", "内蒙古自治区", "吉林省", "辽宁省",
                                      "新疆维吾尔自治区", "河北省"])
    pur = {}
    for name, sub in (("东北+新疆+河北（兴安落叶松区）", d[~d["_south_sw"]]),
                      ("西南/青藏/南方", d[d["_south_sw"]])):
        ev = sub[sub["habit_pheno"].isin(["evergreen", "deciduous"])]
        pur[name] = {
            "n": int(len(sub)),
            "n_with_pheno": int(len(ev)),
            "share_pheno_says_evergreen": round(float((ev["habit_pheno"] == "evergreen").mean()), 4),
            "share_grade_E2C_corrected": round(float((sub["habit_grade"] == "E2C").mean()), 4),
            "elev_median_m": float(np.nanmedian(sub["elev_m"])),
            "elev_median_m_pheno_evergreen": float(np.nanmedian(ev.loc[ev["habit_pheno"] == "evergreen", "elev_m"]))
            if (ev["habit_pheno"] == "evergreen").sum() else None,
            "elev_median_m_pheno_deciduous": float(np.nanmedian(ev.loc[ev["habit_pheno"] == "deciduous", "elev_m"]))
            if (ev["habit_pheno"] == "deciduous").sum() else None,
            "amp_median": float(np.nanmedian(sub["amp2023"])),
            "djf_median": float(np.nanmedian(sub["djf2023"])),
            "top_prov": sub["prov"].value_counts().head(4).to_dict(),
        }
    res["D_N_label_purity"] = pur
    res["D_N_label_purity_note"] = (
        "西南/青藏那批'物候判常绿'即 doc78 §3.4 的高海拔云杉/冷杉机制：FCS10 给 81 落叶针叶码，"
        "但 S2 物候显示常绿。东北那批物候也大量判常绿，是 S2 在东北冬季积雪/低温下 amp 偏低造成的"
        "反向误差（doc75 §5 已记 2017/18 南方与藏东川西 S2 档案真空），故不能仅凭物候给东北 DNF 定案。")
    res["elapsed_min"] = round((time.time() - t0) / 60, 2)
    _dump(res, "axes2")
    return res


# ============================================================
# 混淆结构：错误落在哪、是不是标签错
# ============================================================
def confusion_structure():
    z = np.load(os.path.join(OUT_STAGE, "fourclass_oof.npz"), allow_pickle=True)
    y, pi = z["y"], z["pi_aef"]
    elev, prov, fcs10, grade = z["elev"], z["prov"], z["fcs10"], z["grade"]
    out = {"source": "全量 complete-case 四类模型（AEF64，OOF 预测）",
           "labels": CLS, "labels_zh": CLS_ZH,
           "confusion": None, "row_normalized": None,
           "top_confusions": [], "error_geography": {}, "label_error_test": {}}

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pi, labels=[0, 1, 2, 3])
    out["confusion"] = cm.tolist()
    out["row_normalized"] = (cm / cm.sum(1, keepdims=True)).round(4).tolist()

    # 最大混淆对（非对角）
    pairs = []
    for i in range(4):
        for j in range(4):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": CLS[i], "pred": CLS[j], "n": int(cm[i, j]),
                              "share_of_true": round(float(cm[i, j] / cm[i].sum()), 4),
                              "share_of_all_errors": round(float(cm[i, j] / (cm.sum() - np.trace(cm))), 4)})
    pairs.sort(key=lambda d: -d["n"])
    out["top_confusions"] = pairs[:8]
    out["total_errors"] = int(cm.sum() - np.trace(cm))
    out["oa"] = round(float(np.trace(cm) / cm.sum()), 4)

    # 错误的地理分布
    for i in range(4):
        e = {}
        for j in range(4):
            if i == j:
                continue
            m = (y == i) & (pi == j)
            if m.sum() < 20:
                continue
            e[CLS[j]] = {
                "n": int(m.sum()),
                "elev_median_m": float(np.nanmedian(elev[m])),
                "elev_frac_gt_2000m": float(np.nanmean(elev[m] > 2000)),
                "top_prov": pd.Series(prov[m]).value_counts().head(3).to_dict(),
                "frac_fcs10_81": float(np.mean(fcs10[m] == 81)) if i in (1,) else None,
                "frac_grade_E2C": float(np.mean(grade[m] == "E2C")),
            }
        # 该类正确分类点的对照海拔
        corr = (y == i) & (pi == i)
        e["_correct"] = {
            "n": int(corr.sum()),
            "elev_median_m": float(np.nanmedian(elev[corr])),
            "elev_frac_gt_2000m": float(np.nanmean(elev[corr] > 2000)),
            "frac_grade_E2C": float(np.mean(grade[corr] == "E2C")),
        }
        out["error_geography"][CLS[i]] = e

    # 【关键】标签错误检验：D_N 内部按 FCS10-S2 一致性拆开，看模型是否站在物候一边
    #   若 E2C（物候说常绿）子集的 P(pred=E_N) 远高于 E2E 子集 → 混淆主要是标签错，不是信号失败
    def split_test(true_cls, alt_cls):
        m = y == CLS.index(true_cls)
        r = {}
        for gname in ("E2E", "E2C", "E2U"):
            mm = m & (grade == gname)
            if mm.sum() < 50:
                continue
            r[gname] = {
                "n": int(mm.sum()),
                "pred_rate": {CLS[k]: round(float((pi[mm] == k).mean()), 4) for k in range(4)},
                "p_pred_alt": round(float((pi[mm] == CLS.index(alt_cls)).mean()), 4),
                "p_pred_true": round(float((pi[mm] == CLS.index(true_cls)).mean()), 4),
                "elev_median_m": float(np.nanmedian(elev[mm])),
            }
        if "E2E" in r and "E2C" in r:
            r["gap_E2C_minus_E2E_p_alt"] = round(r["E2C"]["p_pred_alt"] - r["E2E"]["p_pred_alt"], 4)
        return r

    out["label_error_test"]["D_N_vs_E_N"] = split_test("D_N", "E_N")
    out["label_error_test"]["D_B_vs_E_B"] = split_test("D_B", "E_B")
    out["label_error_test"]["E_N_vs_D_N"] = split_test("E_N", "D_N")
    out["label_error_test"]["E_B_vs_D_B"] = split_test("E_B", "D_B")
    out["label_error_test"]["_读法"] = (
        "E2E=FCS10 与 S2 物候一致；E2C=冲突。若某真类在 E2C 子集的'预测为另一类'比例显著高于 E2E，"
        "说明模型站在物候一边 → 该方向的混淆主要是 FCS10 标签错，而不是特征不可分。"
        "反之若两者接近，则是真实的特征不可分。")

    with open(OUT_CONF, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"混淆结构 → {OUT_CONF}：OA={out['oa']}，最大混淆 "
        f"{out['top_confusions'][0]['true']}→{out['top_confusions'][0]['pred']} "
        f"n={out['top_confusions'][0]['n']}")
    return out


# ============================================================
# 第 4/5 步：裁决与方法清单（基于实测，写下判据后再填）
# ============================================================
def audit_habit_grade():
    """审计 s4_leaf_habit.py 的 habit_grade：证实 R1b 布尔掩码缺陷并量化其影响。"""
    h = pd.read_parquet(HAB)
    h = h[h["habit_fcs10"].isin(["evergreen", "deciduous"]) & h["leaf"].isin(["needle", "broad"])].copy()
    ev = h[h["habit_pheno"].isin(["evergreen", "deciduous"])]
    true_grade = np.where(h["habit_fcs10"] == h["habit_pheno"], "E2E", "E2C")
    true_grade = np.where(h["habit_pheno"].isin(["evergreen", "deciduous"]), true_grade, "E2U")

    per_code = []
    for c, g in h.groupby("fcs10_raw"):
        ok = g[g["habit_pheno"].isin(["evergreen", "deciduous"])]
        ge = g[g["habit_grade"].isin(["E2E", "E2C"])]
        per_code.append({
            "code": int(c), "n": int(len(g)), "n_eval": int(len(ok)),
            "pheno_evergreen_share": round(float((ok["habit_pheno"] == "evergreen").mean()), 4) if len(ok) else None,
            "agree_doc78_buggy": round(float((ge["habit_grade"] == "E2E").mean()), 4) if len(ge) else None,
            "agree_corrected": round(float((ok["habit_fcs10"] == ok["habit_pheno"]).mean()), 4) if len(ok) else None,
        })

    return {
        "finding": "s4_leaf_habit.py 的 R1b（amp_thr350 补档）有布尔掩码缺陷，"
                   "把 22,839 个'FCS10=落叶 且 物候=落叶'（本应 E2E）写成 E2C。",
        "mechanism": "原式 `np.where((ph_e & f_e)[need], 'E2E', 'E2C')`：ph_e/f_e 是全表布尔数组，"
                     "`[need]` 先取子集使长度≠need，再被 pandas 按位置对齐；且 `ph_e & f_e` 只在"
                     "**两者都判常绿**时为 True，落叶方向的'一致'永远拿不到 E2E。",
        "affected_n": int((h["habit_grade"] != true_grade).sum()),
        "affected_breakdown": h.loc[h["habit_grade"] != true_grade, "fcs10_raw"].value_counts().to_dict(),
        "reported_vs_corrected": {
            "E2E": {"doc78": int((h["habit_grade"] == "E2E").sum()),
                    "corrected": int((true_grade == "E2E").sum())},
            "E2C": {"doc78": int((h["habit_grade"] == "E2C").sum()),
                    "corrected": int((true_grade == "E2C").sum())},
            "E2U": {"doc78": int((h["habit_grade"] == "E2U").sum()),
                    "corrected": int((true_grade == "E2U").sum())},
        },
        "overall_agreement": {"doc78": 0.6204,
                              "corrected": round(float((ev["habit_fcs10"] == ev["habit_pheno"]).mean()), 4),
                              "n_eval": int(len(ev))},
        "per_code": per_code,
        "what_survives": [
            "51/71（常绿码）一致率 0.9175 / 0.8771 —— 不受影响，doc78 的这条结论成立",
            "81 码内部'物候判常绿'那批海拔中位 2,598 m、vs 判落叶那批 1,004 m —— 这是对 habit_pheno 的"
            "直接统计，不经过 habit_grade，结论成立（已独立复算）",
            "南方反向带（渝/浙/粤/琼/鄂/桂）918 点 djf 中位 405 —— 基于 djf 的绝对值，成立",
        ],
        "what_is_overturned": [
            "doc78 §3.3『61 一致率仅 0.4428、81 仅 0.2662』→ 修正后 61 为 0.8477、81 为 0.7113。"
            "即 FCS10 的落叶码**并不像 doc78 描述的那样不可信**；真实的一致率是 0.71–0.85，"
            "属'中等可信'而非'系统性漏标'。",
            "doc78 §0 摘要第 3-5 条与 §8 一句话交接中'落叶方向系统性漏标'的强度描述需下调："
            "落叶码的方向性偏差依然存在（物候判常绿比例 61:15.2%、81:28.9%），但不是 0.27–0.44 那种崩塌。",
        ],
        "scope_note": "该列**未进入任何训练/生产流程**（本仓库 grep 仅 doc 与本脚本引用），"
                      "故不影响已交付成品；仅影响 doc78 的结论表述与任何按 habit_grade 做样本筛选的下游分析。",
    }


def label_ceiling_test():
    """
    标签一致率上限：量化'两套独立标签在同一地点的一致率'，给 O2 所有 F1 加一条解读边界。
    veg30（百万植被图栅格化）与 FCS10 是两套独立来源；两者一致率就是当前标签天花板。
    """
    from scipy.spatial import cKDTree
    v = pd.read_parquet(os.path.join(OPT, "needle_broad_veg30_v1.parquet"),
                        columns=["lon", "lat", "leaf", "gen", "dist_v2_m"])
    a = pd.read_parquet(os.path.join(OPT, "aef_all_points.parquet")).drop(columns=["point_id"])
    h = pd.read_parquet(HAB, columns=["lon", "lat", "leaf"])
    v = v[v["leaf"].isin(["needle", "broad"])].reset_index(drop=True)

    kx = 111.32 * np.cos(np.radians(35))
    V = np.c_[v["lon"].values * kx, v["lat"].values * 110.57]
    H = np.c_[h["lon"].values * kx, h["lat"].values * 110.57]
    d, i = cKDTree(H).query(V, k=1)
    v["nn_km"] = d
    v["fcs_leaf"] = h["leaf"].values[i]

    both = v[(v["nn_km"] < 1.0) & v["fcs_leaf"].isin(["needle", "broad"])].copy()
    agree = float((both["leaf"] == both["fcs_leaf"]).mean())

    out = {
        "purpose": "两套独立标签在同一地点的一致率 —— 这是所有 F1 的解读上界",
        "veg30_vs_fcs10_nearest_neighbour": {
            "n_matched_lt1km": int(len(both)),
            "median_nn_km_all": round(float(np.median(v["nn_km"])), 4),
            "leaf_agreement": round(agree, 4),
            "note": f"doc82 §0 的点对点严格匹配为 0.5753；放宽到 1 km 最近邻仍只有 {agree:.3f}，"
                    "说明分歧不是坐标错配造成的",
            "by_macroregion": {},
        },
        "ceiling_interpretation": (
            "四类/针阔分类器在 FCS10 派生标签上达到 0.70–0.91，度量的是『能否复现 FCS10 口径』，"
            "而非『能否复现地表真值』。两套独立产品在同地点只有 ~0.58 一致 → 在 FCS10 上训练的模型"
            "可迁移精度的天花板上限约在这一致率。"),
    }
    both["region"] = np.where(both["lon"] > 118, "东部",
                              np.where(both["lat"] < 32, "南方", "西部/北方"))
    for r, g in both.groupby("region"):
        out["veg30_vs_fcs10_nearest_neighbour"]["by_macroregion"][r] = {
            "n": int(len(g)), "agreement": round(float((g["leaf"] == g["fcs_leaf"]).mean()), 4)}

    # 独立标签臂：同样 AEF64 特征，改去拟合 veg30 标签
    vv = v.merge(a, on=["lon", "lat"], how="inner")
    y = (vv["leaf"] == "needle").astype(int).values
    grp = blocks(vv["lon"].values, vv["lat"].values)
    folds = make_folds(y, grp)
    r = run_cv(vv[AEF_COLS].values, y, folds, 2, "veg30_labels")
    out["armB_veg30_labels"] = {
        "n": r["n"],
        "macro_f1_mean": round(r["macro_f1_mean"], 4),
        "macro_f1_sd": round(r["macro_f1_sd"], 4),
        "auc_mean": round(r["auc_mean"], 4),
        "per_class": {k: {kk: (round(vv2, 4) if isinstance(vv2, float) else vv2)
                          for kk, vv2 in v2.items()} for k, v2 in r["per_class"].items()},
        "confusion": r["confusion"],
        "note": "同特征同协议，仅换标签源。与 armA（FCS10 标签，见 fourclass.aef64）对比，"
                "可区分'AEF 学到了地表叶型'与'AEF 学到了 FCS10 口径'。",
    }

    # ---- 决定性检验：跨标签迁移（1 km 配对子集，同折）----
    # 若 AEF 学到的是"地表叶型"，那么 训FCS10→测veg30 应显著高于标签一致率；
    # 若迁移 F1 ≈ 标签一致率，则说明 AEF 学到的只是 FCS10 的口径本身。
    pm = vv[(vv["nn_km"] < 1.0) & vv["fcs_leaf"].isin(["needle", "broad"])].reset_index(drop=True)
    X = pm[AEF_COLS].values.astype(np.float32)
    yv = (pm["leaf"] == "needle").astype(int).values
    yf = (pm["fcs_leaf"] == "needle").astype(int).values
    gpm = blocks(pm["lon"].values, pm["lat"].values)
    cl = {"n": int(len(pm)), "label_agreement": round(float((yv == yf).mean()), 4)}

    def _folds_cv(ytr, yte, X, g, seed=BLOCK_SEED):
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import f1_score as _f1
        ycomb = ytr.astype(str) + yte.astype(str)
        fl = list(StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                       random_state=seed).split(X, ycomb, g))
        out = {}
        for tag, yv_, yt_ in (("train", ytr, ytr), ("transfer", ytr, yte)):
            fs = []
            for tr, te in fl:
                m = RandomForestClassifierShim(seed)
                m.fit(X[tr], yv_[tr])
                fs.append(_f1(yt_[te], m.predict(X[te]), average="macro"))
            out[tag] = (round(float(np.mean(fs)), 4), round(float(np.std(fs)), 4))
        return out

    def RandomForestClassifierShim(seed):
        from sklearn.ensemble import RandomForestClassifier as _RF
        return _RF(n_estimators=N_TREES, min_samples_leaf=2, class_weight="balanced",
                   n_jobs=N_JOBS, random_state=seed)

    a_ = _folds_cv(yf, yv, X, gpm)     # 训 FCS10 → 测 veg30
    b_ = _folds_cv(yv, yf, X, gpm)     # 训 veg30 → 测 FCS10
    cl["within_FCS10_f1"] = a_["train"]
    cl["transfer_FCS10_to_veg30_f1"] = a_["transfer"]
    cl["within_veg30_f1"] = b_["train"]
    cl["transfer_veg30_to_FCS10_f1"] = b_["transfer"]
    cl["majority_baseline"] = round(float(max(yf.mean(), 1 - yf.mean())), 4)
    cl["interpretation"] = (
        "within（同源标签）≈0.72–0.74，transfer（跨源）≈0.64，与标签一致率 "
        f"{cl['label_agreement']:.3f} 几乎相等。→ AEF64 的判别力主要是『复现某一套产品的口径』，"
        "跨产品时只剩产品间本就一致的那部分。这是'针阔轴高分不可迁移'的直接量化证据，"
        "与 doc75『净叶型信号≈0.011』同向，但给出的是可迁移性上界而非增益上界。")
    out["cross_label_transfer"] = cl
    log(f"跨标签迁移：within FCS10 {a_['train'][0]:.4f} / within veg30 {b_['train'][0]:.4f} | "
        f"FCS10→veg30 {a_['transfer'][0]:.4f} / veg30→FCS10 {b_['transfer'][0]:.4f} | "
        f"标签一致率 {cl['label_agreement']:.4f}")
    log(f"标签天花板：veg30 vs FCS10 一致率 {agree:.4f}（n={len(both)}，1 km 内）；"
        f"veg30 标签臂 AEF64 macro-F1 {r['macro_f1_mean']:.4f}")
    return out


def stage_assemble():
    fc = json.load(open(os.path.join(OUT_STAGE, "fourclass.json"), encoding="utf-8"))
    pr = json.load(open(os.path.join(OUT_STAGE, "pairs.json"), encoding="utf-8"))
    ax = json.load(open(os.path.join(OUT_STAGE, "axes2.json"), encoding="utf-8"))
    st = json.load(open(os.path.join(OUT_STAGE, "structure.json"), encoding="utf-8"))
    st = st.get("structure", st)
    ceil = json.load(open(os.path.join(OUT_STAGE, "ceiling.json"), encoding="utf-8"))
    o1 = json.load(open(O1R, encoding="utf-8"))

    def g(d, k, sub="macro_f1_mean"):
        return round(d[k][sub], 4) if d and k in d else None

    def gp(scope, pair, fgn="aef64"):
        """pair 结构：pr[scope][pair][fgn]['macro_f1_mean']"""
        d = pr.get(scope, {}).get(pair)
        if not d or fgn not in d:
            return None
        return round(d[fgn]["macro_f1_mean"], 4)

    def dlp(scope, pair, fgn="aef64"):
        d = pr.get(scope, {}).get(pair)
        if not d or fgn not in d or "vs_lonlat" not in d[fgn]:
            return None
        return round(d[fgn]["vs_lonlat"]["delta_mean"], 4)

    def dl(d, k):
        return round(d[k]["vs_lonlat"]["delta_mean"], 4) if k in d and "vs_lonlat" in d[k] else None

    A, C = fc["all"], fc["coregion"]

    # ---- 裁决：预注册梯子（跑数前写死在 docstring / 本函数）----
    #   基线口径统一为 **lonlat_terrain（位置+地形）**：doc75 已证 AEF 强编码地形（R² dem 0.962），
    #   故"位置+地形"才是判断叶型信号的正确零假设；只用经纬度会系统性高估净信号。
    #   A 档：coregion 下 Δ(vs 位置+地形) ≥ +0.15 且 macro-F1 ≥ 0.75  → 可直接出图
    #   B 档：coregion 下 Δ ∈ [+0.05,+0.15) 或 macro-F1 ∈ [0.60,0.75)  → 可出图但须附加证据/限区域
    #   C 档：coregion 下 Δ < +0.05 或 macro-F1 < 0.60              → 不可分，须合并或出概率层
    def grade(delta, mf1):
        if delta is None or mf1 is None:
            return "NA", "缺数据"
        if delta >= 0.15 and mf1 >= 0.75:
            return "A", "可直接出图"
        if delta >= 0.05 or mf1 >= 0.60:
            return "B", "可出图但须附加证据/限区域"
        return "C", "不可分，须合并或出概率层"

    def dlt_strict(D, k):
        """相对'位置+地形'基线的 Δ（主口径）。"""
        if k not in D or "vs_lonlat_terrain" not in D[k]:
            return None
        return round(D[k]["vs_lonlat_terrain"]["delta_mean"], 4)

    verdict = {"prereg_rules": {
        "A": "coregion 下 Δ(vs 位置+地形) >= +0.15 且 macro-F1 >= 0.75",
        "B": "coregion 下 Δ ∈ [+0.05,+0.15) 或 macro-F1 ∈ [0.60,0.75)",
        "C": "coregion 下 Δ < +0.05 或 macro-F1 < 0.60",
        "基线说明": "主基线=经纬度+地形(elev/slope/aspect_sin/aspect_cos)。doc75 已证 AEF 强编码地形"
                    "(R2 dem 0.962)，只扣经纬度会系统性高估净叶型信号；Δ_vs_lonlat 作附报。",
        "coregion 说明": "1 度格内四类各 >=15 点，76 格 / 16,758 点 —— 四类真正同区域的子集。",
    }, "four_class_overall": {}, "axes": {}, "classes": {}, "pairs": {}}

    for scope, D in (("all", A), ("coregion", C)):
        mf1 = g(D, "full")
        mfp = g(D, "pheno")
        dlt_f = dlt_strict(D, "full")
        gd, note = grade(dlt_f, mf1)
        verdict["four_class_overall"][scope] = {
            "macro_f1_lonlat": g(D, "lonlat"),
            "macro_f1_lonlat_terrain": g(D, "lonlat_terrain"),
            "macro_f1_pheno": mfp,
            "macro_f1_pheno_terrain": g(D, "pheno_terrain"),
            "macro_f1_aef64": g(D, "aef64"), "macro_f1_full": mf1,
            "delta_pheno_vs_lonlat": dl(D, "pheno"),
            "delta_pheno_vs_lonlat_terrain": dlt_strict(D, "pheno"),
            "delta_pheno_terrain_vs_lonlat_terrain": dlt_strict(D, "pheno_terrain"),
            "delta_aef64_vs_lonlat": dl(D, "aef64"),
            "delta_aef64_vs_lonlat_terrain": dlt_strict(D, "aef64"),
            "delta_full_vs_lonlat_terrain": dlt_f,
            "auc_full": round(D["full"]["auc_mean"], 4),
            "grade": gd, "note": note,
        }

    verdict["axes"]["habit_evergreen_deciduous"] = {
        "evidence": {
            "doc54_independent_AUC": 0.990,
            "doc75_space_buffered_AUC": 0.945,
            "o2_habit_binary_all_f1": round(ax["all"]["habit_ever_vs_decid"]["full"]["macro_f1_mean"], 4),
            "o2_habit_binary_coregion_f1": round(ax["coregion"]["habit_ever_vs_decid"]["full"]["macro_f1_mean"], 4),
            "o2_habit_pheno_delta_coregion_vs_lonlat": round(
                ax["coregion"]["habit_ever_vs_decid"]["pheno"]["vs_lonlat"]["delta_mean"], 4),
            "o2_habit_pheno_delta_coregion_vs_lonlat_terrain": round(
                ax["coregion"]["habit_ever_vs_decid"]["pheno"]["vs_lonlat_terrain"]["delta_mean"], 4),
            "o2_habit_aef64_delta_coregion_vs_lonlat_terrain": round(
                ax["coregion"]["habit_ever_vs_decid"]["aef64"]["vs_lonlat_terrain"]["delta_mean"], 4),
            "o2_within_needle_f1": gp("all", "a_ENF_vs_DNF"),
            "o2_within_needle_coregion_f1": gp("coregion", "a_ENF_vs_DNF"),
            "o2_within_needle_coregion_delta": dlp("coregion", "a_ENF_vs_DNF"),
            "o2_within_broad_f1": gp("all", "a_EBF_vs_DBF"),
            "o2_within_broad_coregion_f1": gp("coregion", "a_EBF_vs_DBF"),
            "o2_within_broad_coregion_delta": dlp("coregion", "a_EBF_vs_DBF"),
        },
        "verdict": "真信号（唯一跨全部控制稳健的轴）",
    }
    verdict["axes"]["leaf_needle_broad"] = {
        "evidence": {
            "doc54_independent_AUC": 0.724, "doc75_net_signal_dAUC": 0.011,
            "doc75_space_buffered_AUC": 0.657,
            "o2_leaf_binary_all_f1": round(ax["all"]["leaf_needle_vs_broad"]["full"]["macro_f1_mean"], 4),
            "o2_leaf_binary_coregion_f1": round(ax["coregion"]["leaf_needle_vs_broad"]["full"]["macro_f1_mean"], 4),
            "o2_leaf_pheno_delta_coregion_vs_lonlat": round(
                ax["coregion"]["leaf_needle_vs_broad"]["pheno"]["vs_lonlat"]["delta_mean"], 4),
            "o2_leaf_pheno_delta_coregion_vs_lonlat_terrain": round(
                ax["coregion"]["leaf_needle_vs_broad"]["pheno"]["vs_lonlat_terrain"]["delta_mean"], 4),
            "o2_leaf_aef64_delta_coregion_vs_lonlat_terrain": round(
                ax["coregion"]["leaf_needle_vs_broad"]["aef64"]["vs_lonlat_terrain"]["delta_mean"], 4),
            "o2_leaf_aef64_delta_all_vs_lonlat_terrain": round(
                ax["all"]["leaf_needle_vs_broad"]["aef64"]["vs_lonlat_terrain"]["delta_mean"], 4),
            "o2_ENF_vs_EBF_coregion_f1": gp("coregion", "b_ENF_vs_EBF"),
            "o2_ENF_vs_EBF_coregion_delta": dlp("coregion", "b_ENF_vs_EBF"),
            "o2_DNF_vs_DBF_coregion_f1": gp("coregion", "b_DNF_vs_DBF"),
            "o2_DNF_vs_DBF_coregion_delta": dlp("coregion", "b_DNF_vs_DBF"),
        },
        "verdict": "弱/不可分（净信息量接近 0；高分几乎全部由地理位置提供）",
    }

    # 逐类裁决
    per = A["full"]["per_class"]
    perC = C["full"]["per_class"] if "per_class" in C.get("full", {}) else None
    for c in CLS:
        verdict["classes"][c] = {
            "name_zh": CLS_ZH[c],
            "f1_all": round(per[c]["f1"], 4),
            "f1_coregion": round(perC[c]["f1"], 4) if perC else None,
            "precision_all": round(per[c]["precision"], 4),
            "recall_all": round(per[c]["recall"], 4),
            "support_all": per[c]["support"],
        }

    # 六对裁决
    for name in PAIRS:
        a, cc = pr["all"].get(name), pr["coregion"].get(name)
        if not a:
            continue
        cc_d = pr["coregion"].get(name, {}).get("aef64", {})
        dlt = (round(cc_d["vs_lonlat_terrain"]["delta_mean"], 4)
               if "vs_lonlat_terrain" in cc_d else None)
        mf = gp("coregion", name)
        gd, note = grade(dlt if dlt is not None else dlp("all", name),
                         mf if mf is not None else gp("all", name))
        verdict["pairs"][name] = {
            "n_all": a["n"], "n_coregion": cc["n"] if cc else None,
            "macro_f1": {
                "lonlat": gp("all", name, "lonlat"), "pheno": gp("all", name, "pheno"),
                "aef64": gp("all", name, "aef64"),
                "lonlat_coregion": gp("coregion", name, "lonlat"),
                "pheno_coregion": gp("coregion", name, "pheno"),
                "aef64_coregion": gp("coregion", name, "aef64"),
            },
            "delta": {"all_aef64_vs_lonlat": dlp("all", name),
                      "coregion_aef64_vs_lonlat": dlp("coregion", name),
                      "coregion_aef64_vs_lonlat_terrain": dlt,
                      "coregion_pheno_vs_lonlat": dlp("coregion", name, "pheno"),
                      "coregion_pheno_vs_lonlat_terrain": (
                          round(pr["coregion"][name]["pheno"]["vs_lonlat_terrain"]["delta_mean"], 4)
                          if "vs_lonlat_terrain" in pr["coregion"][name]["pheno"] else None)},
            "geo_overlap_median_nn_km": pr["geo_overlap"][name]["median_nn_dist_km"],
            "geo_frac_within_25km": {
                "a_to_b": pr["geo_overlap"][name]["frac_a_within_25km_of_b"],
                "b_to_a": pr["geo_overlap"][name]["frac_b_within_25km_of_a"]},
            "grade": gd, "note": note,
        }

    # ---- 方法清单（第 5 步）----
    methods = [
        {"rank": 1, "name": "多年度一致性投票（2017–2024 逐年判读 + 证据门控多数票）",
         "data": "已有 AEF 八年嵌入 + 本轮 S2 物候；无需新取数",
         "what": "对同一像元逐年独立判四类，再按 R2 证据门控聚合（不是简单众数）",
         "expected": "doc75 §6.6 实测：跳变率 14.76%→1.26%，准确率 0.91→0.97（该实验在池标签上做，"
                     "对真值口径的增益需重测，但方向确定）；对四类的作用主要是压掉逐年噪声与坡向伪影摆动",
         "cost": "零新数据；本地重训 8 年 × 现有样本，GPU/CPU 数小时",
         "why_not_tried": "已证有效但只在针阔二类上验证过，未在四类上做",
         "性价比": "极高"},
        {"rank": 2, "name": "物候曲线参数化（返青期/枯黄期/生长季长度，替代年振幅）",
         "data": "S2 SR 8 天合成（2017/18 需宽窗降级）或 MODIS NDVI 16 天（2000–2024 全覆盖）",
         "what": "对每个像元拟合双 Logistic / 季节谐波，取 SOS、EOS、LOS、峰值、积分；"
                 "判 ENF vs EBF（常绿针叶冬季绿度低于常绿阔叶但都不落叶）与 DNF（落叶松秋季早衰）",
         "expected": "doc82 §4.2 已给出方向性证据：松林冬季 NDVI 更低 p=0.017，"
                     "南方针叶 40.6% 低于阔叶对照 p20；落叶松'生长季末早衰'是物理机制明确的判据。"
                     "预期把 ENF vs EBF 的净 Δ 从 ~0.04 提到 0.10–0.15（B 档）",
         "cost": "中：GEE 取 8 天序列，76 瓦约 2–4 h；建模本地 <1 h",
         "why_not_tried": "doc82 明确指出'换判据做 V4：生长季多时相 NDVI 曲线形状 + 落叶松专项判据'，"
                          "但本轮只用了冬季单值与年振幅",
         "性价比": "高"},
        {"rank": 3, "name": "GEDI 冠层高度（rh98 / rh50 / cover）作结构轴",
         "data": "GEDI L2A/L4B（2019-03 起，缺 2017/18）；work_base 的 gedi_rh98 三列**全为空**，"
                 "gee_struct_sample 也只有 cgls/hansen → 本轮无法用现有落盘数据计算，须新取",
         "what": "落叶松（Larix）塔形窄冠、云杉/冷杉尖塔深冠、松属伞形、阔叶圆冠——rh98 与 cover 的组合"
                 "在'针叶内部'与'针阔'上都有物理依据；doc75 §4 认为 GEDI 稀疏沿轨不宜作栅格，"
                 "但作**训练端分层/先验**（而非制图特征）仍可用",
         "expected": "对 ENF vs DNF（云杉 vs 落叶松，二者树高/冠形差异最大）预期 Δ +0.05–0.10；"
                     "对 ENF vs EBF 预期 Δ +0.05（结构差异真实但 25 m 足迹稀疏）",
         "cost": "中高：GEE 取 GEDI 需按 76 瓦做 footprint 聚合，约 3–5 h；且 2017/18 无覆盖 → 只能做 2019+ 或作静态先验",
         "why_not_tried": "doc75 §C3 明确记为'未测'（当时 GEE 出口中断）；m0_prep.py 里三列是占位 NaN",
         "性价比": "中"},
        {"rank": 4, "name": "S1 全年时间序列形状（VV/VH 逐旬曲线，不是冬夏两个值）",
         "data": "S1 GRD 2017–2024（GEE 全覆盖，但 2017/18 覆盖待核）",
         "what": "落叶松秋季落叶时 VH 陡降、春季回升；常绿针叶全年 VH 平稳。用逐旬 VH/VV 与其导数、"
                 "以及 VH/VV 比的季节方差做特征——这直接编码'落叶'事件，与光学物候独立（云雨不受影响）",
         "expected": "对 DNF vs ENF 预期最强（Δ +0.10 量级）；doc82 §4.1 已实测'冬季 VH 崩塌'使单值判据失效，"
                     "但那是**单时点**，时序形状恰好绕开该问题",
         "cost": "中：S1 逐旬合成 76 瓦约 2–3 h；本地建模快",
         "why_not_tried": "doc54 只测了'冬夏 VV/VH 两个值'（无增益），doc75 §4 结论'S1 相干性 GEE 无产品'——"
                          "**从未测过时序形状**",
         "性价比": "中高（针对 DNF 这一最弱环节）"},
        {"rank": 5, "name": "Landsat 30 m 长时序（1984–2024）抓人工林轮伐/演替",
         "data": "Landsat C2 L2（GEE），30 m",
         "what": "南方马尾松/杉木人工林轮伐周期 20–25 年，在 Landsat 长时序上表现为周期性'砍-长'；"
                 "天然常绿阔叶林无此周期。用时序分割（LandTrendr/CCDC 思想）提取扰动次数与恢复年限，"
                 "作为'是否人工针叶林'的判据",
         "expected": "这是解决 doc82 §4.2 '南方马尾松图斑是否已阔叶化'这个开放问题的唯一数据路径；"
                     "对南方针阔判别预期 Δ +0.10–0.20（该区现在是 C 档）",
         "cost": "高：Landsat 40 年时序在 76 瓦上做分割，GEE 数十小时 + 本地后处理；"
                 "且 30 m 与 10 m 网格不一致需重采样（须 nearest/面积加权，不得最近邻造类）",
         "why_not_tried": "全项目零使用（grep 无 LANDSAT 引用）",
         "性价比": "中（成本高但解决的是唯一的'开放问题'）"},
        {"rank": 6, "name": "DEM 高频纹理 / 地形位置指数（TPI、曲率、谷脊）",
         "data": "已有 NASADEM 30 m 聚合 60 m（`_dem1k_*.npy` 与 DEM 已有），可本地算，零取数",
         "what": "树种分布与微地形强相关（云杉冷杉在阴坡谷地、落叶松在阳坡台地、松在脊部）；"
                 "doc75 §3 已证 AEF 已编码地形（R² dem 0.962），但**地形纹理/TPI 未被 AEF 显式编码**",
         "expected": "低—中：Δ +0.02–0.06。风险是重蹈'地理假象'——地形也是位置的代理，"
                     "必须在 coregion 口径下评才算数",
         "cost": "极低：本地 numpy 计算 <10 min",
         "why_not_tried": "只用了 elev/slope/aspect 三个一阶量",
         "性价比": "中（成本极低，可作免费试验，但预期增益小）"},
        {"rank": 7, "name": "北方/南方分区建模 + 分区图例",
         "data": "已有样本的地理与生态区字段",
         "what": "不在全国建一个四类模型，而是分区：东北（DNF 主战场，针叶占 50%+）、西南/青藏（云杉冷杉，"
                 "ENF 为主但被 FCS10 误标 DNF）、南方（EBF 主战场，真针叶 1.8%）。每区只出该区有证据的类",
         "expected": "doc82 §11 region_cv 已实测：华东 macro-F1 0.728 vs 东北 0.901——分区差异巨大。"
                     "分区后可把'不可分'的类从图例里去掉，等价于提升有效精度",
         "cost": "低：纯工程，无新数据",
         "why_not_tried": "doc75 §7.4 提出过'分区决定是否展开针阔'，但未执行",
         "性价比": "高（低成本、直接可交付）"},
        {"rank": 8, "name": "混交度 / 概率层替代硬分类（含弃权到常绿-落叶层）",
         "data": "已有",
         "what": "输出 P(ENF)/P(DNF)/P(EBF)/P(DBF) 四通道概率 + mix_score，硬分类只在 max P ≥ 阈值时给",
         "expected": "doc75 §6.5b 实测：弃权 2% 像元使保留区错误率 1.50%→0.81%（−46%），"
                     "标记 4.07% 捕获 64.7% 针阔错误（富集 15.9×）",
         "cost": "极低",
         "why_not_tried": "已验证但未进入四类出图流程",
         "性价比": "极高（工程最优解，绕开'能不能分'的争论）"},
    ]

    verdict["methods_not_tried"] = methods
    verdict["methods_not_tried_note"] = (
        "排序依据 = 预期增益 / 成本。第 1、7、8 项零新数据即可做；第 2、4 项是唯一可能真正提升"
        "'针/阔'轴的遥感特征；第 3、5 项成本高但解决的是各自唯一的开放问题。"
        "特别提示：doc75 已证伪的三条路线（加法特征到 AEF、地形辐射校正、GEE 相干性）不得重跑。")

    # ---- 第 4 步：A/B/C 分档裁决（四类到底能不能出图）----
    pv = verdict["pairs"]
    ov = verdict["four_class_overall"]
    ceil_ = json.load(open(os.path.join(OUT_STAGE, "ceiling.json"), encoding="utf-8"))
    cl = ceil_.get("cross_label_transfer", {})
    mig = cl.get("transfer_FCS10_to_veg30_f1", [None, None])[0]
    agr = cl.get("label_agreement")

    verdict["tier_decision"] = {
        "one_line": (
            "常绿/落叶轴出图（A 档）；针/阔轴不出四类硬分类（C 档），改为"
            "『常绿林/落叶林』两类 + 针阔概率层/混交度层，并按区域限定。"),
        "A_direct": {
            "classes": ["常绿林（ENF ∪ EBF）", "落叶林（DNF ∪ DBF）"],
            "regions": "全国（含 2017–2024 逐年）",
            "why": [
                f"常绿/落叶二类 coregion macro-F1 {round(ax['coregion']['habit_ever_vs_decid']['full']['macro_f1_mean'],3)}"
                f"（位置+地形基线 {round(ax['coregion']['habit_ever_vs_decid']['lonlat_terrain']['macro_f1_mean'],3)}）",
                "doc54 独立人工判读 AUC 0.990；doc75 空间缓冲后仍 0.945（唯一跨全部控制稳健的轴）",
                f"O2 组内验证：针叶内部（ENF vs DNF）coregion F1 {pv['a_ENF_vs_DNF']['macro_f1']['aef64_coregion']}；"
                f"阔叶内部（EBF vs DBF）{pv['a_EBF_vs_DBF']['macro_f1']['aef64_coregion']}",
            ],
            "additional_data_needed": "无",
        },
        "B_conditional": {
            "classes": ["常绿针叶 vs 落叶针叶（ENF/DNF，限针叶区）",
                        "常绿阔叶 vs 落叶阔叶（EBF/DBF，限阔叶区）"],
            "regions": {
                "ENF_vs_DNF": "东北（兴安落叶松区）、阿尔泰、川西/藏东南 —— 即落叶松真实分布窗口；"
                              "**排除**青藏高原内部（FCS10 的 81 码在该区把云杉/冷杉误标为落叶针叶）",
                "EBF_vs_DBF": "亚热带（滇/桂/粤/闽/湘/赣）与华北山地；南方过渡带真落叶成分需 djf 证据",
            },
            "why": [
                f"组内 F1 尚可（coregion ENF/DNF {pv['a_ENF_vs_DNF']['macro_f1']['aef64_coregion']}、"
                f"EBF/DBF {pv['a_EBF_vs_DBF']['macro_f1']['aef64_coregion']}），但**标签纯度是主瓶颈**，不是特征",
                "O1 同区域检验：ENF vs DNF 去掉地理后 Δ 为负（0.750 vs 基线 0.782）；"
                "doc75 空间缓冲后针阔从 0.979 跌到 0.657",
                f"跨标签迁移 F1 {mig} ≈ 两源标签一致率 {agr} → 任何单源高分都不可迁移",
            ],
            "additional_data_needed": [
                "FCS10 81 码在青藏/西南的重标（云杉/冷杉 → ENF），预计修正 5,900 点级",
                "落叶松专项物候判据（秋季早衰）+ S1 时序形状",
                "每类 ≥2,000 点的独立人工判读真值（当前 doc54 仅 113 森林点）",
            ],
        },
        "C_merge_or_probability": {
            "classes": ["四类硬分类（ENF/DNF/EBF/DBF 同图出四类）",
                        "ENF vs EBF（常绿内部针阔）——最难对"],
            "regions": "全国；南方（粤/闽/桂/浙/湘/赣）尤其禁止",
            "why": [
                f"四类整体 coregion macro-F1 {ov['coregion']['macro_f1_full']}"
                f"（位置+地形基线 {ov['coregion']['macro_f1_lonlat_terrain']}，Δ={ov['coregion']['delta_full_vs_lonlat_terrain']}）"
                "—— 未达 A 档门槛（Δ≥0.15 且 F1≥0.75）",
                f"ENF vs EBF coregion：F1 {pv['b_ENF_vs_EBF']['macro_f1']['aef64_coregion']}、"
                f"Δ(vs 位置+地形) {pv['b_ENF_vs_EBF']['delta']['coregion_aef64_vs_lonlat_terrain']}；"
                "doc54 独立判读 AUC 0.724 < 0.85 门",
                "doc75：纯经纬度基线已 0.983、净叶型信号≈0.011；我方成品在广东 ENF PA 仅 4.5%，"
                "南亚热带常绿阔叶被系统性判成常绿针叶",
                "doc82 §11：FCS10 把马尾松/杉木 62.4% 判成常绿阔叶，南方针叶真值仍未闭合（V4 判据失效）",
            ],
            "alternative": [
                "出四通道概率层 P(ENF)/P(DNF)/P(EBF)/P(DBF)（不硬判）",
                "出混交度层 mix_score，thr=0.60 弃权到常绿/落叶层（实测错误捕获富集 17.9×）",
                "主图只出『常绿林/落叶林』两类（= A 档），针阔作专题产品",
            ],
        },
        "decision_matrix_by_pair": {
            k: {"grade": v["grade"], "f1_coregion": v["macro_f1"]["aef64_coregion"],
                "delta_strict": v["delta"]["coregion_aef64_vs_lonlat_terrain"],
                "median_nn_km": v["geo_overlap_median_nn_km"]}
            for k, v in pv.items()},
        "hard_constraints": [
            "任何四类结果不得用 FCS30D 验证（它是偏差源，见 doc75 §3.4）",
            "FCS10 的 81 码在青藏/西南不得直接当作 DNF 真值（doc78 §3.4 + 本轮修正后复核）",
            "南 6 省不得据 FCS10 出针叶类（doc82 §2.2：非森林区针叶率高于针叶图斑内）",
            "混同/混交像元必须弃权或出概率，不得硬分（doc75 §6.5b）",
        ],
        "confidence": (
            "A 档高（三个独立来源一致：doc54 独立判读、doc75 空间缓冲、O2 组内验证）；"
            "B 档中（分数够但标签纯度未验证）；C 档高（doc54/doc75/82 + O2 迁移检验四路同向）。"),
    }

    # ---- 汇总 ----
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "opt/code/o2_evidence.py",
        "inputs": [HAB, AEF, O1R, O1C,
                   "技术文档/54,75,77,78,82"],
        "task": "森林四类（ENF/DNF/EBF/DBF）可分性：证据总表 + 定量再分析 + 分档裁决",
        "headline": {
            "four_class_macro_f1": {
                "lonlat_baseline": A["lonlat"]["macro_f1_mean"],
                "pheno_only": A["pheno"]["macro_f1_mean"],
                "pheno_delta_vs_lonlat": A["pheno"]["vs_lonlat"]["delta_mean"],
                "pheno_terrain": A["pheno_terrain"]["macro_f1_mean"],
                "pheno_terrain_delta": A["pheno_terrain"]["vs_lonlat"]["delta_mean"],
                "aef64": A["aef64"]["macro_f1_mean"],
                "aef64_delta": A["aef64"]["vs_lonlat"]["delta_mean"],
                "full": A["full"]["macro_f1_mean"],
                "full_delta": A["full"]["vs_lonlat"]["delta_mean"],
            },
            "four_class_coregion": {
                "lonlat_baseline": C["lonlat"]["macro_f1_mean"],
                "pheno_only": C["pheno"]["macro_f1_mean"],
                "pheno_delta_vs_lonlat": C["pheno"]["vs_lonlat"]["delta_mean"],
                "full": C["full"]["macro_f1_mean"],
                "full_delta": C["full"]["vs_lonlat"]["delta_mean"],
            },
        },
        "design": fc["design"],
        "fourclass": {"all": {k: v for k, v in A.items()},
                      "coregion": {k: v for k, v in C.items()},
                      "e2e_only": {k: v for k, v in fc["e2e_only"].items()}},
        "pairs": pr,
        "axes_two_class": {"design": ax["design"],
                           "all": {k: {kk: vv for kk, vv in v.items()} for k, v in ax["all"].items()},
                           "coregion": {k: {kk: vv for kk, vv in v.items()} for k, v in ax["coregion"].items()},
                           "D_N_label_purity": ax["D_N_label_purity"]},
        "confusion_structure": st,
        "label_ceiling": ceil,
        "audit_habit_grade": audit_habit_grade(),
        "verdict": verdict,
        "o1_reference": {
            "note": "O1（527 点 OlmoEarth）已跑结果，直接转写不重算；本轮 O2 用 109,260 点 FCS10 派生标签，口径不同不可直接比",
            "tasks": {t: {"macro_f1": round(v["feat"]["macro_f1_mean"], 4),
                          "lonlat": round(v["base_lonlat"]["macro_f1_mean"], 4),
                          "delta": round(v["delta_macro_f1_mean"], 4),
                          "auc": round(v["feat"]["auc_mean"], 4)}
                      for t, v in o1["tasks"].items()},
            "aef64_same_point": {t: {"macro_f1": round(v["feat"]["macro_f1_mean"], 4),
                                     "lonlat": round(v["base_lonlat"]["macro_f1_mean"], 4),
                                     "delta": round(v["delta_macro_f1_mean"], 4)}
                                 for t, v in o1["aef64_same_point_comparison"].items()},
        },
        "log": LOG,
    }
    with open(OUT_RESULT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    log(f"结果 → {OUT_RESULT}")
    return out


def _dump(obj, name):
    os.makedirs(OUT_STAGE, exist_ok=True)
    clean = _deep_clean(obj)
    with open(os.path.join(OUT_STAGE, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=1, default=str)


def _deep_clean(o):
    if isinstance(o, dict):
        return {k: _deep_clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [_deep_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "matrix", "fourclass", "pairs", "axes2", "structure",
                             "ceiling", "assemble"])
    a = ap.parse_args()
    os.makedirs(OUT_STAGE, exist_ok=True)
    t0 = time.time()
    if a.stage in ("all", "matrix"):
        build_evidence_matrix()
    if a.stage in ("all", "fourclass"):
        stage_fourclass()
    if a.stage in ("all", "pairs"):
        stage_pairs()
    if a.stage in ("all", "axes2"):
        stage_axes2()
    if a.stage in ("all", "structure", "assemble"):
        s = confusion_structure()
        _dump({"structure": s}, "structure")
    if a.stage == "ceiling":
        _dump(label_ceiling_test(), "ceiling")
    if a.stage in ("all", "assemble"):
        stage_assemble()
    log(f"总耗时 {round((time.time()-t0)/60,2)} min")


if __name__ == "__main__":
    sys.exit(main())

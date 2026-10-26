# -*- coding: utf-8 -*-
"""prereg.py — 阶段A 机器可读预注册配置的加载 / 校验 / 摘要（control 专属）

* 输入：control/prereg_stageA.json（预注册配置本体）
* 规则源：doc69 §8.2 阶段A（本地单因素、三 seed 配对、≤20k/100树/minLeaf2/
  maxNodes5000）+ §8.3 预注册验收门（组级非劣 CI 下界 −2pp 且点估计 ≥ −1pp；
  目标类 ΔF1 ≥ +0.03 且 CI 支持改善方向；有效参考 n<30 只作诊断）
* 门槛：本模块只做「配置自洽性」校验，不跑模型、不读样本
* 输出：校验结果 dict；CLI 打印摘要
* 用法：
    python 代码/11.改进实施与小区域实验/control/prereg.py --check
    python 代码/11.改进实施与小区域实验/control/prereg.py --print
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREREG = os.path.join(HERE, "prereg_stageA.json")

# 冻结的硬约束（改这里必须同步主 agent 的实施文档 70 §10）
FROZEN = {
    "year": 2023,
    "seeds": [7, 11, 23],
    "n_train_limit": 20000,
    "max_points_per_window": 20000,
    "n_trees": 100,
    "min_leaf": 2,
    "max_nodes": 5000,
    "limit_s": 1200,
    "n_jobs": 10,
    "cpu_cap": 15,
    "mem_cap_gb": 10,
    "orchard_class": 2,
    "orchard_fraction": 0.01,
    "orchard_max_points": 200,
    "group_noninf_point_estimate_pp": -1.0,
    "group_noninf_ci_lower_pp": -2.0,
    "target_class_delta_f1": 0.03,
    "class_level_min_n": 30,
    "new_regression_delta_f1": -0.05,
}


def load_prereg(path: str = DEFAULT_PREREG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    errs: List[str] = []
    warns: List[str] = []

    # --- 年份 / 类别空间 ---
    if cfg.get("year") != FROZEN["year"]:
        errs.append(f"year={cfg.get('year')} != 冻结 {FROZEN['year']}")
    cs = cfg.get("class_space", {})
    if cs.get("missing_codes") != [8]:
        errs.append("class_space.missing_codes 必须 = [8]（第二轮池缺类8）")

    # --- 规模/容量硬约束 ---
    model = cfg.get("model", {})
    for k in ("n_trees", "min_leaf", "max_nodes"):
        if model.get(k) != FROZEN[k]:
            errs.append(f"model.{k}={model.get(k)} != 冻结值 {FROZEN[k]}")
    if cfg.get("seeds") != FROZEN["seeds"]:
        errs.append(f"seeds={cfg.get('seeds')} != 冻结 {FROZEN['seeds']}（须三 seed 配对）")
    if cfg.get("n_train_limit") != FROZEN["n_train_limit"]:
        errs.append("n_train_limit 必须 = 20000")
    samp = cfg.get("sampling", {})
    if samp.get("max_points_per_window") != FROZEN["max_points_per_window"]:
        errs.append("sampling.max_points_per_window 必须 = 20000")
    if samp.get("window_expand_deg") != 2.0:
        errs.append("sampling.window_expand_deg 必须 = 2.0（沿用生产借样框）")

    # --- O-CAP 园地上限 ---
    ocap = cfg.get("candidate_rules", {}).get("O-CAP", {})
    if ocap.get("class_code") != FROZEN["orchard_class"]:
        errs.append("O-CAP.class_code 必须 = 2（乔灌园地）")
    if ocap.get("fraction") != FROZEN["orchard_fraction"]:
        errs.append("O-CAP.fraction 必须 = 0.01（1%）")
    if ocap.get("max_points") != FROZEN["orchard_max_points"]:
        errs.append("O-CAP.max_points 必须 = 200")

    # --- 时间/资源 ---
    if cfg.get("limit_s") != FROZEN["limit_s"]:
        errs.append("limit_s 必须 = 1200")
    if cfg.get("n_jobs") != FROZEN["n_jobs"]:
        errs.append("n_jobs 必须 = 10（CPU≤15 保守）")
    if cfg.get("cpu_cap") != FROZEN["cpu_cap"]:
        errs.append("cpu_cap 必须 = 15")
    if cfg.get("mem_cap_gb") != FROZEN["mem_cap_gb"]:
        errs.append("mem_cap_gb 必须 = 10")

    # --- 判据 ---
    rules = cfg.get("decision_rules", {})
    ni = rules.get("group_non_inferiority", {})
    if ni.get("metric") != "macro9":
        errs.append("组级非劣 metric 必须 = macro9")
    if ni.get("point_estimate_pp") != FROZEN["group_noninf_point_estimate_pp"]:
        errs.append("组级非劣点估计阈值必须 = -1.0 pp")
    if ni.get("ci_lower_pp") != FROZEN["group_noninf_ci_lower_pp"]:
        errs.append("组级非劣 CI 下界必须 = -2.0 pp")
    tc = rules.get("target_class_improvement", {})
    if tc.get("delta_f1") != FROZEN["target_class_delta_f1"]:
        errs.append("目标类 ΔF1 阈值必须 = +0.03")
    if not tc.get("require_ci_supports_direction", False):
        errs.append("目标类改善必须要求 CI 支持改善方向")
    if rules.get("class_level_min_n") != FROZEN["class_level_min_n"]:
        errs.append("类级最小 n 必须 = 30")
    if rules.get("new_regression", {}).get("delta_f1") != FROZEN["new_regression_delta_f1"]:
        errs.append("新退化 ΔF1 阈值必须 = -0.05")

    # --- 设计纪律 ---
    if cfg.get("design") != "local_single_factor":
        errs.append("阶段A design 必须 = local_single_factor（单因素，多因素不得一次全开）")
    if not cfg.get("dev_not_independent", False):
        warns.append("应显式声明 dev_not_independent=true（开发集非独立真值）")
    if not cfg.get("holdout_frozen", False):
        errs.append("必须声明 holdout_frozen=true（评估点/留出块冻结）")
    if not cfg.get("confirmation_single_use", False):
        warns.append("应声明 confirmation_single_use=true（确认集只用一次）")
    hs = cfg.get("holdout", {}).get("split", {})
    if "60" not in str(hs.get("target", "")):
        warns.append("holdout.split.target 应约 60%开发/40%确认")

    return {"ok": not errs, "errors": errs, "warnings": warns,
            "n_errors": len(errs), "n_warnings": len(warns)}


def summarize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": cfg.get("stage"),
        "spec_doc": cfg.get("spec_doc"),
        "design": cfg.get("design"),
        "year": cfg.get("year"),
        "class_space": cfg.get("class_space"),
        "provinces": cfg.get("provinces"),
        "regions_planned": cfg.get("regions_planned"),
        "seeds": cfg.get("seeds"),
        "limit_s": cfg.get("limit_s"),
        "n_jobs": cfg.get("n_jobs"),
        "cpu_cap": cfg.get("cpu_cap"),
        "mem_cap_gb": cfg.get("mem_cap_gb"),
        "n_train_limit": cfg.get("n_train_limit"),
        "model": cfg.get("model"),
        "sampling": cfg.get("sampling"),
        "candidate_rules": cfg.get("candidate_rules"),
        "decision_rules": cfg.get("decision_rules"),
        "acceptance_grades": cfg.get("acceptance_grades"),
        "holdout": cfg.get("holdout"),
        "dev_not_independent": cfg.get("dev_not_independent"),
        "holdout_frozen": cfg.get("holdout_frozen"),
        "confirmation_single_use": cfg.get("confirmation_single_use"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段A 预注册配置校验/摘要")
    ap.add_argument("--prereg", default=DEFAULT_PREREG)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print", dest="do_print", action="store_true")
    args = ap.parse_args()
    cfg = load_prereg(args.prereg)
    if args.do_print:
        print(json.dumps(summarize(cfg), ensure_ascii=False, indent=2))
    res = validate(cfg)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

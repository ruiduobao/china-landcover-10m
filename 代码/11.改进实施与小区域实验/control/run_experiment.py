# -*- coding: utf-8 -*-
"""run_experiment.py — 20 分钟硬截止实验 runner（control 专属）

* 输入：一份实验 spec JSON（见 control/spec_template.json）
* 规则源：doc69 §8.2/§8.3 阶段A；主 agent 冻结判据
* 门槛：
    - 单实验「train 调用开始 → eval 完成」窗口 ≤ limit_s（默认 1200 s）
    - prepare 独立计时、独立上限 PREP_CAP（默认 1200 s），不并入窗口；
      单次调用只跑 1 个实验，禁止把一次实验拆成多次 invoke 规避超时
    - 超时只杀本实验派生的子进程树（deadline.kill_process_tree）
    - 全局单实例锁（control/locks/runner.lock）：同一时刻只允许 1 个 runner
      —— 与「计算重活串行、RF 单代理」的约定一致
    - 结果 JSON 原子写
* 输出：control/run_results/<exp_id>.json（含逐阶段时长与 deadline_window_s）
        追加/更新共享注册表 registry/experiments.json
* 用法：
    python 代码/11.改进实施与小区域实验/control/run_experiment.py \
        --spec F:/lc_work/improve_2023_pilot_20261015/control/specs/A1.json \
        --workspace F:/lc_work/improve_2023_pilot_20261015
    # 干跑（只校验 spec 与预算，不启动任何子进程）：
    python .../run_experiment.py --spec ... --workspace ... --dry-run

时间证据可独立审核：结果文件里 `stage_times_s.{prepare,train,infer,eval}` 为各阶段
实测秒数；`deadline_window_s` = eval.end - train.start，`limit_s` 为其上限；
`budget_ok` 为布尔判定。审核者无需信任本 runner 的内部状态，直接比较两者即可。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deadline import (  # noqa: E402
    DEFAULT_CPU_CAP,
    DEFAULT_LIMIT_S,
    DEFAULT_MEM_CAP_GB,
    DEFAULT_PREP_CAP_S,
    DEFAULT_TRIGGER_MARGIN_S,
    DeadlineBudget,
    FileLock,
    atomic_write_json,
    preflight_memory,
    run_stage,
    system_mem_gb,
)

STAGE_ORDER = ["prepare", "train", "infer", "eval"]
WINDOW_STAGES = ["train", "infer", "eval"]  # 计入 1200s 窗口的阶段
DEFAULT_SLOT_WAIT_S = 1800  # 等待 RF slot 的最长时间

# doc70 §4.1 要求的注册字段（由实验代理在 spec 中提供，runner 原样落盘留痕）
REG_FIELDS = [
    "experiment_id", "region_bbox", "tile_id", "training_bbox", "input_identity",
    "class_map_hash", "candidate_rule", "seeds", "n_train_limit", "rf_params",
    "deadline_s", "development_ids", "confirmation_ids", "target_metrics",
    "expected_effect", "stop_rules", "code_hash", "registered_at",
]


def _code_hash(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return None


def _acquire_with_wait(lock: FileLock, wait_s: float, poll_s: float = 10.0) -> bool:
    """带等待地获取锁（用于 RF slot）。只等待，不杀任何进程。"""
    t0 = time.monotonic()
    while True:
        if lock.acquire():
            return True
        if time.monotonic() - t0 >= wait_s:
            return False
        time.sleep(poll_s)


# --------------------------------------------------------------------------- #
# spec 加载与校验
# --------------------------------------------------------------------------- #
def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return spec


def validate_spec(spec: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if not spec.get("exp_id"):
        errs.append("缺少 exp_id")
    stages = spec.get("stages") or {}
    if not isinstance(stages, dict):
        errs.append("stages 必须是对象")
        return errs
    for s in STAGE_ORDER:
        st = stages.get(s)
        if st is None:
            errs.append(f"缺少阶段 {s}")
            continue
        if not isinstance(st, dict):
            errs.append(f"阶段 {s} 必须是对象")
            continue
        cmd = st.get("cmd")
        if not isinstance(cmd, list) or not cmd:
            errs.append(f"阶段 {s}.cmd 必须是非空数组（命令+参数）")
        if s == "prepare" and st.get("enabled", True) is False:
            errs.append("prepare 不允许 disabled：数据准备必须显式计时上报")
    limit = spec.get("limit_s", DEFAULT_LIMIT_S)
    if not isinstance(limit, (int, float)) or limit <= 0:
        errs.append("limit_s 必须为正数")
    if limit > DEFAULT_LIMIT_S:
        errs.append(f"limit_s={limit} 超过硬上限 {DEFAULT_LIMIT_S}s（阶段A 判据）")
    if spec.get("seeds") is not None:
        seeds = spec["seeds"]
        if not isinstance(seeds, list) or len(seeds) < 1:
            errs.append("seeds 必须是非空数组")
        if len(set(seeds)) != len(seeds):
            errs.append("seeds 存在重复（须为配对种子）")
    return errs


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(spec: Dict[str, Any], workspace: str, dry_run: bool = False) -> Dict[str, Any]:
    exp_id = spec["exp_id"]
    limit_s = float(spec.get("limit_s", DEFAULT_LIMIT_S))
    prep_cap_s = float(spec.get("prep_cap_s", DEFAULT_PREP_CAP_S))
    stages = spec["stages"]

    ctrl = os.path.join(workspace, "control")
    logs = os.path.join(ctrl, "logs", exp_id)
    results_dir = os.path.join(ctrl, "run_results")
    reg_path = os.path.join(workspace, "registry", "experiments.json")
    lock_path = os.path.join(ctrl, "locks", "runner.lock")
    rf_lock_path = os.path.join(ctrl, "locks", "rf.lock")

    mem_cap_gb = float(spec.get("mem_cap_gb", DEFAULT_MEM_CAP_GB))
    uses_rf = bool(spec.get("uses_rf", True))
    slot_wait_s = float(spec.get("slot_wait_s", DEFAULT_SLOT_WAIT_S))

    record: Dict[str, Any] = {
        "exp_id": exp_id,
        "owner": spec.get("owner", "control"),
        "hypothesis": spec.get("hypothesis"),
        "province": spec.get("province"),
        "stage": spec.get("stage", "A"),
        "seeds": spec.get("seeds"),
        "arms": spec.get("arms"),
        "limit_s": limit_s,
        "prep_cap_s": prep_cap_s,
        "trigger_margin_s": DEFAULT_TRIGGER_MARGIN_S,
        "n_jobs": spec.get("n_jobs"),
        "uses_rf": uses_rf,
        "mem_cap_gb": mem_cap_gb,
        "cpu_cap": spec.get("cpu_cap", DEFAULT_CPU_CAP),
        "dry_run": dry_run,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system_mem_at_start_gb": system_mem_gb(),
        "stage_times_s": {},
        "stage_rc": {},
        "stage_timed_out": {},
        "stage_mem_killed": {},
        "stage_peak_rss_gb": {},
        "killed_pids": {},
        "deadline_window_s": None,
        "budget_ok": None,
        "status": "running",
    }
    # doc70 §4.1 注册字段留痕
    record["registration"] = {k: spec.get(k) for k in REG_FIELDS}
    record["code_hash"] = spec.get("code_hash") or _code_hash(os.path.abspath(__file__))

    budget = DeadlineBudget(limit_s=limit_s, prep_cap_s=prep_cap_s)
    stage_results: Dict[str, Any] = {}

    if dry_run:
        # 只校验与预估，不启动任何子进程、不写锁、不写共享注册表
        record["status"] = "dry_run"
        record["checks"] = "spec 通过校验；未启动子进程"
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(os.path.join(results_dir, f"{exp_id}.json"), record)
        return record

    with FileLock(lock_path) as lk:
        rf_lock: Optional[FileLock] = None
        try:
            # ---- prepare（独立计时、独立上限）----
            prep = stages["prepare"]
            prep_to = float(prep.get("timeout_s", prep_cap_s))
            prep_to = min(prep_to, prep_cap_s)
            r = run_stage(
                "prepare", prep["cmd"], timeout_s=prep_to,
                cwd=prep.get("cwd"), env=_stage_env(prep),
                log_path=os.path.join(logs, "prepare.log"),
                mem_cap_gb=mem_cap_gb,
            )
            stage_results["prepare"] = r
            _record_stage(record, "prepare", r)
            if r.timed_out or r.mem_killed or r.rc != 0:
                record["status"] = "failed_prepare"
                record["fail_reason"] = (
                    "prepare 超时（>prep_cap_s）" if r.timed_out
                    else "prepare 超内存上限" if r.mem_killed
                    else f"prepare rc={r.rc}"
                )
                return _finalize(record, stage_results, results_dir, reg_path)

            # ---- 内存预检（启动重活前）----
            pf = preflight_memory(mem_cap_gb=mem_cap_gb)
            record["mem_preflight"] = pf
            if not pf["ok"]:
                record["status"] = "blocked_mem"
                record["fail_reason"] = pf["reason"]
                return _finalize(record, stage_results, results_dir, reg_path)

            # ---- RF slot（重活串行；非 RF 实验可跳过）----
            if uses_rf:
                rf_lock = FileLock(rf_lock_path)
                if not _acquire_with_wait(rf_lock, slot_wait_s):
                    record["status"] = "blocked_slot"
                    record["fail_reason"] = (
                        f"等待 RF slot 超时（{slot_wait_s}s），holder_pid="
                        f"{rf_lock.holder_pid()}；不杀任何进程，如实记录"
                    )
                    return _finalize(record, stage_results, results_dir, reg_path)

            # ---- 窗口开始：训练调用开始 ----
            budget.start_window()
            record["train_start_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")

            for s in WINDOW_STAGES:
                st = stages[s]
                remaining = budget.remaining()
                # 硬截止：在 limit_s 前预留 TRIGGER_MARGIN，确保终止+日志尽量 ≤ limit_s
                to = float(st.get("timeout_s", limit_s))
                to = min(to, remaining - DEFAULT_TRIGGER_MARGIN_S)
                r = run_stage(
                    s, st["cmd"], timeout_s=to,
                    cwd=st.get("cwd"), env=_stage_env(st),
                    log_path=os.path.join(logs, f"{s}.log"),
                    mem_cap_gb=mem_cap_gb,
                )
                stage_results[s] = r
                _record_stage(record, s, r)
                if r.mem_killed:
                    record["status"] = "failed"
                    record["fail_reason"] = f"阶段 {s} 超内存上限 {mem_cap_gb}GB（只杀本树）"
                    budget.mark_eval_end()
                    return _finalize(record, stage_results, results_dir, reg_path)
                if r.timed_out:
                    record["status"] = "timeout"
                    record["fail_reason"] = (
                        f"阶段 {s} 触发截止（窗口剩余 {remaining:.1f}s，预留 "
                        f"{DEFAULT_TRIGGER_MARGIN_S:.0f}s 清理）"
                    )
                    budget.mark_eval_end()
                    return _finalize(record, stage_results, results_dir, reg_path)
                if r.rc != 0:
                    record["status"] = "failed"
                    record["fail_reason"] = f"阶段 {s} rc={r.rc}"
                    budget.mark_eval_end()
                    return _finalize(record, stage_results, results_dir, reg_path)

            budget.mark_eval_end()
            record["status"] = "done"
            return _finalize(record, stage_results, results_dir, reg_path)
        finally:
            if rf_lock is not None:
                rf_lock.release()


def _record_stage(record: Dict[str, Any], name: str, r: Any) -> None:
    record["stage_times_s"][name] = round(r.duration_s, 3)
    record["stage_rc"][name] = r.rc
    record["stage_timed_out"][name] = r.timed_out
    record["stage_mem_killed"][name] = r.mem_killed
    record["stage_peak_rss_gb"][name] = r.peak_rss_gb
    if r.killed_pids:
        record["killed_pids"][name] = r.killed_pids


def _stage_env(stage: Dict[str, Any]) -> Optional[Dict[str, str]]:
    env = None
    if stage.get("env"):
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in stage["env"].items()})
    return env


def _finalize(record, stage_results, results_dir, reg_path) -> Dict[str, Any]:
    window = None
    train = stage_results.get("train")
    evalr = stage_results.get("eval")
    if train is not None and evalr is not None and evalr.end_mono > 0:
        # 窗口 = eval 结束 - train 开始（仅当两阶段都真正启动过）
        window = evalr.end_mono - train.start_mono
    record["deadline_window_s"] = round(window, 3) if window is not None else None
    if window is None:
        record["budget_ok"] = None
    else:
        record["budget_ok"] = bool(window <= record["limit_s"] + 1e-6)
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    # 阶段日志摘要（只留尾部，避免结果文件膨胀）
    record["stage_tails"] = {
        k: {"rc": v.rc, "timed_out": v.timed_out, "mem_killed": v.mem_killed,
            "peak_rss_gb": v.peak_rss_gb, "log": v.log_path,
            "stderr_tail": v.stderr_tail[-600:]}
        for k, v in stage_results.items()
    }
    atomic_write_json(os.path.join(results_dir, f"{record['exp_id']}.json"), record)
    _update_registry(reg_path, record)
    return record


def _update_registry(reg_path: str, record: Dict[str, Any]) -> None:
    """把本次运行摘要并入共享注册表（按 exp_id 覆盖；原子写）。"""
    os.makedirs(os.path.dirname(reg_path), exist_ok=True)
    if os.path.exists(reg_path):
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
        except Exception:
            reg = {"schema": "improve_pilot_registry/v1", "experiments": {}}
    else:
        reg = {"schema": "improve_pilot_registry/v1", "experiments": {}}
    reg.setdefault("experiments", {})
    reg["experiments"][record["exp_id"]] = {
        "owner": record.get("owner"),
        "hypothesis": record.get("hypothesis"),
        "province": record.get("province"),
        "stage": record.get("stage"),
        "status": record.get("status"),
        "seeds": record.get("seeds"),
        "limit_s": record.get("limit_s"),
        "uses_rf": record.get("uses_rf"),
        "mem_cap_gb": record.get("mem_cap_gb"),
        "stage_times_s": record.get("stage_times_s"),
        "stage_peak_rss_gb": record.get("stage_peak_rss_gb"),
        "deadline_window_s": record.get("deadline_window_s"),
        "budget_ok": record.get("budget_ok"),
        "registration": record.get("registration"),
        "code_hash": record.get("code_hash"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
    }
    reg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(reg_path, reg)


def main() -> int:
    ap = argparse.ArgumentParser(description="20 分钟硬截止实验 runner")
    ap.add_argument("--spec", required=True, help="实验 spec JSON 路径")
    ap.add_argument("--workspace", required=True, help="工作区根目录")
    ap.add_argument("--dry-run", action="store_true", help="只校验，不启动子进程")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    errs = validate_spec(spec)
    if errs:
        print("[FAIL] spec 校验未通过：")
        for e in errs:
            print("   -", e)
        return 2
    rec = run(spec, args.workspace, dry_run=args.dry_run)
    print(json.dumps({
        "exp_id": rec["exp_id"],
        "status": rec["status"],
        "stage_times_s": rec.get("stage_times_s"),
        "deadline_window_s": rec.get("deadline_window_s"),
        "budget_ok": rec.get("budget_ok"),
    }, ensure_ascii=False, indent=2))
    ok = rec["status"] in ("done", "dry_run")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

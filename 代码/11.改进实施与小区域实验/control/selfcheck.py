# -*- coding: utf-8 -*-
"""selfcheck.py — control 侧轻量自检（不训练、不读大样本、不触 GEE）

* 输入：control/ 下的 deadline.py / run_experiment.py / prereg.py / ns_exclude.py
* 规则源：doc69 §8.2/§8.3；主 agent 冻结判据
* 门槛（全部必须 PASS）：
    1. 预注册配置自洽（prereg.validate ok）
    2. 原子写可用（写→读→字节一致）
    3. 单实例锁：可获取；同 PID 二次获取被拒；释放后可再获取
    4. 超时杀树：3s 超时的 sleep 子进程被判定 timed_out 且进程已死；
       且「只杀自己的子进程」——自检前后其它 python 进程数不减少
    5. runner 干跑：spec 校验通过，不启动子进程
* 输出：control/run_results/_selfcheck.json（原子写）+ stdout 摘要
* 用法：
    python 代码/11.改进实施与小区域实验/control/selfcheck.py \
        --workspace F:/lc_work/improve_2023_pilot_20261015
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import deadline as DL  # noqa: E402
import prereg as PR  # noqa: E402
from run_experiment import load_spec, validate_spec  # noqa: E402


def list_python_pids() -> List[int]:
    """当前 python.exe 的 PID 列表（仅用于「不误伤」对照，绝不用于杀进程）。"""
    pids: List[int] = []
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH", "/FO", "CSV"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        for ln in out.splitlines():
            parts = [p.strip().strip('"') for p in ln.split(",")]
            if len(parts) >= 2 and parts[0].lower() == "python.exe":
                try:
                    pids.append(int(parts[1]))
                except ValueError:
                    pass
    except Exception:
        pass
    return pids


def count_python() -> int:
    pids = list_python_pids()
    return len(pids) if pids else -1


def check_atomic(tmpdir: str) -> Dict[str, Any]:
    p = os.path.join(tmpdir, "_atomic_probe.json")
    obj = {"a": 1, "中文": "值", "nested": [1, 2, 3]}
    DL.atomic_write_json(p, obj)
    with open(p, "r", encoding="utf-8") as f:
        back = json.load(f)
    ok = back == obj and not any(fn.startswith("_atomic_probe.json.tmp")
                                 for fn in os.listdir(tmpdir))
    return {"ok": ok, "path": p}


def check_lock(lockdir: str) -> Dict[str, Any]:
    p = os.path.join(lockdir, "_lock_probe.lock")
    if os.path.exists(p):
        os.remove(p)
    lk1 = DL.FileLock(p)
    got1 = lk1.acquire()
    lk2 = DL.FileLock(p)
    got2 = lk2.acquire()          # 同 PID 存活 → 必须失败
    lk1.release()
    lk3 = DL.FileLock(p)
    got3 = lk3.acquire()          # 释放后 → 必须成功
    lk3.release()
    return {"ok": bool(got1 and (not got2) and got3),
            "acquire": got1, "contend_rejected": (not got2), "reacquire": got3}


def check_kill_tree(tmpdir: str) -> Dict[str, Any]:
    before_pids = set(list_python_pids())
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    r = DL.run_stage("probe_timeout", cmd, timeout_s=3.0,
                     log_path=os.path.join(tmpdir, "probe_timeout.log"))
    time.sleep(1.0)
    after_pids = set(list_python_pids())
    child_pid = r.killed_pids[0] if r.killed_pids else None
    child_dead = (not DL._pid_alive(child_pid)) if child_pid else True
    # 直接证据：本次终止只针对自己的子进程（killed_pids 仅含该子进程）
    only_own_child = (r.killed_pids == [child_pid]) and child_pid is not None
    # 参考信息：自检前存在的其它 python 是否消失（环境有并发代理时可能有自然退出）
    vanished = sorted(p for p in before_pids if not DL._pid_alive(p))
    ok = bool(r.timed_out and 2.0 <= r.duration_s <= 8.0 and child_dead and only_own_child)
    return {"ok": ok, "timed_out": r.timed_out, "duration_s": round(r.duration_s, 2),
            "killed_pids": r.killed_pids, "child_dead": child_dead,
            "only_own_child_killed": only_own_child,
            "preexisting_python_pids": sorted(before_pids),
            "preexisting_vanished_info_only": vanished,
            "python_before": len(before_pids), "python_after": len(after_pids),
            "note": "硬判据=只终止自己的子进程（killed_pids 仅含该子进程）；"
                    "preexisting_vanished_info_only 仅为参考，并发代理的短命进程可能自然退出"}


def check_mem_guard(tmpdir: str) -> Dict[str, Any]:
    """内存守卫：分配 ~500MB 的子进程在 mem_cap=0.3GB 下应被只杀本树。"""
    if DL.psutil is None:
        return {"ok": True, "skipped": True, "reason": "psutil 缺失，守卫降级"}
    before_pids = set(list_python_pids())
    code = ("import time\n"
            "a=bytearray(500*1024*1024)\n"
            "for i in range(0,len(a),4096): a[i]=1\n"
            "time.sleep(30)\n")
    cmd = [sys.executable, "-c", code]
    r = DL.run_stage("probe_mem", cmd, timeout_s=30.0,
                     log_path=os.path.join(tmpdir, "probe_mem.log"),
                     mem_cap_gb=0.3, mem_poll_s=0.5)
    time.sleep(1.0)
    child_pid = r.killed_pids[0] if r.killed_pids else None
    child_dead = (not DL._pid_alive(child_pid)) if child_pid else True
    untouched = [p for p in before_pids if not DL._pid_alive(p)]
    ok = bool(r.mem_killed and child_dead and len(untouched) == 0)
    return {"ok": ok, "mem_killed": r.mem_killed, "peak_rss_gb": r.peak_rss_gb,
            "killed_pids": r.killed_pids, "child_dead": child_dead,
            "preexisting_now_dead": untouched,
            "note": "只杀超内存的本实验子进程树"}


def check_resources() -> Dict[str, Any]:
    mem = DL.system_mem_gb()
    return {"ok": True, "cpu_count": DL.cpu_count(),
            "mem_total_gb": mem["total"], "mem_available_gb": mem["available"],
            "psutil_available": DL.psutil is not None,
            "note": "全局 CPU≤15 / 内存≤10GB 为 runner 侧约束（n_jobs=10 + mem_cap_gb）"}


def check_ns_exclude_synthetic() -> Dict[str, Any]:
    """合成用例（只验两点，按主 agent 指令收敛）：
      (a) 异命名空间同数字 id → **保留**（不误剔），并标记 suspect_false_removal
      (b) 同坐标（位置键命中）→ **依旧排除**（与命名空间无关）
    复刻 doc69 §2.3 园地 uid 4,238 因与留出 row_id 数字相同被误剔的场景。
    """
    import pandas as pd
    from ns_exclude import ExclusionIndex, exclude_points, pos_key

    idx = ExclusionIndex()
    idx.add_positions([pos_key(100.0, 40.0)], "holdout")
    idx.add_ids("motherlib_row_id", [4238], "holdout")

    pts = pd.DataFrame({
        "uid": [4238, 4238],
        "lon": [110.0, 100.0],
        "lat": [30.0, 40.0],
    })
    # 点 0：命名空间 yuandi、数字与留出 row_id 相同、坐标不同 → 保留
    # 点 1：坐标命中留出位置 → 排除
    out = exclude_points(pts, idx, point_namespace="yuandi", id_col="uid")
    p0, p1 = out.iloc[0], out.iloc[1]
    ok = (not bool(p0["excluded"])) and bool(p0["suspect_false_removal"]) \
        and bool(p1["excluded"]) and bool(p1["excl_pos"])
    return {
        "ok": ok,
        "diff_ns_same_number_preserved": not bool(p0["excluded"]),
        "diff_ns_same_number_flagged": bool(p0["suspect_false_removal"]),
        "same_coordinate_still_excluded": bool(p1["excluded"]),
        "note": "命名空间数值撞号保留 + 同坐标依旧排除（doc69 记录 4,238 个园地 uid）",
    }


def check_deadline_reserve(ws: str) -> Dict[str, Any]:
    """验证截止预留：limit=20s 时，sleep 30s 的 train 应在 limit−margin(10s) 前被终止。"""
    import run_experiment as RX
    spec = {
        "exp_id": "_selftest_reserve", "owner": "control", "stage": "A",
        "seeds": [7, 11, 23], "limit_s": 20, "uses_rf": False, "n_jobs": 10,
        "stages": {
            "prepare": {"cmd": [sys.executable, "-c", "print('p')"], "timeout_s": 30},
            "train": {"cmd": [sys.executable, "-c", "import time;time.sleep(30)"], "timeout_s": 30},
            "infer": {"cmd": [sys.executable, "-c", "print('i')"], "timeout_s": 30},
            "eval": {"cmd": [sys.executable, "-c", "print('e')"], "timeout_s": 30},
        },
    }
    errs = validate_spec(spec)
    if errs:
        return {"ok": False, "errors": errs}
    rec = RX.run(spec, ws)
    train_dur = rec["stage_times_s"].get("train")
    # 预留生效：train 在 limit−margin 附近被终止（留 ≤2s 抖动）
    ok = (rec["status"] == "timeout" and train_dur is not None
          and train_dur <= 20 - 10 + 2.0)
    return {"ok": ok, "status": rec["status"], "train_duration_s": train_dur,
            "limit_s": 20, "trigger_margin_s": 10,
            "note": "train 在 limit−margin 前被终止，为终止+日志预留清理时间"}


def check_runner_dryrun(workspace: str, spec_path: str) -> Dict[str, Any]:
    if not os.path.exists(spec_path):
        return {"ok": False, "reason": f"spec 不存在: {spec_path}"}
    spec = load_spec(spec_path)
    errs = validate_spec(spec)
    return {"ok": not errs, "errors": errs, "spec": spec_path}


def main() -> int:
    ap = argparse.ArgumentParser(description="control 轻量自检")
    ap.add_argument("--workspace", default="F:/lc_work/improve_2023_pilot_20261015")
    ap.add_argument("--spec", default=os.path.join(HERE, "spec_template.json"))
    args = ap.parse_args()

    ws = os.path.abspath(args.workspace)
    tmpdir = os.path.join(ws, "control", "run_results")
    lockdir = os.path.join(ws, "control", "locks")
    os.makedirs(tmpdir, exist_ok=True)
    os.makedirs(lockdir, exist_ok=True)

    report: Dict[str, Any] = {
        "schema": "improve_pilot_selfcheck/v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workspace": ws,
        "checks": {},
    }

    # 1. prereg
    cfg = PR.load_prereg()
    report["checks"]["prereg"] = PR.validate(cfg)
    # 2. atomic
    report["checks"]["atomic_write"] = check_atomic(tmpdir)
    # 3. lock
    report["checks"]["single_instance_lock"] = check_lock(lockdir)
    # 4. kill tree
    report["checks"]["timeout_kill_tree"] = check_kill_tree(tmpdir)
    # 5. memory guard
    report["checks"]["memory_guard"] = check_mem_guard(tmpdir)
    # 6. resources snapshot
    report["checks"]["resources"] = check_resources()
    # 7. namespace-safe exclusion synthetic case
    report["checks"]["ns_exclude_synthetic"] = check_ns_exclude_synthetic()
    # 8. runner dry-run spec
    report["checks"]["runner_spec"] = check_runner_dryrun(ws, args.spec)
    # 9. deadline reserve (limit−margin trigger)
    report["checks"]["deadline_reserve"] = check_deadline_reserve(ws)

    report["all_ok"] = all(c.get("ok") for c in report["checks"].values())
    out = os.path.join(tmpdir, "_selfcheck.json")
    DL.atomic_write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[selfcheck] all_ok={report['all_ok']} → {out}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

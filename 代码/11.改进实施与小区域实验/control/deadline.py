# -*- coding: utf-8 -*-
"""deadline.py — 20 分钟硬截止实验的可复用时间/进程原语（control 专属）

* 输入：无（库模块，被 run_experiment.py 与各实验代理 import）
* 规则源：doc69 §8.2/§8.3 阶段A（≤20 分钟单实验）；主 agent 冻结判据
* 门槛：
    - 单实验「训练调用开始 → 评价完成」窗口 ≤ 1200 s（train→eval deadline）
    - 数据准备（prepare）独立计时上报，另设 PREP_CAP 上限，不得靠拆分 prepare 规避超时
    - 超时只杀「自己派生的子进程树」，绝不 taskkill /IM python.exe 或匹配 bash.exe
    - 单实例锁（文件锁，O_CREAT|O_EXCL + PID 活性校验）
    - 结果 JSON 原子写（tmp + os.replace）
* 输出：StageResult / ExperimentTiming 数据类；atomic_write_json；FileLock
* 用法：
    from deadline import DeadlineBudget, run_stage, atomic_write_json, FileLock
    budget = DeadlineBudget(limit_s=1200)
    budget.start_window()                 # 训练调用开始那一刻
    r = run_stage('train', cmd, timeout_s=budget.remaining())
    budget.mark_eval_end()

设计要点（供独立审核时间证据）：
- 所有计时用 time.monotonic()，不用墙钟，避免 NTP 跳变。
- 每个 StageResult 记录 start/end/duration_s/rc/timed_out/killed_pids。
- 窗口时长 = eval_end_mono - train_start_mono，由 run_experiment.py 落盘为
  `deadline_window_s`，与 limit_s 直接比较即可审核。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

IS_WINDOWS = os.name == "nt"

# 可选 psutil（用于内存守卫；缺失则守卫降级为「不可用」并显式告警）
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# Windows: 新建进程组，便于只杀自己的树（不影响其它 python/bash）
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000

# 默认上限（秒 / GB）
DEFAULT_LIMIT_S = 1200          # 单实验 train→eval 窗口
DEFAULT_PREP_CAP_S = 1200       # 数据准备独立上限（上报，不并入窗口）
DEFAULT_KILL_GRACE_S = 5.0      # 发 kill 后等待回收的宽限
DEFAULT_MEM_CAP_GB = 10.0       # 单实验内存目标上限（doc70 §3.1）
DEFAULT_CPU_CAP = 15            # 全局 CPU 上限（doc70 §3.1）
DEFAULT_MEM_MIN_FREE_GB = 12.0  # 启动重活前要求的最小系统可用内存
DEFAULT_MEM_POLL_S = 5.0        # 内存守卫轮询间隔
DEFAULT_TRIGGER_MARGIN_S = 10.0  # 在 1200s 截止前预留的终止+清理余量（→ 1190s 触发）


# --------------------------------------------------------------------------- #
# 原子写 / 锁
# --------------------------------------------------------------------------- #
def atomic_write_json(path: str, obj: Any, indent: int = 2) -> None:
    """原子写 JSON：写同目录临时文件后 os.replace（Windows 同盘原子）。"""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    """判断 PID 是否存活（跨平台，不误伤）。"""
    if pid <= 0:
        return False
    if IS_WINDOWS:
        # tasklist 精确按 PID 查询，不按镜像名
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return f'"{pid}"' in out or f",{pid}," in out
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


class FileLock:
    """单实例锁：O_CREAT|O_EXCL 创建 lock 文件，写入 PID；陈旧锁（PID 已死）可回收。

    只锁自己声明的 lock 路径，不触碰任何其它进程。
    """

    def __init__(self, path: str, stale_ok: bool = True):
        self.path = os.path.abspath(path)
        self.stale_ok = stale_ok
        self.acquired = False
        self._fh = None

    def _try_create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if self._try_create():
            self.acquired = True
            return True
        if not self.stale_ok:
            return False
        # 读取旧 PID，若已死则回收
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                old = int((f.read() or "0").strip() or "0")
        except Exception:
            old = 0
        if not _pid_alive(old):
            try:
                os.remove(self.path)
            except OSError:
                return False
            if self._try_create():
                self.acquired = True
                return True
        return False

    def holder_pid(self) -> int:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return int((f.read() or "0").strip() or "0")
        except Exception:
            return 0

    def release(self) -> None:
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"单实例锁被占用：{self.path} (pid={self.holder_pid()})")
        return self

    def __exit__(self, *exc):
        self.release()
        return False


# --------------------------------------------------------------------------- #
# 只杀自己的子进程树
# --------------------------------------------------------------------------- #
def kill_process_tree(pid: int, grace_s: float = DEFAULT_KILL_GRACE_S) -> List[int]:
    """终止 pid 及其后代，只针对该进程树；返回实际被杀的 PID 列表。

    安全约束：Windows 用 `taskkill /F /T /PID <pid>`（精确 PID + 树），
    POSIX 用 killpg(进程组)。绝不使用 `taskkill /IM python.exe`。
    """
    killed: List[int] = []
    if pid <= 0:
        return killed
    if IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, timeout=15,
            )
            killed.append(pid)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except OSError:
                pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            killed.append(pid)
            time.sleep(min(grace_s, 2.0))
            if _pid_alive(pid):
                os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
    # 宽限回收
    t0 = time.monotonic()
    while time.monotonic() - t0 < grace_s and _pid_alive(pid):
        time.sleep(0.2)
    return killed


# --------------------------------------------------------------------------- #
# 内存守卫（只作用于自己派生的进程树）
# --------------------------------------------------------------------------- #
def system_mem_gb() -> Dict[str, Optional[float]]:
    """系统内存快照（GB）；psutil 缺失时返回 None。"""
    if psutil is None:
        return {"total": None, "available": None}
    vm = psutil.virtual_memory()
    return {"total": round(vm.total / 1e9, 2), "available": round(vm.available / 1e9, 2)}


def cpu_count() -> int:
    return os.cpu_count() or 0


def proc_tree_rss_gb(pid: int) -> Optional[float]:
    """自己进程树的常驻内存（GB）。只读，不杀进程。"""
    if psutil is None:
        return None
    try:
        p = psutil.Process(pid)
        rss = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except Exception:
                pass
        return rss / 1e9
    except Exception:
        return None


def preflight_memory(min_free_gb: float = DEFAULT_MEM_MIN_FREE_GB,
                     mem_cap_gb: float = DEFAULT_MEM_CAP_GB) -> Dict[str, Any]:
    """启动重活前的内存预检。不足时返回 ok=False（由调用方决定 BLOCKED 或等待）。"""
    mem = system_mem_gb()
    if mem["available"] is None:
        return {"ok": True, "available_gb": None, "reason": "psutil 缺失，内存守卫降级"}
    need = max(min_free_gb, mem_cap_gb + 2.0)
    return {"ok": mem["available"] >= need, "available_gb": mem["available"],
            "need_gb": round(need, 2), "mem_cap_gb": mem_cap_gb,
            "reason": "" if mem["available"] >= need else
            f"系统可用内存 {mem['available']}GB < 需求 {need:.1f}GB"}


# --------------------------------------------------------------------------- #
# 计时数据类
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    name: str
    cmd: List[str]
    start_mono: float = 0.0
    end_mono: float = 0.0
    duration_s: float = 0.0
    rc: Optional[int] = None
    timed_out: bool = False
    killed_pids: List[int] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    log_path: str = ""
    skipped: bool = False
    mem_cap_gb: Optional[float] = None
    mem_killed: bool = False
    peak_rss_gb: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("start_mono", None)
        d.pop("end_mono", None)
        return d


@dataclass
class DeadlineBudget:
    """单实验时间预算：以「训练调用开始」为零点，窗口 limit_s。"""

    limit_s: float = DEFAULT_LIMIT_S
    prep_cap_s: float = DEFAULT_PREP_CAP_S
    train_start_mono: Optional[float] = None
    eval_end_mono: Optional[float] = None

    def start_window(self) -> None:
        self.train_start_mono = time.monotonic()

    def mark_eval_end(self) -> None:
        self.eval_end_mono = time.monotonic()

    def elapsed(self) -> float:
        if self.train_start_mono is None:
            return 0.0
        end = self.eval_end_mono if self.eval_end_mono is not None else time.monotonic()
        return end - self.train_start_mono

    def remaining(self) -> float:
        return self.limit_s - self.elapsed()

    def window_s(self) -> Optional[float]:
        if self.train_start_mono is None or self.eval_end_mono is None:
            return None
        return self.eval_end_mono - self.train_start_mono

    def expired(self) -> bool:
        return self.remaining() <= 0


# --------------------------------------------------------------------------- #
# 运行一个阶段（子进程 + 超时杀树 + 内存守卫）
# --------------------------------------------------------------------------- #
class _MemMonitor(threading.Thread):
    """轮询自己派生的进程树内存；超上限只杀该树（绝不匹配 python.exe 全局）。"""

    def __init__(self, pid: int, cap_gb: float, poll_s: float,
                 on_exceed: "callable", grace_s: float):
        super().__init__(daemon=True)
        self.pid = pid
        self.cap_gb = cap_gb
        self.poll_s = poll_s
        self.on_exceed = on_exceed
        self.grace_s = grace_s
        self.peak_gb: Optional[float] = None
        self.exceeded = False
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.poll_s):
            rss = proc_tree_rss_gb(self.pid)
            if rss is None:
                return  # psutil 不可用，降级
            self.peak_gb = rss if self.peak_gb is None else max(self.peak_gb, rss)
            if rss > self.cap_gb:
                self.exceeded = True
                self.on_exceed(self.pid, self.grace_s)
                return

    def stop(self) -> None:
        self._stop.set()


def run_stage(
    name: str,
    cmd: Sequence[str],
    timeout_s: float,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[str] = None,
    tail_chars: int = 4000,
    mem_cap_gb: Optional[float] = None,
    mem_poll_s: float = DEFAULT_MEM_POLL_S,
) -> StageResult:
    """启动一个子进程阶段，超时只杀该进程树。

    timeout_s <= 0 时不启动，直接返回 timed_out=True（预算已耗尽）。
    mem_cap_gb 给定且 psutil 可用时，超上限只杀本阶段进程树（记录 mem_killed）。
    """
    res = StageResult(name=name, cmd=list(cmd))
    if log_path:
        res.log_path = os.path.abspath(log_path)
        os.makedirs(os.path.dirname(res.log_path), exist_ok=True)

    if timeout_s <= 0:
        res.timed_out = True
        res.rc = None
        res.start_mono = res.end_mono = time.monotonic()
        res.duration_s = 0.0
        res.stderr_tail = f"[budget] {name}: remaining<=0, 未启动（硬截止）"
        if log_path:
            with open(res.log_path, "a", encoding="utf-8") as f:
                f.write(res.stderr_tail + "\n")
        return res

    creationflags = 0
    popen_kwargs: Dict[str, Any] = {}
    if IS_WINDOWS:
        creationflags = _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    res.mem_cap_gb = mem_cap_gb
    res.start_mono = time.monotonic()
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **popen_kwargs,
    )

    monitor: Optional[_MemMonitor] = None
    if mem_cap_gb and psutil is not None:
        monitor = _MemMonitor(
            proc.pid, float(mem_cap_gb), mem_poll_s,
            on_exceed=lambda pid, grace: kill_process_tree(pid, grace),
            grace_s=DEFAULT_KILL_GRACE_S,
        )
        monitor.start()

    try:
        out, err = proc.communicate(timeout=timeout_s)
        res.rc = proc.returncode
    except subprocess.TimeoutExpired:
        res.timed_out = True
        res.killed_pids = kill_process_tree(proc.pid)
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = "", ""
        res.rc = proc.returncode

    if monitor is not None:
        monitor.stop()
        res.peak_rss_gb = round(monitor.peak_gb, 3) if monitor.peak_gb is not None else None
        if monitor.exceeded:
            res.mem_killed = True
            if not res.killed_pids:
                res.killed_pids = [proc.pid]

    res.end_mono = time.monotonic()
    res.duration_s = res.end_mono - res.start_mono
    out = out or ""
    err = err or ""
    res.stdout_tail = out[-tail_chars:]
    res.stderr_tail = err[-tail_chars:]

    if log_path:
        with open(res.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n===== stage={name} rc={res.rc} timed_out={res.timed_out} "
                    f"mem_killed={res.mem_killed} peak_rss_gb={res.peak_rss_gb} "
                    f"dur={res.duration_s:.1f}s =====\n")
            f.write("--- stdout ---\n" + out + "\n")
            f.write("--- stderr ---\n" + err + "\n")
    return res

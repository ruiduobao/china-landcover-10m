# control/ — 20 分钟硬截止实验控制与预注册（子 agent 专属）

> 实施依据：`技术文档/70_样本与分类改进实施方案及20分钟小区域实验规范.md`（v1.0 冻结）
> 证据来源：`技术文档/69_三省两轮分类与样本质量综合评价及改进建议.md`
> 归属：本目录由「实施准备、限时控制与实验注册」子 agent 独占；其他实验组只读，不修改。

## 1. 冻结参数（doc70 §10）

| 项 | 值 |
|---|---|
| 年份 / 类码空间 | 2023 / 24 类紧凑码（第二轮池实际 23 类，缺类 8） |
| 种子 | **7, 11, 23**（配对，三 seed；临界结果加至五 seed） |
| 单实验窗口 | 训练调用开始 → 评价与本实验结果落盘 **≤1200 s**；在 **1190 s 触发终止**，预留 ≤10 s 清理 |
| 每模型训练点数 | **≤20,000** |
| RF | **100 树 / minLeaf=2 / maxNodes=5000** / AEF64 |
| 园地上限 O-CAP | 类 2 ≤ **1%（最多 200 点）**，不足保留全部 |
| 资源 | `n_jobs=10`；CPU 全局 ≤15；内存目标 ≤10 GB |
| 评价 | 同点、全点＋FCS10／非FCS10 分层；类参考 n<30 只作诊断 |
| 采纳门 | macro9 配对 Δ 点估 ≥−1pp 且块 bootstrap 95% 下界 ≥−2pp；目标类 ΔF1 ≥+0.03 且 CI 支持正向；无 F1≤−0.05 退化 |
| 只读 | 原池、母库、第一/二轮交付、生产配置一律不改 |

## 2. 文件

| 文件 | 作用 |
|---|---|
| `deadline.py` | 可复用原语：`DeadlineBudget`（窗口计时）、`run_stage`（子进程+超时+**只杀自己子树**+内存守卫）、`FileLock`（单实例锁）、`atomic_write_json`、`preflight_memory` |
| `run_experiment.py` | **统一 runner（CLI）**：读 spec，跑 prepare/train/infer/eval，强制 1200s 窗口，写原子 JSON 与共享注册表 |
| `prereg.py` + `prereg_stageA.json` | 机器可读预注册配置与校验（硬约束对齐 doc70 §10） |
| `ns_exclude.py` | **namespace-safe 排除 helper**（位置键无条件 + id 键仅同命名空间生效）+ 池自检 |
| `selfcheck.py` | 轻量自检（不训练）：锁 / 原子写 / 超时杀树 / 内存守卫 / 合成命名空间用例 |
| `spec_template.json` | 单实验 spec 模板（含 doc70 §4.1 注册字段） |
| `registry_template.json` | 共享注册表模板（`registry/experiments.json`） |

## 3. CLI 用法

```bash
# 绝对路径，可直接复制运行
CTRL="F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/11.改进实施与小区域实验/control"
WS="F:/lc_work/improve_2023_pilot_20261015"

# 预注册配置校验（硬约束）
python "$CTRL/prereg.py" --check

# 轻量自检（不训练；输出 control/run_results/_selfcheck.json）
python "$CTRL/selfcheck.py" --workspace "$WS"

# 干跑校验 spec（不启动任何子进程）
python "$CTRL/run_experiment.py" --spec "$CTRL/spec_template.json" --workspace "$WS" --dry-run

# 跑一个实验（单次 invoke 只跑 1 个 region×seed×arm）
python "$CTRL/run_experiment.py" --spec "$WS/control/specs/<exp>.json" --workspace "$WS"

# namespace-safe 排除自检（只读现有池，不重建、不改 s1 原池）
python "$CTRL/ns_exclude.py" --selfcheck \
  --pool "F:/lc_work/prod2nd_2023/data/train2023_v2.parquet" \
  --out "$WS/control/run_results/ns_pool_selfcheck.json"
```

## 4. 其他实验组如何接入

1. 把实验登记成一份 spec（复制 `spec_template.json`），填好 doc70 §4.1 字段
   （`region_bbox / tile_id / training_bbox / input_identity / class_map_hash /
   candidate_rule / seeds / n_train_limit / rf_params / deadline_s /
   development_ids / confirmation_ids / target_metrics / expected_effect /
   stop_rules / code_hash / registered_at`）。
2. 四个阶段各自写成独立脚本（`prepare/train/infer/eval`），在 spec 的 `stages.*.cmd` 里给出
   绝对命令；阶段之间用工作区文件传递中间产物。
3. 需要 RF 的置 `"uses_rf": true` → runner 会先取 `rf.lock`（**重活串行 slot**，等待上限
   `slot_wait_s`，默认 1800s）；取不到记 `blocked_slot`，**不杀任何进程**。几何等非 RF
   实验可置 `"uses_rf": false` 直接跑。
4. **若已有等效控制（自建超时/锁），不强制迁移**；迁移时只需保证：窗口定义一致、
   超时只杀自己子树、结果含逐阶段时长。

## 5. 时间证据如何独立审核

每次运行写两份原子 JSON：
- `control/run_results/<exp_id>.json`：完整记录
- `registry/experiments.json`：全实验汇总

审核者无需信任 runner 内部状态，直接看：
- `stage_times_s.{prepare,train,infer,eval}`：各阶段实测秒数
- `deadline_window_s` = `eval.end − train.start`（**唯一的 1200s 判定量**）
- `limit_s` 与 `budget_ok`：`deadline_window_s ≤ limit_s` 的布尔结论
- `trigger_margin_s`：截止前预留的终止+清理余量（默认 10s → 1190s 触发）
- `stage_peak_rss_gb` / `stage_mem_killed`：内存守卫证据
- `killed_pids`：超时/超内存时被终止的**本实验**子进程 PID（证明未误伤）

截止语义：单阶段超时预算 = `min(阶段timeout, 窗口剩余 − trigger_margin)`；触发后终止本实验子树
并完成清理与落盘。若清理后整体 elapsed 仍 > limit，仍记 `TIMEOUT`，不伪称成功（`budget_ok` 为
`deadline_window_s ≤ limit_s` 的严格判定）。1200 s 是**上限**，不要求请求卡满。

`prepare` 独立计时上报，不并入窗口；runner **一次只跑一个实验**，禁止把一次实验拆成多次
invoke 规避 1200s（拆分即需按 doc70 §3.2 另立**新的预注册降级实验**）。

## 6. 安全边界

- 超时/超内存：Windows 用 `taskkill /F /T /PID <pid>`（精确 PID+树），POSIX 用 `killpg`；
  **绝不** `taskkill /IM python.exe`、不匹配 `bash.exe`。
- 单实例：`control/locks/runner.lock`；RF 重活：`control/locks/rf.lock`（带等待）。
- 无 GEE 登录/任务；不动母库、生产配置与既有交付。
- 内存守卫依赖 `psutil`（项目已装）；缺失时降级为「不可用」并显式告警，不静默通过。

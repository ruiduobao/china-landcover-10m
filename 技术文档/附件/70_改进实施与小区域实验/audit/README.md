# audit — 独立方法验收与最终汇总（工作区）

> 建立 2026-10-15。角色：独立方法验收 + 最终汇总子 agent。
> 依据：`技术文档/70_样本与分类改进实施方案及20分钟小区域实验规范.md`（**冻结 v1.0 最终 md5 `749017cb73ba7e66b5816b1bbd7486bf`**，18,429 B，mtime 2026-10-15 14:05:47；陈旧 `9d635870…` 已被取代）、`技术文档/69_…md`、doc69 附件 `methodology/03`、`experiment/q7`。
> 主层口径：组级非劣门与目标类 F1 以**非FCS10内部确认层**为主；全点/FCS10 完整报告，不得以主层过门称地表真值提升。

## 职责边界

**做**：全结果独立复算（UA/PA/F1、macro9、来源分层、块 bootstrap、McNemar）、20 分钟 walltime 计时验收、泄漏/块排除证据核查、manifest 与哈希归档、几何覆盖证据核查。
**不做**：不重训 RF、不启 GEE、不改训练池/母库/交付、不改实验原结果、不动 `代码/9.*`；发现问题只报告主 agent。
**纪律**：本轮不要求统计优势普适保证，采纳仅限小区；不预先新增 n≥100 硬门（n30–99 只作线索，可判 `PROMISING`）；不重复全池统计；不重复大栅格审查。

## 目录

| 目录 | 用途 |
|---|---|
| `registry/` | 冻结记录（doc70 指纹、注册表 hash） |
| `review/` | 只读评审与逻辑审核记录 |
| `inbox/` | 从实验代理接收的**只读**输入（注册表 JSON、逐点预测、计时日志、几何证据） |
| `scripts/` | 轻量审计脚本（等 outputs，不重训 RF） |
| `out/` | 审计产物（复算表、计时核验、manifest、结论） |

## 脚本（已就绪，待实验结果）

| 脚本 | 作用 | 输入契约 |
|---|---|---|
| `scripts/a0_hash_manifest.py` | 任意文件/目录哈希清单 | 路径列表 |
| `scripts/a1_timing_audit.py` | 计时/超时/残留核验 | runner 记录 JSON/JSONL |
| `scripts/a2_recompute_metrics.py` | 独立复算指标 + 配对 CI | 逐点预测表 + class/macro9 映射 |

运行前先确认 `inbox/` 内 schema 与脚本契约一致；不一致脚本会显式报错而非猜测。

## 判定口径（最终采用）

`audit/out/verdicts_2026-10-15.json`（schema `audit_verdict/v1`）三字段：`execution`（DONE/TIMEOUT/BLOCKED/FAILED）、`adoption`（ADOPT/REJECT）、`exploration`（组级方向描述，不作采纳依据）。
**规则**：目标类 NOT_IMPROVED 或 INSUFFICIENT_SUPPORT → `adoption=REJECT`；不得用 PROMISING/HOLD 充当正结论；REJECT＝本轮未证明目标类改善，非组级退化。

## 关键验收项（详见 `review/70_逻辑审核_2026-10-15.md`）

1. **A1** 确认空间块须从**训练**中排除（不只评估拆分），基线与候选同一排除规则；与生产设计不同须如实写明。
2. **A2/A3** n≥30 可报 F1，n30–99 仅线索（可 `PROMISING`）；确认档 n 不足不判 `ACCEPT_EXPERIMENTAL`。
3. **A4** O-CAP 补足规则须明确并输出每类实际训练数。
4. **A5** 超时后验证子进程终止（无孤儿）、无半写文件、不误杀他会话。
5. **A6** 几何楔外一致率应 =1.0（或注册容差）。

# 归档脚本运行说明（_archive）

> 2026-10-15 ｜ 状态：**脚本就绪、尚未执行复制**（等主 agent"开始归档"指令）

## 命令

```bash
# 预览（默认 dry-run，不写任何文件）
python F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/_archive/archive_to_repo.py

# 执行归档（落 manifest 与 README；幂等可重跑）
python F:/lc_work/prod2nd_2023/reports/quality_review_2026-10-15/_archive/archive_to_repo.py --apply
```

## 映射

| 源（quality_review_2026-10-15/<组>/） | 目标 |
|---|---|
| `*.py` | `代码/10.样本与实验综合评价/2026-10-15/<组>/` |
| `*.md` / `*.csv` / `*.json` | `技术文档/附件/69_样本与实验综合评价/<组>/` |

分组：`experiment` / `terrain` / `seams` / `samples` / `methodology`；另 `_archive/` 的脚本只进代码侧同名目录。

## 规则（硬）

1. 白名单扩展名 `.py .md .csv .json`；**排除** gpkg/parquet/tif/png/jpg/ovr/npz/zip 与 `chips/__pycache__/vis/sheets/zoom/images` 目录。
2. 文件名命中 `cred|token|secret|password|passwd|_key|.key|account|accounts|cfg|config`（大小写不敏感）一律排除——**任何凭证/账号配置不进归档**。
3. 单件 > 50 MB 跳过（轻量附件约束）。
4. 目标已存在：**同 sha256 → 跳过并记 same；内容不同 → 记 CONFLICT 并拒绝覆盖**（进程退出码 3，便于发现问题）。
5. `--apply` 时写入：
   - `技术文档/附件/69_样本与实验综合评价/_manifest_archive_2026-10-15.csv`（逐文件：源相对路径、目标相对路径、大小、sha256、action）
   - 同 manifest 副本到 `代码/10.样本与实验综合评价/2026-10-15/`
   - `技术文档/附件/69_样本与实验综合评价/README_附件说明.md` 与代码侧 `README_代码说明.md`
6. 不删除/不移动任何原件；脚本本身可重跑（幂等）。

## 已完成的验证（未复制）

- `python -m py_compile archive_to_repo.py` → PASS；
- dry-run 列表 PASS（首次 48 件候选；`methodology/` 新增 3 件后应在 51 件左右，以实际输出为准）；
- 若输出出现 `[conflict]`，先人工比对再处理，不要强写。

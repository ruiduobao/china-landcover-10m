# experiments/ — 盲判修复与小分类·实验驱动（R/V 视图 + 受门控 RF）

> 依据：`技术文档/72_盲判证据驱动样本修复与小分类实施协议.md`（主 agent 冻结）
> 单一配置源：`../rc_conf.py`（路径/角色/配额/码表/中性 ID）；本目录不另设第二套常量。
> 工作区：`F:/lc_work/label_repair_2023_20261015/`（materials 只读、cache 视图、experiments 预测、logs 计时、prereg 冻结）

## 脚本

| 脚本 | 作用 | 是否训练 RF |
|---|---|---|
| `lr_common.py` | 共用原语：路径、pos_key/block_id、冻结校验、结构组/轴映射、指标、块 bootstrap | 否 |
| `r0_prereg.py` | 预注册校验 / 冻结（sidecar）/ 目标冻结（R QC 后） | 否 |
| `r1_selfcheck.py` | 零训练自检（合成夹具：修复规则、C/R 同点、泄漏、同 count、映射、pred schema） | 否 |
| `r2_build_view.py` | 构建两臂视图 `cache/view_<region>_<size>.npz`（C + R；剔 V 块+已用参考块） | 否 |
| `r3_run_unit.py` | 受门控 RF 运行器：**必须 `--gate`**；只写 `pred`，不写真值；≤1200s 硬截止 | 是 |

## 关键纪律（写死在代码里）

- 两臂 `baseline.v2 = C ∪ R(label_before)` / `view-repair = C ∪ R(label_after/isolate+同窗同序回填)`；
  **非变更点（C 与 relabel/keep 的 R）ID 与特征逐值一致**；isolate 后同 count 回填来自同窗同一 rng 顺序，
  类别先验变化落 `class_prior_delta`。
- `selected unique R == 全部合格 R` 且 `>0`；否则 `NO_REPAIR_SUPPORT`（不假造改善，允许负结果）。
- R 占比 = `unique_R / N_total ≤ 0.10`（probe N=2000；real20k 仅一次旁证）。
- 训练窗剔**全部 V blocks + 已用 reference blocks**；`record_leak=0`。
- 修复规则机械执行：**针阔变更须显式高分结构复判一致**（`evidence_type=highres_structure_recheck`，不得由 S2/物候/坡向推针阔）/ 常绿落叶须当年物候 / 园地须管理时序证据 / `scope=point`。
- 训练窗剔除规则：上下文 C 候选剔 **V 块 + 已用参考块**；**R 即使落在已用参考块内也保留**（训练审计组，否则修复无实际作用），仅不得落 V 块；断言 R∩V 位置与块均不相交、C 无禁止块残留。
- 预测表 schema `region,arm,seed,vid,lon,lat,pred`，**禁含真值列**（V 真值由 audit 一次解盲）。

## 命令

```bash
CTRL="F:/地理所/论文/中国土地覆盖数据_2017-2024/代码/12.盲判修复与小分类/experiments"
WS="F:/lc_work/label_repair_2023_20261015"

python "$CTRL/r0_prereg.py" --check                 # 校验预注册（改前必跑）
python "$CTRL/r1_selfcheck.py"                      # 零训练自检（ALL_OK=True 才算通过）

# —— 以下三步等主 agent 下达『盲判及QC过门、修复规则冻结、开始实验』后再跑 ——
python "$CTRL/r0_prereg.py" --freeze --by "<main-agent-id>"
python "$CTRL/r0_prereg.py" --freeze-targets --by "<main-agent-id>"   # R QC 后
python "$CTRL/r2_build_view.py" --size probe --regions SC_F,SC_O
python "$CTRL/r3_run_unit.py" --region SC_F --size probe --gate
python "$CTRL/r3_run_unit.py" --region SC_O --size probe --gate
# real20k 只允许一次：
python "$CTRL/r2_build_view.py" --size real20k --regions SC_F,SC_O
python "$CTRL/r3_run_unit.py" --region SC_F --size real20k --gate
python "$CTRL/r3_run_unit.py" --region SC_O --size real20k --gate
```

规划 fits：2 size × 2 region × 2 arm × 3 seed = **24**。
V 真值解盲与指标计算归 audit；本目录不读真值、不调阈值、不扫权重。

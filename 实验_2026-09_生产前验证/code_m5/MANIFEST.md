# D0 材料链路代码清单（MD5）

> 归档时间：2026-10-08 01:54 ｜ 工作副本：`F:/lc_work/v31_exp/code/` ｜ git 基线：533ce5d（2026-10-07，仓库根）

| 文件 | MD5 | 字节 |
|---|---|---|
| m5_build_points.py | `690df68a9cbda5285ec1d9a6594d6307` | 4326 |
| m5a_fast_probe.py | `98deeab7d5ebd2abc021fca2ca007df9` | 13469 |
| m5a_materials.py | `6c5a1a29ab25dc7c052d9eca09d60c20` | 15468 |
| m5b_fleet.py | `cd052ba3f9de3d1f0d1672de35b9a391` | 5865 |
| m5c_ningxia_dissect.py | `12b5172a60c3f98768d72209a5db3722` | 8287 |
| m5f_merge_materials.py | `9ed051eb3dd9921bc50b63c730a0320b` | 8358 |

- 依赖外部件：geefast-download 技能（`C:/Users/Administrator/.agents/skills/geefast-download/`，xcomp/dl_tool.py 为带补丁副本）
- GEE 数据资产：`COPERNICUS/S2_SR_HARMONIZED`、`UMD/hansen/global_forest_change_2025_v1_13`、`LARSE/GEDI/GEDI02_A_002_MONTHLY`(rh98)
- 账号映射：`F:/lc_work/v31_exp/config/m5_pids.json`（63 账号，由 gee_accounts/*/_任务登记.md 挖取）
| m3c_analyze.py（含 --form-file，2026-10-08） | `7a76c9ba841a3cf12dbf62e1448f9579` | 14923 |
| m5g_review_min.py（最小复核清单生成+判后合并，2026-10-08） | `10b3e89d86d826d949f43134f9eb36ef` | 8533 |
| m5i_m3b_shp.py（复核点转 shp：WGS84 + GCJ-02 双版，2026-10-08） | `6853b75b4e1ba84bcff8fc971fe8c237` | 8096 |

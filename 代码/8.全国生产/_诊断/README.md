# 一次性诊断脚本（生产链路验证过程中用）

| 脚本 | 作用 | 结论 |
|---|---|---|
| `asset_train_test.py` | 样本 → GEE 表资产（分片导出） | ✅ 5,650 点 ~5 min COMPLETED；**同名资产需先删** |
| `b_only.py` | GEE 端 `smileRandomForest` 训练+分类耗时 | ✅ 100 树 15 s / 150 树 19 s / maxNodes 11 s |
| `acl_test.py` / `acl_read.py` | 资产跨账号 ACL 共享 | ✅ `setAssetAcl({'readers':['user:<email>']})` 成功；被授权账号可读可训 |
| `p5_call_test.py` | `getDownloadURL` 子块尺寸实测 | ✅ 0.15° 可交互分类下载；0.25° 撞 `User memory limit exceeded`；0.5° 超 48 MB |
| `stage_test_tile.py` | 交互式产测试瓦片喂 p6 | ⚠️ 0.3° 交互分类超限，改用 v7 已产栅格 |
| `poll_test.py` | 轮询单个导出任务状态 | ✅ 台账/状态可读 |

> 生产链路正式代码在上一级目录（`p0`–`p6` + `prod_conf.py` + `pw_template.tpl` + `fleet/`）。

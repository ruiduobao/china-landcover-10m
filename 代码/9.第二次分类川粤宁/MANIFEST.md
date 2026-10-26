# 代码 MD5 清单（2026-10-14 建，生产收官后再补终稿）

| 文件 | MD5 | 字节 | 说明 |
|---|---|---|---|
| `c0_health.py` | `428d59a79f50d025aae7754a54b26e17` | 2728 | 账号健康复检（串行；并发会串号） |
| `c1_upload_pool_v2.py` | `6cc4e475d27e19d6187021764d4d419d` | 14298 | 池上传链（分片→组→宿主 merge；QCAP=3） |
| `c2_prep_prod.py` | `c536e2503e4060d8f2185da8fe57b9a5` | 3085 | 生产前置（prod2nd 目录 + ACL 共享 + 跨账号校验） |
| `c3_driver_prod2.py` | `2af4f46063c52eca63aec17952d89b31` | 32468 | 生产驱动运行副本（工作区 F:/lc_work/prod2nd_2023/run/prod2_driver.py 为执行体，两者同 MD5） |
| `c4_eval_second.py` | `ed05ca74532b6f80ddd8b3a61e8a9395` | 43664 | 评估（同窗同口径 + 与第一次对比） |
| `c5_mosaic_compare.py` | `56463773107a3feb93f98ffb8ea32c7a` | 76454 | 拼接对比（mosaic/compare/qa） |
| `c6_deliver.py` | `0baad2cf27b12257fdf4476bcc86f3a8` | 72210 | 交付打包（**2026-10-15 正式运行三处最小修复**：台账汇总排除 board*、清单排除 `_` 前缀草稿件、41 瓦动态计数；详 `README_c6.md` 正式运行留痕） |
| `c7_local_ab.py` | `254494babbcb44c1266259fba9207105` | 31693 |  |
| `mkm.py` | `6a634a6112bd69a38a4ec711c1c342b7` | 1403 |  |
| `ov_build_副本.py` | `69ab0f3a4fb8bf79ccf15a75df1c6497` | 7375 | 概览重建+PIL 验收（工作区 run/ 同 MD5，doc43 纪律）；**2026-10-15 BigTIFF 修复**：TileOffsets/TileByteCounts 元素宽度改为逐 tag 取实际 TIFF 类型（原恒 4 字节 → BigTIFF 幅面写坏，四川实跑暴露；回归 `reports/mosaic2/rerun/regress.json`） |

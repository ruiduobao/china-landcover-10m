# 15. Agent 复现手册——从零复现全流程的逐步操作指南

> 读者：接手的 agent 或新成员。按本文顺序执行可完整复现：r1 样本库重建 → 嵌入提取 → 清洗训练 → 逐年分类 → 精度验证。
> 环境：Windows + Git Bash + Anaconda（F:\anaconda，python 即 F:\anaconda\python.exe）+ 代理 socks5h://127.0.0.1:7890（常开）。
> 工作根目录：`Z:\Mywork\论文\中国土地覆盖数据`（**所有命令均在此目录执行**——旧脚本用相对路径）。

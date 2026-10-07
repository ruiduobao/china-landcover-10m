@echo off
rem launch_resume.bat — T3 主嵌入续跑一键启动（13 账号编队 + 自愈监督；待用户确认后使用）
rem 前置: fleet_new_13accounts.json 已生成；禁动 19 账号未包含；代理 socks5h://127.0.0.1:7890 已通
cd /d F:\地理所\论文\中国土地覆盖数据_2017-2024
python _tools\gen_fleet_new.py
python _tools\supervise_resume.py
pause

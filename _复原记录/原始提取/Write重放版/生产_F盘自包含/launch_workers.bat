@echo off
rem 一键启动 12 账号 r7 年度嵌入提取（F 盘自包含版）
rem 前置: 本机代理 127.0.0.1:7890 在线；账号凭证在 F:\地理所\...\gee_accounts
cd /d F:\r7_prod
if not exist logs mkdir logs
for %%A in (zsi8emo s4ezbd w2qe4hiu kitmyfaceplease2 berk95733 chengruiduobao ughwvm7968.med zhnagningdan1 5vqm9g save456jr e0p36771 7lus57it) do (
  start "e1b_%%A" /min cmd /c "F:\anaconda\python.exe -u F:\r7_prod\e1b_worker_f.py %%A >> F:\r7_prod\logs\e1b_w_%%A.log 2>&1"
)
echo 12 个 worker 已启动，日志在 F:\r7_prod\logs\

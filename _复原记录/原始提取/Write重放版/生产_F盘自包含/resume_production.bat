@echo off
rem resume_production.bat — r7 提取断点续跑（F盘版）
rem ①容错抢救Z盘幸存块 ②清杀新旧两版worker ③启动12账号(F盘输出)
echo [1/3] 抢救 Z 盘幸存块到 F 盘（Z不可读则自动跳过）...
F:\anaconda\python.exe F:\r7_prod\rescue_copy.py

echo [2/3] 清理残留 worker（新旧脚本名都杀，不碰其他进程）...
powershell -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*e1b_worker_f.py*' -or $_.CommandLine -like '*e1b_worker_r7.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output ('killed ' + $_.ProcessId + ' ' + $_.CommandLine) }"

echo [3/3] 启动 12 账号 worker（F 盘输出）...
cd /d F:\r7_prod
if not exist logs mkdir logs
for %%A in (zsi8emo s4ezbd w2qe4hiu kitmyfaceplease2 berk95733 chengruiduobao ughwvm7968.med zhnagningdan1 5vqm9g save456jr e0p36771 7lus57it) do (
  start "e1b_%%A" /min cmd /c "F:\anaconda\python.exe -u F:\r7_prod\e1b_worker_f.py %%A >> F:\r7_prod\logs\e1b_w_%%A.log 2>&1"
)
echo 完成。日志: F:\r7_prod\logs\  进度: F:\r7_prod\emb_parts_r7\
pause

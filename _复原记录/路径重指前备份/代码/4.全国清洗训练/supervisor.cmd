@echo off
REM CNLC 样本嵌入提取守护器：每 10 分钟检查，worker 不足自动补启
cd /d "Z:\Mywork\论文\中国土地覆盖数据"
set LOG=数据\本地处理\全国清洗训练\supervisor.log
:loop
echo %date% %time% supervisor tick >> "%LOG%"
set COUNT=0
for /f %%i in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object {$_.CommandLine -match 'e1_worker'} | Measure-Object).Count"') do set COUNT=%%i
echo   e1_worker count: %COUNT% >> "%LOG%"
if %COUNT% LSS 3 (
  echo   不足 3 个，补启 3 个 worker >> "%LOG%"
  cd /d "Z:\Mywork\论文\中国土地覆盖数据"
  start "" /b python -u "代码\4.全国清洗训练\e1_worker.py" zsi8emo
  start "" /b python -u "代码\4.全国清洗训练\e1_worker.py" s4ezbd
  start "" /b python -u "代码\4.全国清洗训练\e1_worker.py" w2qe4hiu
)
for /f %%i in ('powershell -NoProfile -Command "(Get-ChildItem '数据\本地处理\全国清洗训练\emb_parts' -Filter 'chunk_*.parquet' | Measure-Object).Count"') do set DONE=%%i
echo   done %DONE%/709 >> "%LOG%"
if %DONE% GEQ 704 (
  echo   extraction complete, supervisor exit >> "%LOG%"
  goto end
)
timeout /t 600 /nobreak >nul
goto loop
:end

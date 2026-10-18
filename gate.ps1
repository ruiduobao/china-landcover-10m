<#
  gate.ps1 — 项目闸门。每个项目根目录放一份。
  用法：  .\gate.ps1            跑全部检查
          .\gate.ps1 -Scope 1  只跑第 1 组
  输出：每行 PASS|FAIL|WARN | 检查名 | 详情；末尾汇总。
  退出码：0 = 无 FAIL；1 = 有 FAIL。
#>
param([int]$Scope = 0)

$ErrorActionPreference = "Continue"
$script:Fails = 0
$script:Warns = 0

function Check([string]$Name, [bool]$Ok, [string]$Detail) {
  if ($Ok) { Write-Output ("PASS | " + $Name + " | " + $Detail) }
  else { Write-Output ("FAIL | " + $Name + " | " + $Detail); $script:Fails++ }
}
function Warn([string]$Name, [string]$Detail) {
  Write-Output ("WARN | " + $Name + " | " + $Detail); $script:Warns++
}
function Group([string]$Title) { Write-Output ""; Write-Output ("---- " + $Title + " ----") }

# ---------- 1. 盘位红线 ----------
if ($Scope -eq 0 -or $Scope -eq 1) {
  Group "1 盘位红线"
  $red = [ordered]@{ "C" = 40; "D" = 50; "F" = 300 }
  foreach ($k in $red.Keys) {
    $d = Get-PSDrive -Name $k -ErrorAction SilentlyContinue
    if ($null -eq $d) { Warn ("disk_" + $k) "盘不存在，跳过"; continue }
    $free = [math]::Round($d.Free / 1GB, 1)
    Check ("disk_" + $k) ($free -ge $red[$k]) ("free=" + $free + "GB  redline=" + $red[$k] + "GB")
  }
}

# ---------- 2. 六件套 ----------
if ($Scope -eq 0 -or $Scope -eq 2) {
  Group "2 项目文档六件套"
  foreach ($f in @("LATEST.md","STATUS.md","DECISIONS.md","BLOCKERS.md","RUNLOG.md","MANIFEST.csv")) {
    $p = Join-Path $PSScriptRoot $f
    Check ("doc_" + $f) (Test-Path $p) $f
  }
}

# ---------- 3. LATEST.md 登记的路径必须真实存在 ----------
if ($Scope -eq 0 -or $Scope -eq 3) {
  Group "3 LATEST.md 路径有效性"
  $latest = Join-Path $PSScriptRoot "LATEST.md"
  if (Test-Path $latest) {
    $text = Get-Content $latest -Raw -Encoding UTF8
    $paths = [regex]::Matches($text, '`([A-Za-z]:\\[^`]+)`') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
    if ($paths.Count -eq 0) { Warn "latest_paths" "LATEST.md 里没有登记绝对路径" }
    foreach ($p in $paths) {
      if ($p -match 'sha256|hash') { continue }
      Check "path_exists" (Test-Path $p) $p
    }
  } else { Warn "latest_paths" "缺 LATEST.md，跳过" }
}

# ---------- 4. 禁词检查 ----------
if ($Scope -eq 0 -or $Scope -eq 4) {
  Group "4 禁词检查（无运行范围的结论性措辞）"
  $bad = @("终验通过","全部收口","检查无误","应该没问题","基本完成")
  $hits = @()
  Get-ChildItem $PSScriptRoot -Filter *.md -File -ErrorAction SilentlyContinue | ForEach-Object {
    foreach ($b in $bad) {
      $m = Select-String -Path $_.FullName -Pattern $b -SimpleMatch -ErrorAction SilentlyContinue
      if ($m) { $hits += ($_.Name + ":" + $m.LineNumber + " " + $b) }
    }
  }
  if ($hits.Count -eq 0) { Check "banned_words" $true "未发现禁词" }
  else { Check "banned_words" $false ($hits -join " ; ") }
}

# ---------- 汇总 ----------
Write-Output ""
Write-Output ("SUMMARY | FAIL=" + $script:Fails + " WARN=" + $script:Warns + " | " + $PSScriptRoot)
if ($script:Fails -gt 0) { exit 1 } else { exit 0 }

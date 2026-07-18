# =============================================================================
# run.ps1 - 运行脚本
# =============================================================================

param(
    [int]$Episodes = 100,
    [int]$Steps = 500
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcDir = Split-Path -Parent $ScriptDir
$Exe = Join-Path $SrcDir "build\Release\snn_dialogue.exe"

if (-not (Test-Path $Exe)) {
    Write-Host "可执行文件不存在，请先运行 build.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "运行 SNN Dialogue..." -ForegroundColor Green
Write-Host "Episodes: $Episodes"
Write-Host "Steps:    $Steps"
Write-Host ""

& $Exe

Write-Host ""
Write-Host "运行结束。" -ForegroundColor Green

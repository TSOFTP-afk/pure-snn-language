# Build snn_stage2_analyze
# Per project memory: must use x64 cl.exe to avoid cudafe++ crash

$vsShell = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"

Write-Output "=== Launching VS DevShell (x64) ==="
& $vsShell -HostArch amd64 -Arch amd64
Write-Output "VsDevShell exit: $LASTEXITCODE"

$buildDir = "f:\项目\THE TRUE AI\src\stage2\build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
}
Set-Location $buildDir
Write-Output "Build dir: $(Get-Location)"

Write-Output ""
Write-Output "=== cmake configure (if needed) ==="
if (-not (Test-Path "CMakeCache.txt")) {
    & cmake .. -G Ninja 2>&1 | Select-Object -Last 10
} else {
    Write-Output "CMakeCache.txt exists, skipping configure"
}
Write-Output "cmake exit: $LASTEXITCODE"

Write-Output ""
Write-Output "=== ninja build snn_stage2_analyze ==="
& ninja snn_stage2_analyze 2>&1 | Select-Object -Last 40
$ninjaExit = $LASTEXITCODE
Write-Output "ninja exit: $ninjaExit"

Write-Output ""
Write-Output "=== Check exe ==="
if (Test-Path "snn_stage2_analyze.exe") {
    $info = Get-Item "snn_stage2_analyze.exe"
    Write-Output "Built: $($info.FullName)"
    Write-Output "Size: $([math]::Round($info.Length/1MB,2)) MB"
} else {
    Write-Output "FAILED: snn_stage2_analyze.exe not found"
}

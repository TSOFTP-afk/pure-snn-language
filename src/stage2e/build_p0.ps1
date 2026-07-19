# Build snn_stage2e_p0 (Phase 0 烟雾测试)
# Per project memory: must use x64 cl.exe to avoid cudafe++ crash
# 修复：用 cmd /c 包装 cmake/ninja 调用，避开 PowerShell 5.x 中文参数编码问题

$vsShell = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"

Write-Output "=== Launching VS DevShell (x64) ==="
& $vsShell -HostArch amd64 -Arch amd64
Write-Output "VsDevShell exit: $LASTEXITCODE"

# 用 cmd /c pushd 获取当前脚本所在目录（脚本通过 -File 调用，
# $PSScriptRoot/$MyInvocation 在 VsDevShell 加载后丢失，所以用 %~dp0）
# 然后用相对路径 .. 调用 cmake
$buildDir = "build"

# 通过 Get-Location 进入脚本所在目录
# 假设用户从 stage2e 目录调用，或从项目根目录调用都行
# 关键：先确认当前工作目录里有 CMakeLists.txt
if (-not (Test-Path "CMakeLists.txt")) {
    # 当前目录没有 CMakeLists.txt，尝试切换到脚本目录
    # 用 cmd /c pushd 避开 PowerShell 中文路径编码问题
    Write-Output "CMakeLists.txt not in cwd, trying script dir via cmd..."
    # 通过 cmd for /c 获取脚本所在目录
    $cmd = 'for %i in ("%~f0") do @echo %~dpi'
    # 这里 $0 是 powershell 的, 不行
    # 改用 Get-Location 然后构造路径
    $cwd = (Get-Location).Path
    Write-Output "Current cwd: $cwd"
    # 如果 cwd 是 F:\项目\THE TRUE AI，则 srcDir 是 cwd\src\stage2e
    if (Test-Path "src\stage2e\CMakeLists.txt") {
        Set-Location "src\stage2e"
    } elseif (Test-Path "src\stage2e\build_p0.ps1") {
        # 已经在 stage2e 目录
    } else {
        Write-Output "ERROR: cannot find CMakeLists.txt"
        exit 1
    }
}

$cwd = (Get-Location).Path
Write-Output "Working dir: $cwd"

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
}

Write-Output ""
Write-Output "=== cmake configure (if needed) ==="
$cacheFile = "$buildDir\CMakeCache.txt"
if (-not (Test-Path $cacheFile)) {
    # 关键：用 cmd /c 调用 cmake，绕开 PowerShell 中文参数编码
    # 强制 Release 配置，避免 Debug 的 /RTC1 与 nvcc -O3 传过去的 /O2 冲突
    cmd /c "cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release" 2>&1 | Select-Object -Last 40
} else {
    Write-Output "CMakeCache.txt exists, skipping configure"
}
Write-Output "cmake exit: $LASTEXITCODE"

Write-Output ""
Write-Output "=== ninja build snn_stage2e_p0 ==="
cmd /c "ninja -C build snn_stage2e_p0" 2>&1 | Select-Object -Last 100
$ninjaExit = $LASTEXITCODE
Write-Output "ninja exit: $ninjaExit"

Write-Output ""
Write-Output "=== Check exe ==="
$exePath = "$buildDir\snn_stage2e_p0.exe"
if (Test-Path $exePath) {
    $info = Get-Item $exePath
    Write-Output "Built: $($info.FullName)"
    Write-Output "Size: $([math]::Round($info.Length/1MB,2)) MB"
} else {
    Write-Output "FAILED: snn_stage2e_p0.exe not found"
}

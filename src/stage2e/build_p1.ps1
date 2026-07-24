# Build snn_stage2e_p1 (Phase 1 烟雾测试)
# Per project memory: must use x64 cl.exe to avoid cudafe++ crash
# 修复：用 cmd /c 包装 cmake/ninja 调用，避开 PowerShell 5.x 中文参数编码问题

$ErrorActionPreference = "Stop"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found" }
$vsInstall = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsInstall) { throw "Visual C++ x64 toolchain not found" }
$vsShell = Join-Path $vsInstall "Common7\Tools\Launch-VsDevShell.ps1"

Write-Output "=== Launching VS DevShell (x64) ==="
& $vsShell -HostArch amd64 -Arch amd64
if ($LASTEXITCODE -ne 0) { throw "VsDevShell failed: $LASTEXITCODE" }

$bundledCmake = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$bundledNinja = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (Test-Path (Join-Path $bundledCmake "cmake.exe")) { $env:PATH = "$bundledCmake;$env:PATH" }
if (Test-Path (Join-Path $bundledNinja "ninja.exe")) { $env:PATH = "$bundledNinja;$env:PATH" }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw "cmake not found" }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw "ninja not found" }

$buildDir = "build"

# 切换到脚本所在目录 (CMakeLists.txt 所在)
if (-not (Test-Path "CMakeLists.txt")) {
    Write-Output "CMakeLists.txt not in cwd, trying script dir via cmd..."
    $cwd = (Get-Location).Path
    Write-Output "Current cwd: $cwd"
    if (Test-Path "src\stage2e\CMakeLists.txt") {
        Set-Location "src\stage2e"
    } elseif (Test-Path "src\stage2e\build_p1.ps1") {
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
    cmd /c "cmake -S . -B build -G Ninja -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=ON" 2>&1 | Select-Object -Last 60
} else {
    Write-Output "CMakeCache.txt exists, skipping configure"
}
Write-Output "cmake exit: $LASTEXITCODE"

Write-Output ""
Write-Output "=== ninja build snn_stage2e_p1 ==="
cmd /c "cmake --build build --target snn_stage2e_p1 stage2e_run_config_test --parallel" 2>&1 | Select-Object -Last 150
$ninjaExit = $LASTEXITCODE
Write-Output "ninja exit: $ninjaExit"
if ($ninjaExit -ne 0) { throw "build failed: $ninjaExit" }
cmd /c "ctest --test-dir build --output-on-failure"
if ($LASTEXITCODE -ne 0) { throw "tests failed: $LASTEXITCODE" }

Write-Output ""
Write-Output "=== Check exe ==="
$exePath = "$buildDir\snn_stage2e_p1.exe"
if (Test-Path $exePath) {
    $info = Get-Item $exePath
    Write-Output "Built: $($info.FullName)"
    Write-Output "Size: $([math]::Round($info.Length/1MB,2)) MB"
} else {
    Write-Output "FAILED: snn_stage2e_p1.exe not found"
}

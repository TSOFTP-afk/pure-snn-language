$ErrorActionPreference = "Continue"

# 0. 切到非中文路径
Set-Location C:\

# 1. junction 已存在, 验证
$srcJunction = "C:\stage2e_src"
$buildJunction = "C:\stage2e_build"
if (-not (Test-Path "$srcJunction\CMakeLists.txt")) { throw "source junction invalid" }
Write-Output "Source junction OK"

# 刷新 build junction
if (Test-Path $buildJunction) { (Get-Item $buildJunction).Delete() }
$realBuild = "F:\项目\THE TRUE AI\src\stage2e\build"
New-Item -ItemType Junction -Path $buildJunction -Target $realBuild -Force | Out-Null
Write-Output "Build junction OK"

# 2. 用 vcvars64.bat 设置 MSVC 环境 (更可靠)
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found" }
Write-Output "Setting up MSVC env via vcvars64.bat..."
$envOut = & cmd /c "`"$vcvars`" && set" 2>&1
foreach ($line in $envOut) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# 3. 添加 cmake/ninja 到 PATH
$vsInstall = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$bundledCmake = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$bundledNinja = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
$env:PATH = "$bundledCmake;$bundledNinja;$env:PATH"

# 4. 验证工具链
$clCmd = Get-Command cl -ErrorAction SilentlyContinue
if ($clCmd) { Write-Output "cl: OK ($($clCmd.Source))" } else { Write-Output "cl NOT FOUND"; exit 1 }
$nvccCmd = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvccCmd) { Write-Output "nvcc: OK ($($nvccCmd.Source))" } else { Write-Output "nvcc NOT FOUND"; exit 1 }
$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCmd) { Write-Output "cmake: OK" } else { Write-Output "cmake NOT FOUND"; exit 1 }
$ninjaCmd = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninjaCmd) { Write-Output "ninja: OK" } else { Write-Output "ninja NOT FOUND"; exit 1 }

# 5. 切到 build junction
Set-Location $buildJunction
Write-Output "Build dir: $(Get-Location)"

# 6. 清理旧 cache
Write-Output "=== Cleaning old cache ==="
if (Test-Path CMakeCache.txt) { Remove-Item CMakeCache.txt -Force }
if (Test-Path CMakeFiles) { Remove-Item CMakeFiles -Force -Recurse -ErrorAction SilentlyContinue }
if (Test-Path build.ninja) { Remove-Item build.ninja -Force }

# 7. CMAKE configure
Write-Output "=== CMAKE configure ==="
& cmd /c "cmake -S C:\stage2e_src -B C:\stage2e_build -G Ninja -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=ON" 2>&1 | Select-Object -Last 50
$cfgExit = $LASTEXITCODE
Write-Output "cmake configure exit: $cfgExit"
if ($cfgExit -ne 0) { exit $cfgExit }

# 8. 编译 snn_stage2e_p1
Write-Output ""
Write-Output "=== Building snn_stage2e_p1 ==="
& cmd /c "cmake --build C:\stage2e_build --target snn_stage2e_p1 --parallel" 2>&1 | Select-Object -Last 200
$buildExit = $LASTEXITCODE
Write-Output "ninja build exit: $buildExit"
if ($buildExit -ne 0) { exit $buildExit }

# 9. 验证产物
Write-Output ""
Write-Output "=== Check exe ==="
$exePath = "$buildJunction\snn_stage2e_p1.exe"
if (Test-Path $exePath) {
    $info = Get-Item $exePath
    Write-Output "OK: $($info.FullName)"
    Write-Output "Size: $([math]::Round($info.Length/1MB,2)) MB"
    Write-Output "Modified: $($info.LastWriteTime)"
} else {
    Write-Output "FAILED: snn_stage2e_p1.exe not found"
    exit 1
}

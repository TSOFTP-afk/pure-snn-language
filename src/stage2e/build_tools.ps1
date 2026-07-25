$ErrorActionPreference = "Stop"

# 1. 切到非中文路径 (避免 VsDevShell 中文路径启动失败)
Set-Location C:\

# 2. 启动 VS DevShell (x64)
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

# 3. 添加 cmake/ninja 到 PATH
$bundledCmake = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$bundledNinja = Join-Path $vsInstall "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (Test-Path "$bundledCmake\cmake.exe") { $env:PATH = "$bundledCmake;$bundledNinja;$env:PATH" }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw "cmake not found" }
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) { throw "ninja not found" }

# 4. 切到 junction 路径 (无中文) 编译
Set-Location C:\stage2e_build
Write-Output ""
Write-Output "=== Building inspect_ckpt + snn_stage2e_decoder ==="
cmake --build . --target inspect_ckpt snn_stage2e_decoder --parallel 2>&1
$ninjaExit = $LASTEXITCODE
Write-Output "ninja exit: $ninjaExit"
if ($ninjaExit -ne 0) { throw "build failed: $ninjaExit" }

# 5. 验证产物
Write-Output ""
Write-Output "=== Check exes ==="
foreach ($exe in @("inspect_ckpt.exe", "snn_stage2e_decoder.exe")) {
    if (Test-Path $exe) {
        $info = Get-Item $exe
        Write-Output "OK: $($info.FullName) ($([math]::Round($info.Length/1KB,2)) KB, $($info.LastWriteTime))"
    } else {
        Write-Output "MISSING: $exe"
    }
}

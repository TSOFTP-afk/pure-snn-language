# =============================================================================
# build.ps1 - 构建脚本（Windows + MSVC + CUDA）
# =============================================================================
#
# 使用方法：
#   cd f:\项目\THE TRUE AI\src
#   .\scripts\build.ps1
#
# 前置要求：
#   - Visual Studio 2022（含 C++ 工具链）
#   - CUDA Toolkit 12.x 或 13.x
#   - CMake 3.18+
# =============================================================================

param(
    [string]$BuildType = "Release",
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"

# 路径
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SrcDir = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $SrcDir "build"

Write-Host "=== SNN Dialogue 构建脚本 ===" -ForegroundColor Green
Write-Host "源码目录: $SrcDir"
Write-Host "构建目录: $BuildDir"
Write-Host "构建类型: $BuildType"
Write-Host ""

# 清理
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host "[1/4] 清理旧构建..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BuildDir
}

# 检查 CUDA
Write-Host "[2/4] 检查 CUDA..." -ForegroundColor Yellow
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if (-not $nvcc) {
    Write-Host "错误：找不到 nvcc，请确认 CUDA Toolkit 已安装并加入 PATH" -ForegroundColor Red
    exit 1
}
Write-Host "  nvcc: $($nvcc.Source)"

# 检查 CMake
Write-Host "[3/4] 检查 CMake..." -ForegroundColor Yellow
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "错误：找不到 cmake，请安装 CMake 3.18+" -ForegroundColor Red
    exit 1
}
Write-Host "  cmake: $($cmake.Source)"

# 创建构建目录
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
}

# CMake 配置
Write-Host "[4/4] CMake 配置与编译..." -ForegroundColor Yellow
Push-Location $BuildDir

try {
    # 生成项目（使用 Ninja 或 MSVC）
    Write-Host "  生成构建文件..."
    & cmake $SrcDir -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=$BuildType
    if ($LASTEXITCODE -ne 0) {
        throw "CMake 配置失败"
    }

    # 编译
    Write-Host "  编译中（可能需要几分钟）..."
    & cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "编译失败"
    }

    Write-Host ""
    Write-Host "=== 构建成功 ===" -ForegroundColor Green
    Write-Host "可执行文件: $BuildDir\$BuildType\snn_dialogue.exe"

} catch {
    Write-Host "错误: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

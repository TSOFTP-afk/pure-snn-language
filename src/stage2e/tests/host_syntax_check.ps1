$ErrorActionPreference = "Stop"

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found"
}
$install = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $install) {
    throw "Visual C++ toolchain not found"
}
$devcmd = Join-Path $install "Common7\Tools\VsDevCmd.bat"
$stage2e = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stub = Join-Path $PSScriptRoot "stubs"
$tempDir = Join-Path $env:TEMP "stage2e-host-syntax"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

$command = "call `"$devcmd`" -arch=amd64 -host_arch=amd64 >nul" +
    " && cl /nologo /utf-8 /std:c++17 /EHsc /W4 `"$stage2e\run_config.cpp`" `"$stage2e\tests\run_config_test.cpp`" /I`"$stage2e`" /Fe:`"$tempDir\run_config_test.exe`"" +
    " && `"$tempDir\run_config_test.exe`"" +
    " && cl /nologo /utf-8 /std:c++17 /EHsc /W4 /TP /c `"$stage2e\scheduler_checkpoint.cu`" /I`"$stub`" /I`"$stage2e`" /Fo:`"$tempDir\scheduler_checkpoint.obj`"" +
    " && cl /nologo /utf-8 /std:c++17 /EHsc /W4 /c `"$stage2e\main.cpp`" /I`"$stub`" /I`"$stage2e`" /Fo:`"$tempDir\main.obj`""

cmd.exe /d /c $command
if ($LASTEXITCODE -ne 0) {
    throw "host syntax check failed with exit code $LASTEXITCODE"
}
Write-Output "Host syntax checks passed"

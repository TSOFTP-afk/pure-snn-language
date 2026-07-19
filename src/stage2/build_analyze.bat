@echo off
REM Build snn_stage2_analyze using VS DevShell x64
REM Per project memory: must use HostX64/x64 to avoid cudafe++ crash

setlocal

set "VSBT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VSCMD=%VSBT%\Common7\Tools\VsDevShell.bat"

REM Launch VsDevShell with x64 host and target architecture
call "%VSCMD%" -host_arch=x64 -arch=x64 >nul 2>&1

if not exist "f:\项目\THE TRUE AI\src\stage2\build" (
    mkdir "f:\项目\THE TRUE AI\src\stage2\build"
)

cd /d "f:\项目\THE TRUE AI\src\stage2\build"

echo === cmake configure ===
cmake .. -G Ninja 2>&1 | findstr /R "error Configuring done Generating done"
echo cmake exit: %ERRORLEVEL%

echo.
echo === ninja build snn_stage2_analyze ===
ninja snn_stage2_analyze 2>&1 | findstr /R "error Error Building"
echo ninja exit: %ERRORLEVEL%

echo.
echo === Check exe ===
if exist "snn_stage2_analyze.exe" (
    for %%A in (snn_stage2_analyze.exe) do echo Built: %%~fA  Size: %%~zA bytes
) else (
    echo FAILED: snn_stage2_analyze.exe not found
)

endlocal

@echo off
chcp 65001 >nul
REM Stage 2e build script - direct F: Chinese path (no junction needed)
REM This file is at F:\项目\THE TRUE AI\src\stage2e\build_p1_cmd.bat

REM 1. Setup MSVC env
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed
    exit /b 1
)

REM 2. Add bundled cmake/ninja
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

REM 3. Verify tools
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl not found
    exit /b 1
)
echo cl OK
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found
    exit /b 1
)
echo nvcc OK
where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found
    exit /b 1
)
echo cmake OK
where ninja >nul 2>&1
if errorlevel 1 (
    echo ERROR: ninja not found
    exit /b 1
)
echo ninja OK

REM 4. Clean old cache
echo === Cleaning old cache ===
cd /d "F:\项目\THE TRUE AI\src\stage2e\build"
if exist CMakeCache.txt del /q CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles
if exist build.ninja del /q build.ninja

REM 5. CMAKE configure
echo === CMAKE configure ===
cmake -S "F:\项目\THE TRUE AI\src\stage2e" -B "F:\项目\THE TRUE AI\src\stage2e\build" -G Ninja -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTING=ON
if errorlevel 1 (
    echo ERROR: cmake configure failed
    exit /b 1
)
echo cmake configure OK

REM 6. Build snn_stage2e_p1
echo.
echo === Building snn_stage2e_p1 ===
cmake --build "F:\项目\THE TRUE AI\src\stage2e\build" --target snn_stage2e_p1 --parallel
if errorlevel 1 (
    echo ERROR: build failed
    exit /b 1
)

REM 7. Verify exe
echo.
echo === Check exe ===
if exist "F:\项目\THE TRUE AI\src\stage2e\build\snn_stage2e_p1.exe" (
    echo OK: snn_stage2e_p1.exe built
    for %%I in ("F:\项目\THE TRUE AI\src\stage2e\build\snn_stage2e_p1.exe") do echo Size: %%~zI bytes
) else (
    echo FAILED: snn_stage2e_p1.exe not found
    exit /b 1
)
@echo off
chcp 65001 >nul
REM Run 40K step training with proper output capture
REM Triggers sleep replay at steps 30000, 40000 (REPLAY_WARMUP_STEPS=20000)
cd /d "F:\项目\THE TRUE AI\src\stage2e\build"

REM Remove old capture files
if exist training_motor_40k.csv del /q training_motor_40k.csv
if exist stdout_40k.log del /q stdout_40k.log
if exist stderr_40k.log del /q stderr_40k.log

REM Run training with output redirection (cmd's redirection is more reliable for native exes)
REM checkpoint every 10K steps -> 4 checkpoints expected (10K, 20K, 30K, 40K)
REM Sleep replay triggers at step 30000 and 40000 (REPLAY_WARMUP_STEPS=20000)
REM --text points to project-level data dir (relative path resolves from build/)
snn_stage2e_p1.exe --steps 40000 --checkpoint-interval 10000 --text "..\..\..\data\lccc_sample_1mb.txt" > stdout_40k.log 2> stderr_40k.log
echo Exit code: %ERRORLEVEL%

REM Show file sizes
for %%I in (stdout_40k.log) do echo stdout size: %%~zI bytes
for %%I in (stderr_40k.log) do echo stderr size: %%~zI bytes

REM Show last 80 lines of stdout
echo.
echo === stdout last 80 lines ===
powershell -NoProfile -Command "Get-Content stdout_40k.log -Tail 80"

echo.
echo === stderr last 30 lines ===
powershell -NoProfile -Command "Get-Content stderr_40k.log -Tail 30"
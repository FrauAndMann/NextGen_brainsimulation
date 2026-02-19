@echo off
REM ============================================================
REM SYNAPSE - Start Life
REM ============================================================
REM
REM Starts the "life" of SYNAPSE:
REM - Continuous learning (brain-like training)
REM - Dashboard (real-time visualization)
REM - Auto-save progress
REM
REM The AI will learn continuously until you close this window.
REM Progress is saved automatically every 5 minutes.
REM
REM Usage:
REM   run_life.bat              - Start fresh or resume latest
REM   run_life.bat new          - Start fresh (ignore checkpoints)
REM   run_life.bat resume xxx   - Resume specific checkpoint
REM   run_life.bat --hours 8    - Stop after 8 hours
REM
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo    SYNAPSE - Starting Life
echo ============================================================
echo.

REM Parse arguments
set MODE=%1
set CHECKPOINT=%2
set EXTRA_ARGS=

if "%MODE%"=="new" (
    set MODE=start_fresh
)
if "%MODE%"=="resume" (
    if not "%CHECKPOINT%"=="" (
        set RESUME_ARG=--resume %CHECKPOINT%
    )
)

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Virtual environment not found!
    echo.
    echo Please run setup.bat first:
    echo   setup.bat        (for CPU)
    echo   setup.bat cuda   (for GPU)
    echo.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Check if model exists
if not exist "files\model\self_aware_ai.py" (
    echo.
    echo ERROR: Model files not found!
    echo Please ensure the project is complete.
    echo.
    pause
    exit /b 1
)

REM Find latest checkpoint if resuming
if not defined RESUME_ARG (
    if not "%MODE%"=="start_fresh" (
        echo Looking for latest checkpoint...
        for /f "delims=" %%i in ('dir /b /o-d files\checkpoints\continuous_*.pt 2^>nul') do (
            set LATEST=%%i
            goto :found_checkpoint
        )
        :found_checkpoint
        if defined LATEST (
            echo    Found: !LATEST!
            set /p RESUME="Resume from this checkpoint? (y/n): "
            if /i "!RESUME!"=="y" (
                set RESUME_ARG=--resume !LATEST!
                echo    Resuming...
            ) else (
                echo    Starting fresh...
            )
        ) else (
            echo    No checkpoints found, starting fresh...
        )
    )
)

echo.
echo ============================================================
echo.
echo    Starting SYNAPSE Life...
echo.
echo    Training: CONTINUOUS (brain-like learning)
echo    Dashboard: http://localhost:8000
echo    Auto-save: Every 5 minutes
echo.
echo    Press Ctrl+C to stop. Progress will be saved.
echo.
echo ============================================================
echo.

REM Start API server in background
echo Starting API server...
start "SYNAPSE API" /min cmd /c "cd files && python api.py"

REM Wait for server to start
timeout /t 3 /nobreak >nul

REM Open dashboard in browser
echo Opening dashboard...
start "" "dashboard\index.html"

REM Wait a bit
timeout /t 2 /nobreak >nul

REM Start continuous training
echo Starting continuous learning...
echo.

cd files
python train_continuous.py %RESUME_ARG% --save-interval 300 --keep-checkpoints 10
cd ..

echo.
echo ============================================================
echo.
echo    Life session ended.
echo.
echo    Checkpoints saved in: files\checkpoints\
echo.
echo    To resume:
echo      run_life.bat resume latest
echo.
echo ============================================================
echo.

REM Close API server
taskkill /FI "WINDOWTITLE eq SYNAPSE API*" /F >nul 2>&1

pause

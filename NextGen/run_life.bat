@echo off
chcp 65001 >nul 2>&1

echo ============================================================
echo    SYNAPSE - Starting Life
echo ============================================================
echo.

REM Check venv
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Run setup.bat first!
    pause
    exit /b 1
)

REM Activate
call venv\Scripts\activate.bat

echo.
echo Starting SYNAPSE...
echo.
echo    - Continuous learning: ON
echo    - Dashboard: http://localhost:8000
echo    - Auto-save: Every 5 minutes
echo.
echo    Press Ctrl+C to stop (progress saved)
echo.
echo ============================================================
echo.

REM Start API in background
echo Starting API server...
start "SYNAPSE-API" /min cmd /c "cd files && python api.py"

REM Wait for API
timeout /t 3 /nobreak >nul

REM Open dashboard
echo Opening dashboard...
start "" "dashboard\index.html"

timeout /t 2 /nobreak >nul

REM Start training
echo Starting continuous learning...
echo.

cd files
python train_continuous.py --save-interval 300 --keep-checkpoints 10
cd ..

echo.
echo ============================================================
echo Session ended. Checkpoints saved in files\checkpoints\
echo To resume: run_life.bat
echo ============================================================
echo.

REM Kill API
taskkill /FI "WINDOWTITLE eq SYNAPSE-API*" /f >nul 2>&1

pause

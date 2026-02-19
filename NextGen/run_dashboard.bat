@echo off
REM SYNAPSE Dashboard Launcher
REM Starts both API server and opens dashboard

echo ============================================================
echo SYNAPSE Dashboard Launcher
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if in correct directory
if not exist "files\api.py" (
    echo ERROR: Please run this script from the NextGen directory
    pause
    exit /b 1
)

echo Starting SYNAPSE API Server...
echo.

REM Start API server in background
start "SYNAPSE API" cmd /k "cd files && python api.py"

REM Wait for server to start
timeout /t 3 /nobreak >nul

echo.
echo API Server started on http://localhost:8000
echo.
echo Opening Dashboard...

REM Open dashboard in default browser
start "" "dashboard\index.html"

echo.
echo ============================================================
echo SYNAPSE is running!
echo.
echo - API: http://localhost:8000
echo - Dashboard: dashboard/index.html (opened in browser)
echo - API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C in API window to stop the server.
echo ============================================================
echo.

pause

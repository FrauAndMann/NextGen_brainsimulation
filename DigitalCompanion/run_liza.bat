@echo off
chcp 65001 >nul 2>&1
chcp 1251 >nul 2>&1
title Liza - Digital Companion

echo.
echo ====================================================
echo            LIZA - Digital Companion v2.0
echo ====================================================
echo.

cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Run setup_all.bat
    pause
    exit /b 1
)

:: Check Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not responding! Run: ollama serve
    pause
    exit /b 1
)

echo [OK] System ready
echo.
echo ====================================================
echo   Select mode:
echo.
echo   [1] Voice mode       - Speak with Liza
echo   [2] Text chat        - Type messages
echo   [3] Voice + Camera   - Liza sees your emotions
echo   [4] Avatar demo      - View emotions
echo   [5] Setup
echo   [Q] Exit
echo ====================================================
echo.

set /p choice="Your choice: "

if "%choice%"=="" set choice=1
if /i "%choice%"=="1" goto voice
if /i "%choice%"=="2" goto chat
if /i "%choice%"=="3" goto camera
if /i "%choice%"=="4" goto demo
if /i "%choice%"=="5" goto setup
if /i "%choice%"=="q" goto end

:voice
echo.
echo ====================================================
echo   VOICE MODE
echo   Speak with Liza - she hears and responds
echo   Ctrl+C to exit
echo ====================================================
echo.
python live_liza.py --mode voice
goto end

:chat
echo.
echo ====================================================
echo   TEXT CHAT
echo ====================================================
echo.
python live_liza.py --mode chat --no-voice
goto end

:camera
echo.
echo ====================================================
echo   VOICE + CAMERA MODE
echo ====================================================
echo.
python live_liza.py --mode voice --camera
goto end

:demo
echo.
echo ====================================================
echo   AVATAR DEMO
echo ====================================================
echo.
python live_liza.py --mode demo --no-voice
goto end

:setup
echo.
call setup_all.bat
goto end

:end
echo.
pause

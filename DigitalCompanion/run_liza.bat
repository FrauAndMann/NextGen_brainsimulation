@echo off
title Liza

echo ====================================================
echo            LIZA - Voice Companion v2.1
echo ====================================================
echo.

cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
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
echo   Speak to Liza. She listens and responds with voice.
echo   Say "exit" or "quit" to stop.
echo ====================================================
echo.

python simple_liza.py

echo.
pause

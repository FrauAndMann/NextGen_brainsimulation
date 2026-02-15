@echo off
chcp 65001 >nul 2>&1
chcp 1251 >nul 2>&1
title ANIMA - Setup

echo.
echo ====================================================
echo         ANIMA v2.0 - Installing dependencies
echo ====================================================
echo.

:: Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python not installed!
    echo Download Python 3.10+ from https://python.org
    start "" "https://www.python.org/downloads/"
    pause
    exit /b 1
)
python --version
echo        OK!

:: Check pip
echo [2/5] Updating pip...
python -m pip install --upgrade pip --quiet 2>nul
echo        OK!

:: Install ffmpeg
echo [3/5] Checking ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ffmpeg not found. Installing...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [WARNING] Could not install ffmpeg automatically!
        echo Download from https://www.gyan.dev/ffmpeg/builds/
        echo Extract to C:\ffmpeg and add C:\ffmpeg\bin to PATH
        echo.
        start "" "https://www.gyan.dev/ffmpeg/builds/"
        pause
    )
) else (
    echo        OK!
)

:: Install Python packages
echo [4/5] Installing Python packages...
echo        (this may take a minute)

pip install numpy requests edge-tts --quiet 2>nul
pip install sounddevice soundfile --quiet 2>nul
pip install SpeechRecognition --quiet 2>nul
pip install openai --quiet 2>nul
pip install customtkinter --quiet 2>nul
pip install opencv-python --quiet 2>nul

echo        OK!

:: Check Ollama connection
echo [5/5] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo [WARNING] Ollama not responding on localhost:11434
    echo Make sure Ollama is running: ollama serve
    echo.
) else (
    echo        OK! Server found.
)

echo.
echo ====================================================
echo               INSTALLATION COMPLETE!
echo ====================================================
echo.
echo Now run run_liza.bat
echo.
pause

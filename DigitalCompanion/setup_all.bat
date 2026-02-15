@echo off
chcp 65001 >nul
title ANIMA - Установка зависимостей

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║         ANIMA v2.0 - Установка зависимостей                ║
echo ║         Живой Цифровой Компаньон Лиза                      ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: Проверка Python
echo [1/5] Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Python не установлен!
    echo Скачайте Python 3.10+ с https://python.org
    start "" "https://www.python.org/downloads/"
    pause
    exit /b 1
)
python --version
echo         OK!

:: Проверка pip
echo [2/5] Обновление pip...
python -m pip install --upgrade pip --quiet 2>nul
echo         OK!

:: Установка ffmpeg
echo [3/5] Проверка ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ffmpeg не найден. Устанавливаю...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ВАЖНО] Не удалось установить ffmpeg автоматически!
        echo Скачайте с https://www.gyan.dev/ffmpeg/builds/
        echo Распакуйте в C:\ffmpeg и добавьте C:\ffmpeg\bin в PATH
        echo.
        start "" "https://www.gyan.dev/ffmpeg/builds/"
        pause
    )
) else (
    echo         OK!
)

:: Установка Python пакетов
echo [4/5] Установка Python пакетов...
echo         (это займёт минуту)

pip install numpy requests edge-tts --quiet 2>nul
pip install sounddevice soundfile --quiet 2>nul
pip install SpeechRecognition --quiet 2>nul
pip install openai --quiet 2>nul
pip install customtkinter --quiet 2>nul
pip install opencv-python --quiet 2>nul

echo         OK!

:: Проверка связи с Ollama (уже должно быть запущено)
echo [5/5] Проверка Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ВНИМАНИЕ] Ollama не отвечает на localhost:11434
    echo Убедитесь что Ollama запущена: ollama serve
    echo.
) else (
    echo         OK! Сервер найден.
)

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                 УСТАНОВКА ЗАВЕРШЕНА!                       ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Теперь запустите run_liza.bat
echo.
pause

@echo off
chcp 65001 >nul
title ANIMA - Полная установка

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║         ANIMA v2.0 - Полная установка                      ║
echo ║         Живой Цифровой Компаньон Лиза                      ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: Проверка Python
echo [1/7] Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Python не установлен!
    echo.
    echo Скачайте и установите Python 3.10+ с https://python.org
    echo При установке ОБЯЗАТЕЛЬНО отметьте "Add Python to PATH"
    echo.
    start "" "https://www.python.org/downloads/"
    pause
    exit /b 1
)
python --version
echo         OK!

:: Проверка pip
echo [2/7] Проверка pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Обновление pip...
    python -m ensurepip --default-pip
    python -m pip install --upgrade pip
)
echo         OK!

:: Установка ffmpeg (нужен для Whisper)
echo [3/7] Проверка ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ffmpeg не найден. Устанавливаю через winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ВАЖНО] Не удалось установить ffmpeg автоматически!
        echo.
        echo Установите вручную:
        echo 1. Скачайте с https://www.gyan.dev/ffmpeg/builds/
        echo 2. Распакуйте в C:\ffmpeg
        echo 3. Добавьте C:\ffmpeg\bin в PATH
        echo.
        start "" "https://www.gyan.dev/ffmpeg/builds/"
        pause
    )
) else (
    echo         OK!
)

:: Установка основных зависимостей
echo [4/7] Установка Python-пакетов...
echo         Это может занять несколько минут...

:: Основные пакеты
pip install numpy requests edge-tts --quiet 2>nul

:: Аудио
pip install sounddevice soundfile --quiet 2>nul

:: Whisper для распознавания речи (локально)
pip install openai-whisper --quiet 2>nul

:: SpeechRecognition как fallback
pip install SpeechRecognition --quiet 2>nul

:: OpenAI совместимый клиент
pip install openai --quiet 2>nul

echo         OK!

:: Проверка Ollama
echo [5/7] Проверка Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo Ollama не запущена. Проверяю установку...

    where ollama >nul 2>&1
    if errorlevel 1 (
        echo.
        echo [ВАЖНО] Ollama не установлена!
        echo.
        echo Скачайте с https://ollama.com/download
        echo После установки выполните: ollama pull dolphin-mistral:7b
        echo.
        start "" "https://ollama.com/download/windows"
        pause
    ) else (
        echo Запускаю Ollama...
        start "" ollama serve
        timeout /t 5 >nul
    )
) else (
    echo         OK! Сервер запущен.
)

:: Проверка модели
echo [6/7] Проверка модели dolphin-mistral:7b...
ollama list 2>nul | findstr "dolphin-mistral" >nul
if errorlevel 1 (
    echo.
    echo Модель не найдена. Загружаю dolphin-mistral:7b...
    echo         Это займет несколько минут (4.1 GB)...
    echo.
    ollama pull dolphin-mistral:7b
)
echo         OK!

:: Установка опциональных пакетов
echo [7/7] Установка опциональных пакетов...

:: CustomTkinter для красивого GUI
pip install customtkinter --quiet 2>nul

:: OpenCV для камеры
pip install opencv-python --quiet 2>nul

:: DeepFace для детекции эмоций (опционально, большое)
:: pip install deepface --quiet 2>nul

echo         OK!

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                 УСТАНОВКА ЗАВЕРШЕНА!                       ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo Теперь можно запускать Лизу!
echo.
pause

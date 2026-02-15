@echo off
chcp 65001 >nul
title ANIMA - Цифровой Компаньон

echo ========================================
echo    ANIMA - Цифровой Компаньон v2.0
echo ========================================
echo.

:: Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден. Установите Python 3.10+
    pause
    exit /b 1
)

:: Проверка Ollama
echo [1/3] Проверка Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ПРЕДУПРЕЖДЕНИЕ] Ollama не запущена. Запустите 'ollama serve'
    echo.
    start "" "https://ollama.com/download"
    pause
)

:: Установка зависимостей
echo [2/3] Проверка зависимостей...
pip install -q numpy requests edge-tts 2>nul
pip install -q customtkinter 2>nul

:: Запуск
echo [3/3] Запуск ANIMA...
echo.
echo Выберите режим:
echo   1. Консольный режим (рекомендуется для первого запуска)
echo   2. GUI режим (с аватаром)
echo   3. Только аватар (демо эмоций)
echo.

set /p choice="Ваш выбор [1]: "

if "%choice%"=="" set choice=1
if "%choice%"=="1" (
    python unified_anima.py --model dolphin-mistral:7b
) else if "%choice%"=="2" (
    python anima_app.py --model dolphin-mistral:7b
) else if "%choice%"=="3" (
    python -c "from avatar.advanced_avatar import AdvancedAvatar; a = AdvancedAvatar('Лиза'); a.start()"
) else (
    echo Неверный выбор. Запуск консольного режима...
    python unified_anima.py --model dolphin-mistral:7b
)

pause

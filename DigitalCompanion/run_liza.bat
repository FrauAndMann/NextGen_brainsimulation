@echo off
chcp 65001 >nul
title Лиза - Живой Цифровой Компаньон

:: Цвета (если поддерживаются)
color 0B

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║              ЛИЗА - Живой Цифровой Компаньон               ║
echo ║                       v2.0                                 ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

:: Переход в директорию скрипта
cd /d "%~dp0"

:: Быстрая проверка
echo Проверка системы...

:: Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python не найден! Запустите setup_all.bat
    pause
    exit /b 1
)

:: Ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [!] Ollama не запущена, запускаю...
    start /min ollama serve
    timeout /t 3 >nul
)

:: ffmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [!] ffmpeg не найден - распознавание речи может не работать
    echo     Запустите setup_all.bat для установки
)

echo [OK] Система готова
echo.
echo ══════════════════════════════════════════════════════════════
echo   Выберите режим:
echo.
echo   [1] Голосовой режим        - Говорите с Лизой голосом
echo   [2] Текстовый чат          - Пишите сообщения
echo   [3] Голосовой + Камера     - Лиза видит ваши эмоции
echo   [4] Демо аватара          - Посмотреть эмоции
echo   [5] Настройка/Установка    - Установить зависимости
echo   [Q] Выход
echo ══════════════════════════════════════════════════════════════
echo.

set /p choice="Ваш выбор: "

if "%choice%"=="" set choice=1
if /i "%choice%"=="1" goto voice
if /i "%choice%"=="2" goto chat
if /i "%choice%"=="3" goto camera
if /i "%choice%"=="4" goto demo
if /i "%choice%"=="5" goto setup
if /i "%choice%"=="q" goto end
if /i "%choice%"=="й" goto voice
if /i "%choice%"=="ц" goto chat
if /i "%choice%"=="ы" goto camera
if /i "%choice%"=="в" goto demo
if /i "%choice%"=="й" goto end

:voice
echo.
echo ══════════════════════════════════════════════════════════════
echo   ГОЛОСОВОЙ РЕЖИМ
echo   Говорите с Лизой - она слышит и отвечает голосом
echo   Нажмите Ctrl+C для выхода
echo ══════════════════════════════════════════════════════════════
echo.
python live_liza.py --mode voice
goto end

:chat
echo.
echo ══════════════════════════════════════════════════════════════
echo   ТЕКСТОВЫЙ ЧАТ
echo   Пишите сообщения, Лиза отвечает текстом
echo ══════════════════════════════════════════════════════════════
echo.
python live_liza.py --mode chat --no-voice
goto end

:camera
echo.
echo ══════════════════════════════════════════════════════════════
echo   ГОЛОСОВОЙ РЕЖИМ + КАМЕРА
echo   Лиза видит ваше лицо и реагирует на эмоции
echo ══════════════════════════════════════════════════════════════
echo.
python live_liza.py --mode voice --camera
goto end

:demo
echo.
echo ══════════════════════════════════════════════════════════════
echo   ДЕМО АВАТАРА
echo   Нажимайте на аватар для смены эмоции
echo ══════════════════════════════════════════════════════════════
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

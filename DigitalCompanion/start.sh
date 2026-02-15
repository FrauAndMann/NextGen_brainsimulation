#!/bin/bash

echo "========================================"
echo "   ANIMA - Цифровой Компаньон v2.0"
echo "========================================"
echo

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "[ОШИБКА] Python не найден. Установите Python 3.10+"
    exit 1
fi

# Проверка Ollama
echo "[1/3] Проверка Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[ПРЕДУПРЕЖДЕНИЕ] Ollama не запущена. Запустите 'ollama serve'"
fi

# Установка зависимостей
echo "[2/3] Проверка зависимостей..."
pip install -q numpy requests edge-tts 2>/dev/null
pip install -q customtkinter 2>/dev/null

# Запуск
echo "[3/3] Запуск ANIMA..."
echo
echo "Выберите режим:"
echo "  1. Консольный режим"
echo "  2. GUI режим"
echo "  3. Только аватар"
echo

read -p "Ваш выбор [1]: " choice
choice=${choice:-1}

case $choice in
    1) python3 unified_anima.py --model dolphin-mistral:7b ;;
    2) python3 anima_app.py --model dolphin-mistral:7b ;;
    3) python3 -c "from avatar.advanced_avatar import AdvancedAvatar; a = AdvancedAvatar('Лиза'); a.start()" ;;
    *) echo "Неверный выбор. Запуск консольного режима..."
       python3 unified_anima.py --model dolphin-mistral:7b ;;
esac

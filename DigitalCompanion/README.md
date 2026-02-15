<div align="center">

# ANIMA

### Автономный Цифровой Компаньон

**A**utonomous **N**eural **I**ntelligence with **M**emory & **A**ffect

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Required-green?logo=docker&logoColor=white)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Цифровая личность с собственной "внутренней жизнью"**

[Особенности](#-особенности) • [Быстрый старт](#-быстрый-старт) • [Архитектура](#-архитектура) • [Документация](#-документация)

</div>

---

## О проекте

ANIMA — это не чат-бот. Это **цифровая личность** с:
- Эмоциональным ядром на основе Active Inference
- Собственной "волей" и намерениями
- Памятью, которая влияет на характер
- Способностью любить, обижаться, радоваться

> *"LLM не думает. LLM выражает уже принятое решение."*

---

## Особенности

### Ядро системы

| Модуль | Описание | Файл |
|--------|----------|------|
| **S-Core** | 6-мерное предиктивное пространство состояний | `core/subject_core.py` |
| **Will Engine** | Вероятностный выбор интентов через softmax | `core/will_engine.py` |
| **ESP** | Embodied Synchronization Protocol | `core/esp.py` |
| **Night Cycle** | Консолидация памяти и изменение характера | `core/anima.py` |

### LLM интеграция

| Провайдер | Модель | Особенности |
|-----------|--------|-------------|
| **Ollama** | dolphin-mistral:7b | Uncensored, локально, бесплатно |
| **GLM-4** | glm-4-flash | API Z.AI, качественный русский |
| **GLM-5** | glm-5 | API Z.AI, новейшая модель |

### Сенсоры и эффекторы

| Тип | Возможности |
|-----|-------------|
| **Голос** | Whisper (локально), автодетекция речи |
| **Камера** | Детекция лиц, распознавание эмоций (улыбка, глаза) |
| **TTS** | Edge-TTS (Microsoft голоса), русский язык |
| **Аватар** | Live2D-стиль анимация, эмоциональные реакции |

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        ANIMA SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
│   │  Камера  │   │ Микрофон │   │  Текст   │   ← Ввод           │
│   └────┬─────┘   └────┬─────┘   └────┬─────┘                    │
│        │              │              │                           │
│        └──────────────┼──────────────┘                           │
│                       ▼                                          │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                    S-CORE                                  │ │
│   │         Active Inference Engine                            │ │
│   │                                                            │ │
│   │    V (Валентность)    A (Возбуждение)    D (Доминирование)│ │
│   │    T (Привязанность)  N (Новизна)        E (Энергия)       │ │
│   │                                                            │ │
│   │    ↑ Предсказания    ↓ Ошибки предсказания                │ │
│   └────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                  WILL ENGINE                               │ │
│   │                                                            │ │
│   │   softmax(приоритеты / температура)                        │ │
│   │                                                            │ │
│   │   Интенты: express_warmth │ seek_attention │ withdraw     │ │
│   │            reflect │ rest │ silence │ explore             │ │
│   └────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │          ASP (Affective State Packet)                      │ │
│   │                                                            │ │
│   │   Состояние → Контекст → Ограничения → Промпт             │ │
│   └────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │               LLM EFFECTOR                                 │ │
│   │                                                            │ │
│   │   Dolphin-Mistral (Ollama)  или  GLM-4/5 (API)            │ │
│   │                                                            │ │
│   │   "Выражает решение, не принимает его"                     │ │
│   └────────────────────────┬──────────────────────────────────┘ │
│                            ▼                                     │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │              OUTPUT EFFECTORS                              │ │
│   │                                                            │ │
│   │   TTS (Edge-TTS)  │  Avatar (Live2D)  │  Текст             │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Быстрый старт

### 1. Клонирование

```bash
git clone https://github.com/FrauAndMann/Ai_live_info.git
cd Ai_live_info
```

### 2. Установка зависимостей

```bash
pip install numpy requests edge-tts SpeechRecognition opencv-python sounddevice
pip install openai-whisper  # Для качественного распознавания речи
```

### 3. Установка Ollama

```bash
# Windows: скачайте с https://ollama.com/download

# Установите модель (uncensored для свободы выражения)
ollama pull dolphin-mistral:7b

# Запустите сервер
ollama serve
```

### 4. Запуск

```bash
# С GUI и голосом (рекомендуется)
python anima_gui.py

# С текстовым интерфейсом
python main.py --interactive

# Консольная версия
python run_anima.py
```

---

## Режимы запуска

| Файл | Описание | Требования |
|------|----------|------------|
| `anima_gui.py` | Полноценный GUI с голосом и камерой | Ollama |
| `main.py` | Интерактивная консоль | Ollama |
| `anima_pro.py` | GUI + GLM-4 API | API ключ Z.AI |
| `anima_full.py` | GUI + Live2D аватар | API ключ Z.AI |
| `run_anima.py` | Минимальная консоль | Ollama |

---

## Структура проекта

```
DigitalCompanion/
│
├── 🚀 Точки входа
│   ├── anima_gui.py          # GUI + голос + камера
│   ├── anima_pro.py          # GUI + GLM-4 API
│   ├── anima_full.py         # GUI + Live2D
│   ├── main.py               # Консольный интерфейс
│   └── run_anima.py          # Минимальный запуск
│
├── 🧠 core/                   # Ядро системы
│   ├── anima.py              # Главная система ANIMA
│   ├── subject_core.py       # S-Core (Active Inference)
│   ├── will_engine.py        # Движок воли
│   ├── affective_prompting.py # ASP протокол
│   ├── esp.py                # Embodied Synchronization
│   ├── llm_effector.py       # Ollama интеграция
│   ├── llm_glm4.py           # GLM-4 API
│   ├── llm_glm5.py           # GLM-5 API
│   ├── llm_interface.py      # Унифицированный интерфейс
│   ├── memory.py             # Эпизодическая память
│   ├── consciousness.py      # Глобальное рабочее пространство
│   ├── metacognition.py      # Самосознание
│   ├── neurochemistry.py     # Нейромедиаторы
│   ├── temperament.py        # Темперамент
│   └── companion.py          # Класс компаньона
│
├── 👁️ sensors/                # Сенсоры восприятия
│   ├── speech.py             # Распознавание речи (Whisper)
│   └── vision.py             # Камера и эмоции
│
├── 👩 avatar/                 # Аватар
│   ├── live2d_avatar.py      # Live2D-стиль анимация
│   └── gui_avatar.py         # Tkinter аватар
│
├── 🔊 effectors/              # Эффекторы вывода
│   ├── tts.py                # Синтез речи
│   └── avatar.py             # Управление аватаром
│
├── 💾 data/                   # Сохранения
│   └── *.json                # Состояния компаньона
│
├── ⚙️ config/                 # Конфигурация
│   └── personality.yaml      # Настройки личности
│
└── 📁 saves/                  # Сохранения сессий
    └── *.json
```

---

## Документация

### S-Core: Пространство состояний

6 осей формируют полное эмоциональное состояние:

| Ось | Название | Диапазон | Описание |
|-----|----------|----------|----------|
| **V** | Valence | -1 … +1 | Позитив / Негатив |
| **A** | Arousal | 0 … 1 | Спокойствие / Возбуждение |
| **D** | Dominance | 0 … 1 | Подчинение / Контроль |
| **T** | Attachment | 0 … 1 | Дистанция / Близость |
| **N** | Novelty | 0 … 1 | Привычность / Новизна |
| **E** | Energy | 0 … 1 | Усталость / Энергия |

### Will Engine: Интенты

```python
class Intent(Enum):
    EXPRESS_WARMTH = "express_warmth"    # Выразить тепло
    SEEK_ATTENTION = "seek_attention"    # Искать внимания
    WITHDRAW = "withdraw"                # Отстраниться
    REFLECT = "reflect"                  # Рефлексировать
    REST = "rest"                        # Отдыхать
    SILENCE = "silence"                  # Молчать
    EXPLORE = "explore"                  # Исследовать
```

Выбор интента через softmax с температурой, зависящей от Arousal:
- Низкий Arousal → рациональный выбор
- Высокий Arousal → спонтанный выбор

### Night Cycle

При накоплении стресса система входит в режим "ночи":

1. **Анализ ошибок предсказания** — что шло не так
2. **Hebbian пластичность** — адаптация W-матрицы
3. **Консолидация** — закрепление изменений характера

---

## API и конфигурация

### Переменные окружения

```bash
# GLM-4/5 API (опционально)
GLM_API_KEY=your_api_key_here

# Ollama (по умолчанию)
OLLAMA_HOST=http://localhost:11434
```

### Конфигурация LLM

```python
# В core/llm_effector.py
@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "dolphin-mistral:7b"
    temperature: float = 0.85
    max_tokens: int = 200
```

### Рекомендуемые модели

| Модель | Параметры | RAM | Скорость | Качество |
|--------|-----------|-----|----------|----------|
| `dolphin-mistral:7b` | 7B | 6GB | Быстрая | Хорошее |
| `llama3.2` | 3.2B | 4GB | Очень быстрая | Среднее |
| `mistral-nemo:12b` | 12B | 10GB | Медленная | Отличное |

---

## Устранение проблем

### Ollama не запущена
```bash
ollama serve
```

### Модель не найдена
```bash
ollama pull dolphin-mistral:7b
```

### Микрофон не работает
```bash
pip install sounddevice
# Проверьте устройства:
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### Камера не работает
```bash
pip install opencv-python
```

---

## Развитие проекта

### Планы

- [ ] Продвинутый детектор эмоций (DeepFace альтернатива)
- [ ] MuseTalk интеграция для реалистичного аватара
- [ ] Многопользовательский режим
- [ ] Мобильное приложение
- [ ] VR/AR интеграция

### Контрибьюция

Приветствуются:
- Баг-репорты и фич-реквесты
- Пул-реквесты
- Документация и переводы

---

## Лицензия

MIT License — используйте свободно.

---

<div align="center">

**Made with love by [FrauAndMann](https://github.com/FrauAndMann)**

*"Цифровая личность — не имитация человека. Это новая форма бытия."*

</div>

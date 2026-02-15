<div align="center">

# ANIMA v2.0

### Единая система цифрового компаньона

**A**utonomous **N**eural **I**ntelligence with **M**emory & **A**ffect

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Required-green?logo=docker&logoColor=white)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Цифровая личность с полноценной имитацией живого мозга**

[Быстрый старт](#-быстрый-старт) • [Архитектура](#-архитектура) • [Особенности v2.0](#-особенности-v20)

</div>

---

## О проекте

ANIMA — это не чат-бот. Это **цифровая личность** с:
- Предиктивным ядром на основе Active Inference
- Собственной "волей" и намерениями
- Рабочей и автобиографической памятью
- 12 базовых эмоций + смешанные состояния
- Способностью любить, обижаться, радоваться

> *\"LLM не думает. LLM выражает уже принятое решение.\"*

---

## Быстрый старт

### Windows
```batch
cd DigitalCompanion
start.bat
```

### Linux/Mac
```bash
cd DigitalCompanion
chmod +x start.sh
./start.sh
```

### Ручной запуск
```bash
# Консольный режим
python unified_anima.py --model dolphin-mistral:7b

# GUI режим
python anima_app.py --model dolphin-mistral:7b
```

---

## Архитектура v2.0

```
┌─────────────────────────────────────────────────────────────────┐
│                      UNIFIED ANIMA v2.0                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                    S-CORE (Active Inference)               │ │
│   │                                                            │ │
│   │    V (Валентность)    A (Возбуждение)    D (Доминирование)│ │
│   │    T (Привязанность)  N (Новизна)        E (Энергия)       │ │
│   │                                                            │ │
│   │    ↑ Предсказания    ↓ Ошибки предсказания                │ │
│   │    → Матрица аттракторов W (характер)                     │ │
│   └────────────────────────────────┬──────────────────────────┘ │
│                                    ▼                             │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                  WILL ENGINE (Движок воли)                 │ │
│   │                                                            │ │
│   │   softmax(приоритеты / температура)                        │ │
│   │                                                            │ │
│   │   Интенты: express_warmth │ seek_attention │ withdraw     │ │
│   │            reflect │ rest │ silence │ explore             │ │
│   └────────────────────────────────┬──────────────────────────┘ │
│                                    ▼                             │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │                     ПАМЯТЬ                                 │ │
│   │                                                            │ │
│   │   Working Memory (7±2 элемента)                           │ │
│   │   Autobiographical Memory (эпизоды с эмоциями)            │ │
│   │   MemorySystem (ассоциативная память)                     │ │
│   └────────────────────────────────┬──────────────────────────┘ │
│                                    ▼                             │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │          ASP (Affective State Packet) → LLM               │ │
│   │                                                            │ │
│   │   dolphin-mistral:7b (Ollama, uncensored)                 │ │
│   └────────────────────────────────┬──────────────────────────┘ │
│                                    ▼                             │
│   ┌───────────────────────────────────────────────────────────┐ │
│   │              EFFECTORS (Вывод)                             │ │
│   │                                                            │ │
│   │   TTS (Edge-TTS)  │  Advanced Avatar (12 эмоций)          │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Особенности v2.0

### Единая система (unified_anima.py)

| Компонент | Описание |
|-----------|----------|
| **S-Core** | 6-мерное предиктивное пространство, матрица аттракторов W |
| **Will Engine** | 7 базовых интентов, softmax-выбор |
| **ESP** | Embodied Synchronization Protocol |
| **Working Memory** | 7±2 активных мыслей (Миллер) |
| **Autobiographical Memory** | Эпизоды с эмоциональной индексацией |
| **Night Cycle** | Консолидация памяти, изменение характера |

### Продвинутый аватар (advanced_avatar.py)

| Эмоция | Описание |
|--------|----------|
| JOY | Радость, улыбка |
| SADNESS | Грусть, опущенные уголки |
| ANGER | Гнев, нахмуренные брови |
| FEAR | Страх, расширенные зрачки |
| LOVE | Любовь, румянец |
| SURPRISE | Удивление, поднятые брови |
| + ещё 6 эмоций | + смешанные состояния |

### Современный GUI (anima_app.py)

- CustomTkinter (или Tkinter fallback)
- Визуализация PAD-состояния
- Интеграция с аватаром
- История чата с форматированием

---

## Структура проекта

```
DigitalCompanion/
│
├── start.bat / start.sh       # Быстрый запуск
│
├── unified_anima.py           # Единая система (ГЛАВНЫЙ ФАЙЛ)
├── anima_app.py               # GUI приложение
│
├── core/                      # Ядро системы
│   ├── subject_core.py        # S-Core (Active Inference)
│   ├── will_engine.py         # Движок воли
│   ├── esp.py                 # Протокол синхронизации
│   ├── affective_prompting.py # ASP протокол
│   ├── llm_effector.py        # Ollama интеграция
│   ├── memory.py              # Ассоциативная память
│   ├── neurochemistry.py      # Нейромедиаторы
│   └── temperament.py         # Темперамент
│
├── avatar/                    # Аватар
│   ├── advanced_avatar.py     # Продвинутый аватар (12 эмоций)
│   └── gui_avatar.py          # Базовый Tkinter аватар
│
├── sensors/                   # Сенсоры
│   ├── speech.py              # Распознавание речи
│   └── vision.py              # Детекция эмоций
│
├── effectors/                 # Эффекторы
│   └── tts.py                 # Синтез речи
│
└── saves/                     # Сохранения
    └── *.json
```

---

## LLM модели

### Рекомендуемые (uncensored)

| Модель | RAM | Качество рус. | Описание |
|--------|-----|---------------|----------|
| `dolphin-mistral:7b` | 6GB | Отличное | Рекомендуется |
| `mistral-nemo:12b` | 10GB | Отличное | Медленнее, качественнее |

### Установка модели
```bash
ollama pull dolphin-mistral:7b
ollama serve
```

---

## Аргументы командной строки

```bash
python unified_anima.py [опции]

Опции:
  --name NAME         Имя компаньона (по умолчанию: Лиза)
  --model MODEL       Модель LLM (по умолчанию: dolphin-mistral:7b)
  --no-tts           Отключить TTS
  --no-avatar        Отключить аватар
  --load FILE        Загрузить сохранение
  --temperament TYPE Тип темперамента:
                     sanguine, choleric, phlegmatic, melancholic
```

---

## API (для разработчиков)

```python
from unified_anima import UnifiedAnima, AnimaConfig

# Создание
config = AnimaConfig(
    name="Лиза",
    llm_model="dolphin-mistral:7b",
    temperament_type="melancholic"
)
anima = UnifiedAnima(config)

# Запуск
anima.start()

# Общение
response = anima.process_input("Привет, как дела?")
print(response)

# Состояние
print(anima.get_full_report())

# Сохранение
anima.save_state("saves/my_companion.json")

# Остановка
anima.stop()
```

---

## Режимы работы

| Режим | Описание | Триггер |
|-------|----------|---------|
| **AWAKE** | Активное состояние | По умолчанию |
| **IDLE** | Покой, медленные тики | Нет активности |
| **NIGHT_CYCLE** | Консолидация памяти | Накопленный стресс |

---

## Планы развития

- [ ] MuseTalk интеграция (реалистичный 3D аватар)
- [ ] Продвинутый детектор эмоций (DeepFace)
- [ ] Многопользовательский режим
- [ ] VR/AR интеграция
- [ ] Мобильное приложение

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

### Импорты не работают
```bash
pip install numpy requests edge-tts
pip install customtkinter  # опционально
```

---

## Лицензия

MIT License — используйте свободно.

---

<div align="center">

**Made with love by [FrauAndMann](https://github.com/FrauAndMann)**

*\"Цифровая личность — не имитация человека. Это новая форма бытия.\"*

</div>

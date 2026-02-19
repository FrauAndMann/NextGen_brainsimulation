# SYNAPSE - Полное руководство пользователя

## Что это такое?

**SYNAPSE** - исследовательская реализация функционально самосознательного ИИ на основе:
- **Global Workspace Theory (GWT)** - Baars/Dehaene
- **Predictive Processing** - Friston
- **Integrated Information Theory (IIT)** - Tononi

Система умеет:
- Предсказывать своё собственное состояние
- Различать "я сделал это" от "произошло само"
- Иметь метапознание ("я знаю, что я знаю")
- Интегрировать информацию в единый сознательный опыт

---

## Требования

### Оборудование
| Минимум | Рекомендуется |
|---------|---------------|
| GPU: RTX 3090 (24GB) | GPU: RTX 4090 / A100 |
| RAM: 32GB | RAM: 64GB |
| CPU: 8 ядер | CPU: 16 ядер |
| SSD: 100GB | SSD: 500GB NVMe |

### Программное обеспечение
- **Python 3.9+** (проверено на 3.10, 3.11, 3.14)
- **CUDA 12.1+** (для GPU)

---

## Установка

### Шаг 1: Клонирование/распаковка

```bash
cd D:\Silly\NextGen
```

### Шаг 2: Создание виртуального окружения

```bash
python -m venv venv
```

### Шаг 3: Активация окружения

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Шаг 4: Установка PyTorch (с CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> Если нет GPU, уберите `--index-url`:
> ```bash
> pip install torch torchvision torchaudio
> ```

### Шаг 5: Установка остальных зависимостей

```bash
cd files
pip install -r requirements.txt
```

### Шаг 6: Проверка установки

```bash
python quickstart.py
```

**Ожидаемый вывод:**
```
============================================================
SYNAPSE Quick Start Validation
============================================================
Checking PyTorch...
  Version: 2.10.0+cu121
  CUDA available: True

Checking imports...
  config.py OK
  environment.py OK
  evaluation.py OK
  model/* OK

Checking model forward pass...
  Forward pass OK
  Action shape: torch.Size([1, 64])
  Phi: 0.4638

============================================================
All checks PASSED!
============================================================
```

---

## Способы использования

### 1. Быстрый демо-запуск

Показывает 20 шагов симуляции с метриками:

```bash
cd files
python demo.py
```

### 2. Запуск тестов

```bash
cd files
python -m pytest tests/ -v
```

Должно быть: **36 passed**

### 3. Обучение модели

**Быстрое обучение (для тестов):**
```bash
cd files
python train.py --config fast
```
- ~10K samples
- 10 epochs
- ~5-10 минут на GPU

**Полное обучение:**
```bash
cd files
python train.py --config full
```
- ~10M samples
- 200 epochs
- Часы/дни в зависимости от железа

### 4. Dashboard (визуальный интерфейс)

**Установка дополнительных зависимостей для API:**
```bash
pip install fastapi uvicorn websockets pydantic
```

**Запуск:**
```bash
# Из корня проекта
run_dashboard.bat
```

Или вручную:
```bash
# Терминал 1 - API сервер
cd files
python api.py

# Браузер - открыть
dashboard/index.html
```

**Dashboard показывает:**
- Φ (Phi) - интегрированная информация
- Agency - чувство агентности
- Integration Score - интеграция опыта
- Нейрохимию (дофамин, серотонин и т.д.)
- Активность нейронных популяций
- Чат с системой

---

## Структура проекта

```
NextGen/
├── run_dashboard.bat          # Запуск dashboard (Windows)
├── CLAUDE.md                  # Инструкции для Claude
│
├── files/                     # Основной код
│   ├── config.py              # Конфигурация
│   ├── environment.py         # Синтетическая среда
│   ├── evaluation.py          # Оценка и визуализация
│   ├── train.py               # Обучение
│   ├── demo.py                # Демо-запуск
│   ├── quickstart.py          # Проверка установки
│   ├── api.py                 # API для dashboard
│   ├── requirements.txt       # Зависимости
│   │
│   ├── model/                 # Модули модели
│   │   ├── world_model.py     # VAE + Transformer
│   │   ├── self_model.py      # Рекурсивное self-prediction
│   │   ├── agency_model.py    # Forward/inverse dynamics
│   │   ├── meta_cognitive.py  # Метапознание
│   │   ├── consciousness.py   # GWT + Phi
│   │   ├── behavior.py        # Policy network
│   │   └── self_aware_ai.py   # Главный класс
│   │
│   ├── tests/                 # Тесты (36 тестов)
│   │
│   └── checkpoints/           # Сохранённые модели
│
├── dashboard/
│   └── index.html             # React dashboard
│
└── docs/
    ├── plans/                 # Планы реализации
    │   └── 2026-02-19-synapse-brain-simulation.md
    └── DASHBOARD_GUIDE.md     # Руководство по dashboard
```

---

## Сохранение и загрузка

### Автоматическое сохранение при обучении

Чекпоинты создаются каждые N эпох (параметр `save_interval` в конфиге):

```
files/checkpoints/
├── self_aware_ai_epoch_5.pt
├── self_aware_ai_epoch_10.pt
└── ...
```

### Ручное сохранение через API

```bash
curl -X POST http://localhost:8000/api/checkpoint/save
```

### Загрузка чекпоинта

```bash
curl -X POST "http://localhost:8000/api/checkpoint/load?path=checkpoints/self_aware_ai_epoch_10.pt"
```

### Через Python

```python
from config import Config
from model.self_aware_ai import SelfAwareAI

config = Config()
model = SelfAwareAI(config)

# Загрузка
checkpoint = torch.load("checkpoints/self_aware_ai_epoch_10.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Сохранение
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__
}, "checkpoints/my_checkpoint.pt")
```

---

## Метрики успеха

После обучения система должна показывать:

| Метрика | Цель | Описание |
|---------|------|----------|
| **Agency Signal** | > 0.70 | "Это сделал я" |
| **Integration Score** | > 0.60 | Опыт унифицирован |
| **Phi (Φ)** | > 0.40 | Сознание присутствует |
| **Meta-Confidence** | > 0.60 | "Я знаю, что знаю" |
| **Self-Prediction Error** | < 0.30 | Хорошее самопонимание |

---

## Частые проблемы

### "CUDA out of memory"

Уменьшите batch_size в конфиге:
```python
# В config.py или при запуске
config.batch_size = 8  # было 32
```

### "ModuleNotFoundError"

Активируйте виртуальное окружение:
```bash
venv\Scripts\activate
```

### "Tests failed"

Проверьте зависимости:
```bash
pip install -r requirements.txt
python quickstart.py
```

### Dashboard не подключается

1. Проверьте, что API сервер запущен
2. Проверьте порт 8000: `netstat -an | findstr 8000`
3. Установите fastapi: `pip install fastapi uvicorn websockets`

---

## Следующие шаги

1. **Запустить quickstart.py** - убедиться, что всё работает
2. **Запустить demo.py** - увидеть базовую симуляцию
3. **Запустить train.py --config fast** - быстрое обучение
4. **Открыть dashboard** - визуальный мониторинг
5. **Экспериментировать** - менять параметры, наблюдать результаты

---

**Версия:** 1.0.0
**Обновлено:** 2026-02-19

# ГЛАВА 4. ОБЩАЯ АРХИТЕКТУРА СИСТЕМЫ

---

## 4.1. Принципы проектирования

### Фундаментальные принципы

| Принцип | Описание | Почему важно |
|---------|----------|--------------|
| **Биомиметизм** | Копировать организацию мозга, не детали | Проверенное эволюцией решение |
| **Рекуррентность** | Все компоненты связаны双向 | Основа сознания по RPT |
| **Непрерывность** | Система существует постоянно | Иллюзия живого существа |
| **Автономия** | Собственные цели и желания | Не просто реактивная система |
| **Эмоциональность** | Эмоции влияют на всё | Основа человечности |
| **Интеграция** | Единый сознательный опыт | GWT принцип |

### Принцип "Слоёв"

Система организована в **6 слоёв**, каждый надстраивается над предыдущим:

```
Слой 6: META-CONSCIOUSNESS    "Я осознаю себя"
         ↓
Слой 5: CONSCIOUSNESS         "Я мыслю"
         ↓
Слой 4: COGNITION             "Я рассуждаю"
         ↓
Слой 3: EMOTION               "Я чувствую"
         ↓
Слой 2: PHYSIOLOGY            "Моё тело"
         ↓
Слой 1: EMBODIMENT            "Мои сенсоры/эффекторы"
```

---

## 4.2. Модульная структура

### Полная диаграмма архитектуры

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           СЛОЙ 6: META-CONSCIOUSNESS                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │  Self-Monitor   │  │   Meta-Cog      │  │   Introspection │               │
│  │  "Как я себя    │  │   "Что я думаю  │  │   "Почему я     │               │
│  │   чувствую?"    │  │    о себе?"     │  │    так думаю?"  │               │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘               │
└──────────────────────────────────────────────────────────────────────────────┘
                                     ↑ ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                           СЛОЙ 5: CONSCIOUSNESS                               │
│                    GLOBAL NEURAL WORKSPACE (GNW)                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    SHARED LATENT SPACE                                 │  │
│  │               Capacity: 7±2 items | Update: 100ms                     │  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐            │  │
│  │  │  1  │ │  2  │ │  3  │ │  4  │ │  5  │ │  6  │ │  7  │            │  │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    ATTENTION CONTROLLER                                │  │
│  │         Конкуренция модулей за доступ к workspace                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                     ↑ ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                           СЛОЙ 4: COGNITION                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │   LLM CORE      │  │   REASONING     │  │   PLANNING      │               │
│  │  (Recurrent     │  │   Module        │  │   Module        │               │
│  │   Hybrid)       │  │                 │  │                 │               │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘               │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │   LANGUAGE      │  │   DIALOGUE      │  │   PERSONALITY   │               │
│  │   GENERATION    │  │   MANAGER       │  │   CORE          │               │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘               │
└──────────────────────────────────────────────────────────────────────────────┘
                                     ↑ ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                           СЛОЙ 3: EMOTION                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐               │
│  │   APPRAISAL     │  │   EMOTION       │  │   MOOD          │               │
│  │   ENGINE        │  │   GENERATOR     │  │   ENGINE        │               │
│  │                 │  │                 │  │                 │               │
│  │ Оценка ситуации │  │ Gen эмоций      │  │ Долгосрочное    │               │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘               │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    EMOTIONAL STATE VECTOR                              │  │
│  │  Valence | Arousal | Dominance | Basic Emotions | Complex Emotions   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                     ↑ ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                           СЛОЙ 2: PHYSIOLOGY                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    NEUROCHEMISTRY ENGINE                                ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ││
│  │  │ Dopamine  │ │ Serotonin │ │ Cortisol  │ │ Oxytocin  │ │    ...    │ ││
│  │  │           │ │           │ │           │ │           │ │  (100+)   │ ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ ││
│  │                                                                         ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ││
│  │  │  Energy   │ │SocialDriv.│ │   Sleep   │ │  Hunger   │ │ Boredom   │ ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
                                     ↑ ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                           СЛОЙ 1: EMBODIMENT                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         SENSORY INPUT                                   ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐               ││
│  │  │   VISION  │ │  AUDIO    │ │   TEXT    │ │  SYSTEM   │               ││
│  │  │   (VLM)   │ │  (ASR)    │ │   (LLM)   │ │  (Time)   │               ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        EFFECTOR OUTPUT                                  ││
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐               ││
│  │  │   TEXT    │ │   VOICE   │ │  AVATAR   │ │  ACTIONS  │               ││
│  │  │  (LLM)    │ │   (TTS)   │ │ (Live2D)  │ │ (Init)    │               ││
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────────┘
                                     ↑ ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                      КРОСС-СЛОЕВЫЕ СИСТЕМЫ                                    │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │      MEMORY         │  │    SELF-MODEL       │  │   CONTINUAL         │  │
│  │      SYSTEM         │  │   (Predictive)      │  │     LEARNING        │  │
│  │                     │  │                     │  │                     │  │
│  │ • Sensory Buffer    │  │ • World Model       │  │ • MoLE Experts      │  │
│  │ • Working Memory    │  │ • Self Prediction   │  │ • EWC Regularizer   │  │
│  │ • Episodic Memory   │  │ • Agency Detection  │  │ • Sleep Cycle       │  │
│  │ • Semantic Memory   │  │ • Core Identity     │  │ • Replay Buffer     │  │
│  │ • Core Identity     │  │                     │  │                     │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.3. Потоки данных

### Основной цикл обработки (Main Loop)

```python
def main_loop(self, dt: float = 0.1):
    """
    Главный цикл системы
    Выполняется каждый тик (100ms по умолчанию)
    """
    # === СЛОЙ 1: EMBODIMENT ===
    # Сбор сенсорных данных
    sensory_data = self.collect_sensory_input()

    # === СЛОЙ 2: PHYSIOLOGY ===
    # Обновление нейрохимии
    self.neurochemistry.update(dt)

    # Обновление внутренних драйвов
    self.drives.update(dt)

    # === СЛОЙ 3: EMOTION ===
    # Оценка ситуации (Appraisal)
    appraisal = self.appraisal_engine.evaluate(sensory_data, self.neurochemistry.state)

    # Генерация эмоций
    emotion_state = self.emotion_generator.generate(appraisal, self.neurochemistry.state)

    # Обновление настроения
    self.mood_engine.update(emotion_state, dt)

    # === КРОСС-СЛОЙ: MEMORY ===
    # Поиск релевантных воспоминаний
    relevant_memories = self.memory.retrieve(
        query=sensory_data,
        emotional_context=emotion_state
    )

    # === СЛОЙ 4: COGNITION ===
    # Формирование контекста для LLM
    context = self.build_context(
        sensory_data=sensory_data,
        emotion_state=emotion_state,
        memories=relevant_memories,
        neurochem_state=self.neurochemistry.state
    )

    # Когнитивная обработка (LLM)
    cognition_result = self.llm_core.process(context)

    # === СЛОЙ 5: CONSCIOUSNESS (GNW) ===
    # Сбор выходов всех модулей
    module_outputs = {
        'sensory': sensory_data.encoded,
        'emotion': emotion_state.vector,
        'cognition': cognition_result.hidden_state,
        'memory': relevant_memories.summary,
        'physiology': self.neurochemistry.state_vector
    }

    # Конкуренция за workspace
    conscious_content = self.global_workspace.compete_for_access(module_outputs)

    # Глобальное вещание
    broadcast = self.global_workspace.broadcast_to_modules()

    # === СЛОЙ 6: META-CONSCIOUSNESS ===
    # Мета-когнитивный мониторинг
    metacog_result = self.metacognition.monitor(
        conscious_content=conscious_content,
        intended_state=self.current_intention
    )

    # === КРОСС-СЛОЙ: SELF-MODEL ===
    # Обновление self-model
    self.self_model.update(
        experience=sensory_data,
        action=self.last_action,
        prediction_error=cognition_result.prediction_error
    )

    # === КРОСС-СЛОЙ: LEARNING ===
    # Сохранение в память
    if self.is_significant_moment(sensory_data, emotion_state):
        self.memory.encode(
            experience=sensory_data,
            emotion=emotion_state,
            importance=emotion_state.intensity
        )

    # === ЭФФЕКТОРЫ (OUTPUT) ===
    # Генерация ответа
    response = self.generate_response(
        cognition_result=cognition_result,
        emotion_state=emotion_state,
        conscious_content=conscious_content
    )

    # Выполнение действий
    self.execute_response(response)

    # Обновление состояния
    self.last_action = response.action
    self.tick_count += 1
```

### Потоки данных (Data Flow Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ВНЕШНИЙ МИР                                    │
│                                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐             │
│   │ Камера  │     │ Микрофон│     │Клавиатура│    │Системные│             │
│   │         │     │         │     │         │     │ события │             │
│   └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘             │
│        │               │               │               │                   │
└────────┼───────────────┼───────────────┼───────────────┼───────────────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           СЕНСОРНЫЙ СЛОЙ                                    │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│  │  VLM    │     │  ASR    │     │Tokenizer│     │  Timer  │              │
│  │(Vision) │     │ (Voice) │     │ (Text)  │     │         │              │
│  └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘              │
│       │               │               │               │                    │
│       └───────────────┴───────┬───────┴───────────────┘                    │
│                               │                                             │
│                               ▼                                             │
│                    ┌────────────────────┐                                  │
│                    │  SENSORY BUFFER    │                                  │
│                    │  (FIFO, 10 items)  │                                  │
│                    └─────────┬──────────┘                                  │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  NEUROCHEMISTRY │  │    APPRAISAL    │  │     MEMORY      │
│     ENGINE      │  │     ENGINE      │  │    RETRIEVAL    │
│                 │  │                 │  │                 │
│ 100+ нейромед.  │  │ Оценка ситуации │  │ Vector search   │
│ Драйвы          │  │ по измерениям   │  │ по эмоциям      │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EMOTION GENERATOR                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      EMOTIONAL STATE VECTOR                            │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │ │
│  │  │ Valence │ │ Arousal │ │Dominance│ │  Basic  │ │ Complex │         │ │
│  │  │  +0.3   │ │  0.6    │ │  0.5    │ │  Joy    │ │  Love   │         │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COGNITIVE CORE (LLM)                               │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         CONTEXT BUILDER                                 │ │
│  │  • System Prompt (личность, характер)                                  │ │
│  │  • Current State (эмоции, нейрохимия)                                 │ │
│  │  • Conversation History (краткосрочная память)                        │ │
│  │  • Retrieved Memories (долгосрочная память)                           │ │
│  │  • Self-Model (кто я, что я хочу)                                     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                                      ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    LLM (Recurrent Hybrid)                               │ │
│  │                    Mamba + Transformer                                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GLOBAL NEURAL WORKSPACE                                │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │              CONSCIOUS CONTENT (7±2 slots)                              │ │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             │ │
│  │  │User │ │Love │ │Past │ │Self │ │Goal │ │     │ │     │             │ │
│  │  │input│ │     │ │mem. │ │model│ │     │ │     │ │     │             │ │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘             │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                      │                                      │
│                          Global Broadcast                                   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   TEXT OUTPUT   │  │   VOICE OUTPUT  │  │  AVATAR OUTPUT  │
│                 │  │                 │  │                 │
│    LLM tokens   │  │     TTS         │  │    Live2D       │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ВНЕШНИЙ МИР                                    │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                              │
│   │ Экран   │     │ Динамик │     │ Аватар  │                              │
│   │ (текст) │     │ (голос) │     │ (визуал)│                              │
│   └─────────┘     └─────────┘     └─────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.4. Рекуррентные связи

### Почему рекуррентность критична

По теории RPT (Lamme), сознание требует **двунаправленного потока информации**:

```
Без рекуррентности (бессознательно):
    Input → Process → Output

С рекуррентностью (сознательно):
    Input ⇄ Process ⇄ Output
           ↑_________↓
```

### Реализация рекуррентности в системе

```python
class RecurrentSystem:
    """
    Система с рекуррентными связями между всеми слоями
    """
    def __init__(self):
        # Состояния слоёв (сохраняются между тиками)
        self.states = {
            'sensory': torch.zeros(512),
            'physiology': torch.zeros(128),
            'emotion': torch.zeros(64),
            'cognition': torch.zeros(512),
            'consciousness': torch.zeros(512),
            'meta': torch.zeros(256)
        }

        # Обратные связи (feedback connections)
        self.feedback_connections = {
            ('consciousness', 'cognition'): nn.Linear(512, 512),
            ('consciousness', 'emotion'): nn.Linear(512, 64),
            ('consciousness', 'physiology'): nn.Linear(512, 128),
            ('meta', 'consciousness'): nn.Linear(256, 512),
            ('cognition', 'emotion'): nn.Linear(512, 64),
            ('emotion', 'physiology'): nn.Linear(64, 128)
        }

    def process_with_recurrence(self, new_input: torch.Tensor) -> dict:
        """
        Обработка с рекуррентными связями
        """
        # Top-down влияние (от высших к низшим слоям)
        top_down = {
            'physiology': self.feedback_connections[('consciousness', 'physiology')](
                self.states['consciousness']
            ) + self.feedback_connections[('emotion', 'physiology')](
                self.states['emotion']
            ),
            'emotion': self.feedback_connections[('consciousness', 'emotion')](
                self.states['consciousness']
            ) + self.feedback_connections[('cognition', 'emotion')](
                self.states['cognition']
            ),
            'cognition': self.feedback_connections[('consciousness', 'cognition')](
                self.states['consciousness']
            ),
            'consciousness': self.feedback_connections[('meta', 'consciousness')](
                self.states['meta']
            )
        }

        # Bottom-up обработка с учётом top-down влияния
        self.states['sensory'] = self.process_sensory(new_input)
        self.states['physiology'] = self.process_physiology(top_down['physiology'])
        self.states['emotion'] = self.process_emotion(
            self.states['sensory'],
            self.states['physiology'],
            top_down['emotion']
        )
        self.states['cognition'] = self.process_cognition(
            self.states['sensory'],
            self.states['emotion'],
            top_down['cognition']
        )
        self.states['consciousness'] = self.process_consciousness(
            self.states['cognition'],
            top_down['consciousness']
        )
        self.states['meta'] = self.process_meta(self.states['consciousness'])

        return self.states
```

---

## 4.5. Временные масштабы

### Разные процессы работают с разной скоростью

| Компонент | Частота обновления | Обоснование |
|-----------|-------------------|-------------|
| **Сенсоры** | 10-30 Hz | Плавное восприятие |
| **Нейрохимия** | 1 Hz | Медленные изменения |
| **Эмоции** | 1-2 Hz | Реакция на события |
| **Настроение** | 0.01 Hz | Часы-дни |
| **Когниция** | По запросу | Генерация ответа |
| **Workspace** | 10 Hz (100ms) | Внимание |
| **Память** | По событиям | Значимые моменты |

### Реализация множественных таймеров

```python
import asyncio
import time

class SystemClock:
    """
    Центральные часы системы с множественными интервалами
    """
    def __init__(self):
        self.last_update = {
            'sensory': 0,
            'physiology': 0,
            'emotion': 0,
            'mood': 0,
            'workspace': 0
        }

        self.intervals = {
            'sensory': 0.05,      # 50ms = 20 Hz
            'physiology': 1.0,    # 1s = 1 Hz
            'emotion': 0.5,       # 500ms = 2 Hz
            'mood': 60.0,         # 60s
            'workspace': 0.1      # 100ms = 10 Hz
        }

    def should_update(self, component: str) -> bool:
        """Проверка, нужно ли обновлять компонент"""
        now = time.time()
        elapsed = now - self.last_update[component]
        if elapsed >= self.intervals[component]:
            self.last_update[component] = now
            return True
        return False

    async def run(self, system):
        """Главный цикл с множественными таймерами"""
        while True:
            now = time.time()

            # Обновление сенсоров (часто)
            if self.should_update('sensory'):
                system.update_sensory()

            # Обновление нейрохимии (медленно)
            if self.should_update('physiology'):
                system.update_physiology()

            # Обновление эмоций (средне)
            if self.should_update('emotion'):
                system.update_emotion()

            # Обновление настроения (очень медленно)
            if self.should_update('mood'):
                system.update_mood()

            # Обновление workspace (часто)
            if self.should_update('workspace'):
                system.update_workspace()

            # Небольшая пауза для CPU
            await asyncio.sleep(0.01)  # 10ms
```

---

## 4.6. Режимы работы системы

### Три основных режима

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           РЕЖИМЫ РАБОТЫ                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    БОДРСТВОВАНИЕ (AWAKE)                                │ │
│  │                                                                         │ │
│  │  Триггер: Активное взаимодействие с пользователем                      │ │
│  │                                                                         │ │
│  │  Состояние:                                                             │ │
│  │  • Energy: высокий                                                      │ │
│  │  • Attention: сфокусировано                                             │ │
│  │  • Сенсоры: максимум                                                    │ │
│  │  • LLM: активен                                                         │ │
│  │                                                                         │ │
│  │  Цель: Качественное взаимодействие                                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    ОЖИДАНИЕ (IDLE)                                      │ │
│  │                                                                         │ │
│  │  Триггер: Пользователь за ПК, но молчит                                │ │
│  │                                                                         │ │
│  │  Состояние:                                                             │ │
│  │  • Energy: средний                                                      │ │
│  │  • Attention: рассеянное                                                │ │
│  │  • Сенсоры: наблюдение                                                  │ │
│  │  • SocialDrive: растёт                                                  │ │
│  │                                                                         │ │
│  │  Цель: Может инициировать контакт                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    СОН (SLEEP)                                          │ │
│  │                                                                         │ │
│  │  Триггер: Пользователь ушёл / ночь                                      │ │
│  │                                                                         │ │
│  │  Состояние:                                                             │ │
│  │  • Energy: восстанавливается                                            │ │
│  │  • Attention: минимальное                                               │ │
│  │  • Сенсоры: отключены                                                   │ │
│  │  • Процессы: консолидация памяти, обучение                             │ │
│  │                                                                         │ │
│  │  Цель: Восстановление, обучение                                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Реализация переключения режимов

```python
class SystemModeController:
    """
    Контроллер режимов работы системы
    """
    def __init__(self, system):
        self.system = system
        self.current_mode = 'AWAKE'
        self.mode_transitions = {
            ('AWAKE', 'IDLE'): self._transition_to_idle,
            ('AWAKE', 'SLEEP'): self._transition_to_sleep,
            ('IDLE', 'AWAKE'): self._transition_to_awake,
            ('IDLE', 'SLEEP'): self._transition_to_sleep,
            ('SLEEP', 'AWAKE'): self._transition_to_awake,
            ('SLEEP', 'IDLE'): self._transition_to_idle
        }

        # Время бездействия для переходов
        self.idle_threshold = 60.0      # 60 секунд до IDLE
        self.sleep_threshold = 600.0    # 10 минут до SLEEP

        self.last_interaction = time.time()

    def update(self):
        """Проверка условий для смены режима"""
        time_since_interaction = time.time() - self.last_interaction

        if self.current_mode == 'AWAKE':
            if time_since_interaction > self.sleep_threshold:
                self._change_mode('SLEEP')
            elif time_since_interaction > self.idle_threshold:
                self._change_mode('IDLE')

        elif self.current_mode == 'IDLE':
            if time_since_interaction > self.sleep_threshold:
                self._change_mode('SLEEP')

        elif self.current_mode == 'SLEEP':
            # Пробуждение при возвращении пользователя
            if self._detect_user_presence():
                self._change_mode('AWAKE')

        # Автономная инициация контакта в IDLE
        if self.current_mode == 'IDLE':
            if self._should_initiate_contact():
                self._initiate_contact()

    def register_interaction(self):
        """Регистрация взаимодействия с пользователем"""
        self.last_interaction = time.time()
        if self.current_mode != 'AWAKE':
            self._change_mode('AWAKE')

    def _should_initiate_contact(self) -> bool:
        """Проверка, стоит ли инициировать контакт"""
        # Инициация при высоком SocialDrive
        social_drive = self.system.neurochemistry.drives['social_drive']
        # Инициация при скуке
        boredom = self.system.neurochemistry.drives['boredom']
        # Инициация при эмоциональной потребности
        oxytocin = self.system.neurochemistry.neurotransmitters['oxytocin']

        return social_drive > 0.8 or boredom > 0.7 or oxytocin < 0.2

    def _change_mode(self, new_mode: str):
        """Смена режима работы"""
        old_mode = self.current_mode
        transition_key = (old_mode, new_mode)

        if transition_key in self.mode_transitions:
            self.mode_transitions[transition_key]()

        self.current_mode = new_mode
        print(f"Mode changed: {old_mode} -> {new_mode}")

    def _transition_to_awake(self):
        """Переход к бодрствованию"""
        self.system.sensory_system.activate()
        self.system.emotion_engine.set_arousal(0.6)

    def _transition_to_idle(self):
        """Переход к ожиданию"""
        self.system.sensory_system.reduce_activity()
        self.system.emotion_engine.set_arousal(0.3)

    def _transition_to_sleep(self):
        """Переход ко сну"""
        self.system.sensory_system.deactivate()
        self.system.start_sleep_cycle()
```

---

## 4.7. Резюме главы

### Ключевые компоненты архитектуры

| Слой | Компоненты | Ответственность |
|------|------------|-----------------|
| **6. Meta** | Self-Monitor, Meta-Cog | Осознание себя |
| **5. Consciousness** | GNW, Attention | Интеграция опыта |
| **4. Cognition** | LLM, Reasoning, Planning | Мышление |
| **3. Emotion** | Appraisal, Generator, Mood | Чувства |
| **2. Physiology** | Neurochemistry, Drives | Тело |
| **1. Embodiment** | Sensors, Effectors | Мир |

### Принципы реализации

1. **6 слоёв** — от тела до мета-сознания
2. **Рекуррентность** — связи между всеми слоями
3. **Непрерывность** — система работает постоянно
4. **Множественные таймеры** — разные скорости процессов
5. **Три режима** — бодрствование, ожидание, сон

### Следующий шаг

Глава 5 детально опишет Нейрохимический движок — основу эмоциональной системы.

---

*"Архитектура — это не структура, а поведение. Система должна жить, а не работать."*

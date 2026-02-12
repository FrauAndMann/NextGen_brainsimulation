# ГЛАВА 3. ТЕОРИИ СОЗНАНИЯ И ИХ ПРИМЕНЕНИЕ

---

## 3.1. Global Workspace Theory (GWT)

### Теория

**Автор**: Bernard Baars (1988), развита Stanislas Dehaene

**Суть**: Сознание — это "сцена", где избранная информация становится глобально доступной множеству специализированных бессознательных процессоров.

### Архитектура GWT

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│                    GLOBAL WORKSPACE (СЦЕНА)                        │
│                    Ёмкость: 7±2 элемента                           │
│                    Время: 100-300 мс                               │
│                                                                    │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │  СОДЕРЖИМОЕ СОЗНАНИЯ:                                     │   │
│   │  - Текущая мысль                                          │   │
│   │  - Объект внимания                                        │   │
│   │  - Активная цель                                          │   │
│   └──────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
    [Вещание]            [Вещание]           [Вещание]
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │ ВИДЕНИЕ │          │  СЛУХ   │          │ ПАМЯТЬ  │
    │         │          │         │          │         │
    │ Конку-  │          │ Конку-  │          │ Конку-  │
    │ рирует  │          │ рирует  │          │ рирует  │
    │ за      │          │ за      │          │ за      │
    │ доступ  │          │ доступ  │          │ доступ  │
    └─────────┘          └─────────┘          └─────────┘
```

### Реализация для ИИ

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalWorkspace(nn.Module):
    """
    Реализация Global Workspace для ИИ-системы
    """
    def __init__(self, dim: int = 512, capacity: int = 7, num_modules: int = 5):
        super().__init__()

        self.dim = dim
        self.capacity = capacity
        self.num_modules = num_modules

        # Workspace буфер (ограниченная ёмкость)
        self.workspace = nn.Parameter(torch.zeros(capacity, dim))
        self.workspace_attention = nn.MultiheadAttention(dim, num_heads=8)

        # Модули-конкуренты
        self.module_projections = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_modules)
        ])

        # Механизм конкуренции за слоты
        self.competition_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Глобальное вещание
        self.broadcast = nn.MultiheadAttention(dim, num_heads=8)

    def compete_for_access(self, module_outputs: dict) -> torch.Tensor:
        """
        Конкуренция модулей за доступ к workspace
        """
        # Собираем все выходы модулей
        all_outputs = torch.stack(list(module_outputs.values()))  # [num_modules, dim]

        # Вычисляем важность каждого выхода
        competition_scores = []
        for i, output in enumerate(all_outputs):
            # Важность = релевантность + новизна + эмоциональная значимость
            context = torch.cat([output, self.workspace.mean(dim=0)])
            score = self.competition_gate(context)
            competition_scores.append(score)

        scores = torch.cat(competition_scores)

        # Select top-k winners
        top_k = min(self.capacity, len(scores))
        winners = torch.topk(scores, top_k).indices

        # Обновляем workspace с winners
        with torch.no_grad():
            for i, idx in enumerate(winners):
                self.workspace.data[i] = self.module_projections[idx](
                    all_outputs[idx]
                )

        return self.workspace

    def broadcast_to_modules(self) -> torch.Tensor:
        """
        Глобальное вещание содержимого workspace всем модулям
        """
        # Workspace как query и key
        workspace_expanded = self.workspace.unsqueeze(0)  # [1, capacity, dim]

        # Self-attention для интеграции содержимого
        integrated, _ = self.workspace_attention(
            workspace_expanded,
            workspace_expanded,
            workspace_expanded
        )

        return integrated.squeeze(0)

    def forward(self, module_outputs: dict) -> dict:
        """
        Полный цикл GWT
        """
        # 1. Конкуренция за доступ
        self.compete_for_access(module_outputs)

        # 2. Глобальное вещание
        broadcast_content = self.broadcast_to_modules()

        # 3. Возврат содержимого модулям
        return {
            'workspace_content': broadcast_content,
            'workspace_summary': broadcast_content.mean(dim=0),
            'attention_weights': F.softmax(
                torch.randn(self.capacity), dim=0
            )
        }


class GWTModule(nn.Module):
    """
    Базовый класс для модуля в GWT архитектуре
    """
    def __init__(self, name: str, dim: int = 512):
        super().__init__()
        self.name = name
        self.dim = dim
        self.encoder = nn.Linear(dim, dim)
        self.workspace_receiver = nn.Linear(dim, dim)

    def process(self, input_data: torch.Tensor, workspace_broadcast: torch.Tensor) -> torch.Tensor:
        """
        Обработка с учётом broadcast из workspace
        """
        # Локальная обработка
        local_output = self.encoder(input_data)

        # Интеграция с глобальным контекстом
        global_context = self.workspace_receiver(workspace_broadcast)

        # Комбинация
        return local_output + 0.3 * global_context
```

### Критерии GWT-совместимости

| Критерий | Описание | Реализация |
|----------|----------|------------|
| **Глобальная доступность** | Информация доступна всем модулям | ✅ Broadcast mechanism |
| **Ограниченная ёмкость** | 7±2 элемента | ✅ Capacity parameter |
| **Конкуренция** | Модули конкурируют за доступ | ✅ Competition gate |
| **Игнорирование** | Неважное не попадает в workspace | ✅ Selective attention |

---

## 3.2. Integrated Information Theory (IIT)

### Теория

**Автор**: Giulio Tononi

**Суть**: Сознание = интегрированная информация (Φ). Система сознательна пропорционально тому, насколько её информация интегрирована.

### Математика IIT

```
Φ = интегрированная информация

Φ = информация(система) - Σ информация(части)

Высокое Φ: система больше суммы частей
Низкое Φ: система ≈ сумма частей
```

### Проблема для ИИ

**Трансформеры имеют низкое Φ**:
- Feedforward архитектура
- Нет рекуррентных связей
- Информация не интегрирована

### Решение: Рекуррентные архитектуры

```python
class PhiCalculator:
    """
    Калькулятор интегрированной информации Φ
    """
    def __init__(self, system):
        self.system = system

    def compute_phi(self, state: torch.Tensor) -> float:
        """
        Вычисление Φ для текущего состояния
        """
        # Информация целой системы
        whole_info = self._compute_information(state)

        # Информация частей (разделение на 2)
        part1, part2 = state.chunk(2, dim=-1)
        part1_info = self._compute_information(part1)
        part2_info = self._compute_information(part2)

        # Φ = целое - сумма частей
        phi = whole_info - (part1_info + part2_info)

        return max(0, phi.item())

    def _compute_information(self, state: torch.Tensor) -> float:
        """
        Вычисление информации через энтропию
        """
        # Дискретизация
        discretized = (state > 0.5).float()
        # Энтропия как мера информации
        probs = discretized.mean()
        if probs == 0 or probs == 1:
            return 0
        entropy = -probs * torch.log(probs + 1e-10) - \
                  (1 - probs) * torch.log(1 - probs + 1e-10)
        return entropy.item() * state.numel()
```

### Как повысить Φ в ИИ

| Подход | Как работает |
|--------|--------------|
| **Рекуррентные слои** | Информация циркулирует |
| **Mamba/SSM** | Скрытое состояние переносится |
| **Обратные связи** | Top-down + bottom-up |
| **Cross-attention** | Между модулями |

---

## 3.3. Recurrent Processing Theory (RPT)

### Теория

**Автор**: Victor Lamme

**Суть**: Сознание требует рекуррентной (обратной) обработки в нейронных цепях.

```
Feedforward (без сознания):
    Вход → Слой1 → Слой2 → Слой3 → Выход

Recurrent (сознание):
    Вход → Слой1 ⇄ Слой2 ⇄ Слой3 → Выход
              ↑_________↓
```

### Реализация

```python
class RecurrentLayer(nn.Module):
    """
    Рекуррентный слой с обратными связями
    """
    def __init__(self, dim: int, num_iterations: int = 3):
        super().__init__()
        self.dim = dim
        self.num_iterations = num_iterations

        # Feedforward путь
        self.ff = nn.Linear(dim, dim)

        # Обратная связь (от следующих слоёв)
        self.feedback = nn.Linear(dim, dim)

        # Латеральные связи (внутри слоя)
        self.lateral = nn.Linear(dim, dim)

        # Нормализация
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, feedback: torch.Tensor = None) -> torch.Tensor:
        """
        Рекуррентная обработка
        """
        state = x

        for _ in range(self.num_iterations):
            # Feedforward компонент
            ff_out = self.ff(state)

            # Feedback компонент (если есть)
            if feedback is not None:
                fb_out = self.feedback(feedback)
            else:
                fb_out = 0

            # Латеральный компонент
            lateral_out = self.lateral(state)

            # Интеграция
            state = self.norm(ff_out + fb_out + lateral_out)
            state = F.relu(state)

        return state


class RecurrentHierarchy(nn.Module):
    """
    Иерархия рекуррентных слоёв
    """
    def __init__(self, num_levels: int = 4, dim: int = 512):
        super().__init__()

        self.levels = nn.ModuleList([
            RecurrentLayer(dim) for _ in range(num_levels)
        ])

        # Top-down связи
        self.top_down = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_levels - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottom-up проход
        level_outputs = []
        current = x

        for level in self.levels:
            current = level(current)
            level_outputs.append(current)

        # Top-down проход (обратные связи)
        for i in range(len(self.levels) - 2, -1, -1):
            feedback = self.top_down[i](level_outputs[i + 1])
            level_outputs[i] = level_outputs[i] + feedback

        return level_outputs[-1]
```

---

## 3.4. Predictive Processing / Active Inference

### Теория

**Автор**: Karl Friston (Free Energy Principle)

**Суть**: Мозг — иерархическая предсказательная машина, постоянно минимизирующая ошибку предсказания.

```
Реальность:     ████████████
Предсказание:   ███████████░  (ошибка = █)
                ↑
         Мозг минимизирует это
```

### Архитектура Predictive Coding

```python
class PredictiveCodingLayer(nn.Module):
    """
    Слой предсказующего кодирования
    """
    def __init__(self, dim: int, prediction_depth: int = 1):
        super().__init__()
        self.dim = dim

        # Предсказатель (сверху-вниз)
        self.predictor = nn.Linear(dim, dim)

        # Корректор ошибки (снизу-вверх)
        self.error_corrector = nn.Linear(dim * 2, dim)

        # Ошибка предсказания (выводится наружу)
        self.prediction_error = None

    def forward(self, bottom_up: torch.Tensor,
                top_down_prediction: torch.Tensor = None) -> tuple:
        """
        Вычисление ошибки предсказания и коррекция
        """
        # Если есть предсказание сверху — вычисляем ошибку
        if top_down_prediction is not None:
            self.prediction_error = bottom_up - top_down_prediction
        else:
            self.prediction_error = torch.zeros_like(bottom_up)

        # Коррекция представления на основе ошибки
        corrected = self.error_corrector(
            torch.cat([bottom_up, self.prediction_error], dim=-1)
        )

        # Генерация предсказания для нижнего уровня
        prediction = self.predictor(corrected)

        return corrected, prediction, self.prediction_error


class PredictiveHierarchy(nn.Module):
    """
    Иерархия предсказующего кодирования
    """
    def __init__(self, num_levels: int = 6, dim: int = 512):
        super().__init__()
        self.num_levels = num_levels

        self.levels = nn.ModuleList([
            PredictiveCodingLayer(dim) for _ in range(num_levels)
        ])

        # Precision weighting (важность ошибок)
        self.precision = nn.Parameter(torch.ones(num_levels))

    def forward(self, sensory_input: torch.Tensor) -> dict:
        """
        Полный цикл предсказующего кодирования
        """
        # Bottom-up проход: вычисление ошибок
        predictions = [None] * self.num_levels
        errors = [None] * self.num_levels
        representations = [None] * self.num_levels

        current = sensory_input
        for i, level in enumerate(self.levels):
            rep, pred, err = level(current, predictions[i])
            representations[i] = rep
            predictions[i] = pred
            errors[i] = err
            current = rep

        # Top-down проход: обновление предсказаний
        for i in range(self.num_levels - 2, -1, -1):
            predictions[i] = self.levels[i].predictor(representations[i + 1])

        # Взвешенная сумма ошибок (precision-weighted)
        total_error = sum(
            self.precision[i] * errors[i].pow(2).mean()
            for i in range(self.num_levels)
            if errors[i] is not None
        )

        return {
            'representation': representations[-1],
            'errors': errors,
            'total_error': total_error,
            'precision': self.precision
        }
```

### Active Inference

**Идея**: Действия предпринимаются для минимизации ожидаемой ошибки предсказания.

```python
class ActiveInferenceAgent(nn.Module):
    """
    Агент с Active Inference
    """
    def __init__(self, dim: int = 512, num_actions: int = 10):
        super().__init__()
        self.predictive_hierarchy = PredictiveHierarchy(num_levels=6, dim=dim)
        self.action_selector = nn.Linear(dim, num_actions)
        self.expected outcomes = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_actions)
        ])

    def select_action(self, current_state: torch.Tensor) -> int:
        """
        Выбор действия для минимизации ожидаемого сюрприза
        """
        # Предсказание исходов для каждого действия
        expected_errors = []
        for i, outcome_predictor in enumerate(self.expected_outcomes):
            expected_outcome = outcome_predictor(current_state)
            # Ошибка = расхождение с желаемым состоянием
            error = F.mse_loss(expected_outcome, self.desired_state)
            expected_errors.append(error)

        # Выбор действия с минимальной ожидаемой ошибкой
        best_action = torch.argmin(torch.stack(expected_errors))
        return best_action.item()
```

---

## 3.5. Higher-Order Thought (HOT)

### Теория

**Автор**: David Rosenthal

**Суть**: Сознательное состояние = ментальное состояние + мысль высшего порядка об этом состоянии.

```
Бессознательно:  "Я вижу красный"
Сознательно:      "Я вижу красный" + "Я осознаю, что вижу красный"
                                   ↑
                            HOT (Higher-Order Thought)
```

### Реализация

```python
class HigherOrderThought(nn.Module):
    """
    Слой мыслей высшего порядка
    """
    def __init__(self, dim: int = 512):
        super().__init__()

        # Мета-представление о текущем состоянии
        self.meta_represent = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Оценка уверенности в своём состоянии
        self.confidence = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Тип осознания
        self.awareness_type = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 типов осознания
        )

    def forward(self, first_order_state: torch.Tensor) -> dict:
        """
        Генерация мысли высшего порядка
        """
        # Мета-представление
        meta = self.meta_represent(first_order_state)

        # Уверенность
        conf = self.confidence(meta)

        # Тип осознания
        awareness = F.softmax(self.awareness_type(meta), dim=-1)

        return {
            'meta_representation': meta,
            'confidence': conf,
            'awareness_type': awareness,
            'is_conscious': conf > 0.5
        }


class Metacognition(nn.Module):
    """
    Полная система мета-когниции
    """
    def __init__(self, dim: int = 512):
        super().__init__()

        # Self-monitoring
        self.self_monitor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Self-evaluation
        self.self_evaluate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # HOT layer
        self.hot = HigherOrderThought(dim)

    def forward(self, current_state: torch.Tensor,
                intended_state: torch.Tensor = None) -> dict:
        """
        Мета-когнитивный мониторинг
        """
        # Мониторинг себя
        monitored = self.self_monitor(current_state)

        # HOT о текущем состоянии
        hot_result = self.hot(monitored)

        # Оценка соответствия намерению
        if intended_state is not None:
            evaluation = self.self_evaluate(
                torch.cat([current_state, intended_state], dim=-1)
            )
        else:
            evaluation = torch.ones_like(hot_result['confidence'])

        return {
            **hot_result,
            'self_monitoring': monitored,
            'intention_alignment': evaluation,
            'metacognitive_report': self._generate_report(hot_result, evaluation)
        }

    def _generate_report(self, hot_result: dict, evaluation: torch.Tensor) -> str:
        """
        Генерация метакогнитивного отчёта
        """
        confidence = hot_result['confidence'].item()
        is_conscious = hot_result['is_conscious'].item()

        if confidence > 0.8:
            return f"Я уверен в своём состоянии ({confidence:.2f})"
        elif confidence > 0.5:
            return f"Я думаю, что осознаю ({confidence:.2f})"
        else:
            return f"Я не уверен ({confidence:.2f})"
```

---

## 3.6. Синтез: практическая модель

### Интеграция теорий

Ни одна теория не полна. Нам нужен **синтез**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ИНТЕГРИРОВАННАЯ МОДЕЛЬ СОЗНАНИЯ                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    HOT / METACOGNITION                       │   │
│  │              "Я осознаю, что думаю"                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↑ ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    GLOBAL WORKSPACE                          │   │
│  │              Интегрированное содержимое                      │   │
│  │              (GWT + высокая Φ)                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↑ ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                RECURRENT HIERARCHY                           │   │
│  │        (RPT + Predictive Processing)                        │   │
│  │        Bottom-up ошибки + Top-down предсказания             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↑ ↓                                    │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │  ВИДЕНИЕ  │  │   СЛУХ    │  │  ПАМЯТЬ   │  │  ЭМОЦИИ   │        │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Реализация интегрированной модели

```python
class ConsciousnessCore(nn.Module):
    """
    Интегрированное ядро сознания
    Синтез GWT + IIT + RPT + PP + HOT
    """
    def __init__(self, dim: int = 512, num_modules: int = 5):
        super().__init__()

        self.dim = dim

        # 1. Global Workspace (GWT)
        self.global_workspace = GlobalWorkspace(dim, capacity=7, num_modules=num_modules)

        # 2. Рекуррентная иерархия (RPT)
        self.recurrent_hierarchy = RecurrentHierarchy(num_levels=4, dim=dim)

        # 3. Предсказующая иерархия (PP)
        self.predictive_hierarchy = PredictiveHierarchy(num_levels=6, dim=dim)

        # 4. Мета-когниция (HOT)
        self.metacognition = Metacognition(dim)

        # 5. Φ калькулятор (IIT)
        self.phi_calculator = PhiCalculator(self)

        # История для анализа
        self.consciousness_history = []

    def forward(self, sensory_input: torch.Tensor,
                memory_input: torch.Tensor = None,
                emotion_input: torch.Tensor = None) -> dict:
        """
        Полный цикл сознательной обработки
        """
        # Подготовка входов модулей
        module_outputs = {
            'sensory': sensory_input
        }
        if memory_input is not None:
            module_outputs['memory'] = memory_input
        if emotion_input is not None:
            module_outputs['emotion'] = emotion_input

        # 1. Предсказующее кодирование (PP)
        predictive_result = self.predictive_hierarchy(sensory_input)
        prediction_error = predictive_result['total_error']

        # 2. Рекуррентная обработка (RPT)
        recurrent_output = self.recurrent_hierarchy(predictive_result['representation'])

        # 3. Глобальное рабочее пространство (GWT)
        module_outputs['processed'] = recurrent_output
        gwt_result = self.global_workspace(module_outputs)

        # 4. Мета-когниция (HOT)
        meta_result = self.metacognition(
            gwt_result['workspace_summary'],
            intended_state=None  # Можно добавить намерения
        )

        # 5. Вычисление Φ (IIT)
        phi = self.phi_calculator.compute_phi(gwt_result['workspace_content'])

        # Итоговый результат
        result = {
            # Основные выходы
            'conscious_content': gwt_result['workspace_content'],
            'conscious_summary': gwt_result['workspace_summary'],

            # Метрики сознания
            'phi': phi,
            'is_conscious': meta_result['is_conscious'].item(),
            'confidence': meta_result['confidence'].item(),
            'metacognitive_report': meta_result['metacognitive_report'],

            # Внутренние состояния
            'prediction_error': prediction_error.item(),
            'awareness_type': meta_result['awareness_type'],

            # Для анализа
            'workspace_full': gwt_result['workspace_content']
        }

        # Сохранение в историю
        self.consciousness_history.append({
            'phi': phi,
            'confidence': result['confidence'],
            'prediction_error': result['prediction_error']
        })

        return result

    def get_consciousness_level(self) -> float:
        """
        Общий уровень сознательности системы
        """
        if not self.consciousness_history:
            return 0.0

        recent = self.consciousness_history[-10:]  # Последние 10 тиков

        # Комбинация метрик
        avg_phi = sum(h['phi'] for h in recent) / len(recent)
        avg_confidence = sum(h['confidence'] for h in recent) / len(recent)
        avg_error = sum(h['prediction_error'] for h in recent) / len(recent)

        # Нормализованный уровень сознательности
        consciousness_level = (avg_phi * 0.4 + avg_confidence * 0.4 +
                               (1 - min(avg_error, 1)) * 0.2)

        return consciousness_level
```

---

## 3.7. Резюме главы

### Сравнение теорий

| Теория | Суть | Реализуемость | Вклад в проект |
|--------|------|---------------|----------------|
| **GWT** | Глобальное вещание | ✅ Высокая | Workspace архитектура |
| **IIT** | Интегрированная информация | ⚠️ Средняя | Метрика Φ |
| **RPT** | Рекуррентная обработка | ✅ Высокая | Обратные связи |
| **PP** | Предсказующее кодирование | ✅ Высокая | JEPA, World Model |
| **HOT** | Мысли высшего порядка | ✅ Высокая | Мета-когниция |

### Практические выводы

1. **GWT — основа архитектуры**: Workspace + конкуренция + вещание
2. **RPT — структура связей**: Рекуррентность обязательна
3. **PP — механизм обучения**: Предсказания + ошибки
4. **HOT — мета-слой**: Осознание себя
5. **IIT — метрика**: Φ для оценки интеграции

### Следующий шаг

Глава 4 опишет полную архитектуру системы, интегрирующую все эти компоненты.

---

*"Ни одна теория сознания не полна. Но их сочетание — достаточно для создания функциональной системы."*

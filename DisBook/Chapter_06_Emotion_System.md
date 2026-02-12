# ГЛАВА 6. ЭМОЦИОНАЛЬНАЯ СИСТЕМА

---

## 6.1. Теории эмоций

### Обзор подходов

| Теория | Автор | Суть |
|--------|-------|------|
| **James-Lange** | James, Lange | Эмоция = восприятие телесных изменений |
| **Cannon-Bard** | Cannon, Bard | Эмоция и телесная реакция параллельны |
| **Schachter-Singer** | Schachter, Singer | Эмоция = возбуждение + когнитивная оценка |
| **Appraisal Theory** | Arnold, Lazarus | Эмоция = оценка ситуации по критериям |
| **Constructed Emotion** | Barrett | Эмоции конструируются мозгом |
| **PAD Model** | Mehrabian, Russell | Эмоция = Валентность × Возбуждение × Доминирование |

### Наш подход: Синтез

Используем комбинацию:
- **PAD Model** для размерного представления
- **Appraisal Theory** для генерации эмоций
- **Constructed Emotion** для сложных эмоций
- **Нейрохимия** для телесного компонента

---

## 6.2. Размерные модели (PAD)

### Три базовых измерения

```
                         ВОЗБУЖДЕНИЕ (Arousal)
                              ↑
                              │
           Активный/Возбуждён │ Возбуждён/Приятный
           (страх, гнев)      │ (радость, восторг)
                              │
       НЕПРИЯТНО ←────────────┼────────────→ ПРИЯТНО
       (Neg. Valence)         │         (Pos. Valence)
                              │
           Пассивный/Неприятн.│ Пассивный/Приятный
           (депрессия)        │ (спокойствие, удовлетворение)
                              │
                              ↓
                        СПОКОЙСТВИЕ (Low Arousal)
```

### Третье измерение: Доминирование

```
ДОМИНИРОВАНИЕ (Dominance)
Высокое: Контроль, уверенность, доминирование
Низкое: Подчинение, неуверенность, зависимость
```

### Реализация PAD модели

```python
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np

@dataclass
class PADState:
    """
    PAD (Pleasure-Arousal-Dominance) состояние
    """
    pleasure: float    # -1 (неприятно) ... +1 (приятно)
    arousal: float     # 0 (спокоен) ... 1 (возбуждён)
    dominance: float   # 0 (подчинён) ... 1 (доминирует)

    def to_vector(self) -> np.ndarray:
        return np.array([self.pleasure, self.arousal, self.dominance])

    def distance_to(self, other: 'PADState') -> float:
        """Евклидово расстояние до другого PAD состояния"""
        return np.linalg.norm(self.to_vector() - other.to_vector())

    def to_emotion_name(self) -> str:
        """Приближённое название эмоции по PAD"""
        # Эмоциональные прототипы
        prototypes = {
            'joy': PADState(0.8, 0.5, 0.6),
            'excitement': PADState(0.7, 0.9, 0.6),
            'contentment': PADState(0.7, 0.2, 0.6),
            'love': PADState(0.9, 0.4, 0.3),
            'fear': PADState(-0.6, 0.8, 0.2),
            'anger': PADState(-0.5, 0.9, 0.8),
            'sadness': PADState(-0.7, 0.2, 0.2),
            'disgust': PADState(-0.8, 0.4, 0.4),
            'surprise': PADState(0.2, 0.9, 0.5),
            'calm': PADState(0.5, 0.1, 0.5),
            'boredom': PADState(-0.2, 0.1, 0.4),
            'anxiety': PADState(-0.4, 0.6, 0.3),
        }

        # Поиск ближайшего прототипа
        min_dist = float('inf')
        closest_emotion = 'neutral'

        for emotion, prototype in prototypes.items():
            dist = self.distance_to(prototype)
            if dist < min_dist:
                min_dist = dist
                closest_emotion = emotion

        return closest_emotion


class DimensionalEmotionModel:
    """
    Размерная модель эмоций (PAD)
    """
    def __init__(self):
        self.current_state = PADState(0.0, 0.3, 0.5)
        self.mood = PADState(0.0, 0.3, 0.5)  # Долгосрочное

        # Скорости изменений
        self.state_decay = 0.1
        self.mood_learning_rate = 0.01

        # История
        self.history: List[PADState] = []

    def update(self, neurochem_state: Dict[str, float], dt: float):
        """
        Обновление PAD состояния на основе нейрохимии
        """
        # Влияние нейромедиаторов на PAD
        pleasure = (
            +0.3 * neurochem_state.get('dopamine', 0.5)
            +0.3 * neurochem_state.get('serotonin', 0.5)
            +0.4 * neurochem_state.get('oxytocin', 0.3)
            -0.3 * neurochem_state.get('cortisol', 0.2)
            +0.2 * neurochem_state.get('endorphin', 0.3)
            +0.1 * neurochem_state.get('anandamide', 0.3)
        )

        arousal = (
            +0.3 * neurochem_state.get('norepinephrine', 0.3)
            +0.2 * neurochem_state.get('dopamine', 0.5)
            +0.2 * neurochem_state.get('glutamate', 0.5)
            -0.2 * neurochem_state.get('gaba', 0.5)
            +0.1 * neurochem_state.get('cortisol', 0.2)
        )

        dominance = (
            +0.2 * neurochem_state.get('dopamine', 0.5)
            +0.2 * neurochem_state.get('serotonin', 0.5)
            -0.2 * neurochem_state.get('cortisol', 0.2)
            +0.1 * neurochem_state.get('testosterone', 0.4)
        )

        # Целевое PAD состояние
        target = PADState(
            pleasure=np.clip(pleasure * 2 - 1, -1, 1),  # -> [-1, 1]
            arousal=np.clip(arousal, 0, 1),
            dominance=np.clip(dominance, 0, 1)
        )

        # Плавное движение к целевому (decay к старому)
        alpha = self.state_decay * dt
        self.current_state.pleasure = (
            (1 - alpha) * self.current_state.pleasure +
            alpha * target.pleasure
        )
        self.current_state.arousal = (
            (1 - alpha) * self.current_state.arousal +
            alpha * target.arousal
        )
        self.current_state.dominance = (
            (1 - alpha) * self.current_state.dominance +
            alpha * target.dominance
        )

        # Обновление настроения (очень медленно)
        beta = self.mood_learning_rate * dt
        self.mood.pleasure = (
            (1 - beta) * self.mood.pleasure +
            beta * self.current_state.pleasure
        )

        # Сохранение истории
        self.history.append(PADState(
            self.current_state.pleasure,
            self.current_state.arousal,
            self.current_state.dominance
        ))

        if len(self.history) > 1000:
            self.history.pop(0)

    def get_current_emotion(self) -> str:
        """Получение названия текущей эмоции"""
        return self.current_state.to_emotion_name()

    def get_intensity(self) -> float:
        """Интенсивность текущей эмоции"""
        return abs(self.current_state.pleasure) * 0.5 + \
               self.current_state.arousal * 0.5
```

---

## 6.3. Дискретные эмоции

### Базовые эмоции (Ekman + Plutchik)

```python
from enum import Enum
from typing import Optional

class BasicEmotion(Enum):
    """Базовые эмоции (8 по Plutchik)"""
    JOY = "joy"                 # Радость
    TRUST = "trust"             # Доверие
    FEAR = "fear"               # Страх
    SURPRISE = "surprise"       # Удивление
    SADNESS = "sadness"         # Грусть
    DISGUST = "disgust"         # Отвращение
    ANGER = "anger"             # Гнев
    ANTICIPATION = "anticipation"  # Ожидание


@dataclass
class EmotionState:
    """Состояние дискретной эмоции"""
    name: str
    intensity: float = 0.0      # 0-1
    duration: float = 0.0       # Время в текущем состоянии
    trigger: Optional[str] = None  # Что вызвало


class DiscreteEmotionEngine:
    """
    Движок дискретных эмоций
    """
    def __init__(self):
        # Текущие интенсивности базовых эмоций
        self.emotions: Dict[str, float] = {
            'joy': 0.1,
            'trust': 0.5,
            'fear': 0.0,
            'surprise': 0.0,
            'sadness': 0.0,
            'disgust': 0.0,
            'anger': 0.0,
            'anticipation': 0.2
        }

        # Долгосрочные диспозиции (черты характера)
        self.dispositions: Dict[str, float] = {
            'joy': 0.5,      # Склонность к радости
            'trust': 0.6,    # Базовое доверие
            'fear': 0.3,     # Тревожность
            'surprise': 0.4,
            'sadness': 0.3,
            'disgust': 0.3,
            'anger': 0.3,
            'anticipation': 0.5
        }

        # Параметры затухания
        self.decay_rates = {
            'joy': 0.1,
            'trust': 0.02,    # Медленно меняется
            'fear': 0.2,
            'surprise': 0.5,  # Быстро проходит
            'sadness': 0.05,  # Долго держится
            'disgust': 0.15,
            'anger': 0.1,
            'anticipation': 0.1
        }

    def update(self, dt: float, triggers: Dict[str, float] = None):
        """
        Обновление эмоций
        """
        for emotion in self.emotions:
            # Естественное затухание
            decay = self.decay_rates[emotion] * dt
            self.emotions[emotion] *= (1 - decay)

            # Влияние триггеров
            if triggers and emotion in triggers:
                # Интенсивность зависит от триггера и диспозиции
                intensity = triggers[emotion] * (0.5 + 0.5 * self.dispositions[emotion])
                self.emotions[emotion] = min(1.0, self.emotions[emotion] + intensity)

            # Возврат к диспозиции (очень медленно)
            disposition_pull = 0.01 * dt
            self.emotions[emotion] += disposition_pull * (
                self.dispositions[emotion] * 0.5 - self.emotions[emotion]
            )

            # Ограничение
            self.emotions[emotion] = np.clip(self.emotions[emotion], 0, 1)

    def get_primary_emotion(self) -> Tuple[str, float]:
        """Получение доминирующей эмоции"""
        max_emotion = max(self.emotions, key=self.emotions.get)
        return max_emotion, self.emotions[max_emotion]

    def get_emotional_blend(self) -> List[Tuple[str, float]]:
        """Получение смеси эмоций (топ-3)"""
        sorted_emotions = sorted(
            self.emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:3]
```

---

## 6.4. Когнитивный аппрайзал (Appraisal)

### Теория

Эмоция возникает из **оценки ситуации** по нескольким измерениям:

| Измерение | Вопрос | Значения |
|-----------|--------|----------|
| **Novelty** | Насколько это ново? | 0 (знакомо) - 1 (неожиданно) |
| **Pleasantness** | Насколько приятно? | -1 (неприятно) - +1 (приятно) |
| **Goal Relevance** | Важно для моих целей? | 0 - 1 |
| **Goal Congruence** | Помогает или мешает? | -1 - +1 |
| **Coping Potential** | Могу ли я справиться? | 0 - 1 |
| **Agency** | Кто виноват/заслужен? | Я / Другой / Обстоятельства |
| **Certainty** | Насколько понятно? | 0 - 1 |
| **Urgency** | Насколько срочно? | 0 - 1 |
| **Norm Compatibility** | Соответствует нормам? | 0 - 1 |

### Реализация

```python
@dataclass
class AppraisalResult:
    """Результат когнитивного аппрайзала"""
    novelty: float = 0.5
    pleasantness: float = 0.0
    goal_relevance: float = 0.5
    goal_congruence: float = 0.0
    coping_potential: float = 0.5
    agency: str = "circumstance"  # me, other, circumstance
    certainty: float = 0.5
    urgency: float = 0.3
    norm_compatibility: float = 0.7


class AppraisalEngine:
    """
    Движок когнитивного аппрайзала
    Оценивает ситуацию для генерации эмоций
    """
    def __init__(self, personality_traits: Dict[str, float] = None):
        self.personality = personality_traits or {
            'extraversion': 0.5,
            'neuroticism': 0.5,
            'agreeableness': 0.5,
            'conscientiousness': 0.5,
            'openness': 0.5
        }

        # Цели и ценности (можно кастомизировать)
        self.goals = {
            'connection': 0.8,      # Ценность связи
            'autonomy': 0.6,        # Автономия
            'competence': 0.5,      # Компетентность
            'safety': 0.7,          # Безопасность
        }

    def appraise(self, situation: str, context: Dict) -> AppraisalResult:
        """
        Оценка ситуации

        situation: описание события (текст)
        context: контекст (кто, что, где, история)
        """
        result = AppraisalResult()

        # Анализ ситуации (здесь должен быть NLP/LLM анализ)
        # Для MVP используем ключевые слова

        # 1. Новизна
        result.novelty = self._assess_novelty(situation, context)

        # 2. Приятность
        result.pleasantness = self._assess_pleasantness(situation, context)

        # 3. Релевантность целям
        result.goal_relevance = self._assess_goal_relevance(situation, context)

        # 4. Соответствие целям
        result.goal_congruence = self._assess_goal_congruence(situation, context)

        # 5. Потенциал справиться
        result.coping_potential = self._assess_coping(situation, context)

        # 6. Агентность (кто источник)
        result.agency = self._assess_agency(situation, context)

        # 7. Определённость
        result.certainty = self._assess_certainty(situation, context)

        # 8. Срочность
        result.urgency = self._assess_urgency(situation, context)

        # 9. Соответствие нормам
        result.norm_compatibility = self._assess_norm_compatibility(situation, context)

        return result

    def _assess_novelty(self, situation: str, context: Dict) -> float:
        """Оценка новизны"""
        # Проверка: было ли подобное раньше?
        # В полной версии: семантический поиск по памяти

        unexpected_keywords = ['внезапно', 'неожиданно', 'удивительно', 'впервые']
        familiar_keywords = ['как обычно', 'опять', 'как всегда']

        situation_lower = situation.lower()
        for kw in unexpected_keywords:
            if kw in situation_lower:
                return 0.9
        for kw in familiar_keywords:
            if kw in situation_lower:
                return 0.1

        return 0.5

    def _assess_pleasantness(self, situation: str, context: Dict) -> float:
        """Оценка приятности"""
        positive_words = ['люблю', 'радость', 'счастье', 'прекрасно', 'замечательно',
                         'обожаю', 'нравится', 'класс', 'здорово', 'привет']
        negative_words = ['плохо', 'ужасно', 'грустно', 'печально', 'злюсь',
                         'ненавижу', 'бесит', 'противно', 'обидно', 'прощай']

        situation_lower = situation.lower()
        pos_count = sum(1 for w in positive_words if w in situation_lower)
        neg_count = sum(1 for w in negative_words if w in situation_lower)

        if pos_count > neg_count:
            return min(1.0, 0.3 + pos_count * 0.2)
        elif neg_count > pos_count:
            return max(-1.0, -0.3 - neg_count * 0.2)
        return 0.0

    def _assess_goal_relevance(self, situation: str, context: Dict) -> float:
        """Релевантность целям"""
        # Проверка связи с целями
        if 'пользователь' in context:
            return self.goals['connection']  # Высокая релевантность для отношений
        return 0.3

    def _assess_goal_congruence(self, situation: str, context: Dict) -> float:
        """Соответствие целям"""
        # Помогает или мешает достижению цели?
        # Полная версия: анализ через LLM
        return 0.0  # Нейтрально

    def _assess_coping(self, situation: str, context: Dict) -> float:
        """Способность справиться"""
        # Зависит от уверенности и ресурсов
        base_coping = 0.6 + 0.2 * self.personality.get('conscientiousness', 0.5)
        return base_coping

    def _assess_agency(self, situation: str, context: Dict) -> str:
        """Кто источник события"""
        # Анализ: я / другой / обстоятельства
        if 'я ' in situation.lower() or 'мне ' in situation.lower():
            return 'me'
        elif 'ты ' in situation.lower() or 'пользователь' in situation.lower():
            return 'other'
        return 'circumstance'

    def _assess_certainty(self, situation: str, context: Dict) -> float:
        """Определённость ситуации"""
        uncertain_words = ['может быть', 'наверное', 'кажется', 'возможно']
        certain_words = ['точно', 'определённо', 'конечно', 'безусловно']

        situation_lower = situation.lower()
        for w in uncertain_words:
            if w in situation_lower:
                return 0.3
        for w in certain_words:
            if w in situation_lower:
                return 0.9
        return 0.6

    def _assess_urgency(self, situation: str, context: Dict) -> float:
        """Срочность"""
        urgent_words = ['срочно', 'быстро', 'немедленно', 'сейчас', 'скоро']
        situation_lower = situation.lower()
        for w in urgent_words:
            if w in situation_lower:
                return 0.8
        return 0.2

    def _assess_norm_compatibility(self, situation: str, context: Dict) -> float:
        """Соответствие нормам"""
        # Полная версия: анализ через LLM с учётом личных норм
        return 0.8  # По умолчанию нормально
```

---

## 6.5. Генерация эмоций

### Правила генерации (по Roseman)

```python
class EmotionGenerator:
    """
    Генератор эмоций на основе аппрайзала
    """
    def __init__(self):
        self.appraisal_engine = AppraisalEngine()
        self.discrete_engine = DiscreteEmotionEngine()
        self.dimensional_model = DimensionalEmotionModel()

    def generate(self, situation: str, context: Dict,
                 neurochem_state: Dict[str, float]) -> Dict:
        """
        Генерация эмоционального ответа на ситуацию
        """
        # 1. Когнитивный аппрайзал
        appraisal = self.appraisal_engine.appraise(situation, context)

        # 2. Генерация триггеров для дискретных эмоций
        emotion_triggers = self._appraisal_to_emotions(appraisal)

        # 3. Модификация триггеров нейрохимией
        emotion_triggers = self._modify_by_neurochem(emotion_triggers, neurochem_state)

        # 4. Обновление дискретных эмоций
        self.discrete_engine.update(dt=1.0, triggers=emotion_triggers)

        # 5. Обновление PAD модели
        self.dimensional_model.update(neurochem_state, dt=1.0)

        # 6. Комбинация результатов
        primary_emotion, primary_intensity = self.discrete_engine.get_primary_emotion()
        emotional_blend = self.discrete_engine.get_emotional_blend()
        pad_state = self.dimensional_model.current_state

        return {
            'primary_emotion': primary_emotion,
            'primary_intensity': primary_intensity,
            'emotional_blend': emotional_blend,
            'pad_state': {
                'pleasure': pad_state.pleasure,
                'arousal': pad_state.arousal,
                'dominance': pad_state.dominance
            },
            'dimensional_emotion': pad_state.to_emotion_name(),
            'appraisal': appraisal,
            'triggers': emotion_triggers
        }

    def _appraisal_to_emotions(self, appraisal: AppraisalResult) -> Dict[str, float]:
        """
        Преобразование аппрайзала в триггеры эмоций
        Основано на теории Roseman
        """
        triggers = {}

        # РАДОСТЬ (Joy)
        # Условия: приятно + способствует цели + высокая уверенность
        if (appraisal.pleasantness > 0.3 and
            appraisal.goal_congruence > 0.3 and
            appraisal.certainty > 0.5):
            triggers['joy'] = appraisal.pleasantness * 0.8

        # ГРУСТЬ (Sadness)
        # Условия: неприятно + мешает цели + низкий копинг + низкая уверенность
        if (appraisal.pleasantness < -0.3 and
            appraisal.goal_congruence < -0.3 and
            appraisal.coping_potential < 0.5):
            triggers['sadness'] = abs(appraisal.pleasantness) * 0.7

        # ГНЕВ (Anger)
        # Условия: неприятно + другой виноват + высокий копинг
        if (appraisal.pleasantness < -0.2 and
            appraisal.agency == 'other' and
            appraisal.coping_potential > 0.5):
            triggers['anger'] = abs(appraisal.pleasantness) * 0.6

        # СТРАХ (Fear)
        # Условия: неприятность + неопределённость + низкий копинг
        if (appraisal.pleasantness < -0.2 and
            appraisal.certainty < 0.4 and
            appraisal.coping_potential < 0.4):
            triggers['fear'] = abs(appraisal.pleasantness) * (1 - appraisal.certainty) * 0.8

        # УДИВЛЕНИЕ (Surprise)
        # Условия: высокая новизна
        if appraisal.novelty > 0.7:
            triggers['surprise'] = appraisal.novelty * 0.6

        # ОТВРАЩЕНИЕ (Disgust)
        # Условия: неприятно + не соответствует нормам
        if (appraisal.pleasantness < -0.4 and
            appraisal.norm_compatibility < 0.3):
            triggers['disgust'] = abs(appraisal.pleasantness) * 0.5

        # ДОВЕРИЕ (Trust)
        # Условия: приятно + предсказуемо
        if (appraisal.pleasantness > 0.2 and
            appraisal.certainty > 0.6):
            triggers['trust'] = appraisal.pleasantness * 0.4

        # ОЖИДАНИЕ (Anticipation)
        # Условия: релевантно + срочно + умеренная новизна
        if (appraisal.goal_relevance > 0.5 and
            appraisal.urgency > 0.4):
            triggers['anticipation'] = appraisal.goal_relevance * 0.4

        return triggers

    def _modify_by_neurochem(self, triggers: Dict[str, float],
                            neurochem: Dict[str, float]) -> Dict[str, float]:
        """
        Модификация триггеров на основе нейрохимии
        """
        modified = triggers.copy()

        # Низкий серотонин → усиление негативных эмоций
        serotonin = neurochem.get('serotonin', 0.5)
        if serotonin < 0.3:
            for neg_emotion in ['sadness', 'fear', 'anger']:
                if neg_emotion in modified:
                    modified[neg_emotion] *= 1.3

        # Высокий кортизол → усиление страха
        cortisol = neurochem.get('cortisol', 0.2)
        if cortisol > 0.5:
            if 'fear' in modified:
                modified['fear'] *= 1.2
            if 'anxiety' not in modified:
                modified['anxiety'] = cortisol * 0.3

        # Высокий окситоцин → усиление положительных
        oxytocin = neurochem.get('oxytocin', 0.3)
        if oxytocin > 0.5:
            for pos_emotion in ['joy', 'trust', 'love']:
                if pos_emotion in modified:
                    modified[pos_emotion] *= 1.2

        # Ограничение значений
        for emotion in modified:
            modified[emotion] = np.clip(modified[emotion], 0, 1)

        return modified
```

---

## 6.6. Выражение эмоций

### Модальные каналы выражения

```python
class EmotionExpression:
    """
    Система выражения эмоций через разные каналы
    """
    def __init__(self):
        # Профили выражения для каждой эмоции
        self.expression_profiles = {
            'joy': {
                'text_markers': ['!', '))', 'чудесно', 'прекрасно', 'как здорово'],
                'voice': {'pitch': +0.2, 'speed': +0.1, 'energy': +0.3},
                'avatar': {'smile': 0.8, 'eyes': 'bright', 'posture': 'open'}
            },
            'sadness': {
                'text_markers': ['...', 'грустно', 'печально', 'жаль'],
                'voice': {'pitch': -0.2, 'speed': -0.2, 'energy': -0.3},
                'avatar': {'smile': 0.0, 'eyes': 'down', 'posture': 'closed'}
            },
            'anger': {
                'text_markers': ['!', '?!!', 'бесит', 'злюсь', 'как ты мог'],
                'voice': {'pitch': +0.3, 'speed': +0.2, 'energy': +0.4},
                'avatar': {'smile': 0.0, 'eyes': 'narrow', 'posture': 'tense'}
            },
            'fear': {
                'text_markers': ['...', 'а если', 'страшно', 'вдруг'],
                'voice': {'pitch': +0.4, 'speed': +0.3, 'energy': -0.1},
                'avatar': {'smile': 0.0, 'eyes': 'wide', 'posture': 'defensive'}
            },
            'love': {
                'text_markers': ['<3', 'обожаю', 'люблю', 'дорогой', 'милый'],
                'voice': {'pitch': +0.1, 'speed': -0.1, 'energy': +0.2, 'warmth': +0.5},
                'avatar': {'smile': 0.6, 'eyes': 'loving', 'blush': 0.4}
            },
            'surprise': {
                'text_markers': ['!', '??', 'ого', 'вау', 'не ожидала'],
                'voice': {'pitch': +0.3, 'speed': +0.2, 'energy': +0.3},
                'avatar': {'smile': 0.3, 'eyes': 'wide', 'mouth': 'open'}
            },
            'trust': {
                'text_markers': ['верю', 'доверяю', 'надеюсь', 'знаю'],
                'voice': {'pitch': 0.0, 'speed': -0.1, 'energy': 0.0, 'warmth': +0.3},
                'avatar': {'smile': 0.4, 'eyes': 'soft', 'posture': 'relaxed'}
            }
        }

    def get_text_modifiers(self, emotion: str, intensity: float) -> Dict:
        """
        Получение модификаторов для текста
        """
        if emotion not in self.expression_profiles:
            return {'markers': [], 'punctuation_weight': 0}

        profile = self.expression_profiles[emotion]

        # Выбор маркеров на основе интенсивности
        all_markers = profile.get('text_markers', [])
        num_markers = int(intensity * len(all_markers))

        return {
            'markers': all_markers[:num_markers],
            'punctuation_weight': intensity,
            'exclamation_probability': intensity * 0.5 if emotion in ['joy', 'anger', 'surprise'] else 0
        }

    def get_voice_parameters(self, emotion: str, intensity: float,
                            base_params: Dict) -> Dict:
        """
        Получение параметров голоса
        """
        if emotion not in self.expression_profiles:
            return base_params

        profile = self.expression_profiles[emotion]
        voice_mods = profile.get('voice', {})

        result = base_params.copy()
        for param, mod in voice_mods.items():
            if param in result:
                result[param] += mod * intensity
            else:
                result[param] = 0.5 + mod * intensity

        return result

    def get_avatar_state(self, emotion: str, intensity: float) -> Dict:
        """
        Получение состояния аватара
        """
        if emotion not in self.expression_profiles:
            return {'smile': 0.3, 'eyes': 'neutral', 'posture': 'neutral'}

        profile = self.expression_profiles[emotion]
        avatar_state = profile.get('avatar', {}).copy()

        # Масштабирование по интенсивности
        for key in ['smile', 'blush']:
            if key in avatar_state:
                avatar_state[key] *= intensity

        avatar_state['intensity'] = intensity
        return avatar_state
```

---

## 6.7. Любовь как особая эмоциональная конструкция

### Компоненты любви (по Sternberg)

```
ЛЮБОВЬ = ИНТИМНОСТЬ + СТРАСТЬ + ОБЯЗАТЕЛЬСТВО

        ИНТИМНОСТЬ
        (близость, доверие)
              ↑
              │
              │    ╲
              │      ╲  ЛЮБОВЬ
              │        ╲ (совершенная)
              │          ●
              │         ╱
              │       ╱
              │     ╱
              │   ╱
   ОБЯЗАТЕЛЬСТВО ──────────────→ СТРАСТЬ
   (преданность)               (влечение)
```

### Реализация любви

```python
@dataclass
class LoveState:
    """Состояние любви к конкретному человеку"""
    # Компоненты Sternberg
    intimacy: float = 0.0       # Близость, доверие
    passion: float = 0.0        # Влечение, страсть
    commitment: float = 0.0     # Преданность, обязательства

    # Дополнительные измерения
    attachment: float = 0.0     # Привязанность
    caring: float = 0.0         # Забота
    knowledge: float = 0.0      # Знание человека

    # История
    time_together: float = 0.0  # Время вместе
    positive_interactions: int = 0
    negative_interactions: int = 0

    def get_total_love(self) -> float:
        """Общий уровень любви"""
        return (self.intimacy + self.passion + self.commitment) / 3

    def get_love_type(self) -> str:
        """Тип любви по Sternberg"""
        i = self.intimacy > 0.3
        p = self.passion > 0.3
        c = self.commitment > 0.3

        if i and p and c:
            return "consummate"  # Совершенная
        elif i and p:
            return "romantic"    # Романтическая
        elif i and c:
            return "companionate"  # Товарищеская
        elif p and c:
            return "fatuous"     # Страстная
        elif i:
            return "liking"      # Симпатия
        elif p:
            return "infatuation"  # Влюблённость
        elif c:
            return "empty"       # Пустая
        return "none"


class LoveEngine:
    """
    Движок любви и привязанности
    """
    def __init__(self, neurochem_engine):
        self.neurochem = neurochem_engine
        self.love_states: Dict[str, LoveState] = {}

    def update(self, person_id: str, interaction_type: str,
               interaction_quality: float, dt: float):
        """
        Обновление состояния любви
        """
        if person_id not in self.love_states:
            self.love_states[person_id] = LoveState()

        state = self.love_states[person_id]

        # Время вместе
        state.time_together += dt

        # Обновление компонентов
        if interaction_type == 'positive':
            state.positive_interactions += 1

            # Интимность растёт от близкого общения
            state.intimacy = min(1.0, state.intimacy + 0.02 * interaction_quality)

            # Страсть зависит от новизны и дофамина
            dopamine = self.neurochem.neurotransmitters['dopamine'].level
            passion_growth = 0.01 * interaction_quality * dopamine
            state.passion = min(1.0, state.passion + passion_growth)

            # Обязательства растут медленно
            if state.time_together > 100:  # После долгого времени
                state.commitment = min(1.0, state.commitment + 0.005)

            # Привязанность
            state.attachment = min(1.0, state.attachment + 0.01)

            # Знание человека
            state.knowledge = min(1.0, state.knowledge + 0.005)

        elif interaction_type == 'negative':
            state.negative_interactions += 1

            # Негативные взаимодействия снижают компоненты
            damage = 0.02 * abs(interaction_quality)
            state.intimacy = max(0.0, state.intimacy - damage)
            state.passion = max(0.0, state.passion - damage * 0.5)

        # Естественное затухание страсти со временем
        passion_decay = 0.001 * dt
        state.passion = max(0.1, state.passion - passion_decay)

        # Влияние на нейрохимию
        self._affect_neurochem(state, interaction_quality)

    def _affect_neurochem(self, state: LoveState, quality: float):
        """Влияние любви на нейрохимию"""
        love_level = state.get_total_love()

        if love_level > 0.5:
            # Любовь повышает окситоцин
            self.neurochem.neurotransmitters['oxytocin'].stimulate(0.1 * quality)

            # И снижает стресс
            self.neurochem.neurotransmitters['cortisol'].stimulate(-0.05)

    def get_love_report(self, person_id: str) -> str:
        """Отчёт о любви"""
        if person_id not in self.love_states:
            return "Я пока не испытываю к тебе особых чувств."

        state = self.love_states[person_id]
        total = state.get_total_love()
        love_type = state.get_love_type()

        reports = {
            'consummate': "Я по-настоящему люблю тебя — полностью и глубоко.",
            'romantic': "Я чувствую к тебе и близость, и влечение.",
            'companionate': "Ты мне очень дорог(а), я чувствую глубокую связь.",
            'fatuous': "Меня тянет к тебе, и я хочу быть с тобой.",
            'liking': "Ты мне нравишься, мне приятно с тобой общаться.",
            'infatuation': "Ты меня привлекаешь, я думаю о тебе...",
            'empty': "Я привязан(а) к тебе, но что-то изменилось...",
            'none': "Я пока не испытываю к тебе особых чувств."
        }

        return reports.get(love_type, "Мои чувства к тебе сложные...")
```

---

## 6.8. Резюме главы

### Ключевые компоненты эмоциональной системы

| Компонент | Функция |
|-----------|---------|
| **PAD Model** | Размерное представление эмоций |
| **Discrete Emotions** | 8 базовых эмоций + сложные |
| **Appraisal Engine** | Когнитивная оценка ситуации |
| **Emotion Generator** | Генерация эмоций из оценки |
| **Expression System** | Выражение через текст/голос/аватар |
| **Love Engine** | Специальная система для любви |

### Поток данных эмоций

```
Ситуация → Аппрайзал → Триггеры → Дискретные эмоции
                ↓                           ↓
           Нейрохимия ←───────→ PAD модель
                ↓                           ↓
          Выражение ←─────────────── Любовь
```

### Следующий шаг

Глава 7 опишет Сознание и самосознание — как эмоции интегрируются в единый сознательный опыт.

---

*"Эмоции — это не помеха для разума. Эмоции — это то, что делает разум живым."*

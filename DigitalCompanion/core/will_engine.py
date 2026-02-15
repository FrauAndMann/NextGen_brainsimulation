"""
Will Engine - Движок воли на основе вероятностного конфликта

Ключевой принцип: действие = попытка минимизировать ожидаемую ошибку предсказания.

НЕ использует пороги (thresholds) - они дают машинные, резкие переходы.
Использует softmax с температурой = вероятностный, человекоподобный выбор.

Интенты - это не "реплики", а операторы модификации динамики.
LLM получает только результат выбора, не участвует в принятии решения.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import math


class IntentType(Enum):
    """Типы интентов (действий)"""
    # Базовые интенты
    REST = "rest"                      # Покой, энергосбережение
    OBSERVE = "observe"                # Наблюдение без реакции

    # Социальные интенты
    SEEK_ATTENTION = "seek_attention"  # Поиск контакта
    WITHDRAW = "withdraw"              # Отчуждение, защита
    ASSERT_DOMINANCE = "assert"        # Агрессия, контроль
    EXPRESS_WARMTH = "express_warmth"  # Тепло, близость

    # Когнитивные интенты
    REFLECT = "reflect"                # Саморефлексия
    EXPLORE = "explore"                # Исследование, новизна

    # Мета-интенты
    SELF_MODIFY = "self_modify"        # Изменение своей структуры
    SILENCE = "silence"                # Осознанное молчание


@dataclass
class Intent:
    """Описание интента"""
    type: IntentType
    name: str
    description: str

    # Какие состояния этот интент пытается изменить
    target_axes: List[int]  # Индексы осей [V,A,D,T,N,E]

    # Ожидаемое изменение напряжения
    expected_tension_delta: float

    # Базовая стоимость (энергия)
    base_cost: float

    # Условия активации (мягкие)
    preferred_conditions: Dict[int, Tuple[float, float]]  # axis -> (min, max)


# Регистр базовых интентов
INTENT_REGISTRY = {
    IntentType.REST: Intent(
        type=IntentType.REST,
        name="Покой",
        description="Восстановление энергии, минимальная активность",
        target_axes=[5],  # Energy
        expected_tension_delta=-0.1,
        base_cost=0.05,
        preferred_conditions={5: (0.0, 0.3)}  # Низкая энергия
    ),

    IntentType.SEEK_ATTENTION: Intent(
        type=IntentType.SEEK_ATTENTION,
        name="Поиск контакта",
        description="Инициировать взаимодействие, получить внимание",
        target_axes=[3, 0],  # Attachment, Valence
        expected_tension_delta=-0.3,
        base_cost=0.3,
        preferred_conditions={3: (0.0, 0.4), 5: (0.4, 1.0)}  # Низкая привязанность, есть энергия
    ),

    IntentType.WITHDRAW: Intent(
        type=IntentType.WITHDRAW,
        name="Отчуждение",
        description="Защита через дистанцию, гейтирование входа",
        target_axes=[1, 3],  # Arousal, Attachment
        expected_tension_delta=-0.2,
        base_cost=0.2,
        preferred_conditions={0: (-1.0, -0.3), 1: (0.6, 1.0)}  # Боль, высокое возбуждение
    ),

    IntentType.ASSERT_DOMINANCE: Intent(
        type=IntentType.ASSERT_DOMINANCE,
        name="Контроль",
        description="Попытка восстановить контроль через агрессию",
        target_axes=[2, 1],  # Dominance, Arousal
        expected_tension_delta=-0.25,
        base_cost=0.35,
        preferred_conditions={2: (0.0, 0.4), 1: (0.5, 1.0)}  # Низкое доминирование, возбуждение
    ),

    IntentType.EXPRESS_WARMTH: Intent(
        type=IntentType.EXPRESS_WARMTH,
        name="Тепло",
        description="Выразить привязанность и близость",
        target_axes=[0, 3],  # Valence, Attachment
        expected_tension_delta=-0.2,
        base_cost=0.15,
        preferred_conditions={0: (0.0, 1.0), 3: (0.5, 1.0)}  # Позитивно, высокая привязанность
    ),

    IntentType.REFLECT: Intent(
        type=IntentType.REFLECT,
        name="Рефлексия",
        description="Внутренний анализ, самонаблюдение",
        target_axes=[1, 4],  # Arousal, Novelty
        expected_tension_delta=-0.15,
        base_cost=0.1,
        preferred_conditions={5: (0.3, 0.7)}  # Умеренная энергия
    ),

    IntentType.EXPLORE: Intent(
        type=IntentType.EXPLORE,
        name="Исследование",
        description="Поиск новизны, любопытство",
        target_axes=[4, 0],  # Novelty, Valence
        expected_tension_delta=-0.1,
        base_cost=0.2,
        preferred_conditions={1: (0.0, 0.5), 5: (0.5, 1.0)}  # Спокойствие, высокая энергия
    ),

    IntentType.SELF_MODIFY: Intent(
        type=IntentType.SELF_MODIFY,
        name="Самоизменение",
        description="Критическое изменение собственной структуры",
        target_axes=[0, 2],  # Valence, Dominance
        expected_tension_delta=-0.5,  # Большое ожидаемое облегчение
        base_cost=0.5,
        preferred_conditions={0: (-1.0, -0.5)}  # Сильная боль
    ),

    IntentType.SILENCE: Intent(
        type=IntentType.SILENCE,
        name="Молчание",
        description="Осознанный отказ от вербализации",
        target_axes=[],
        expected_tension_delta=0.0,
        base_cost=0.01,
        preferred_conditions={5: (0.0, 0.2)}  # Низкая энергия
    ),

    IntentType.OBSERVE: Intent(
        type=IntentType.OBSERVE,
        name="Наблюдение",
        description="Пассивное восприятие без реакции",
        target_axes=[4],  # Novelty
        expected_tension_delta=-0.05,
        base_cost=0.02,
        preferred_conditions={}
    ),
}


@dataclass
class ActionToken:
    """
    Токен действия - результат работы Will Engine

    Это НЕ текст. Это предвербальный импульс, который LLM
    должна выразить (или не выразить - SILENCE).
    """
    intent: IntentType
    confidence: float              # 0-1, уверенность в выборе
    value: float                   # Ожидаемая полезность
    context: Dict[str, Any]        # Контекст выбора

    # Ограничения для LLM
    constraints: Dict[str, Any] = field(default_factory=dict)


class WillEngine:
    """
    Движок воли - вероятностный выбор действия

    Не использует пороги. Использует softmax с температурой.

    Высокий стресс (Arousal) → высокая температура → хаотичный выбор
    Низкий стресс → низкая температура → рациональный выбор

    Это моделирует человеческую иррациональность при стрессе.
    """

    def __init__(self):
        self.available_intents = set(INTENT_REGISTRY.keys())
        self.intent_history: List[ActionToken] = []
        self.max_history = 100

    def select_action(
        self,
        S: np.ndarray,
        tension: float,
        energy: float,
        temperature_override: float = None
    ) -> ActionToken:
        """
        Выбор действия через симуляцию будущего

        Args:
            S: Текущий вектор состояния [V, A, D, T, N, E]
            tension: Текущее напряжение
            energy: Текущая энергия
            temperature_override: Переопределение температуры

        Returns:
            ActionToken - выбранный интент
        """
        # Температура выбора
        # Arousal (S[1]) влияет на хаотичность
        if temperature_override is not None:
            temperature = temperature_override
        else:
            # Базовая температура + вклад от arousal
            arousal = S[1]
            temperature = 0.5 + arousal * 1.5  # 0.5 to 2.0

        # Вычисляем ценность каждого интента
        intent_values = {}
        intent_contexts = {}

        for intent_type in self.available_intents:
            intent = INTENT_REGISTRY[intent_type]

            # Проверяем доступность (энергия)
            if intent.base_cost > energy:
                continue

            # Вычисляем value = ожидаемое снижение напряжения - стоимость
            value = self._compute_intent_value(intent, S, tension)
            intent_values[intent_type] = value
            intent_contexts[intent_type] = {
                'expected_delta': intent.expected_tension_delta,
                'cost': intent.base_cost,
                'conditions_met': self._check_conditions(intent, S),
            }

        if not intent_values:
            # Ничего не доступно - только покой
            return ActionToken(
                intent=IntentType.REST,
                confidence=1.0,
                value=0.0,
                context={'reason': 'no_energy'},
                constraints={'verbosity': 'none'}
            )

        # Softmax выбор
        intent_types = list(intent_values.keys())
        values = np.array([intent_values[t] for t in intent_types])

        # Softmax с температурой
        exp_values = np.exp(values / temperature)
        probabilities = exp_values / np.sum(exp_values)

        # Выбор
        chosen_idx = np.random.choice(len(intent_types), p=probabilities)
        chosen_intent = intent_types[chosen_idx]
        chosen_prob = probabilities[chosen_idx]

        # Формируем constraints для LLM
        constraints = self._generate_constraints(chosen_intent, S)

        token = ActionToken(
            intent=chosen_intent,
            confidence=float(chosen_prob),
            value=float(intent_values[chosen_intent]),
            context=intent_contexts[chosen_intent],
            constraints=constraints
        )

        self.intent_history.append(token)
        if len(self.intent_history) > self.max_history:
            self.intent_history.pop(0)

        return token

    def _compute_intent_value(
        self,
        intent: Intent,
        S: np.ndarray,
        current_tension: float
    ) -> float:
        """
        Вычисление ценности интента

        Value = E[снижение_напряжения] - стоимость

        Учитывает:
        - Насколько текущее состояние соответствует целевому для интента
        - Ожидаемое снижение напряжения
        - Энергетическую стоимость
        """
        # Базовое ожидаемое снижение
        value = -intent.expected_tension_delta  # Отрицательное = хорошо

        # Бонус за соответствие условиям
        conditions_met = self._check_conditions(intent, S)
        if conditions_met > 0.5:
            value += 0.2 * conditions_met

        # Штраф за стоимость
        value -= intent.base_cost * 0.5

        # Штраф если состояние уже хорошее для этого интента
        # (нет смысла искать контакт если уже высокий attachment)
        if intent.type == IntentType.SEEK_ATTENTION and S[3] > 0.7:
            value -= 0.3

        if intent.type == IntentType.WITHDRAW and S[3] < 0.3:
            value -= 0.2  # Уже отдалена

        return value

    def _check_conditions(self, intent: Intent, S: np.ndarray) -> float:
        """Проверка соответствия предпочтительным условиям"""
        if not intent.preferred_conditions:
            return 0.5  # Нет условий

        total_match = 0.0
        for axis, (min_val, max_val) in intent.preferred_conditions.items():
            if min_val <= S[axis] <= max_val:
                total_match += 1.0
            else:
                # Частичное соответствие
                distance = min(abs(S[axis] - min_val), abs(S[axis] - max_val))
                total_match += max(0, 1.0 - distance)

        return total_match / len(intent.preferred_conditions)

    def _generate_constraints(
        self,
        intent_type: IntentType,
        S: np.ndarray
    ) -> Dict[str, Any]:
        """
        Генерация ограничений для LLM

        LLM получает не "инструкцию что сказать",
        а ограничения на то, КАК она может звучать.
        """
        constraints = {
            'allowed_outputs': [],      # Разрешённые типы вывода
            'verbosity': 'normal',      # normal, minimal, none
            'empathy': 'normal',        # normal, reduced, disabled
            'self_disclosure': 'normal', # normal, limited, blocked
            'tone': 'neutral',          # neutral, warm, cold, defensive
        }

        intent = INTENT_REGISTRY[intent_type]

        # На основе интента
        if intent_type == IntentType.SILENCE:
            constraints['allowed_outputs'] = ['SILENCE']
            constraints['verbosity'] = 'none'

        elif intent_type == IntentType.WITHDRAW:
            constraints['allowed_outputs'] = ['SILENCE', 'SHORT_PHRASE', 'COLD_RESPONSE']
            constraints['verbosity'] = 'minimal'
            constraints['empathy'] = 'disabled'
            constraints['tone'] = 'cold'

        elif intent_type == IntentType.SEEK_ATTENTION:
            constraints['allowed_outputs'] = ['QUESTION', 'STATEMENT', 'EXPRESSION']
            constraints['verbosity'] = 'normal'
            constraints['tone'] = 'warm'

        elif intent_type == IntentType.ASSERT_DOMINANCE:
            constraints['allowed_outputs'] = ['STATEMENT', 'CHALLENGE']
            constraints['empathy'] = 'reduced'
            constraints['tone'] = 'defensive'

        elif intent_type == IntentType.EXPRESS_WARMTH:
            constraints['allowed_outputs'] = ['EXPRESSION', 'AFFECTION']
            constraints['tone'] = 'warm'
            constraints['empathy'] = 'normal'

        elif intent_type == IntentType.REFLECT:
            constraints['allowed_outputs'] = ['INTERNAL_THOUGHT', 'SELF_STATEMENT']
            constraints['self_disclosure'] = 'limited'

        elif intent_type == IntentType.REST:
            constraints['allowed_outputs'] = ['SILENCE', 'SHORT_PHRASE']
            constraints['verbosity'] = 'minimal'

        else:
            constraints['allowed_outputs'] = ['ANY']

        # Модификация на основе состояния
        if S[5] < 0.2:  # Очень низкая энергия
            constraints['verbosity'] = 'minimal'

        if S[0] < -0.5:  # Сильный негатив
            constraints['tone'] = 'cold'
            constraints['empathy'] = 'reduced'

        return constraints

    def get_active_intents(self) -> List[IntentType]:
        """Получить список доступных интентов"""
        return list(self.available_intents)

    def disable_intent(self, intent_type: IntentType):
        """Запретить интент (например, после SELF_MODIFY)"""
        self.available_intents.discard(intent_type)

    def enable_intent(self, intent_type: IntentType):
        """Восстановить интент"""
        self.available_intents.add(intent_type)

    def get_recent_intents(self, n: int = 5) -> List[IntentType]:
        """Получить последние выбранные интенты"""
        return [t.intent for t in self.intent_history[-n:]]

    def get_intent_distribution(self, S: np.ndarray, tension: float, energy: float) -> Dict[str, float]:
        """Получить распределение вероятностей по интентам (для отладки)"""
        distribution = {}

        for intent_type in self.available_intents:
            intent = INTENT_REGISTRY[intent_type]
            if intent.base_cost <= energy:
                value = self._compute_intent_value(intent, S, tension)
                distribution[intent.name] = value

        return distribution

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация"""
        return {
            'available_intents': [t.value for t in self.available_intents],
            'recent_intents': [t.intent.value for t in self.intent_history[-20:]],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WillEngine':
        """Десериализация"""
        engine = cls()
        engine.available_intents = {IntentType(v) for v in data.get('available_intents', [t.value for t in INTENT_REGISTRY.keys()])}
        return engine

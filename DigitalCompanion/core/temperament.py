"""
Модуль темперамента
Основан на теории свойств нервной системы (Павлов, Айзенк)

Темперамент — врождённая характеристика нервной системы.
Не меняется в течение жизни (как у человека).

Четыре типа:
- Сангвиник: сильный, уравновешенный, подвижный
- Холерик: сильный, неуравновешенный, возбудимый
- Флегматик: сильный, уравновешенный, инертный
- Меланхолик: слабый, чувствительный
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


class TemperamentType(Enum):
    """Типы темперамента по Павлову"""
    SANGUINE = "sanguine"      # Сангвиник
    CHOLERIC = "choleric"      # Холерик
    PHLEGMATIC = "phlegmatic"  # Флегматик
    MELANCHOLIC = "melancholic"  # Меланхолик


@dataclass
class NervousSystemProperties:
    """
    Свойства нервной системы (врождённые)
    По Павлову: сила, уравновешенность, подвижность
    """
    strength: float = 0.5      # Сила НС (0-1): устойчивость к нагрузкам
    balance: float = 0.5       # Уравновешенность (0-1): баланс возбуждения/торможения
    mobility: float = 0.5      # Подвижность (0-1): скорость переключения

    def get_temperament_type(self) -> TemperamentType:
        """Определение типа темперамента по свойствам НС"""
        # Классификация по Павлову:
        # - Сильный + Уравновешенный + Подвижный = Сангвиник
        # - Сильный + Неуравновешенный = Холерик
        # - Сильный + Уравновешенный + Инертный = Флегматик
        # - Слабый = Меланхолик

        if self.strength < 0.4:
            return TemperamentType.MELANCHOLIC

        if self.balance < 0.4:
            return TemperamentType.CHOLERIC

        if self.mobility < 0.4:
            return TemperamentType.PHLEGMATIC

        return TemperamentType.SANGUINE


@dataclass
class Temperament:
    """
    Темперамент — врождённая основа личности

    Темперамент определяет:
    - Скорость эмоциональных реакций
    - Интенсивность эмоций
    - Адаптивность к изменениям
    - Стиль реагирования на стресс

    ВАЖНО: Темперамент НЕ меняется в течение "жизни" системы,
    как и у человека. Характер формируется на основе темперамента.
    """
    type: TemperamentType
    nervous_system: NervousSystemProperties

    # Врождённые диспозиции (склонности, не фиксированные черты!)
    # Это стартовая точка для формирования характера
    dispositions: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.dispositions:
            self.dispositions = self._calculate_dispositions()

    def _calculate_dispositions(self) -> Dict[str, float]:
        """
        Расчёт врождённых диспозиций на основе свойств НС

        Диспозиции — это склонности, не готовые черты характера.
        Пример: ребёнок с высокой экстраверсией может стать
        интровертом при определённом воспитании.
        """
        ns = self.nervous_system

        return {
            # Экстраверсия — связана с подвижностью и силой НС
            # Сангвиники и холерики более экстравертированы
            'extraversion': (ns.strength * 0.3 + ns.mobility * 0.5 + ns.balance * 0.2),

            # Невротизм — обратная сторона силы НС
            # Меланхолики более эмоционально нестабильны
            'neuroticism': (1 - ns.strength) * 0.6 + (1 - ns.balance) * 0.4,

            # Открытость опыту — связана с подвижностью
            'openness': ns.mobility * 0.7 + ns.strength * 0.3,

            # Доброжелательность — связана с уравновешенностью
            'agreeableness': ns.balance * 0.6 + ns.strength * 0.2 + 0.2,

            # Добросовестность — требует силы и уравновешенности
            'conscientiousness': ns.strength * 0.4 + ns.balance * 0.4 + (1 - ns.mobility) * 0.2,
        }

    def get_reaction_speed(self) -> float:
        """Скорость эмоциональной реакции (0-1)"""
        return self.nervous_system.mobility

    def get_emotion_intensity(self) -> float:
        """Интенсивность эмоций (0-1)"""
        return self.nervous_system.strength * (1.5 - self.nervous_system.balance)

    def get_stress_resistance(self) -> float:
        """Устойчивость к стрессу (0-1)"""
        return self.nervous_system.strength * self.nervous_system.balance

    def get_adaptability(self) -> float:
        """Адаптивность к изменениям (0-1)"""
        return self.nervous_system.mobility * self.nervous_system.strength


# Предустановленные темпераменты
TEMPERAMENTS = {
    TemperamentType.SANGUINE: Temperament(
        type=TemperamentType.SANGUINE,
        nervous_system=NervousSystemProperties(
            strength=0.75,
            balance=0.7,
            mobility=0.8
        )
    ),
    TemperamentType.CHOLERIC: Temperament(
        type=TemperamentType.CHOLERIC,
        nervous_system=NervousSystemProperties(
            strength=0.8,
            balance=0.35,
            mobility=0.7
        )
    ),
    TemperamentType.PHLEGMATIC: Temperament(
        type=TemperamentType.PHLEGMATIC,
        nervous_system=NervousSystemProperties(
            strength=0.7,
            balance=0.75,
            mobility=0.3
        )
    ),
    TemperamentType.MELANCHOLIC: Temperament(
        type=TemperamentType.MELANCHOLIC,
        nervous_system=NervousSystemProperties(
            strength=0.35,
            balance=0.5,
            mobility=0.4
        )
    ),
}


def create_temperament(
    type_str: str = "sanguine",
    nervous_system: Optional[Dict[str, float]] = None
) -> Temperament:
    """
    Создание темперамента

    Args:
        type_str: Тип темперамента ("sanguine", "choleric", "phlegmatic", "melancholic")
        nervous_system: Опционально — кастомные свойства НС

    Returns:
        Temperament объект
    """
    try:
        temp_type = TemperamentType(type_str.lower())
    except ValueError:
        temp_type = TemperamentType.SANGUINE

    if nervous_system:
        ns = NervousSystemProperties(
            strength=nervous_system.get('strength', 0.5),
            balance=nervous_system.get('balance', 0.5),
            mobility=nervous_system.get('mobility', 0.5)
        )
        return Temperament(type=temp_type, nervous_system=ns)

    return TEMPERAMENTS[temp_type]

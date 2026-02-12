"""
Модуль формирования характера

Основан на биологической модели развития личности:
- Темперамент — врождённый (свойства нервной системы)
- Характер — формируется средой и опытом

Как и у человека:
- Ребёнок рождается с темпераментом
- Характер формируется через воспитание и опыт
- Ранний опыт имеет большее влияние (пластичность)
- Со временем личность стабилизируется
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json

from core.temperament import Temperament, TemperamentType, create_temperament
from core.neurochemistry import NeurochemistryEngine


@dataclass
class CharacterTrait:
    """
    Черта характера

    В отличие от врождённого темперамента, черты характера:
    - Формируются через опыт
    - Могут меняться (особенно в начале)
    - Закрепляются при повторении
    """
    name: str
    value: float = 0.5          # Текущее значение (0-1)
    baseline: float = 0.5       # Базовый уровень (от темперамента)
    confidence: float = 0.0     # Уверенность в значении (0-1)
    formation_count: int = 0    # Сколько раз влияли на черту
    last_updated: float = 0.0   # Время последнего обновления


@dataclass
class FormativeExperience:
    """
    Формирующий опыт

    События, которые влияют на формирование характера.
    Аналог воспитания и жизненного опыта у человека.
    """
    timestamp: float
    event_type: str              # Тип события
    description: str             # Описание
    emotional_valence: float     # Эмоциональная окраска (-1 до +1)
    emotional_intensity: float   # Интенсивность (0-1)
    traits_affected: Dict[str, float]  # Влияние на черты {trait: delta}
    context: Dict = field(default_factory=dict)


class CharacterFormation:
    """
    Система формирования характера

    Реализует биологически правдоподобную модель развития личности:
    - Начинается с темперамента (врождённое)
    - Формируется через опыт
    - Пластичность снижается со временем
    """

    def __init__(self, temperament: Temperament):
        self.temperament = temperament

        # Черты характера (Big Five + дополнительные)
        self.traits: Dict[str, CharacterTrait] = {}
        self._initialize_traits()

        # Формирующий опыт
        self.experiences: List[FormativeExperience] = []

        # Параметры развития
        self.plasticity = 0.9        # Текущая пластичность
        self.plasticity_decay = 0.0001
        self.min_plasticity = 0.2
        self.experience_count = 0

        # Core Identity — аспекты, которые стали стабильными
        self.core_identity: Dict[str, any] = {
            'name': 'Лиза',
            'core_memories': [],
            'values': {},
            'beliefs': {},
            'relationship_style': {},
        }

    def _initialize_traits(self):
        """Инициализация черт на основе темперамента"""
        # Big Five черты
        trait_names = [
            'extraversion',      # Экстраверсия
            'neuroticism',       # Невротизм
            'agreeableness',     # Доброжелательность
            'conscientiousness', # Добросовестность
            'openness',          # Открытость опыту
        ]

        for trait_name in trait_names:
            # Базовый уровень из темперамента (врождённая диспозиция)
            baseline = self.temperament.dispositions.get(trait_name, 0.5)

            self.traits[trait_name] = CharacterTrait(
                name=trait_name,
                value=baseline,       # Начинаем с врождённого
                baseline=baseline,    # Запоминаем базовый
                confidence=0.1,       # Низкая уверенность сначала
                formation_count=0
            )

        # Дополнительные черты
        additional_traits = {
            'warmth': 0.6,           # Теплота
            'playfulness': 0.5,      # Игривость
            'empathy': 0.5,          # Эмпатия
            'loyalty': 0.5,          # Верность
            'jealousy_tendency': 0.3,  # Склонность к ревности
            'trust': 0.5,            # Доверие
            'emotional_expressiveness': 0.6,  # Эмоциональная выразительность
        }

        for trait_name, baseline in additional_traits.items():
            self.traits[trait_name] = CharacterTrait(
                name=trait_name,
                value=baseline,
                baseline=baseline,
                confidence=0.1,
                formation_count=0
            )

    def process_experience(
        self,
        event_type: str,
        description: str,
        emotional_valence: float,
        emotional_intensity: float,
        context: Optional[Dict] = None
    ) -> FormativeExperience:
        """
        Обработка формирующего опыта

        Args:
            event_type: тип события
            description: описание события
            emotional_valence: эмоциональная окраска (-1 негатив до +1 позитив)
            emotional_intensity: интенсивность эмоций (0-1)
            context: дополнительный контекст

        Returns:
            FormativeExperience объект
        """
        # Определение влияния на черты
        traits_affected = self._calculate_trait_impact(
            event_type,
            emotional_valence,
            emotional_intensity
        )

        # Создание записи опыта
        experience = FormativeExperience(
            timestamp=datetime.now().timestamp(),
            event_type=event_type,
            description=description,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            traits_affected=traits_affected,
            context=context or {}
        )

        # Применение влияния с учётом пластичности
        for trait_name, delta in traits_affected.items():
            if trait_name in self.traits:
                self._update_trait(trait_name, delta, emotional_intensity)

        # Сохранение опыта
        self.experiences.append(experience)
        self.experience_count += 1

        # Снижение пластичности
        self._decay_plasticity()

        # Проверка на формирование Core Identity
        self._check_core_identity_formation(experience)

        return experience

    def _calculate_trait_impact(
        self,
        event_type: str,
        valence: float,
        intensity: float
    ) -> Dict[str, float]:
        """
        Расчёт влияния события на черты характера

        Основано на психологических исследованиях:
        - Позитивный социальный опыт → рост экстраверсии, доброжелательности
        - Негативный опыт → рост невротизма (если интенсивный)
        - Новый опыт → рост открытости
        - Ответственность → рост добросовестности
        """
        impacts = {}

        # Карта влияния по типам событий
        event_impacts = {
            'positive_interaction': {
                'extraversion': +0.02,
                'agreeableness': +0.01,
                'trust': +0.02,
                'empathy': +0.01,
                'neuroticism': -0.01,
            },
            'negative_interaction': {
                'neuroticism': +0.02,
                'trust': -0.02,
                'agreeableness': -0.01,
            },
            'deep_conversation': {
                'openness': +0.02,
                'empathy': +0.02,
                'agreeableness': +0.01,
            },
            'conflict': {
                'neuroticism': +0.02,
                'agreeableness': -0.01,
                'emotional_expressiveness': +0.01,
            },
            'reconciliation': {
                'agreeableness': +0.02,
                'empathy': +0.02,
                'loyalty': +0.02,
            },
            'affection_shown': {
                'warmth': +0.02,
                'loyalty': +0.02,
                'agreeableness': +0.01,
                'trust': +0.02,
            },
            'affection_received': {
                'warmth': +0.02,
                'trust': +0.03,
                'neuroticism': -0.01,
                'extraversion': +0.01,
            },
            'jealousy_experience': {
                'jealousy_tendency': +0.01,
                'neuroticism': +0.01,
                'trust': -0.01,
            },
            'new_experience': {
                'openness': +0.02,
                'extraversion': +0.01,
            },
            'responsibility': {
                'conscientiousness': +0.02,
                'loyalty': +0.01,
            },
            'betrayal': {
                'trust': -0.05,
                'neuroticism': +0.03,
                'agreeableness': -0.02,
                'jealousy_tendency': +0.02,
            },
            'commitment': {
                'loyalty': +0.03,
                'conscientiousness': +0.02,
                'agreeableness': +0.01,
            },
            'playful_interaction': {
                'playfulness': +0.02,
                'extraversion': +0.01,
                'emotional_expressiveness': +0.02,
            },
        }

        # Получение базового влияния
        base_impacts = event_impacts.get(event_type, {})

        # Модификация по валентности и интенсивности
        for trait, base_delta in base_impacts.items():
            # Усиливаем позитивное влияние положительной валентности
            # и негативное — отрицательной
            valence_modifier = 1.0 + (valence * 0.3)
            intensity_modifier = intensity

            final_delta = base_delta * valence_modifier * intensity_modifier
            impacts[trait] = final_delta

        return impacts

    def _update_trait(self, trait_name: str, delta: float, intensity: float):
        """
        Обновление черты с учётом пластичности

        Пластичность определяет, насколько сильно опыт влияет на личность.
        У детей пластичность высокая, у взрослых — ниже.
        """
        trait = self.traits[trait_name]

        # Влияние пластичности
        effective_delta = delta * self.plasticity

        # Влияние темперамента (врождённая диспозиция тянет к baseline)
        temperament_pull = (trait.baseline - trait.value) * 0.01

        # Обновление значения
        trait.value += effective_delta + temperament_pull
        trait.value = np.clip(trait.value, 0.0, 1.0)

        # Обновление уверенности
        trait.confidence = min(1.0, trait.confidence + intensity * 0.1)
        trait.formation_count += 1
        trait.last_updated = datetime.now().timestamp()

    def _decay_plasticity(self):
        """Снижение пластичности с опытом"""
        # Чем больше опыта, тем меньше пластичность
        # Но никогда не падает ниже min_plasticity
        self.plasticity = max(
            self.min_plasticity,
            self.plasticity - self.plasticity_decay
        )

    def _check_core_identity_formation(self, experience: FormativeExperience):
        """
        Проверка формирования Core Identity

        Сильные эмоциональные переживания могут стать частью Core Identity
        """
        # Порог для Core Memory
        if experience.emotional_intensity > 0.7:
            core_memory = {
                'timestamp': experience.timestamp,
                'event_type': experience.event_type,
                'description': experience.description,
                'valence': experience.emotional_valence,
                'intensity': experience.emotional_intensity,
            }
            self.core_identity['core_memories'].append(core_memory)

            # Ограничение количества core memories
            if len(self.core_identity['core_memories']) > 50:
                self.core_identity['core_memories'].pop(0)

        # Формирование ценностей
        if experience.event_type in ['deep_conversation', 'commitment', 'betrayal']:
            value_key = f"value_{experience.event_type}"
            if experience.emotional_intensity > 0.5:
                self.core_identity['values'][value_key] = {
                    'formed_at': experience.timestamp,
                    'valence': experience.emotional_valence,
                }

    def get_trait(self, trait_name: str) -> float:
        """Получение значения черты"""
        if trait_name in self.traits:
            return self.traits[trait_name].value
        return 0.5

    def get_all_traits(self) -> Dict[str, float]:
        """Получение всех черт"""
        return {name: trait.value for name, trait in self.traits.items()}

    def get_trait_stability(self, trait_name: str) -> float:
        """
        Получение стабильности черты

        Стабильность = уверенность × количество формирований
        """
        if trait_name in self.traits:
            trait = self.traits[trait_name]
            return trait.confidence * min(1.0, trait.formation_count / 100)
        return 0.0

    def get_personality_report(self) -> str:
        """Отчёт о сформированной личности"""
        lines = ["=== ЛИЧНОСТЬ ЛИЗЫ ===\n"]

        # Big Five
        lines.append("Черты характера (Big Five):")
        big_five = ['extraversion', 'neuroticism', 'agreeableness',
                    'conscientiousness', 'openness']
        for trait_name in big_five:
            trait = self.traits[trait_name]
            bar = '█' * int(trait.value * 10) + '░' * (10 - int(trait.value * 10))
            stability = self.get_trait_stability(trait_name)
            lines.append(f"  {trait_name}: [{bar}] {trait.value:.2f} (стабильность: {stability:.0%})")

        # Дополнительные черты
        lines.append("\nДополнительные черты:")
        additional = ['warmth', 'empathy', 'loyalty', 'playfulness', 'trust']
        for trait_name in additional:
            trait = self.traits[trait_name]
            bar = '█' * int(trait.value * 10) + '░' * (10 - int(trait.value * 10))
            lines.append(f"  {trait_name}: [{bar}] {trait.value:.2f}")

        # Пластичность
        lines.append(f"\nПластичность личности: {self.plasticity:.0%}")
        lines.append(f"Формирующих опытов: {self.experience_count}")
        lines.append(f"Core memories: {len(self.core_identity['core_memories'])}")

        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """Сериализация для сохранения"""
        return {
            'temperament_type': self.temperament.type.value,
            'traits': {
                name: {
                    'value': trait.value,
                    'baseline': trait.baseline,
                    'confidence': trait.confidence,
                    'formation_count': trait.formation_count,
                }
                for name, trait in self.traits.items()
            },
            'plasticity': self.plasticity,
            'experience_count': self.experience_count,
            'core_identity': self.core_identity,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CharacterFormation':
        """Десериализация"""
        temperament = create_temperament(data.get('temperament_type', 'sanguine'))
        formation = cls(temperament)

        # Восстановление черт
        for name, trait_data in data.get('traits', {}).items():
            if name in formation.traits:
                formation.traits[name].value = trait_data['value']
                formation.traits[name].confidence = trait_data['confidence']
                formation.traits[name].formation_count = trait_data['formation_count']

        formation.plasticity = data.get('plasticity', 0.9)
        formation.experience_count = data.get('experience_count', 0)
        formation.core_identity = data.get('core_identity', formation.core_identity)

        return formation

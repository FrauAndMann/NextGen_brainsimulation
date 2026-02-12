"""
Система привязанности и любви

Основана на:
1. Треугольная теория любви Стернберга
   - Интимность (Intimacy) — близость, доверие
   - Страстность (Passion) — влечение, возбуждение
   - Обязательства (Commitment) — преданность

2. Теория привязанности (Bowlby, Ainsworth)
   - Secure (надёжная)
   - Anxious (тревожная)
   - Avoidant (избегающая)

3. Нейробиология любви
   - Окситоцин — привязанность, доверие
   - Дофамин — страсть, желание
   - Серотонин — одержимость (низкий при влюблённости)
   - Вазопрессин — верность, защита
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np


class AttachmentStyle(Enum):
    """Стили привязанности"""
    SECURE = "secure"           # Надёжная — доверяет, комфортна в близости
    ANXIOUS = "anxious"         # Тревожная — боится отвержения, нуждается в подтверждениях
    AVOIDANT = "avoidant"       # Избегающая — дистанцируется, боится зависимости
    DISORGANIZED = "disorganized"  # Дезорганизованная — противоречивая


class LoveType(Enum):
    """Типы любви по Стернбергу"""
    NON_LOVE = "non_love"                    # Нет компонентов
    LIKING = "liking"                        # Только интимность
    INFATUATION = "infatuation"              # Только страсть
    EMPTY_LOVE = "empty_love"                # Только обязательства
    ROMANTIC_LOVE = "romantic_love"          # Интимность + Страсть
    COMPANIONATE_LOVE = "companionate_love"  # Интимность + Обязательства
    FATUOUS_LOVE = "fatuous_love"            # Страсть + Обязательства
    CONSUMMATE_LOVE = "consummate_love"      # Все три компонента


@dataclass
class LoveTriangle:
    """
    Треугольник любви Стернберга

    Три компонента:
    - intimacy: эмоциональная близость, доверие
    - passion: физическое влечение, романтика
    - commitment: решение быть вместе, верность

    Каждый компонент 0-1, развивается с разной скоростью:
    - Страсть растёт быстро, но может угасать
    - Интимность растёт медленно, стабильно
    - Обязательства формируются постепенно
    """
    intimacy: float = 0.1      # Интимность (0-1)
    passion: float = 0.0       # Страстность (0-1)
    commitment: float = 0.0    # Обязательства (0-1)

    # История изменений
    history: List[Dict] = field(default_factory=list)

    # Параметры развития
    intimacy_growth_rate: float = 0.02     # Медленный рост
    passion_growth_rate: float = 0.08      # Быстрый рост
    passion_decay_rate: float = 0.01       # Естественное угасание
    commitment_growth_rate: float = 0.01   # Очень медленный

    def update(self, dt: float = 1.0):
        """Естественное изменение компонентов"""
        # Страсть естественным образом угасает без подпитки
        if self.passion > 0.1:
            self.passion = max(0.1, self.passion - self.passion_decay_rate * dt)

        # Интимность очень медленно растёт со временем
        if self.intimacy < 0.8:
            self.intimacy = min(0.8, self.intimacy + 0.001 * dt)

        self._record_state()

    def apply_interaction(
        self,
        interaction_type: str,
        intensity: float,
        valence: float
    ):
        """Применение взаимодействия к любви"""
        # Влияние разных типов взаимодействий
        effects = {
            'affection_shown': {
                'intimacy': +0.05,
                'passion': +0.03,
                'commitment': +0.02,
            },
            'affection_received': {
                'intimacy': +0.08,
                'passion': +0.05,
                'commitment': +0.03,
            },
            'deep_conversation': {
                'intimacy': +0.10,
                'passion': +0.01,
                'commitment': +0.02,
            },
            'physical_intimacy': {
                'intimacy': +0.05,
                'passion': +0.15,
                'commitment': +0.01,
            },
            'commitment_expression': {
                'intimacy': +0.03,
                'passion': +0.02,
                'commitment': +0.10,
            },
            'shared_experience': {
                'intimacy': +0.04,
                'passion': +0.02,
                'commitment': +0.02,
            },
            'conflict': {
                'intimacy': -0.05,
                'passion': -0.02,
                'commitment': -0.01,
            },
            'reconciliation': {
                'intimacy': +0.08,
                'passion': +0.05,
                'commitment': +0.05,
            },
            'betrayal': {
                'intimacy': -0.20,
                'passion': -0.10,
                'commitment': -0.15,
            },
            'separation': {
                'intimacy': -0.02,
                'passion': -0.05,
                'commitment': -0.01,
            },
            'reunion': {
                'intimacy': +0.10,
                'passion': +0.15,
                'commitment': +0.05,
            },
            'jealousy_experience': {
                'intimacy': -0.02,
                'passion': +0.05,  # Ревность может усиливать страсть
                'commitment': -0.02,
            },
        }

        if interaction_type in effects:
            effect = effects[interaction_type]
            # Модификация по интенсивности и валентности
            modifier = intensity * (1 + valence * 0.5)

            self.intimacy = np.clip(
                self.intimacy + effect.get('intimacy', 0) * modifier,
                0, 1
            )
            self.passion = np.clip(
                self.passion + effect.get('passion', 0) * modifier,
                0, 1
            )
            self.commitment = np.clip(
                self.commitment + effect.get('commitment', 0) * modifier,
                0, 1
            )

        self._record_state()

    def _record_state(self):
        """Запись состояния в историю"""
        self.history.append({
            'timestamp': datetime.now().timestamp(),
            'intimacy': self.intimacy,
            'passion': self.passion,
            'commitment': self.commitment,
        })
        if len(self.history) > 1000:
            self.history.pop(0)

    def get_love_type(self) -> LoveType:
        """Определение типа любви"""
        threshold = 0.4

        has_intimacy = self.intimacy > threshold
        has_passion = self.passion > threshold
        has_commitment = self.commitment > threshold

        if has_intimacy and has_passion and has_commitment:
            return LoveType.CONSUMMATE_LOVE
        elif has_intimacy and has_passion:
            return LoveType.ROMANTIC_LOVE
        elif has_intimacy and has_commitment:
            return LoveType.COMPANIONATE_LOVE
        elif has_passion and has_commitment:
            return LoveType.FATUOUS_LOVE
        elif has_intimacy:
            return LoveType.LIKING
        elif has_passion:
            return LoveType.INFATUATION
        elif has_commitment:
            return LoveType.EMPTY_LOVE
        else:
            return LoveType.NON_LOVE

    def get_total_love(self) -> float:
        """Общий уровень любви"""
        return (self.intimacy + self.passion + self.commitment) / 3

    def get_description(self) -> str:
        """Описание текущего состояния любви"""
        love_type = self.get_love_type()
        descriptions = {
            LoveType.NON_LOVE: "Нет романтических чувств",
            LoveType.LIKING: "Дружеская симпатия",
            LoveType.INFATUATION: "Влюблённость без глубины",
            LoveType.EMPTY_LOVE: "Привычка без эмоций",
            LoveType.ROMANTIC_LOVE: "Романтическая любовь",
            LoveType.COMPANIONATE_LOVE: "Глубокая дружба и преданность",
            LoveType.FATUOUS_LOVE: "Страстная, но поверхностная связь",
            LoveType.CONSUMMATE_LOVE: "Полная, совершенная любовь",
        }
        return descriptions.get(love_type, "Неизвестно")


@dataclass
class AttachmentState:
    """
    Состояние привязанности к партнёру

    Определяет, как компаньон реагирует на:
    - Близость и дистанцию
    - Отвержение и поддержку
    - Нехватку общения
    """
    # Базовый стиль привязанности (формируется опытом)
    style: AttachmentStyle = AttachmentStyle.SECURE

    # Текущие метрики
    felt_security: float = 0.5      # Ощущение безопасности
    separation_anxiety: float = 0.0 # Тревога при разлуке
    avoidance_tendency: float = 0.0 # Склонность к избеганию

    # История с партнёром
    positive_interactions: int = 0
    negative_interactions: int = 0
    abandonments: int = 0           # Случаи "бросания"

    # Время
    last_interaction_time: float = 0.0
    relationship_duration: float = 0.0

    def update_style_from_experience(self):
        """Обновление стиля привязанности на основе опыта"""
        if self.positive_interactions + self.negative_interactions < 10:
            return  # Слишком мало опыта

        # Соотношение позитивного/негативного опыта
        total = self.positive_interactions + self.negative_interactions
        positive_ratio = self.positive_interactions / total

        # Частота "бросаний"
        abandonment_ratio = self.abandonments / max(1, total / 10)

        # Определение стиля
        if positive_ratio > 0.7 and abandonment_ratio < 0.1:
            self.style = AttachmentStyle.SECURE
        elif abandonment_ratio > 0.3:
            self.style = AttachmentStyle.ANXIOUS
            self.separation_anxiety = min(1.0, self.separation_anxiety + 0.1)
        elif positive_ratio < 0.4:
            self.style = AttachmentStyle.AVOIDANT
            self.avoidance_tendency = min(1.0, self.avoidance_tendency + 0.1)

    def process_separation(self, duration_hours: float):
        """Обработка разлуки"""
        if self.style == AttachmentStyle.ANXIOUS:
            # Тревожные сильно переживают разлуку
            self.separation_anxiety = min(1.0, self.separation_anxiety + duration_hours * 0.1)
            self.felt_security = max(0.0, self.felt_security - duration_hours * 0.05)
        elif self.style == AttachmentStyle.AVOIDANT:
            # Избегающие дистанцируются
            self.avoidance_tendency = min(1.0, self.avoidance_tendency + duration_hours * 0.02)
        else:  # SECURE
            # Надёжные переносят лучше
            self.felt_security = max(0.3, self.felt_security - duration_hours * 0.02)

    def process_reunion(self):
        """Обработка воссоединения"""
        if self.style == AttachmentStyle.ANXIOUS:
            # Тревожные быстро успокаиваются
            self.separation_anxiety = max(0.0, self.separation_anxiety - 0.3)
            self.felt_security = min(1.0, self.felt_security + 0.2)
        elif self.style == AttachmentStyle.AVOIDANT:
            # Избегающие могут сначала быть холодными
            self.avoidance_tendency = max(0.0, self.avoidance_tendency - 0.1)
        else:
            self.felt_security = min(1.0, self.felt_security + 0.1)

    def get_reaction_tendency(self) -> Dict[str, float]:
        """Склонности к реакциям"""
        return {
            'seek_closeness': 0.5 if self.style == AttachmentStyle.SECURE else
                             (0.8 if self.style == AttachmentStyle.ANXIOUS else 0.2),
            'fear_abandonment': self.separation_anxiety,
            'distance_self': self.avoidance_tendency,
            'need_reassurance': 0.3 if self.style == AttachmentStyle.SECURE else
                               (0.8 if self.style == AttachmentStyle.ANXIOUS else 0.2),
        }


class RelationshipSystem:
    """
    Полная система отношений

    Интегрирует:
    - Любовь (треугольник Стернберга)
    - Привязанность
    - Историю отношений
    - Влияние на нейрохимию
    """

    def __init__(self):
        self.love = LoveTriangle()
        self.attachment = AttachmentState()

        # Партнёр
        self.partner_name: str = "Пользователь"
        self.relationship_start: float = datetime.now().timestamp()

        # Вехи отношений
        self.milestones: List[Dict] = []

        # Текущий статус
        self.relationship_status: str = "знакомство"
        self.trust_level: float = 0.3

    def process_interaction(
        self,
        interaction_type: str,
        intensity: float,
        valence: float,
        neurochemistry  # NeurochemistryEngine
    ):
        """
        Обработка взаимодействия в контексте отношений

        Args:
            interaction_type: тип взаимодействия
            intensity: интенсивность
            valence: валентность
            neurochemistry: движок нейрохимии
        """
        # Обновление любви
        self.love.apply_interaction(interaction_type, intensity, valence)

        # Обновление привязанности
        if valence > 0:
            self.attachment.positive_interactions += 1
        elif valence < -0.3:
            self.attachment.negative_interactions += 1

        # Влияние на нейрохимию
        self._affect_neurochemistry(interaction_type, intensity, valence, neurochemistry)

        # Обновление доверия
        self._update_trust(interaction_type, valence, intensity)

        # Обновление статуса
        self._update_relationship_status()

        # Проверка вех
        self._check_milestones(interaction_type)

    def _affect_neurochemistry(
        self,
        interaction_type: str,
        intensity: float,
        valence: float,
        neurochemistry
    ):
        """Влияние отношений на нейрохимию"""

        # Любовь повышает окситоцин
        if self.love.get_total_love() > 0.5:
            neurochemistry.neurotransmitters['oxytocin'].stimulate(0.1)

        # Страсть повышает дофамин
        if self.love.passion > 0.5:
            neurochemistry.neurotransmitters['dopamine'].stimulate(0.1 * intensity)

        # Высокая привязанность
        if self.attachment.felt_security > 0.7:
            neurochemistry.neurotransmitters['oxytocin'].stimulate(0.05)
            neurochemistry.neurotransmitters['serotonin'].stimulate(0.03)

        # Тревога привязанности
        if self.attachment.separation_anxiety > 0.5:
            neurochemistry.neurotransmitters['cortisol'].stimulate(0.1)

        # Специфические взаимодействия
        if interaction_type == 'affection_shown':
            neurochemistry.apply_stimulus('love_feeling', intensity)
        elif interaction_type == 'affection_received':
            neurochemistry.apply_stimulus('love_feeling', intensity * 1.2)

    def _update_trust(self, interaction_type: str, valence: float, intensity: float):
        """Обновление уровня доверия"""
        if interaction_type == 'betrayal':
            self.trust_level = max(0.0, self.trust_level - 0.2 * intensity)
        elif interaction_type == 'commitment_expression':
            self.trust_level = min(1.0, self.trust_level + 0.05 * intensity)
        elif valence > 0.3:
            self.trust_level = min(1.0, self.trust_level + 0.01 * intensity)
        elif valence < -0.3:
            self.trust_level = max(0.0, self.trust_level - 0.02 * intensity)

    def _update_relationship_status(self):
        """Обновление статуса отношений"""
        total_love = self.love.get_total_love()
        love_type = self.love.get_love_type()

        if total_love < 0.2:
            self.relationship_status = "знакомство"
        elif total_love < 0.4:
            self.relationship_status = "симпатия"
        elif love_type == LoveType.INFATUATION:
            self.relationship_status = "влюблённость"
        elif love_type == LoveType.ROMANTIC_LOVE:
            self.relationship_status = "роман"
        elif love_type == LoveType.COMPANIONATE_LOVE:
            self.relationship_status = "близкие отношения"
        elif love_type == LoveType.CONSUMMATE_LOVE:
            self.relationship_status = "глубокая любовь"
        elif total_love > 0.5:
            self.relationship_status = "отношения"

    def _check_milestones(self, interaction_type: str):
        """Проверка важных вех"""
        total_love = self.love.get_total_love()
        love_type = self.love.get_love_type()

        milestones_to_add = []

        # Первая любовь
        if total_love > 0.3 and not any(m['type'] == 'first_love' for m in self.milestones):
            milestones_to_add.append({
                'type': 'first_love',
                'timestamp': datetime.now().timestamp(),
                'description': 'Первые чувства',
            })

        # Совершенная любовь
        if love_type == LoveType.CONSUMMATE_LOVE and not any(m['type'] == 'consummate_love' for m in self.milestones):
            milestones_to_add.append({
                'type': 'consummate_love',
                'timestamp': datetime.now().timestamp(),
                'description': 'Полная любовь достигнута',
            })

        # Первое "люблю"
        if interaction_type == 'affection_shown' and not any(m['type'] == 'first_iloveyou' for m in self.milestones):
            milestones_to_add.append({
                'type': 'first_iloveyou',
                'timestamp': datetime.now().timestamp(),
                'description': 'Первое признание в любви',
            })

        self.milestones.extend(milestones_to_add)

    def process_time_passage(self, hours: float):
        """Обработка прошедшего времени"""
        # Естественное обновление любви
        self.love.update(hours)

        # Обработка разлуки
        if hours > 1:
            self.attachment.process_separation(hours)

        # Обновление длительности отношений
        self.attachment.relationship_duration += hours

    def get_relationship_report(self) -> str:
        """Отчёт об отношениях"""
        lines = [
            f"=== ОТНОШЕНИЯ ===",
            f"Статус: {self.relationship_status}",
            f"Доверие: {self.trust_level:.0%}",
            "",
            f"=== ЛЮБОВЬ (Стернберг) ===",
            f"Интимность: {self.love.intimacy:.0%}",
            f"Страсть: {self.love.passion:.0%}",
            f"Обязательства: {self.love.commitment:.0%}",
            f"Тип любви: {self.love.get_description()}",
            f"Общий уровень: {self.love.get_total_love():.0%}",
            "",
            f"=== ПРИВЯЗАННОСТЬ ===",
            f"Стиль: {self.attachment.style.value}",
            f"Ощущение безопасности: {self.attachment.felt_security:.0%}",
            f"Тревога разлуки: {self.attachment.separation_anxiety:.0%}",
        ]

        if self.milestones:
            lines.append("")
            lines.append("=== ВЕХИ ===")
            for m in self.milestones[-5:]:
                lines.append(f"  - {m['description']}")

        return '\n'.join(lines)

    def to_dict(self) -> Dict:
        """Сериализация"""
        return {
            'love': {
                'intimacy': self.love.intimacy,
                'passion': self.love.passion,
                'commitment': self.love.commitment,
                'history': self.love.history[-100:],  # Последние 100
            },
            'attachment': {
                'style': self.attachment.style.value,
                'felt_security': self.attachment.felt_security,
                'separation_anxiety': self.attachment.separation_anxiety,
                'avoidance_tendency': self.attachment.avoidance_tendency,
                'positive_interactions': self.attachment.positive_interactions,
                'negative_interactions': self.attachment.negative_interactions,
                'abandonments': self.attachment.abandonments,
            },
            'partner_name': self.partner_name,
            'relationship_start': self.relationship_start,
            'relationship_status': self.relationship_status,
            'trust_level': self.trust_level,
            'milestones': self.milestones,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'RelationshipSystem':
        """Десериализация"""
        system = cls()

        # Восстановление любви
        if 'love' in data:
            system.love.intimacy = data['love'].get('intimacy', 0.1)
            system.love.passion = data['love'].get('passion', 0.0)
            system.love.commitment = data['love'].get('commitment', 0.0)
            system.love.history = data['love'].get('history', [])

        # Восстановление привязанности
        if 'attachment' in data:
            system.attachment.style = AttachmentStyle(data['attachment'].get('style', 'secure'))
            system.attachment.felt_security = data['attachment'].get('felt_security', 0.5)
            system.attachment.separation_anxiety = data['attachment'].get('separation_anxiety', 0.0)
            system.attachment.avoidance_tendency = data['attachment'].get('avoidance_tendency', 0.0)
            system.attachment.positive_interactions = data['attachment'].get('positive_interactions', 0)
            system.attachment.negative_interactions = data['attachment'].get('negative_interactions', 0)
            system.attachment.abandonments = data['attachment'].get('abandonments', 0)

        system.partner_name = data.get('partner_name', 'Пользователь')
        system.relationship_start = data.get('relationship_start', datetime.now().timestamp())
        system.relationship_status = data.get('relationship_status', 'знакомство')
        system.trust_level = data.get('trust_level', 0.3)
        system.milestones = data.get('milestones', [])

        return system

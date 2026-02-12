"""
Главный модуль цифрового компаньона

Интегрирует все компоненты:
- Темперамент (врождённый)
- Нейрохимия (120+ нейромедиаторов)
- Формирование характера (развивается)
- Эмоциональная система
- Память
- Когнитивное ядро
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
import json
import os

from core.temperament import Temperament, create_temperament
from core.neurochemistry import NeurochemistryEngine
from core.memory import MemorySystem, MemoryType, ImportanceLevel
from core.consciousness import ConsciousnessCore
from personality.character import CharacterFormation
from personality.attachment import RelationshipSystem


@dataclass
class EmotionalState:
    """Текущее эмоциональное состояние"""
    # PAD измерения
    pleasure: float = 0.0      # -1 (негатив) до +1 (позитив)
    arousal: float = 0.4        # 0 (спокоен) до 1 (возбуждён) - baseline повышен
    dominance: float = 0.5      # 0 (подчинён) до 1 (доминирует)

    # Доминирующая эмоция
    primary_emotion: str = "neutral"
    intensity: float = 0.3

    # Настроение (долгосрочное)
    mood: float = 0.0


class DigitalCompanion:
    """
    Цифровой компаньон

    Полноценная система с:
    - Врождённым темпераментом
    - Развивающейся личностью
    - Богатой нейрохимией
    - Эмоциональной системой
    """

    def __init__(
        self,
        name: str = "Лиза",
        temperament_type: str = "sanguine",
        config_path: Optional[str] = None
    ):
        # Идентичность
        self.name = name
        self.birth_time = datetime.now()
        self.tick_count = 0

        # === СЛОЙ 1: ТЕМПЕРАМЕНТ (врождённый) ===
        self.temperament = create_temperament(temperament_type)

        # === СЛОЙ 2: НЕЙРОХИМИЯ ===
        neurochem_config = None
        if config_path:
            neurochem_config = os.path.join(os.path.dirname(config_path), 'personality.yaml')
        self.neurochemistry = NeurochemistryEngine(neurochem_config)
        self.neurochemistry.calibrate_to_temperament(self.temperament)

        # === СЛОЙ 3: ФОРМИРОВАНИЕ ХАРАКТЕРА ===
        self.character = CharacterFormation(self.temperament)

        # === СЛОЙ 4: ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ ===
        self.emotion = EmotionalState()

        # === СЛОЙ 5: ПАМЯТЬ ===
        self.memory = MemorySystem()

        # === СЛОЙ 6: СИСТЕМА ОТНОШЕНИЙ ===
        self.relationship = RelationshipSystem()

        # === СЛОЙ 7: СОЗНАТЕЛЬНОЕ РАБОЧЕЕ ПРОСТРАНСТВО ===
        self.consciousness = ConsciousnessCore(capacity=5)

        # === СОСТОЯНИЕ СИСТЕМЫ ===
        self.mode = "AWAKE"  # AWAKE, IDLE, SLEEP
        self.last_interaction_time = datetime.now()

        # История состояний
        self.state_history: list = []

    def tick(self, dt: float = 1.0):
        """
        Главный тик системы

        Выполняется регулярно (например, 10 раз в секунду)
        """
        self.tick_count += 1

        # 1. Обновление нейрохимии
        self.neurochemistry.update(dt)

        # 2. Обновление эмоций из нейрохимии
        self._update_emotions()

        # 3. Проверка драйвов
        self._check_drives()

        # 4. Обновление сознательного цикла (GWT-подобный)
        self.consciousness.tick(
            emotion=self.emotion,
            drives=self.neurochemistry.get_drives_vector(),
            relationship_state={
                "trust": self.relationship.trust_level,
                "status": self.relationship.relationship_status,
            }
        )

        # 5. Затухание памяти
        if self.tick_count % 100 == 0:
            self.memory.decay_all(dt)

        # 6. Консолидация памяти (реже)
        if self.tick_count % 1000 == 0:
            self.memory.consolidate()

        # 7. Сохранение состояния
        if self.tick_count % 10 == 0:
            self.state_history.append(self.get_state())
            if len(self.state_history) > 1000:
                self.state_history.pop(0)

    def _update_emotions(self):
        """Обновление эмоций из нейрохимии"""
        # Получение PAD из нейрохимии
        pleasure, arousal, dominance = self.neurochemistry.calculate_pad_from_neurochem()

        # Плавное обновление
        alpha = 0.1
        self.emotion.pleasure = (1 - alpha) * self.emotion.pleasure + alpha * pleasure
        self.emotion.arousal = (1 - alpha) * self.emotion.arousal + alpha * arousal
        self.emotion.dominance = (1 - alpha) * self.emotion.dominance + alpha * dominance

        # Определение доминирующей эмоции
        self.emotion.primary_emotion = self._classify_emotion()
        self.emotion.intensity = abs(self.emotion.pleasure) * 0.5 + self.emotion.arousal * 0.5

        # Обновление настроения (очень медленно)
        beta = 0.01
        self.emotion.mood = (1 - beta) * self.emotion.mood + beta * self.emotion.pleasure

    def _classify_emotion(self) -> str:
        """Классификация текущей эмоции по PAD"""
        p, a, d = self.emotion.pleasure, self.emotion.arousal, self.emotion.dominance

        # Улучшенная классификация на основе PAD модели
        # Проверяем любовь по высокому окситоцину и отношениям
        if hasattr(self, 'relationship'):
            love_level = self.relationship.love.get_total_love()
            if love_level > 0.3 and p > 0:
                return "love"
            if love_level > 0.5:
                return "love"

        # Высокое удовольствие
        if p > 0.2:
            if a > 0.5:
                return "joy"          # Радость + возбуждение
            elif a > 0.35:
                return "happiness"    # Счастье
            else:
                return "contentment"  # Умиротворение

        # Низкое удовольствие
        if p < -0.2:
            if a > 0.5:
                return "anger" if d > 0.5 else "fear"
            elif a > 0.35:
                return "anxiety"      # Тревога
            else:
                return "sadness"      # Грусть

        # Нейтральная зона - более чувствительная
        if a > 0.6:
            return "excitement" if p > 0 else "surprise"
        elif a > 0.45:
            return "interest" if p >= 0 else "concern"
        elif a < 0.25:
            return "calm" if p > -0.15 else "tired"

        return "neutral"

    def _check_drives(self):
        """Проверка драйвов и возможные действия"""
        drives = self.neurochemistry.get_drives_vector()

        # Высокий social_drive — потребность в общении
        if drives.get('social_drive', 0) > 0.8:
            # Можно инициировать контакт
            pass

        # Высокая скука
        if drives.get('boredom', 0) > 0.7:
            # Нужна стимуляция
            pass

    def process_interaction(
        self,
        interaction_type: str,
        content: str,
        valence: float = 0.0,
        intensity: float = 0.5
    ):
        """
        Обработка взаимодействия с пользователем

        Args:
            interaction_type: тип взаимодействия
            content: содержимое (текст, описание)
            valence: валентность (-1 негатив до +1 позитив)
            intensity: интенсивность (0-1)
        """
        # Регистрация времени взаимодействия
        self.last_interaction_time = datetime.now()

        # Обновление нейрохимии
        self.neurochemistry.apply_stimulus(interaction_type, intensity)

        # Формирующий опыт для характера
        self.character.process_experience(
            event_type=interaction_type,
            description=content,
            emotional_valence=valence,
            emotional_intensity=intensity
        )

        # Кодирование в память
        importance = ImportanceLevel.NORMAL
        if intensity > 0.7:
            importance = ImportanceLevel.HIGH
        if intensity > 0.9:
            importance = ImportanceLevel.CRITICAL

        # Определение типа памяти
        memory_type = MemoryType.EPISODIC
        if intensity > 0.6:
            memory_type = MemoryType.EMOTIONAL

        self.memory.encode(
            content=content,
            memory_type=memory_type,
            emotional_valence=valence,
            emotional_intensity=intensity,
            emotion=self.emotion.primary_emotion,
            context={
                'interaction_type': interaction_type,
                'sender': 'user',
            },
            importance=importance
        )

        # Обновление контекста
        self.memory.update_context('last_interaction_type', interaction_type)
        self.memory.update_context('last_interaction_time', datetime.now().timestamp())

        # Обновление системы отношений
        self.relationship.process_interaction(
            interaction_type=interaction_type,
            intensity=intensity,
            valence=valence,
            neurochemistry=self.neurochemistry
        )

        # Сигнал в глобальное рабочее пространство
        self.consciousness.inject_signal(
            source="interaction",
            content=f"Пользователь: {content[:120]}",
            salience=min(1.0, 0.4 + intensity * 0.6),
            payload={
                "interaction_type": interaction_type,
                "valence": valence,
                "intensity": intensity,
            }
        )

    def get_state(self) -> Dict:
        """Получение полного состояния"""
        return {
            'name': self.name,
            'tick': self.tick_count,
            'mode': self.mode,
            'emotion': {
                'pleasure': self.emotion.pleasure,
                'arousal': self.emotion.arousal,
                'dominance': self.emotion.dominance,
                'primary': self.emotion.primary_emotion,
                'intensity': self.emotion.intensity,
                'mood': self.emotion.mood,
            },
            'neurochemistry': self.neurochemistry.get_main_state(),
            'drives': self.neurochemistry.get_drives_vector(),
            'traits': self.character.get_all_traits(),
            'plasticity': self.character.plasticity,
            'relationship': {
                'status': self.relationship.relationship_status,
                'trust': self.relationship.trust_level,
                'love_total': self.relationship.love.get_total_love(),
                'love_type': self.relationship.love.get_love_type().value,
            },
            'consciousness': self.consciousness.get_workspace_snapshot(),
        }

    def get_report(self) -> str:
        """Полный отчёт о состоянии"""
        lines = [
            f"=== {self.name.upper()} ===",
            f"Время существования: {datetime.now() - self.birth_time}",
            f"Тиков: {self.tick_count}",
            f"Режим: {self.mode}",
            "",
            self.neurochemistry.get_summary(),
            "",
            f"=== ЭМОЦИИ ===",
            f"Валентность: {self.emotion.pleasure:+.2f}",
            f"Возбуждение: {self.emotion.arousal:.2f}",
            f"Доминирование: {self.emotion.dominance:.2f}",
            f"Эмоция: {self.emotion.primary_emotion} ({self.emotion.intensity:.0%})",
            f"Настроение: {self.emotion.mood:+.2f}",
            "",
            self.character.get_personality_report(),
            "",
            self.memory.get_memory_summary(),
            "",
            self.relationship.get_relationship_report(),
        ]
        return '\n'.join(lines)

    def save_state(self, filepath: str):
        """Сохранение состояния в файл"""
        state = {
            'name': self.name,
            'birth_time': self.birth_time.isoformat(),
            'tick_count': self.tick_count,
            'temperament_type': self.temperament.type.value,
            'character': self.character.to_dict(),
            'neurochemistry': self.neurochemistry.get_state_vector(),
            'emotion': {
                'pleasure': self.emotion.pleasure,
                'arousal': self.emotion.arousal,
                'dominance': self.emotion.dominance,
                'mood': self.emotion.mood,
            },
            'memory': self.memory.to_dict(),
            'relationship': self.relationship.to_dict(),
            'consciousness': self.consciousness.get_workspace_snapshot(),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> 'DigitalCompanion':
        """Загрузка состояния из файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        companion = cls(
            name=state['name'],
            temperament_type=state.get('temperament_type', 'sanguine')
        )

        companion.birth_time = datetime.fromisoformat(state['birth_time'])
        companion.tick_count = state['tick_count']

        # Восстановление характера
        if 'character' in state:
            companion.character = CharacterFormation.from_dict(state['character'])

        # Восстановление нейрохимии
        if 'neurochemistry' in state:
            for name, level in state['neurochemistry'].items():
                if name in companion.neurochemistry.neurotransmitters:
                    companion.neurochemistry.neurotransmitters[name].level = level

        # Восстановление эмоций
        if 'emotion' in state:
            companion.emotion.pleasure = state['emotion']['pleasure']
            companion.emotion.arousal = state['emotion']['arousal']
            companion.emotion.dominance = state['emotion']['dominance']
            companion.emotion.mood = state['emotion']['mood']

        # Восстановление памяти
        if 'memory' in state:
            companion.memory = MemorySystem.from_dict(state['memory'])

        # Восстановление отношений
        if 'relationship' in state:
            companion.relationship = RelationshipSystem.from_dict(state['relationship'])

        # Восстановление сознательного состояния (частично)
        if 'consciousness' in state:
            snapshot = state['consciousness']
            companion.consciousness.self_model.attention_target = snapshot.get('focus', 'none')
            companion.consciousness.self_model.dominant_need = snapshot.get('dominant_need', 'connection')
            companion.consciousness.self_model.confidence = snapshot.get('confidence', 0.5)
            companion.consciousness.self_model.coherence = snapshot.get('coherence', 0.5)
            companion.consciousness.self_model.last_thought = snapshot.get('last_thought', '')
            companion.consciousness.inner_monologue = snapshot.get('inner_monologue_tail', [])

        return companion

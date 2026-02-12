"""
Система памяти цифрового компаньона

Основана на нейробиологической модели памяти человека:
- Сенсорная память (milliseconds)
- Рабочая память (seconds-minutes)
- Эпизодическая память (события, контекст)
- Семантическая память (факты, знания)
- Процедурная память (навыки, паттерны)
- Эмоциональная память (связанные с эмоциями воспоминания)

Консолидация памяти происходит во время "сна" системы.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import numpy as np
from enum import Enum


class MemoryType(Enum):
    """Типы памяти"""
    SENSORY = "sensory"           # Сенсорная (мигает)
    WORKING = "working"           # Рабочая (краткосрочная)
    EPISODIC = "episodic"         # Эпизодическая (события)
    SEMANTIC = "semantic"         # Семантическая (факты)
    PROCEDURAL = "procedural"     # Процедурная (навыки)
    EMOTIONAL = "emotional"       # Эмоциональная (яркие моменты)
    CORE = "core"                 # Core memories (определяющие)


class ImportanceLevel(Enum):
    """Уровень важности памяти"""
    TRIVIAL = 0      # Тривиальная
    LOW = 1          # Низкая
    NORMAL = 2       # Обычная
    HIGH = 3         # Высокая
    CRITICAL = 4     # Критическая (core memory)


@dataclass
class MemoryEntry:
    """
    Отдельное воспоминание

    Атрибуты основаны на характеристиках человеческой памяти:
    - Эмоциональная окраска усиливает запоминание
    - Повторение укрепляет память
    - Контекст помогает извлечению
    """
    # Идентификация
    id: str
    memory_type: MemoryType
    timestamp: float

    # Содержимое
    content: str
    embedding: Optional[np.ndarray] = None  # Векторное представление

    # Контекст
    context: Dict[str, Any] = field(default_factory=dict)
    related_memories: List[str] = field(default_factory=list)

    # Эмоциональная окраска
    emotional_valence: float = 0.0    # -1 до +1
    emotional_intensity: float = 0.5  # 0 до 1
    emotion_at_encoding: str = "neutral"

    # Параметры памяти
    importance: ImportanceLevel = ImportanceLevel.NORMAL
    access_count: int = 0            # Сколько раз вспомнили
    last_accessed: float = 0.0
    consolidation_strength: float = 0.5  # Сила консолидации

    # Забывание
    decay_rate: float = 0.01
    current_strength: float = 1.0

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp

    def access(self) -> str:
        """Получение доступа к воспоминанию"""
        self.access_count += 1
        self.last_accessed = datetime.now().timestamp()
        # Повторное воспоминание укрепляет память
        self.consolidation_strength = min(1.0, self.consolidation_strength + 0.1)
        self.current_strength = min(1.0, self.current_strength + 0.05)
        return self.content

    def decay(self, dt: float = 1.0):
        """Естественное затухание памяти"""
        # Кривая забывания Эббингауза: R = e^(-t/S)
        # Эмоционально яркие воспоминания забываются медленнее
        emotional_factor = 1.0 + self.emotional_intensity * 0.5
        importance_factor = 1.0 + self.importance.value * 0.2
        access_factor = 1.0 + np.log1p(self.access_count) * 0.1

        effective_decay = self.decay_rate / (emotional_factor * importance_factor * access_factor)
        self.current_strength = max(0.0, self.current_strength - effective_decay * dt)

    def is_forgotten(self) -> bool:
        """Проверка, забыто ли воспоминание"""
        return self.current_strength < 0.1

    def to_dict(self) -> Dict:
        """Сериализация"""
        return {
            'id': self.id,
            'memory_type': self.memory_type.value,
            'timestamp': self.timestamp,
            'content': self.content,
            'context': self.context,
            'related_memories': self.related_memories,
            'emotional_valence': self.emotional_valence,
            'emotional_intensity': self.emotional_intensity,
            'emotion_at_encoding': self.emotion_at_encoding,
            'importance': self.importance.value,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'consolidation_strength': self.consolidation_strength,
            'decay_rate': self.decay_rate,
            'current_strength': self.current_strength,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """Десериализация"""
        return cls(
            id=data['id'],
            memory_type=MemoryType(data['memory_type']),
            timestamp=data['timestamp'],
            content=data['content'],
            context=data.get('context', {}),
            related_memories=data.get('related_memories', []),
            emotional_valence=data.get('emotional_valence', 0.0),
            emotional_intensity=data.get('emotional_intensity', 0.5),
            emotion_at_encoding=data.get('emotion_at_encoding', 'neutral'),
            importance=ImportanceLevel(data.get('importance', 2)),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', data['timestamp']),
            consolidation_strength=data.get('consolidation_strength', 0.5),
            decay_rate=data.get('decay_rate', 0.01),
            current_strength=data.get('current_strength', 1.0),
        )


@dataclass
class SemanticNode:
    """
    Узел семантической памяти

    Представляет концепт или факт с ассоциациями.
    """
    concept: str
    category: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    associations: Dict[str, float] = field(default_factory=dict)  # {concept: weight}
    emotional_valence: float = 0.0
    importance: float = 0.5
    source_memories: List[str] = field(default_factory=list)


class MemorySystem:
    """
    Полная система памяти

    Интегрирует все типы памяти и управляет:
    - Кодированием новых воспоминаний
    - Хранением и организацией
    - Извлечением по запросу
    - Консолидацией во время "сна"
    - Забыванием неважных воспоминаний
    """

    def __init__(self, max_working: int = 7, max_episodic: int = 1000):
        # === РАБОЧАЯ ПАМЯТЬ ===
        # Ограничена 7±2 элементами (как у человека)
        self.working_memory: List[MemoryEntry] = []
        self.max_working = max_working

        # === ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ ===
        # Хронологически организованные события
        self.episodic_memories: List[MemoryEntry] = []
        self.max_episodic = max_episodic

        # === СЕМАНТИЧЕСКАЯ ПАМЯТЬ ===
        # Факты и концепты
        self.semantic_memory: Dict[str, SemanticNode] = {}

        # === ЭМОЦИОНАЛЬНАЯ ПАМЯТЬ ===
        # Яркие эмоциональные моменты
        self.emotional_memories: List[MemoryEntry] = []

        # === CORE MEMORIES ===
        # Определяющие воспоминания (не забываются)
        self.core_memories: List[MemoryEntry] = []

        # === КОНТЕКСТ ===
        self.current_context: Dict[str, Any] = {}
        self.memory_id_counter = 0

        # === ПАРАМЕТРЫ ===
        self.consolidation_threshold = 0.7
        self.forgetting_threshold = 0.1

    def _generate_id(self) -> str:
        """Генерация уникального ID"""
        self.memory_id_counter += 1
        return f"mem_{self.memory_id_counter}_{datetime.now().timestamp():.0f}"

    def encode(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        emotional_valence: float = 0.0,
        emotional_intensity: float = 0.5,
        emotion: str = "neutral",
        context: Optional[Dict] = None,
        importance: ImportanceLevel = ImportanceLevel.NORMAL
    ) -> MemoryEntry:
        """
        Кодирование нового воспоминания

        Args:
            content: содержимое воспоминания
            memory_type: тип памяти
            emotional_valence: эмоциональная валентность
            emotional_intensity: интенсивность эмоций
            emotion: текущая эмоция
            context: контекст
            importance: важность

        Returns:
            Созданное воспоминание
        """
        # Создание записи
        memory = MemoryEntry(
            id=self._generate_id(),
            memory_type=memory_type,
            timestamp=datetime.now().timestamp(),
            content=content,
            context=context or self.current_context.copy(),
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            emotion_at_encoding=emotion,
            importance=importance,
        )

        # Эмоционально яркие моменты запоминаются лучше
        if emotional_intensity > 0.7:
            memory.decay_rate *= 0.5  # Медленнее забываются
            memory.consolidation_strength = 0.7

        # Определение места хранения
        self._store_memory(memory)

        # Обновление рабочей памяти
        self._add_to_working(memory)

        # Обновление семантической памяти
        self._extract_semantics(memory)

        return memory

    def _store_memory(self, memory: MemoryEntry):
        """Размещение воспоминания в соответствующем хранилище"""
        if memory.importance == ImportanceLevel.CRITICAL:
            self.core_memories.append(memory)
        elif memory.memory_type == MemoryType.EMOTIONAL or memory.emotional_intensity > 0.6:
            self.emotional_memories.append(memory)
            self.episodic_memories.append(memory)
        elif memory.memory_type == MemoryType.EPISODIC:
            self.episodic_memories.append(memory)

    def _add_to_working(self, memory: MemoryEntry):
        """Добавление в рабочую память"""
        self.working_memory.append(memory)
        # Ограничение размера (7±2)
        if len(self.working_memory) > self.max_working:
            # Удаляем менее важные
            self.working_memory.sort(key=lambda m: m.importance.value, reverse=True)
            self.working_memory = self.working_memory[:self.max_working]

    def _extract_semantics(self, memory: MemoryEntry):
        """Извлечение семантической информации из воспоминания"""
        # Простая реализация: извлечение ключевых слов
        # В полной версии здесь будет NLP
        words = memory.content.lower().split()
        for word in words:
            if len(word) > 3:  # Игнорируем короткие слова
                if word not in self.semantic_memory:
                    self.semantic_memory[word] = SemanticNode(
                        concept=word,
                        category="extracted",
                        emotional_valence=memory.emotional_valence,
                        source_memories=[memory.id]
                    )
                else:
                    # Усиливаем связь
                    node = self.semantic_memory[word]
                    node.source_memories.append(memory.id)
                    # Обновляем эмоциональную валентность (скользящее среднее)
                    node.emotional_valence = (
                        node.emotional_valence * 0.8 + memory.emotional_valence * 0.2
                    )

    def recall(
        self,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[MemoryType]] = None
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Извлечение воспоминаний по запросу

        Args:
            query: поисковый запрос
            top_k: количество результатов
            memory_types: фильтр по типам памяти

        Returns:
            Список (воспоминание, релевантность)
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Поиск в разных хранилищах
        all_memories = (
            self.core_memories +
            self.emotional_memories +
            self.episodic_memories +
            self.working_memory
        )

        for memory in all_memories:
            # Фильтр по типу
            if memory_types and memory.memory_type not in memory_types:
                continue

            # Пропуск забытых
            if memory.is_forgotten() and memory not in self.core_memories:
                continue

            # Расчёт релевантности
            relevance = self._calculate_relevance(memory, query_words, query_lower)

            if relevance > 0:
                # Влияние силы памяти на извлечение
                retrieval_strength = relevance * memory.current_strength
                # Эмоционально яркие воспоминания легче извлекаются
                emotional_boost = 1.0 + memory.emotional_intensity * 0.3
                final_score = retrieval_strength * emotional_boost

                results.append((memory, final_score))

        # Сортировка по релевантности
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _calculate_relevance(
        self,
        memory: MemoryEntry,
        query_words: set,
        query_lower: str
    ) -> float:
        """Расчёт релевантности воспоминания запросу"""
        content_lower = memory.content.lower()
        content_words = set(content_lower.split())

        # Пересечение слов
        word_overlap = len(query_words & content_words) / max(len(query_words), 1)

        # Прямое вхождение
        direct_match = 1.0 if query_lower in content_lower else 0.0

        # Семантические ассоциации
        semantic_score = 0.0
        for word in query_words:
            if word in self.semantic_memory:
                node = self.semantic_memory[word]
                # Связь с контекстом воспоминания
                for ctx_word in memory.context.values():
                    if isinstance(ctx_word, str) and ctx_word.lower() in node.associations:
                        semantic_score += node.associations[ctx_word.lower()]

        # Итоговая релевантность
        relevance = (
            word_overlap * 0.4 +
            direct_match * 0.4 +
            min(semantic_score * 0.2, 0.2)
        )

        return relevance

    def recall_about_person(self, person_name: str) -> List[MemoryEntry]:
        """Вспомнить всё о конкретном человеке"""
        return [
            mem for mem in
            (self.episodic_memories + self.emotional_memories + self.core_memories)
            if person_name.lower() in mem.content.lower() or
               person_name.lower() in str(mem.context).lower()
        ]

    def get_facts_about(self, subject: str) -> Dict[str, Any]:
        """Получить все факты о субъекте"""
        if subject.lower() in self.semantic_memory:
            node = self.semantic_memory[subject.lower()]
            return {
                'concept': node.concept,
                'category': node.category,
                'attributes': node.attributes,
                'associations': node.associations,
                'emotional_valence': node.emotional_valence,
                'related_memories': len(node.source_memories)
            }
        return {}

    def add_fact(
        self,
        subject: str,
        attribute: str,
        value: Any,
        source: str = "user_input"
    ):
        """Добавление факта в семантическую память"""
        if subject.lower() not in self.semantic_memory:
            self.semantic_memory[subject.lower()] = SemanticNode(
                concept=subject.lower(),
                category="entity"
            )

        node = self.semantic_memory[subject.lower()]
        node.attributes[attribute] = {
            'value': value,
            'source': source,
            'timestamp': datetime.now().timestamp()
        }

    def update_context(self, key: str, value: Any):
        """Обновление текущего контекста"""
        self.current_context[key] = value

    def decay_all(self, dt: float = 1.0):
        """Затухание всех воспоминаний"""
        for memory in self.episodic_memories:
            memory.decay(dt)

        # Удаление забытых воспоминаний (кроме core)
        self.episodic_memories = [
            m for m in self.episodic_memories
            if not m.is_forgotten() or m in self.core_memories
        ]

    def consolidate(self):
        """
        Консолидация памяти (аналог сна)

        Переводит важные воспоминания в долговременную память,
        укрепляет часто используемые связи.
        """
        # Укрепление часто вспоминаемых воспоминаний
        for memory in self.episodic_memories:
            if memory.access_count > 3:
                memory.consolidation_strength = min(
                    1.0,
                    memory.consolidation_strength + 0.1
                )
                memory.decay_rate *= 0.9

        # Промоция эмоционально ярких воспоминаний в core memories
        for memory in self.emotional_memories:
            if (memory.emotional_intensity > 0.8 and
                memory.importance.value >= ImportanceLevel.HIGH.value and
                memory not in self.core_memories):
                self.core_memories.append(memory)

        # Создание ассоциаций между связанными воспоминаниями
        self._create_associations()

        # Ограничение размера хранилищ
        if len(self.episodic_memories) > self.max_episodic:
            self.episodic_memories.sort(
                key=lambda m: m.current_strength * m.importance.value,
                reverse=True
            )
            self.episodic_memories = self.episodic_memories[:self.max_episodic]

    def _create_associations(self):
        """Создание ассоциаций между связанными воспоминаниями"""
        # Простая эвристика: воспоминания с общими словами связаны
        for i, mem1 in enumerate(self.episodic_memories[-100:]):  # Только недавние
            words1 = set(mem1.content.lower().split())
            for mem2 in self.episodic_memories[-100:]:
                if mem1.id != mem2.id:
                    words2 = set(mem2.content.lower().split())
                    overlap = len(words1 & words2)
                    if overlap > 2:
                        if mem2.id not in mem1.related_memories:
                            mem1.related_memories.append(mem2.id)

    def get_memory_summary(self) -> str:
        """Краткое описание состояния памяти"""
        lines = ["=== СОСТОЯНИЕ ПАМЯТИ ==="]
        lines.append(f"Рабочая память: {len(self.working_memory)}/{self.max_working}")
        lines.append(f"Эпизодических воспоминаний: {len(self.episodic_memories)}")
        lines.append(f"Эмоциональных воспоминаний: {len(self.emotional_memories)}")
        lines.append(f"Core memories: {len(self.core_memories)}")
        lines.append(f"Семантических концептов: {len(self.semantic_memory)}")

        # Статистика силы воспоминаний
        if self.episodic_memories:
            avg_strength = np.mean([m.current_strength for m in self.episodic_memories])
            lines.append(f"Средняя сила воспоминаний: {avg_strength:.0%}")

        return '\n'.join(lines)

    def get_recent_context(self, n: int = 5) -> str:
        """Получение недавнего контекста для LLM"""
        recent = self.episodic_memories[-n:] if self.episodic_memories else []
        return '\n'.join([f"- {m.content}" for m in recent])

    def to_dict(self) -> Dict:
        """Полная сериализация"""
        return {
            'working_memory': [m.to_dict() for m in self.working_memory],
            'episodic_memories': [m.to_dict() for m in self.episodic_memories],
            'emotional_memories': [m.to_dict() for m in self.emotional_memories],
            'core_memories': [m.to_dict() for m in self.core_memories],
            'semantic_memory': {
                k: {
                    'concept': v.concept,
                    'category': v.category,
                    'attributes': v.attributes,
                    'associations': v.associations,
                    'emotional_valence': v.emotional_valence,
                    'importance': v.importance,
                    'source_memories': v.source_memories,
                }
                for k, v in self.semantic_memory.items()
            },
            'memory_id_counter': self.memory_id_counter,
            'current_context': self.current_context,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemorySystem':
        """Десериализация"""
        system = cls()

        system.working_memory = [
            MemoryEntry.from_dict(m) for m in data.get('working_memory', [])
        ]
        system.episodic_memories = [
            MemoryEntry.from_dict(m) for m in data.get('episodic_memories', [])
        ]
        system.emotional_memories = [
            MemoryEntry.from_dict(m) for m in data.get('emotional_memories', [])
        ]
        system.core_memories = [
            MemoryEntry.from_dict(m) for m in data.get('core_memories', [])
        ]

        for k, v in data.get('semantic_memory', {}).items():
            system.semantic_memory[k] = SemanticNode(
                concept=v['concept'],
                category=v['category'],
                attributes=v.get('attributes', {}),
                associations=v.get('associations', {}),
                emotional_valence=v.get('emotional_valence', 0.0),
                importance=v.get('importance', 0.5),
                source_memories=v.get('source_memories', []),
            )

        system.memory_id_counter = data.get('memory_id_counter', 0)
        system.current_context = data.get('current_context', {})

        return system

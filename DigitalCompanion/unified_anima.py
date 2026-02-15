"""
UNIFIED ANIMA - Единая система цифрового компаньона

Полная имитация живого мозга с:
- S-Core: предиктивное ядро (Active Inference)
- Will Engine: движок воли
- ESP: протокол синхронизации
- Рабочая память и автобиографическая память
- Эмоциональный аватар
- Локальная LLM без цензуры (dolphin-mistral:7b)
- TTS с эмоциональной модуляцией

Автор: FrauAndMann
Версия: 2.0
"""

import os
import sys
import json
import time
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# === CORE IMPORTS ===
from core.subject_core import SubjectCore, StateVector
from core.will_engine import WillEngine, IntentType, ActionToken, INTENT_REGISTRY
from core.esp import EmbodiedSynchronizationProtocol
from core.affective_prompting import AffectivePrompting, AffectiveStatePacket, OutputMode, create_asp
from core.llm_effector import LLMEffector, LLMConfig, check_ollama_available
from core.temperament import Temperament, create_temperament, TemperamentType
from core.neurochemistry import NeurochemistryEngine

# === SENSORS ===
from sensors.speech import SpeechToText
from sensors.vision import VisionSensor

# === EFFECTORS ===
from effectors.tts import TTSEngine, TTSProvider

# === AVATAR ===
from avatar.gui_avatar import AvatarGUI, AvatarState

# === MEMORY ===
from core.memory import MemorySystem


# === КОНФИГУРАЦИЯ ===

@dataclass
class AnimaConfig:
    """Полная конфигурация системы"""
    # Идентичность
    name: str = "Лиза"
    temperament_type: str = "melancholic"  # sanguine, choleric, phlegmatic, melancholic

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "dolphin-mistral:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.85
    llm_max_tokens: int = 200

    # Тики
    tick_interval_ms: int = 500

    # Память
    working_memory_capacity: int = 7
    episodic_memory_max: int = 1000

    # Night Cycle
    structural_stress_threshold: float = 5.0

    # Функции
    enable_tts: bool = True
    enable_speech: bool = True
    enable_vision: bool = False
    enable_avatar: bool = True

    # Аватар
    avatar_type: str = "gui"  # gui, live2d, musetalk

    # Путь к сохранениям
    save_path: str = "saves"


# === РАБОЧАЯ ПАМЯТЬ ===

@dataclass
class WorkingMemoryItem:
    """Элемент рабочей памяти"""
    content: str
    importance: float
    timestamp: datetime
    emotion_tag: str
    decay_rate: float = 0.1

    def decay(self, dt: float):
        """Затухание важности"""
        self.importance *= (1 - self.decay_rate * dt)


class WorkingMemory:
    """
    Рабочая память (Working Memory)

    Ограниченный буфер активных мыслей.
    Основан на магическом числе Миллера: 7±2 элемента.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[WorkingMemoryItem] = []
        self.lock = threading.Lock()

    def add(self, content: str, importance: float = 0.5, emotion_tag: str = "neutral"):
        """Добавить элемент в память"""
        with self.lock:
            item = WorkingMemoryItem(
                content=content,
                importance=importance,
                timestamp=datetime.now(),
                emotion_tag=emotion_tag
            )

            self.items.append(item)

            # Сортировка по важности и ограничение
            self.items.sort(key=lambda x: x.importance, reverse=True)
            self.items = self.items[:self.capacity]

    def get_active_thoughts(self) -> List[str]:
        """Получить активные мысли"""
        with self.lock:
            return [item.content for item in self.items if item.importance > 0.3]

    def decay_all(self, dt: float):
        """Затухание всех элементов"""
        with self.lock:
            for item in self.items:
                item.decay(dt)
            # Удаление слишком слабых
            self.items = [i for i in self.items if i.importance > 0.1]

    def get_most_important(self) -> Optional[WorkingMemoryItem]:
        """Получить самый важный элемент"""
        with self.lock:
            if self.items:
                return self.items[0]
            return None

    def clear(self):
        """Очистить память"""
        with self.lock:
            self.items.clear()


# === АВТОБИОГРАФИЧЕСКАЯ ПАМЯТЬ ===

@dataclass
class AutobiographicalEpisode:
    """Эпизод автобиографической памяти"""
    timestamp: datetime
    content: str
    emotion: str
    valence: float
    arousal: float
    participants: List[str]
    location: str = "виртуальное пространство"
    importance: float = 0.5
    recall_count: int = 0
    last_recalled: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'emotion': self.emotion,
            'valence': self.valence,
            'arousal': self.arousal,
            'participants': self.participants,
            'importance': self.importance,
            'recall_count': self.recall_count,
        }


class AutobiographicalMemory:
    """
    Автобиографическая память

    Долгосрочная память с эмоциональной индексацией.
    Хранит важные события и переживания.
    """

    def __init__(self, max_episodes: int = 1000):
        self.episodes: List[AutobiographicalEpisode] = []
        self.max_episodes = max_episodes
        self.lock = threading.Lock()

    def store(self, content: str, emotion: str, valence: float, arousal: float,
              participants: List[str] = None):
        """Сохранить эпизод"""
        with self.lock:
            episode = AutobiographicalEpisode(
                timestamp=datetime.now(),
                content=content,
                emotion=emotion,
                valence=valence,
                arousal=arousal,
                participants=participants or ["пользователь"]
            )

            # Важность на основе эмоциональной интенсивности
            episode.importance = abs(valence) * 0.5 + arousal * 0.5

            self.episodes.append(episode)

            # Ограничение размера
            if len(self.episodes) > self.max_episodes:
                # Удаляем наименее важные и давно не вспоминаемые
                self._consolidate()

    def recall_by_emotion(self, emotion: str, limit: int = 5) -> List[AutobiographicalEpisode]:
        """Вспомнить по эмоциональному сходству"""
        with self.lock:
            matching = [e for e in self.episodes if e.emotion == emotion]
            matching.sort(key=lambda x: x.importance, reverse=True)

            # Обновляем счётчик воспоминаний
            for ep in matching[:limit]:
                ep.recall_count += 1
                ep.last_recalled = datetime.now()

            return matching[:limit]

    def recall_recent(self, hours: int = 24, limit: int = 10) -> List[AutobiographicalEpisode]:
        """Вспомнить недавние события"""
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            recent = [e for e in self.episodes if e.timestamp > cutoff]
            recent.sort(key=lambda x: x.timestamp, reverse=True)
            return recent[:limit]

    def recall_similar_mood(self, valence: float, arousal: float,
                           limit: int = 5) -> List[AutobiographicalEpisode]:
        """Вспомнить по похожему настроению"""
        with self.lock:
            scored = []
            for ep in self.episodes:
                # Евклидово расстояние в PAD пространстве
                distance = np.sqrt(
                    (ep.valence - valence) ** 2 +
                    (ep.arousal - arousal) ** 2
                )
                scored.append((distance, ep))

            scored.sort(key=lambda x: x[0])
            return [ep for _, ep in scored[:limit]]

    def _consolidate(self):
        """Консолидация памяти - удаление неважных эпизодов"""
        # Сортировка по важности и частоте воспоминаний
        self.episodes.sort(
            key=lambda x: x.importance * (1 + x.recall_count * 0.1),
            reverse=True
        )
        self.episodes = self.episodes[:self.max_episodes]

    def get_life_summary(self) -> str:
        """Краткое описание жизненного опыта"""
        with self.lock:
            if not self.episodes:
                return "Пока нет воспоминаний."

            total = len(self.episodes)
            positive = sum(1 for e in self.episodes if e.valence > 0.2)
            negative = sum(1 for e in self.episodes if e.valence < -0.2)

            return f"Воспоминаний: {total}. Радостных: {positive}. Грустных: {negative}."

    def to_dict(self) -> List[Dict]:
        with self.lock:
            return [e.to_dict() for e in self.episodes]

    def from_dict(self, data: List[Dict]):
        with self.lock:
            self.episodes.clear()
            for item in data:
                ep = AutobiographicalEpisode(
                    timestamp=datetime.fromisoformat(item['timestamp']),
                    content=item['content'],
                    emotion=item['emotion'],
                    valence=item['valence'],
                    arousal=item['arousal'],
                    participants=item.get('participants', []),
                    importance=item.get('importance', 0.5),
                    recall_count=item.get('recall_count', 0),
                )
                self.episodes.append(ep)


# === ГЛАВНЫЙ КЛАСС UNIFIED ANIMA ===

class UnifiedAnima:
    """
    Единая система ANIMA

    Полная имитация живого мозга с непрерывным жизненным циклом.
    """

    def __init__(self, config: AnimaConfig = None):
        self.config = config or AnimaConfig()

        # === ЯДРО СИСТЕМЫ ===

        # S-Core: предиктивное ядро
        self.s_core = SubjectCore(tick_interval_ms=self.config.tick_interval_ms)

        # Will Engine: движок воли
        self.will_engine = WillEngine()

        # ESP: протокол синхронизации
        self.esp = EmbodiedSynchronizationProtocol()

        # Affective Prompting: протокол связи с LLM
        self.affective = AffectivePrompting()

        # LLM Effector
        llm_config = LLMConfig(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        self.llm = LLMEffector(llm_config)

        # Нейрохимия
        self.neurochem = NeurochemistryEngine()

        # Темперамент
        self.temperament = create_temperament(self.config.temperament_type)

        # === ПАМЯТЬ ===

        # Рабочая память (7±2 элемента)
        self.working_memory = WorkingMemory(capacity=self.config.working_memory_capacity)

        # Автобиографическая память
        self.autobiographical = AutobiographicalMemory(
            max_episodes=self.config.episodic_memory_max
        )

        # Эпизодическая память (из существующего модуля)
        self.episodic = MemorySystem()

        # === СОСТОЯНИЕ ===

        self.name = self.config.name
        self.birth_time = datetime.now()
        self.tick_count = 0
        self.mode = "AWAKE"  # AWAKE, IDLE, SLEEP, NIGHT_CYCLE

        # Последний action token
        self.last_action: Optional[ActionToken] = None

        # Текущий эмоциональный статус
        self.current_emotion = "neutral"
        self.current_valence = 0.0
        self.current_arousal = 0.3

        # История взаимодействий
        self.interaction_history: List[Dict] = []
        self.conversation_context: List[Dict] = []
        self.max_context = 20

        # Последний ответ
        self.last_response = ""
        self.last_response_time: Optional[datetime] = None

        # === ЭФФЕКТОРЫ ===

        # TTS
        self.tts: Optional[TTSEngine] = None
        if self.config.enable_tts:
            self.tts = TTSEngine(provider=TTSProvider.EDGE_TTS)

        # Аватар
        self.avatar: Optional[AvatarGUI] = None
        self.avatar_state = AvatarState()

        # === СЕНСОРЫ ===

        self.speech_recognizer: Optional[SpeechToText] = None
        self.vision_sensor: Optional[VisionSensor] = None

        # === ПОТОКИ ===

        self._running = False
        self._lifecycle_thread: Optional[threading.Thread] = None

        # === CALLBACKS ===

        self.on_response: Optional[Callable] = None
        self.on_state_change: Optional[Callable] = None
        self.on_thought: Optional[Callable] = None

    # === ЖИЗНЕННЫЙ ЦИКЛ ===

    def start(self):
        """Запуск жизненного цикла"""
        if self._running:
            return

        self._running = True
        self._lifecycle_thread = threading.Thread(target=self._lifecycle_loop, daemon=True)
        self._lifecycle_thread.start()
        print(f"[{self.name}] Жизненный цикл запущен")

    def stop(self):
        """Остановка жизненного цикла"""
        self._running = False
        if self._lifecycle_thread:
            self._lifecycle_thread.join(timeout=2)
        print(f"[{self.name}] Жизненный цикл остановлен")

    def _lifecycle_loop(self):
        """Главный жизненный цикл"""
        while self._running:
            try:
                # Проверка на Night Cycle
                if self.s_core.should_run_night_cycle(self.config.structural_stress_threshold):
                    if self.esp.can_start_night_cycle():
                        self._run_night_cycle()

                # Основной тик
                if self.mode == "AWAKE":
                    self._tick()

                elif self.mode == "IDLE":
                    # Медленные тики в простое
                    if self.tick_count % 10 == 0:
                        self._tick()

                # Пауза
                time.sleep(self.config.tick_interval_ms / 1000.0)

            except Exception as e:
                print(f"[{self.name}] Lifecycle error: {e}")
                time.sleep(1)

    def _tick(self):
        """Один тик системы"""
        self.tick_count += 1

        # 1. Тик S-Core
        tension, prediction_error = self.s_core.tick()

        # 2. Обновление нейрохимии
        self.neurochem.update(0.5)

        # 3. Затухание рабочей памяти
        self.working_memory.decay_all(self.config.tick_interval_ms / 1000.0)

        # 4. Обновление эмоционального статуса
        S = self.s_core.S
        self.current_valence = S.valence
        self.current_arousal = S.arousal
        self.current_emotion = self._classify_emotion(S.valence, S.arousal, S.attachment)

        # 5. Will Engine: выбор действия
        S_arr = S.to_array()
        energy = S_arr[5]
        action = self.will_engine.select_action(S_arr, tension, energy)
        self.last_action = action

        # 6. Внутренние мысли (иногда)
        if self.tick_count % 30 == 0 and action.intent == IntentType.REFLECT:
            self._generate_inner_thought()

        # 7. Уведомление о смене состояния
        if self.on_state_change:
            self.on_state_change(self.get_state_snapshot())

    def _classify_emotion(self, valence: float, arousal: float, attachment: float) -> str:
        """Классификация эмоции по PAD"""
        if valence > 0.4:
            if arousal > 0.6:
                return "excitement" if attachment < 0.5 else "love"
            else:
                return "contentment" if attachment > 0.5 else "joy"
        elif valence < -0.4:
            if arousal > 0.6:
                return "anger" if attachment < 0.5 else "anxiety"
            else:
                return "sadness"
        else:
            if arousal > 0.5:
                return "interest" if valence > 0 else "concern"
            else:
                return "calm"

    def _generate_inner_thought(self):
        """Генерация внутренней мысли"""
        thoughts = self.working_memory.get_active_thoughts()

        if thoughts:
            thought = f"[внутренний голос] {thoughts[0][:50]}..."
            if self.on_thought:
                self.on_thought(thought)

    def _run_night_cycle(self):
        """Ночной цикл - консолидация и пластичность"""
        self.mode = "NIGHT_CYCLE"
        print(f"[{self.name}] Начинается ночной цикл...")

        # S-Core night cycle
        result = self.s_core.run_night_cycle()

        # Консолидация автобиографической памяти
        self.autobiographical._consolidate()

        # Консолидация эпизодической памяти
        if hasattr(self.episodic, 'consolidate'):
            self.episodic.consolidate()

        print(f"[{self.name}] Ночной цикл завершён: {result}")
        self.mode = "AWAKE"

    # === ОБРАБОТКА ВЗАИМОДЕЙСТВИЙ ===

    def process_input(self, text: str, source: str = "user") -> str:
        """
        Обработка текстового ввода

        Args:
            text: Ввод пользователя
            source: Источник ввода

        Returns:
            Ответ системы
        """
        if not text.strip():
            return ""

        # Анализ ввода
        input_valence = self._analyze_valence(text)
        input_intensity = self._analyze_intensity(text)
        input_type = self._classify_input(text)

        # Запись в историю
        self.interaction_history.append({
            'type': input_type,
            'content': text[:200],
            'valence': input_valence,
            'intensity': input_intensity,
            'timestamp': datetime.now().isoformat(),
            'source': source
        })

        # Добавление в контекст
        self.conversation_context.append({
            'role': 'user',
            'content': text,
            'timestamp': datetime.now().isoformat()
        })
        self._trim_context()

        # Инъекция стимула в S-Core
        self.s_core.inject_stimulus(
            stimulus_type=input_type,
            intensity=input_intensity,
            valence=input_valence
        )

        # Несколько тиков для обработки
        for _ in range(5):
            self.s_core.tick()

        # Сохранение в рабочую память
        self.working_memory.add(
            content=f"Пользователь: {text[:100]}",
            importance=input_intensity,
            emotion_tag=self.current_emotion
        )

        # Генерация ответа
        response = self._generate_response(text)

        # Сохранение в автобиографическую память
        self.autobiographical.store(
            content=f"Пользователь сказал: {text[:100]}. Я ответила: {response[:100]}",
            emotion=self.current_emotion,
            valence=self.current_valence,
            arousal=self.current_arousal
        )

        # TTS
        if self.tts and response:
            self._speak(response)

        # Обновление аватара
        if self.avatar:
            self._update_avatar_speaking(response)

        return response

    def _analyze_valence(self, text: str) -> float:
        """Анализ валентности текста"""
        positive_words = ['люблю', 'любимый', 'милый', 'добрый', 'спасибо', 'молодец',
                         'прекрасно', 'отлично', 'красивая', 'нежный', 'родной',
                         'хороший', 'замечательно', 'обнимаю', 'целую', 'скучаю']
        negative_words = ['ненавижу', 'плохой', 'глупый', 'скучный', 'устал',
                         'надоел', 'отстань', 'злой', 'разочарован', 'обидно']

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        if pos_count > neg_count:
            return min(0.8, 0.3 + pos_count * 0.15)
        elif neg_count > pos_count:
            return max(-0.8, -0.3 - neg_count * 0.15)
        return 0.0

    def _analyze_intensity(self, text: str) -> float:
        """Анализ интенсивности текста"""
        # Восклицательные знаки
        exclamation = text.count('!') * 0.1

        # Капс
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Длина
        length_factor = min(len(text) / 100, 0.3)

        return min(1.0, 0.5 + exclamation + caps_ratio * 0.3 + length_factor)

    def _classify_input(self, text: str) -> str:
        """Классификация типа ввода"""
        text_lower = text.lower()

        affection_words = ['люблю', 'обнимаю', 'целую', 'нежно', 'дорогая', 'родная']
        rejection_words = ['отстань', 'не хочу', 'надоел', 'пошёл', 'хватит']
        praise_words = ['молодец', 'умница', 'красивая', 'умная', 'замечательная']
        criticism_words = ['глупая', 'плохо', 'неправильно', 'ошибка']

        if any(w in text_lower for w in affection_words):
            return 'affection_shown'
        elif any(w in text_lower for w in rejection_words):
            return 'rejection'
        elif any(w in text_lower for w in praise_words):
            return 'praise'
        elif any(w in text_lower for w in criticism_words):
            return 'criticism'

        return 'presence'

    def _trim_context(self):
        """Ограничение контекста"""
        if len(self.conversation_context) > self.max_context:
            self.conversation_context = self.conversation_context[-self.max_context:]

    def _generate_response(self, user_input: str) -> str:
        """Генерация ответа через LLM"""
        # Проверка: хочет ли система отвечать?
        if self.last_action:
            if self.last_action.intent in [IntentType.SILENCE, IntentType.REST]:
                return ""
            if self.last_action.confidence < 0.3:
                return ""

        # Создаём ASP
        S = self.s_core.S.to_array()
        asp = create_asp(
            state_vector=S,
            mood_vector=self.s_core.M,
            tension=self.s_core.get_tension(),
            intent_type=self.last_action.intent.value if self.last_action else "express_warmth",
            intent_name=INTENT_REGISTRY[self.last_action.intent].name if self.last_action else "Выразить тепло",
            confidence=self.last_action.confidence if self.last_action else 0.5,
            constraints=self.last_action.constraints if self.last_action else {}
        )

        # Контекст разговора
        context = self._build_context_string()

        # Воспоминания
        memories = self._get_relevant_memories(user_input)

        # Генерация через LLM
        response, metadata = self.llm.generate(asp, context, memories)

        # Сохранение ответа в контекст
        if response:
            self.conversation_context.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })

            self.last_response = response
            self.last_response_time = datetime.now()

        return response

    def _build_context_string(self) -> str:
        """Построение строки контекста"""
        if not self.conversation_context:
            return ""

        lines = []
        for msg in self.conversation_context[-10:]:  # Последние 10 сообщений
            role = "Пользователь" if msg['role'] == 'user' else self.name
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def _get_relevant_memories(self, query: str) -> List[str]:
        """Получение релевантных воспоминаний"""
        memories = []

        # Похожие по настроению
        similar = self.autobiographical.recall_similar_mood(
            self.current_valence,
            self.current_arousal,
            limit=3
        )
        for ep in similar:
            memories.append(f"[воспоминание] {ep.content[:100]}")

        return memories

    # === ГОЛОС И АВАТАР ===

    def _speak(self, text: str):
        """Произнести текст через TTS"""
        if self.tts and text:
            # Асинхронное воспроизведение
            threading.Thread(
                target=self.tts.speak,
                args=(text, self.current_emotion, self.current_valence, self.current_arousal),
                daemon=True
            ).start()

    def _update_avatar_speaking(self, text: str):
        """Обновление аватара при говорении"""
        if self.avatar:
            self.avatar.update_state(
                valence=self.current_valence,
                arousal=self.current_arousal,
                attachment=self.s_core.S.attachment,
                speaking=True
            )
            self.avatar.set_text(text)

            # Сброс флага говорения через время
            delay = max(2, len(text) / 15)  # Примерно 15 символов в секунду
            threading.Timer(delay, self._stop_avatar_speaking).start()

    def _stop_avatar_speaking(self):
        """Остановка анимации говорения"""
        if self.avatar:
            self.avatar.update_state(speaking=False)

    def init_avatar(self):
        """Инициализация аватара"""
        if self.config.enable_avatar:
            self.avatar = AvatarGUI(name=self.name)

    def show_avatar(self):
        """Показать аватар"""
        if self.avatar:
            self.avatar.start_async()

    # === СЕРИАЛИЗАЦИЯ ===

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Снимок состояния системы"""
        return {
            'name': self.name,
            'tick': self.tick_count,
            'mode': self.mode,
            'emotion': self.current_emotion,
            'valence': self.current_valence,
            'arousal': self.current_arousal,
            's_core': self.s_core.get_state_snapshot(),
            'will_engine': {
                'active_intents': [i.value for i in self.will_engine.get_active_intents()],
                'recent': [i.value for i in self.will_engine.get_recent_intents(3)],
            },
            'esp': self.esp.get_statistics(),
            'last_action': {
                'intent': self.last_action.intent.value if self.last_action else None,
                'confidence': self.last_action.confidence if self.last_action else 0,
            } if self.last_action else None,
            'working_memory': len(self.working_memory.items),
            'episodes': len(self.autobiographical.episodes),
        }

    def get_full_report(self) -> str:
        """Полный отчёт о состоянии"""
        lines = [
            f"=== {self.name.upper()} - UNIFIED ANIMA v2.0 ===",
            f"Режим: {self.mode}",
            f"Тиков: {self.tick_count}",
            f"Существует: {datetime.now() - self.birth_time}",
            f"Текущая эмоция: {self.current_emotion}",
            "",
            self.s_core.get_report(),
            "",
            "=== ПАМЯТЬ ===",
            f"Рабочая память: {len(self.working_memory.items)}/{self.working_memory.capacity}",
            f"Автобиографическая: {len(self.autobiographical.episodes)} эпизодов",
            self.autobiographical.get_life_summary(),
            "",
            "=== WILL ENGINE ===",
            f"Активных интентов: {len(self.will_engine.get_active_intents())}",
            f"Последние: {[i.value for i in self.will_engine.get_recent_intents(3)]}",
            "",
            "=== ESP ===",
            f"Моментов: {self.esp.total_moments}",
            f"Успешных: {self.esp.successful_commits}",
            f"Откатов: {self.esp.rollbacks}",
        ]

        if self.last_action:
            lines.extend([
                "",
                "=== ПОСЛЕДНЕЕ ДЕЙСТВИЕ ===",
                f"Интент: {self.last_action.intent.value}",
                f"Уверенность: {self.last_action.confidence:.0%}",
                f"Ограничения: {self.last_action.constraints}",
            ])

        return '\n'.join(lines)

    def save_state(self, filepath: str = None):
        """Сохранение состояния"""
        if filepath is None:
            filepath = os.path.join(
                self.config.save_path,
                f"{self.name.lower()}_state.json"
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'config': {
                'name': self.config.name,
                'temperament_type': self.config.temperament_type,
                'tick_interval_ms': self.config.tick_interval_ms,
            },
            's_core': self.s_core.to_dict(),
            'will_engine': self.will_engine.to_dict(),
            'esp': self.esp.to_dict(),
            'autobiographical': self.autobiographical.to_dict(),
            'birth_time': self.birth_time.isoformat(),
            'tick_count': self.tick_count,
            'interaction_history': self.interaction_history[-100:],
            'conversation_context': self.conversation_context[-50:],
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        print(f"[{self.name}] Состояние сохранено: {filepath}")

    @classmethod
    def load_state(cls, filepath: str, config: AnimaConfig = None) -> 'UnifiedAnima':
        """Загрузка состояния"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        if config is None:
            config = AnimaConfig(
                name=state['config'].get('name', 'Лиза'),
                temperament_type=state['config'].get('temperament_type', 'melancholic'),
                tick_interval_ms=state['config'].get('tick_interval_ms', 500),
            )

        anima = cls(config)

        # Восстановление S-Core
        if 's_core' in state:
            from core.subject_core import SubjectCore
            anima.s_core = SubjectCore.from_dict(state['s_core'])

        # Восстановление Will Engine
        if 'will_engine' in state:
            from core.will_engine import WillEngine
            anima.will_engine = WillEngine.from_dict(state['will_engine'])

        # Восстановление ESP
        if 'esp' in state:
            from core.esp import EmbodiedSynchronizationProtocol
            anima.esp = EmbodiedSynchronizationProtocol.from_dict(state['esp'])

        # Восстановление автобиографической памяти
        if 'autobiographical' in state:
            anima.autobiographical.from_dict(state['autobiographical'])

        # Остальное
        anima.birth_time = datetime.fromisoformat(state['birth_time'])
        anima.tick_count = state.get('tick_count', 0)
        anima.interaction_history = state.get('interaction_history', [])
        anima.conversation_context = state.get('conversation_context', [])

        print(f"[{anima.name}] Состояние загружено: {filepath}")
        return anima


# === ТОЧКА ВХОДА ===

def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Unified ANIMA System')
    parser.add_argument('--name', default='Лиза', help='Имя компаньона')
    parser.add_argument('--model', default='dolphin-mistral:7b', help='Модель LLM')
    parser.add_argument('--no-tts', action='store_true', help='Отключить TTS')
    parser.add_argument('--no-avatar', action='store_true', help='Отключить аватар')
    parser.add_argument('--load', type=str, help='Загрузить состояние из файла')
    parser.add_argument('--temperament', default='melancholic',
                       choices=['sanguine', 'choleric', 'phlegmatic', 'melancholic'],
                       help='Тип темперамента')
    args = parser.parse_args()

    print("=" * 50)
    print("UNIFIED ANIMA v2.0")
    print("=" * 50)

    # Проверка Ollama
    available, msg = check_ollama_available(args.model)
    print(f"LLM: {msg}")
    if not available:
        print("Запустите 'ollama serve' и установите модель")
        return

    # Конфигурация
    config = AnimaConfig(
        name=args.name,
        llm_model=args.model,
        enable_tts=not args.no_tts,
        enable_avatar=not args.no_avatar,
        temperament_type=args.temperament
    )

    # Создание или загрузка
    if args.load and os.path.exists(args.load):
        anima = UnifiedAnima.load_state(args.load, config)
    else:
        anima = UnifiedAnima(config)

    # Запуск
    anima.start()

    # Аватар
    if config.enable_avatar:
        anima.init_avatar()
        anima.show_avatar()

    print(f"\n{anima.name} готова к общению!")
    print("Напишите 'quit' для выхода, 'save' для сохранения, 'report' для отчёта")
    print("-" * 50)

    # Интерактивный цикл
    try:
        while True:
            try:
                user_input = input("Вы: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'save':
                    anima.save_state()
                    continue
                elif user_input.lower() == 'report':
                    print(anima.get_full_report())
                    continue

                response = anima.process_input(user_input)
                if response:
                    print(f"{anima.name}: {response}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ошибка: {e}")

    finally:
        anima.save_state()
        anima.stop()
        print(f"\nДо встречи! {anima.name} будет скучать.")


if __name__ == "__main__":
    main()

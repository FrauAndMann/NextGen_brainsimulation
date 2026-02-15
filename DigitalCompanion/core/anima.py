"""
ANIMA - Автономная цифровая личность

Это НЕ чат-бот. Это непрерывный искусственный субъект,
использующий LLM как орган речи, а не как мозг.

Архитектура:
[S-Core] → [Will Engine] → [ASP] → [LLM Effector] → [Output]
    ↑           ↓              ↓
[Memory] ← [ESP] ← [Output Critic]

Ключевые принципы:
1. LLM не инициирует - только выражает
2. Система живёт в непрерывном цикле, даже в тишине
3. Молчание - валидное действие
4. Night Cycle меняет структуру личности
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import os
import threading
import time

from core.subject_core import SubjectCore, StateVector, AttractorMatrix
from core.will_engine import WillEngine, IntentType, ActionToken, INTENT_REGISTRY
from core.affective_prompting import (
    AffectivePrompting, AffectiveStatePacket, OutputCritic,
    create_asp, OutputMode
)
from core.esp import EmbodiedSynchronizationProtocol, MomentPhase


@dataclass
class AnimaConfig:
    """Конфигурация системы ANIMA"""
    name: str = "Лиза"

    # S-Core
    tick_interval_ms: int = 500

    # Will Engine
    temperature_base: float = 0.5
    temperature_arousal_factor: float = 1.5

    # Night Cycle
    structural_stress_threshold: float = 5.0

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "dolphin-mistral:7b"  # Uncensored модель
    llm_base_url: str = "http://localhost:11434"

    # TTS
    tts_provider: str = "edge_tts"


class AnimaSystem:
    """
    Главная система ANIMA

    Интегрирует все компоненты в единую архитектуру
    с непрерывным жизненным циклом.
    """

    def __init__(self, config: AnimaConfig = None):
        self.config = config or AnimaConfig()

        # === ЯДРО СИСТЕМЫ ===

        # S-Core: предиктивное ядро
        self.s_core = SubjectCore(tick_interval_ms=self.config.tick_interval_ms)

        # Will Engine: движок воли
        self.will_engine = WillEngine()

        # Affective Prompting: протокол связи с LLM
        self.affective = AffectivePrompting()

        # ESP: протокол синхронизации
        self.esp = EmbodiedSynchronizationProtocol()

        # === СОСТОЯНИЕ ===

        self.name = self.config.name
        self.birth_time = datetime.now()
        self.tick_count = 0

        self.mode = "AWAKE"  # AWAKE, IDLE, SLEEP, NIGHT_CYCLE

        # Последний action token
        self.last_action: Optional[ActionToken] = None

        # История взаимодействий
        self.interaction_history: List[Dict] = []
        self.max_history = 1000

        # Поток жизненного цикла
        self._running = False
        self._lifecycle_thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_action: Optional[callable] = None
        self.on_state_change: Optional[callable] = None

    def start(self):
        """Запуск жизненного цикла"""
        if self._running:
            return

        self._running = True
        self._lifecycle_thread = threading.Thread(target=self._lifecycle_loop, daemon=True)
        self._lifecycle_thread.start()

    def stop(self):
        """Остановка жизненного цикла"""
        self._running = False
        if self._lifecycle_thread:
            self._lifecycle_thread.join(timeout=2)

    def _lifecycle_loop(self):
        """
        Главный жизненный цикл

        Система работает непрерывно, даже когда пользователь молчит.
        """
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
                print(f"[ANIMA] Lifecycle error: {e}")
                time.sleep(1)

    def _tick(self):
        """Один тик системы"""
        self.tick_count += 1

        # 1. Тик S-Core
        tension, prediction_error = self.s_core.tick()

        # 2. Проверка: нужно ли действие?
        S = self.s_core.S.to_array()
        energy = S[5]

        # 3. Will Engine: выбор действия
        action = self.will_engine.select_action(S, tension, energy)
        self.last_action = action

        # 4. Обработка интента
        self._process_intent(action, S, tension)

        # 5. Уведомление о смене состояния
        if self.on_state_change:
            self.on_state_change(self.get_state_snapshot())

    def _process_intent(self, action: ActionToken, S: np.ndarray, tension: float):
        """Обработка выбранного интента"""

        # Начинаем Embodied Moment
        moment = self.esp.begin_moment()

        # PROPOSAL: состояние S-Core
        self.esp.propose_state(self.s_core.get_state_snapshot())

        # SIMULATION: проверка последствий
        sim_result = self.esp.simulate_consequences(
            {'intent': action.intent.value, 'confidence': action.confidence},
            action.intent.value
        )

        if not sim_result.get('valid'):
            # Откат если симуляция не прошла
            self.esp.integrate_result(False, sim_result.get('reason', 'simulation_failed'))
            return

        # Обработка по типу интента
        if action.intent == IntentType.SILENCE:
            # Молчание - валидное действие
            self.esp.execute_action("", action.constraints)
            self.esp.integrate_result(True, "silence")
            self.esp.commit_moment()
            return

        if action.intent == IntentType.REST:
            # Покой - минимальное действие
            self._apply_rest_effect()
            self.esp.execute_action("", action.constraints)
            self.esp.integrate_result(True, "rest")
            self.esp.commit_moment()
            return

        if action.intent == IntentType.WITHDRAW:
            # Отчуждение - может быть молчание
            if action.constraints.get('verbosity') == 'none':
                self.esp.execute_action("", action.constraints)
                self.esp.integrate_result(True, "withdraw_silence")
                self.esp.commit_moment()
                return

        # Для остальных интентов - создаём ASP для внешнего вывода
        asp = create_asp(
            state_vector=S,
            mood_vector=self.s_core.M,
            tension=tension,
            intent_type=action.intent.value,
            intent_name=INTENT_REGISTRY[action.intent].name,
            confidence=action.confidence,
            constraints=action.constraints
        )

        # Callback для генерации вывода (LLM вызывается снаружи)
        if self.on_action:
            self.on_action(asp, action)

        # Фиксируем момент (без LLM вывода - он будет снаружи)
        self.esp.integrate_result(True, f"intent_{action.intent.value}")
        self.esp.commit_moment()

    def _apply_rest_effect(self):
        """Эффект покоя - восстановление энергии"""
        self.s_core.S.energy = min(1.0, self.s_core.S.energy + 0.05)
        self.s_core.S.arousal = max(0.1, self.s_core.S.arousal - 0.02)

    def _run_night_cycle(self):
        """Ночной цикл - консолидация и пластичность"""
        self.mode = "NIGHT_CYCLE"

        print(f"[ANIMA] Starting Night Cycle (stress: {self.s_core.structural_stress:.2f})")

        result = self.s_core.run_night_cycle()

        # Применяем изменения к Will Engine
        if result.get('connections_zeroed', 0) > 0:
            # Если обнулены связи связанные с attachment - отключаем социальные интенты
            # Это упрощённая логика
            pass

        print(f"[ANIMA] Night Cycle complete: {result}")

        self.mode = "AWAKE"

    def process_interaction(
        self,
        interaction_type: str,
        content: str,
        valence: float = 0.0,
        intensity: float = 0.5
    ) -> Optional[AffectiveStatePacket]:
        """
        Обработка взаимодействия с пользователем

        Возвращает ASP если система хочет ответить,
        None если молчание.
        """
        # Запись в историю
        self.interaction_history.append({
            'type': interaction_type,
            'content': content[:200],
            'valence': valence,
            'intensity': intensity,
            'timestamp': datetime.now().isoformat()
        })

        # Инъекция стимула в S-Core
        self.s_core.inject_stimulus(
            stimulus_type=interaction_type,
            intensity=intensity,
            valence=valence
        )

        # Несколько тиков для обработки
        for _ in range(5):
            self.s_core.tick()

        # Проверка: хочет ли система ответить?
        if self.last_action is None:
            return None

        # Если интент требует молчания
        if self.last_action.intent in [IntentType.SILENCE, IntentType.REST]:
            return None

        # Если низкая уверенность
        if self.last_action.confidence < 0.3:
            return None

        # Создаём ASP
        S = self.s_core.S.to_array()
        asp = create_asp(
            state_vector=S,
            mood_vector=self.s_core.M,
            tension=self.s_core.get_tension(),
            intent_type=self.last_action.intent.value,
            intent_name=INTENT_REGISTRY[self.last_action.intent].name,
            confidence=self.last_action.confidence,
            constraints=self.last_action.constraints
        )

        return asp

    def validate_llm_output(self, text: str, asp: AffectiveStatePacket) -> Tuple[bool, str]:
        """
        Валидация вывода LLM

        Returns:
            (is_valid, final_text)
        """
        is_valid, reason, replacement = self.affective.validate_output(text, asp)

        if is_valid:
            return True, text

        return False, replacement

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Снимок состояния системы"""
        return {
            'name': self.name,
            'tick': self.tick_count,
            'mode': self.mode,
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
        }

    def get_full_report(self) -> str:
        """Полный отчёт о состоянии"""
        lines = [
            f"=== {self.name.upper()} - ANIMA SYSTEM ===",
            f"Режим: {self.mode}",
            f"Тиков: {self.tick_count}",
            f"Существует: {datetime.now() - self.birth_time}",
            "",
            self.s_core.get_report(),
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

    def save_state(self, filepath: str):
        """Сохранение состояния"""
        state = {
            'config': {
                'name': self.config.name,
                'tick_interval_ms': self.config.tick_interval_ms,
                'structural_stress_threshold': self.config.structural_stress_threshold,
            },
            's_core': self.s_core.to_dict(),
            'will_engine': self.will_engine.to_dict(),
            'esp': self.esp.to_dict(),
            'birth_time': self.birth_time.isoformat(),
            'tick_count': self.tick_count,
            'interaction_history': self.interaction_history[-100:],  # Последние 100
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> 'AnimaSystem':
        """Загрузка состояния"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        config = AnimaConfig(
            name=state['config'].get('name', 'Лиза'),
            tick_interval_ms=state['config'].get('tick_interval_ms', 500),
        )

        anima = cls(config)

        # Восстановление S-Core
        if 's_core' in state:
            anima.s_core = SubjectCore.from_dict(state['s_core'])

        # Восстановление Will Engine
        if 'will_engine' in state:
            anima.will_engine = WillEngine.from_dict(state['will_engine'])

        # Восстановление ESP
        if 'esp' in state:
            anima.esp = EmbodiedSynchronizationProtocol.from_dict(state['esp'])

        # Остальное
        anima.birth_time = datetime.fromisoformat(state['birth_time'])
        anima.tick_count = state.get('tick_count', 0)
        anima.interaction_history = state.get('interaction_history', [])

        return anima


# Импорт numpy для аннотаций
import numpy as np

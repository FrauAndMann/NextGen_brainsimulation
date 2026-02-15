"""
Embodied Synchronization Protocol (ESP)

Протокол конвергенции - обеспечивает, что все модули системы
работают как единое "тело", а не рассинхронизированный ансамбль.

Ключевой принцип: ничто не считается "произошедшим",
пока Embodied Moment не закрыт.

Время:
- S-Core: непрерывное, dt ~500ms
- Will Engine: событийное, при выборе действия
- LLM: дискретное, при генерации
- Night Cycle: фазовое, триггерится по structural_stress

ESP синхронизирует эти разные онтологии времени.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import threading
import time


class MomentPhase(Enum):
    """Фазы Embodied Moment"""
    PROPOSAL = "proposal"       # Предложение состояния
    SIMULATION = "simulation"   # Симуляция последствий
    EXECUTION = "execution"     # Выполнение действия
    INTEGRATION = "integration" # Интеграция результата
    COMMIT = "commit"           # Фиксация момента
    ROLLBACK = "rollback"       # Откат при неудаче


class ModuleState(Enum):
    """Состояние модуля в моменте"""
    READY = "ready"
    PROCESSING = "processing"
    BLOCKED = "blocked"
    COMMITTED = "committed"


@dataclass
class EmbodiedMoment:
    """
    Embodied Moment (EM) - атомарная транзакция бытия

    Это минимальная неделимая единица "проживания".
    Ничто не считается реальным для системы, пока EM не закрыт.
    """
    id: str
    timestamp: datetime

    # Фаза момента
    phase: MomentPhase = MomentPhase.PROPOSAL

    # Снимки состояния от каждого модуля
    s_core_snapshot: Optional[Dict] = None
    will_engine_snapshot: Optional[Dict] = None
    llm_output: Optional[str] = None
    memory_write: Optional[Dict] = None

    # Статусы модулей
    module_states: Dict[str, ModuleState] = field(default_factory=dict)

    # Результат валидации
    is_valid: bool = True
    validation_reason: str = ""

    # Время обработки
    processing_time_ms: float = 0.0

    def is_ready_for_commit(self) -> bool:
        """Проверка готовности к коммиту"""
        required = ['s_core', 'will_engine']
        for module in required:
            if self.module_states.get(module) != ModuleState.READY:
                return False
        return True


class SynchronizationBarrier:
    """
    Барьер синхронизации

    Каждый процесс обязан либо подтвердить готовность,
    либо заблокировать момент.
    """

    def __init__(self):
        self.pending_modules: Dict[str, bool] = {}
        self.timeout_ms = 5000  # Максимальное ожидание
        self.lock = threading.Lock()

    def register(self, module_name: str):
        """Регистрация модуля для ожидания"""
        with self.lock:
            self.pending_modules[module_name] = False

    def signal_ready(self, module_name: str):
        """Сигнал готовности от модуля"""
        with self.lock:
            self.pending_modules[module_name] = True

    def wait_for_all(self, timeout_ms: int = None) -> bool:
        """
        Ожидание готовности всех модулей

        Returns:
            True если все готовы, False если таймаут
        """
        timeout = (timeout_ms or self.timeout_ms) / 1000
        start = time.time()

        while time.time() - start < timeout:
            with self.lock:
                if all(self.pending_modules.values()):
                    return True
            time.sleep(0.01)

        return False

    def reset(self):
        """Сброс барьера"""
        with self.lock:
            for module in self.pending_modules:
                self.pending_modules[module] = False


class EmbodiedSynchronizationProtocol:
    """
    ESP - Протокол синхронизации embodied системы

    Гарантирует:
    - Состояние S-Core и действие Will Engine согласованы
    - LLM вывод соответствует текущему состоянию
    - Память записывает только прожитые моменты
    - Night Cycle не стартует при открытых моментах
    """

    def __init__(self):
        self.current_moment: Optional[EmbodiedMoment] = None
        self.moment_history: List[EmbodiedMoment] = []
        self.max_history = 1000

        self.barrier = SynchronizationBarrier()
        self.barrier.register('s_core')
        self.barrier.register('will_engine')
        self.barrier.register('memory')

        self.pending_memory_writes: List[Dict] = []
        self.active_intent: Optional[str] = None

        # Callbacks для модулей
        self.callbacks: Dict[str, Callable] = {}

        # Статистика
        self.total_moments = 0
        self.successful_commits = 0
        self.rollbacks = 0

    def register_callback(self, event: str, callback: Callable):
        """Регистрация callback на события"""
        self.callbacks[event] = callback

    def begin_moment(self) -> EmbodiedMoment:
        """Начало нового Embodied Moment"""
        # Если есть незакрытый момент - это ошибка
        if self.current_moment and self.current_moment.phase != MomentPhase.COMMIT:
            self._rollback_moment(self.current_moment, "previous_moment_not_closed")

        moment_id = f"em_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.current_moment = EmbodiedMoment(
            id=moment_id,
            timestamp=datetime.now(),
            phase=MomentPhase.PROPOSAL
        )

        self.barrier.reset()
        self.total_moments += 1

        return self.current_moment

    def propose_state(self, s_core_snapshot: Dict) -> bool:
        """Фаза PROPOSAL: предложение состояния от S-Core"""
        if not self.current_moment:
            return False

        self.current_moment.phase = MomentPhase.PROPOSAL
        self.current_moment.s_core_snapshot = s_core_snapshot
        self.current_moment.module_states['s_core'] = ModuleState.PROCESSING

        return True

    def simulate_consequences(
        self,
        will_engine_snapshot: Dict,
        intent_type: str
    ) -> Dict[str, Any]:
        """
        Фаза SIMULATION: проверка последствий выбранного интента

        Проверяем:
        - Достаточно ли энергии для интента?
        - Соответствует ли интент текущему состоянию?
        """
        if not self.current_moment:
            return {'valid': False, 'reason': 'no_active_moment'}

        self.current_moment.phase = MomentPhase.SIMULATION
        self.current_moment.will_engine_snapshot = will_engine_snapshot
        self.active_intent = intent_type

        # Проверка энергии
        energy = self.current_moment.s_core_snapshot.get('energy', 0.5) if self.current_moment.s_core_snapshot else 0.5

        # Энергозатратные интенты при низкой энергии
        energy_costly = ['seek_attention', 'assert', 'self_modify']
        if intent_type in energy_costly and energy < 0.3:
            return {
                'valid': False,
                'reason': 'insufficient_energy',
                'suggested': 'rest'
            }

        # Проверка соответствия состояния
        valence = self.current_moment.s_core_snapshot.get('valence', 0) if self.current_moment.s_core_snapshot else 0

        # Тепло при сильном негативе - подозрительно
        if intent_type == 'express_warmth' and valence < -0.5:
            return {
                'valid': False,
                'reason': 'state_mismatch',
                'suggested': 'withdraw'
            }

        self.current_moment.module_states['will_engine'] = ModuleState.PROCESSING
        self.barrier.signal_ready('will_engine')

        return {'valid': True}

    def execute_action(self, llm_output: str, constraints: Dict) -> bool:
        """
        Фаза EXECUTION: выполнение действия (LLM вывод)

        Возвращает True если вывод прошёл валидацию
        """
        if not self.current_moment:
            return False

        self.current_moment.phase = MomentPhase.EXECUTION

        # Проверка на молчание
        if constraints.get('verbosity') == 'none' or not llm_output.strip():
            self.current_moment.llm_output = ""  # Молчание - валидный вывод
            self.current_moment.module_states['llm'] = ModuleState.COMMITTED
            return True

        # Сохраняем вывод
        self.current_moment.llm_output = llm_output
        self.current_moment.module_states['llm'] = ModuleState.PROCESSING

        return True

    def integrate_result(
        self,
        validation_passed: bool,
        validation_reason: str
    ) -> bool:
        """
        Фаза INTEGRATION: интеграция результата в систему

        Если валидация не прошла - момент откатывается
        """
        if not self.current_moment:
            return False

        self.current_moment.phase = MomentPhase.INTEGRATION
        self.current_moment.is_valid = validation_passed
        self.current_moment.validation_reason = validation_reason

        if not validation_passed:
            self._rollback_moment(self.current_moment, validation_reason)
            return False

        # Подготовка записи в память
        self.current_moment.module_states['memory'] = ModuleState.PROCESSING
        self.barrier.signal_ready('s_core')

        return True

    def commit_moment(self, memory_write: Dict = None) -> bool:
        """
        Фаза COMMIT: фиксация момента

        Только после коммита состояние считается "реальным"
        """
        if not self.current_moment:
            return False

        # Проверка готовности барьера
        if not self.barrier.wait_for_all(timeout_ms=1000):
            self._rollback_moment(self.current_moment, "barrier_timeout")
            return False

        self.current_moment.phase = MomentPhase.COMMIT
        self.current_moment.memory_write = memory_write
        self.current_moment.module_states['memory'] = ModuleState.COMMITTED

        # Сохраняем в историю
        self.moment_history.append(self.current_moment)
        if len(self.moment_history) > self.max_history:
            self.moment_history.pop(0)

        self.successful_commits += 1
        self.current_moment.processing_time_ms = (
            datetime.now() - self.current_moment.timestamp
        ).total_seconds() * 1000

        # Уведомляем callbacks
        if 'on_commit' in self.callbacks:
            self.callbacks['on_commit'](self.current_moment)

        self.current_moment = None
        self.active_intent = None

        return True

    def _rollback_moment(self, moment: EmbodiedMoment, reason: str):
        """Откат момента"""
        moment.phase = MomentPhase.ROLLBACK
        moment.is_valid = False
        moment.validation_reason = reason

        self.rollbacks += 1

        # Уведомляем callbacks
        if 'on_rollback' in self.callbacks:
            self.callbacks['on_rollback'](moment, reason)

        # Сохраняем откаченные моменты для анализа
        self.moment_history.append(moment)
        if len(self.moment_history) > self.max_history:
            self.moment_history.pop(0)

        self.current_moment = None
        self.active_intent = None

    def can_start_night_cycle(self) -> bool:
        """
        Проверка возможности запуска ночного цикла

        Night Cycle не может стартовать если:
        - Есть открытый (незакрытый) Embodied Moment
        - Есть активный Intent
        - Есть pending memory writes
        """
        if self.current_moment is not None:
            return False

        if self.active_intent is not None:
            return False

        if self.pending_memory_writes:
            return False

        return True

    def get_unified_state(self) -> Optional[Dict[str, Any]]:
        """
        Получение единого состояния системы

        Возвращает состояние только если все модули синхронизированы
        """
        if self.current_moment:
            return None  # Незавершённый момент

        # Берём последний закоммиченный момент
        if not self.moment_history:
            return None

        last_moment = self.moment_history[-1]
        if last_moment.phase != MomentPhase.COMMIT:
            return None

        return {
            's_core': last_moment.s_core_snapshot,
            'will_engine': last_moment.will_engine_snapshot,
            'last_output': last_moment.llm_output,
            'moment_id': last_moment.id,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Статистика работы ESP"""
        return {
            'total_moments': self.total_moments,
            'successful_commits': self.successful_commits,
            'rollbacks': self.rollbacks,
            'commit_rate': self.successful_commits / max(1, self.total_moments),
            'history_size': len(self.moment_history),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация"""
        return {
            'total_moments': self.total_moments,
            'successful_commits': self.successful_commits,
            'rollbacks': self.rollbacks,
            'can_start_night_cycle': self.can_start_night_cycle(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbodiedSynchronizationProtocol':
        """Десериализация"""
        esp = cls()
        esp.total_moments = data.get('total_moments', 0)
        esp.successful_commits = data.get('successful_commits', 0)
        esp.rollbacks = data.get('rollbacks', 0)
        return esp

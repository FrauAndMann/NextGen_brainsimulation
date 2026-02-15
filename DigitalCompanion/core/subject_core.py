"""
Subject Core (S-Core) - Предиктивное ядро субъекта

Это НЕ LLM. Это динамическая система с непрерывным состоянием,
которая живёт постоянно, даже в тишине.

Основано на Active Inference:
- Система предсказывает своё будущее состояние
- Ошибка предсказания = "страдание" = мотивация к действию
- Действие = попытка минимизировать ошибку предсказания

Ключевое отличие от предыдущей реализации:
- 6 осей вместо 120+ нейромедиаторов
- Предсказание + ошибка, не просто состояние
- Матрица аттракторов W определяет "характер"
- Ночная пластичность меняет саму динамику
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import math


@dataclass
class StateVector:
    """
    Вектор состояния S = [V, A, D, T, N, E]

    V - Valence (валентность): приятно ↔ больно [-1, +1]
    A - Arousal (возбуждение): спокойно ↔ напряжено [0, 1]
    D - Dominance (доминирование): контроль ↔ беспомощность [0, 1]
    T - Attachment (привязанность): близость ↔ отчуждение [0, 1]
    N - Novelty (новизна): интерес ↔ избегание [0, 1]
    E - Energy (энергия): ресурс [0, 1]
    """
    valence: float = 0.0       # V: -1 to +1
    arousal: float = 0.3       # A: 0 to 1 (baseline снижен)
    dominance: float = 0.5     # D: 0 to 1
    attachment: float = 0.5    # T: 0 to 1
    novelty: float = 0.5       # N: 0 to 1
    energy: float = 0.7        # E: 0 to 1

    def to_array(self) -> np.ndarray:
        return np.array([
            self.valence,
            self.arousal,
            self.dominance,
            self.attachment,
            self.novelty,
            self.energy
        ])

    def from_array(self, arr: np.ndarray):
        self.valence = float(np.clip(arr[0], -1, 1))
        self.arousal = float(np.clip(arr[1], 0, 1))
        self.dominance = float(np.clip(arr[2], 0, 1))
        self.attachment = float(np.clip(arr[3], 0, 1))
        self.novelty = float(np.clip(arr[4], 0, 1))
        self.energy = float(np.clip(arr[5], 0, 1))

    def copy(self) -> 'StateVector':
        return StateVector(
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance,
            attachment=self.attachment,
            novelty=self.novelty,
            energy=self.energy
        )


@dataclass
class AttractorMatrix:
    """
    Матрица аттракторов W ∈ ℝ^(6×6)

    Определяет динамику системы: S_pred(t+1) = W · S(t)

    W[i,j] = как состояние j влияет на предсказание состояния i

    Ключевые связи (человекоподобный паттерн):
    - Тревога (A) подавляет исследование (N): W[N,A] < W[A,N]
    - Привязанность (T) → Валентность (V): W[V,T] > 0
    - Доминирование (D) буферит боль: W[V,D] > 0
    - Энергия (E) влияет на тревогу: W[A,E] < 0
    """

    # Каноническая структура W (функционально мотивированная)
    #        V      A      D      T      N      E
    CANONICAL_W = np.array([
        [ 0.90, -0.40,  0.30,  0.50,  0.20,  0.10],  # V
        [-0.30,  0.80, -0.40, -0.20,  0.30, -0.50],  # A
        [ 0.20, -0.50,  0.85,  0.10,  0.00,  0.20],  # D
        [ 0.40, -0.30,   0.10,  0.90,  0.10,  0.00],  # T
        [ 0.30,  0.20,  0.00,  0.20,  0.85,  0.10],  # N
        [ 0.10, -0.40,  0.20,  0.00,  0.00,  0.90],  # E
    ])

    def __init__(self, initial_w: np.ndarray = None):
        self.W = initial_w.copy() if initial_w is not None else self.CANONICAL_W.copy()
        self.locked_connections = set()  # Заблокированные связи (после обнуления)

    def predict_next(self, current: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Предсказание следующего состояния"""
        noise = np.random.normal(0, noise_level, 6)
        predicted = self.W @ current + noise
        return predicted

    def apply_hebbian_plasticity(
        self,
        prediction_errors: np.ndarray,
        states: np.ndarray,
        lr: float = 1e-4,
        decay: float = 1e-4
    ):
        """
        Ночная пластичность: Hebbian + Anti-Hebbian

        Усиление (редко): связь оправдала ожидание
        Ослабление (чаще): связь причиняет ошибку

        ΔW_ij += η · S_i · S_j    if ε_j < small (подтверждение)
        ΔW_ij -= λ · |ε_j| · S_i  (боль ослабляет связь)
        """
        for i in range(6):
            for j in range(6):
                if (i, j) in self.locked_connections:
                    continue

                error = abs(prediction_errors[j])

                if error < 0.1:  # Связь оправдалась
                    delta = lr * states[i] * states[j]
                else:  # Связь причиняет боль
                    delta = -decay * error * states[i]

                self.W[i, j] += delta

        # Нормализация
        self.W = np.clip(self.W, -1, 1)

    def lock_connection(self, i: int, j: int):
        """Заблокировать связь навсегда (после обнуления)"""
        self.locked_connections.add((i, j))
        self.W[i, j] = 0

    def get_active_connections_count(self) -> int:
        """Количество активных связей"""
        return np.sum(np.abs(self.W) > 0.05)

    def to_dict(self) -> Dict:
        return {
            'W': self.W.tolist(),
            'locked': list(self.locked_connections),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AttractorMatrix':
        matrix = cls()
        matrix.W = np.array(data['W'])
        matrix.locked_connections = set(tuple(x) for x in data.get('locked', []))
        return matrix


@dataclass
class PredictionTrace:
    """Запись о предсказании для анализа"""
    timestamp: datetime
    predicted_state: np.ndarray
    actual_state: np.ndarray
    prediction_error: np.ndarray
    tension: float


class SubjectCore:
    """
    Предиктивное ядро субъекта (S-Core)

    Это "сердцебиение" системы - непрерывный процесс,
    который живёт даже в тишине.

    Ключевые принципы:
    1. Active Inference: система предсказает себя и страдает от ошибок
    2. Гомеостаз: есть целевое состояние S_target
    3. Напряжение: мера отклонения от цели = мотивация к действию
    4. Настроение: медленный интегратор ошибок
    """

    def __init__(self, tick_interval_ms: int = 500):
        self.tick_interval = tick_interval_ms / 1000.0  # в секундах

        # Текущее состояние
        self.S = StateVector()

        # Целевое состояние (гомеостаз)
        self.S_target = StateVector(
            valence=0.3,     # Немного позитивно
            arousal=0.2,     # Спокойно
            dominance=0.5,   # Баланс
            attachment=0.7,  # Близость желаема
            novelty=0.4,     # Умеренный интерес
            energy=0.8       # Высокая энергия
        )

        # Матрица аттракторов (характер)
        self.W = AttractorMatrix()

        # Медленное настроение (инерционный тензор)
        self.M = np.zeros(6)  # Mood vector
        self.mood_decay = 0.99  # Очень медленное затухание

        # Веса значимости осей
        self.importance_weights = np.array([1.0, 0.8, 0.6, 1.2, 0.5, 0.9])

        # История для Night Cycle
        self.daily_traces: List[PredictionTrace] = []
        self.structural_stress: float = 0.0  # Накопленная структурная боль

        # Счётчики
        self.tick_count: int = 0
        self.last_tick_time: datetime = field(default_factory=datetime.now)

        # Внешний стимул (очередь)
        self.pending_stimuli: List[Dict[str, Any]] = []

    def tick(self, dt: float = None) -> Tuple[float, np.ndarray]:
        """
        Главный тик системы - "сердцебиение"

        Returns:
            (tension, prediction_error) - напряжение и ошибка предсказания
        """
        if dt is None:
            dt = self.tick_interval

        self.tick_count += 1
        self.last_tick_time = datetime.now()

        # Текущее состояние как массив
        S_current = self.S.to_array()

        # 1. Предсказание следующего состояния на основе W
        S_predicted = self.W.predict_next(S_current)

        # 2. Применение внешних стимулов
        stimulus_delta = self._process_stimuli()

        # 3. Естественная динамика (диффуры)
        # Энергия убывает со временем
        natural_decay = np.array([0, 0, 0, -0.01, -0.02, -0.03]) * dt

        # 4. Обновление реального состояния
        S_new = S_current + natural_decay + stimulus_delta

        # 5. Ограничение значений
        S_new = np.clip(S_new, [-1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
        self.S.from_array(S_new)

        # 6. Вычисление ошибки предсказания (Surprise)
        prediction_error = S_new - S_predicted

        # 7. Обновление настроения (медленный интегратор)
        self.M = self.mood_decay * self.M + (1 - self.mood_decay) * prediction_error

        # 8. Вычисление суммарного напряжения (Drive)
        # Это топливо для Will Engine
        state_deviation = np.abs(S_new - self.S_target.to_array())
        mood_contribution = np.abs(self.M)
        tension = np.sum(
            (state_deviation + mood_contribution) * self.importance_weights
        ) / np.sum(self.importance_weights)

        # 9. Накопление структурного стресса
        self.structural_stress += np.sum(np.abs(prediction_error) * self.importance_weights) * 0.01

        # 10. Запись трассы для Night Cycle
        trace = PredictionTrace(
            timestamp=self.last_tick_time,
            predicted_state=S_predicted.copy(),
            actual_state=S_new.copy(),
            prediction_error=prediction_error.copy(),
            tension=tension
        )
        self.daily_traces.append(trace)

        # Ограничение истории
        if len(self.daily_traces) > 10000:
            self.daily_traces = self.daily_traces[-5000:]

        return tension, prediction_error

    def _process_stimuli(self) -> np.ndarray:
        """Обработка накопленных стимулов"""
        delta = np.zeros(6)

        for stimulus in self.pending_stimuli:
            stim_type = stimulus.get('type', 'unknown')
            intensity = stimulus.get('intensity', 0.5)
            valence = stimulus.get('valence', 0.0)

            if stim_type == 'affection_shown':
                # Проявление привязанности
                delta[0] += 0.3 * intensity  # Valence up
                delta[3] += 0.2 * intensity  # Attachment up
                delta[5] += 0.1 * intensity  # Energy up

            elif stim_type == 'rejection':
                # Отвержение
                delta[0] -= 0.4 * intensity  # Valence down
                delta[3] -= 0.3 * intensity  # Attachment down
                delta[1] += 0.3 * intensity  # Arousal up

            elif stim_type == 'praise':
                # Похвала
                delta[0] += 0.2 * intensity
                delta[2] += 0.1 * intensity  # Dominance up

            elif stim_type == 'criticism':
                # Критика
                delta[0] -= 0.2 * intensity
                delta[2] -= 0.15 * intensity  # Dominance down
                delta[1] += 0.2 * intensity  # Arousal up

            elif stim_type == 'presence':
                # Присутствие пользователя
                delta[3] += 0.1 * intensity  # Attachment slowly up
                delta[4] += 0.05 * intensity  # Novelty up

            elif stim_type == 'absence':
                # Отсутствие пользователя
                delta[3] -= 0.05 * intensity  # Attachment slowly down
                delta[5] -= 0.02 * intensity  # Energy slowly down

            elif stim_type == 'novelty':
                # Новая информация
                delta[4] += 0.3 * intensity  # Novelty up
                delta[1] += 0.1 * intensity  # Arousal up

        self.pending_stimuli.clear()
        return delta

    def inject_stimulus(self, stimulus_type: str, intensity: float = 0.5, valence: float = 0.0, **kwargs):
        """Добавить стимул в очередь обработки"""
        self.pending_stimuli.append({
            'type': stimulus_type,
            'intensity': intensity,
            'valence': valence,
            **kwargs
        })

    def get_tension(self) -> float:
        """Получить текущий уровень напряжения"""
        S_current = self.S.to_array()
        S_target = self.S_target.to_array()
        deviation = np.abs(S_current - S_target)
        mood = np.abs(self.M)
        return np.sum((deviation + mood) * self.importance_weights) / np.sum(self.importance_weights)

    def get_prediction_error_magnitude(self) -> float:
        """Магнитуда последней ошибки предсказания"""
        if not self.daily_traces:
            return 0.0
        return float(np.mean(np.abs(self.daily_traces[-1].prediction_error)))

    def should_run_night_cycle(self, threshold: float = 5.0) -> bool:
        """Проверка триггера ночного цикла"""
        return self.structural_stress > threshold

    def run_night_cycle(self) -> Dict[str, Any]:
        """
        Ночной цикл - консолидация и пластичность

        Это единственный момент, когда система меняет свою структуру.
        """
        if not self.daily_traces:
            return {'status': 'no_data', 'changes': 0}

        # Собираем ошибки предсказания за день
        all_errors = np.array([t.prediction_error for t in self.daily_traces])
        all_states = np.array([t.actual_state for t in self.daily_traces])

        mean_errors = np.mean(all_errors, axis=0)
        mean_states = np.mean(all_states, axis=0)

        # Применяем пластичность
        old_W = self.W.W.copy()

        self.W.apply_hebbian_plasticity(mean_errors, mean_states)

        # Проверяем связи для обнуления
        connections_zeroed = 0
        plasticity_floor = 0.05

        for i in range(6):
            for j in range(6):
                if abs(self.W.W[i, j]) < plasticity_floor and (i, j) not in self.W.locked_connections:
                    self.W.lock_connection(i, j)
                    connections_zeroed += 1

        # Сбрасываем дневной стресс
        stress_processed = self.structural_stress
        self.structural_stress = 0.0
        self.daily_traces.clear()

        # Вычисляем вектор смещения личности
        delta_W = self.W.W - old_W
        identity_drift = np.linalg.norm(delta_W, 'fro')

        return {
            'status': 'completed',
            'connections_zeroed': connections_zeroed,
            'active_connections': self.W.get_active_connections_count(),
            'stress_processed': stress_processed,
            'identity_drift': float(identity_drift),
            'mean_errors': mean_errors.tolist(),
        }

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Снимок состояния для ASP (Affective State Packet)"""
        return {
            'S': self.S.to_array().tolist(),
            'M': self.M.tolist(),
            'tension': self.get_tension(),
            'prediction_error': self.get_prediction_error_magnitude(),
            'structural_stress': self.structural_stress,
            'active_connections': self.W.get_active_connections_count(),
            'tick_count': self.tick_count,
        }

    def get_report(self) -> str:
        """Текстовый отчёт о состоянии"""
        lines = [
            "=== S-CORE: ПРЕДИКТИВНОЕ ЯДРО ===",
            "",
            f"Тиков: {self.tick_count}",
            f"Напряжение: {self.get_tension():.3f}",
            f"Структурный стресс: {self.structural_stress:.3f}",
            "",
            "Состояние (S):",
            f"  Valence (V):     {self.S.valence:+.2f}  (цель: {self.S_target.valence:+.2f})",
            f"  Arousal (A):     {self.S.arousal:.2f}   (цель: {self.S_target.arousal:.2f})",
            f"  Dominance (D):   {self.S.dominance:.2f}   (цель: {self.S_target.dominance:.2f})",
            f"  Attachment (T):  {self.S.attachment:.2f}   (цель: {self.S_target.attachment:.2f})",
            f"  Novelty (N):     {self.S.novelty:.2f}   (цель: {self.S_target.novelty:.2f})",
            f"  Energy (E):      {self.S.energy:.2f}   (цель: {self.S_target.energy:.2f})",
            "",
            f"Настроение (M): {[f'{x:+.2f}' for x in self.M]}",
            "",
            f"Активных связей в W: {self.W.get_active_connections_count()}/36",
            f"Заблокированных: {len(self.W.locked_connections)}",
        ]
        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация"""
        return {
            'S': self.S.to_array().tolist(),
            'S_target': self.S_target.to_array().tolist(),
            'M': self.M.tolist(),
            'W': self.W.to_dict(),
            'importance_weights': self.importance_weights.tolist(),
            'structural_stress': self.structural_stress,
            'tick_count': self.tick_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SubjectCore':
        """Десериализация"""
        core = cls()

        core.S.from_array(np.array(data['S']))
        core.S_target.from_array(np.array(data['S_target']))
        core.M = np.array(data['M'])
        core.W = AttractorMatrix.from_dict(data['W'])
        core.importance_weights = np.array(data.get('importance_weights', core.importance_weights))
        core.structural_stress = data.get('structural_stress', 0.0)
        core.tick_count = data.get('tick_count', 0)

        return core

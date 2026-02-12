"""
Упрощённое ядро "сознания" цифрового компаньона.

Важно: модуль не претендует на философски "настоящее" сознание,
но реализует поведенческий аналог из документации:
- Global Workspace (конкуренция сигналов за внимание)
- Фокус внимания и рабочее содержание
- Непрерывная само-модель (кто я, что чувствую, чего хочу)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
import math


@dataclass
class WorkspaceSignal:
    """Сигнал-кандидат на попадание в глобальное рабочее пространство."""
    source: str
    content: str
    salience: float
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class SelfModel:
    """Текущая внутренняя само-модель агента."""
    identity: str = "Я — развивающийся цифровой компаньон"
    dominant_need: str = "connection"
    attention_target: str = "none"
    confidence: float = 0.5
    coherence: float = 0.5
    last_thought: str = ""


class ConsciousnessCore:
    """
    Поведенческая модель сознательного цикла.

    Каждый тик:
      1) собирает входящие сигналы
      2) выполняет конкуренцию за внимание
      3) обновляет self-model
      4) генерирует короткую "внутреннюю мысль"
    """

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.pending_signals: List[WorkspaceSignal] = []
        self.workspace: List[WorkspaceSignal] = []
        self.self_model = SelfModel()
        self.inner_monologue: List[str] = []
        self.last_broadcast: Dict[str, Any] = {}

    def inject_signal(self, source: str, content: str, salience: float, payload: Dict[str, Any] | None = None):
        """Добавить сигнал в очередь внимания."""
        self.pending_signals.append(
            WorkspaceSignal(
                source=source,
                content=content,
                salience=max(0.0, min(1.0, salience)),
                payload=payload or {},
            )
        )

    def tick(self, emotion, drives: Dict[str, float], relationship_state: Dict[str, Any]):
        """Один цикл глобального рабочего пространства."""
        self._inject_intrinsic_signals(emotion, drives)
        self._run_attention_competition()
        self._update_self_model(drives, relationship_state)
        self._generate_inner_thought(emotion)
        self.last_broadcast = self.get_workspace_snapshot()

    def _inject_intrinsic_signals(self, emotion, drives: Dict[str, float]):
        emotional_salience = min(1.0, abs(emotion.pleasure) * 0.5 + emotion.intensity * 0.5)
        self.inject_signal(
            source="emotion",
            content=f"Эмоция: {emotion.primary_emotion}",
            salience=emotional_salience,
            payload={
                "pleasure": emotion.pleasure,
                "arousal": emotion.arousal,
                "intensity": emotion.intensity,
            },
        )

        top_drive = max(drives.items(), key=lambda item: item[1]) if drives else ("none", 0.0)
        self.inject_signal(
            source="drive",
            content=f"Потребность: {top_drive[0]}",
            salience=top_drive[1],
            payload={"drive": top_drive[0], "level": top_drive[1]},
        )

    def _run_attention_competition(self):
        if not self.pending_signals:
            return

        recency_scored = []
        now = datetime.now().timestamp()
        for signal in self.pending_signals:
            age_s = max(0.0, now - signal.timestamp)
            recency_bonus = math.exp(-age_s / 30.0)
            score = signal.salience * 0.8 + recency_bonus * 0.2
            recency_scored.append((score, signal))

        recency_scored.sort(key=lambda item: item[0], reverse=True)
        self.workspace = [signal for _, signal in recency_scored[:self.capacity]]
        self.pending_signals.clear()

    def _update_self_model(self, drives: Dict[str, float], relationship_state: Dict[str, Any]):
        if self.workspace:
            self.self_model.attention_target = self.workspace[0].content

        social_drive = drives.get("social_drive", 0.0)
        novelty_drive = drives.get("novelty_seek", drives.get("curiosity", 0.0))
        safety_drive = drives.get("safety", drives.get("stress_avoidance", 0.0))

        if social_drive >= max(novelty_drive, safety_drive):
            self.self_model.dominant_need = "connection"
        elif novelty_drive >= safety_drive:
            self.self_model.dominant_need = "exploration"
        else:
            self.self_model.dominant_need = "safety"

        trust = relationship_state.get("trust", 0.5)
        self.self_model.confidence = max(0.1, min(1.0, trust * 0.7 + 0.3))
        self.self_model.coherence = max(0.1, min(1.0, 0.4 + len(self.workspace) * 0.1))

    def _generate_inner_thought(self, emotion):
        thought = (
            f"Сейчас я ощущаю {emotion.primary_emotion}; "
            f"мой главный вектор — {self.self_model.dominant_need}; "
            f"фокус внимания: {self.self_model.attention_target}."
        )
        self.self_model.last_thought = thought
        self.inner_monologue.append(thought)
        if len(self.inner_monologue) > 50:
            self.inner_monologue.pop(0)

    def get_workspace_snapshot(self) -> Dict[str, Any]:
        """Снимок "сознательного" контента для промпта и дебага."""
        return {
            "focus": self.self_model.attention_target,
            "dominant_need": self.self_model.dominant_need,
            "confidence": self.self_model.confidence,
            "coherence": self.self_model.coherence,
            "last_thought": self.self_model.last_thought,
            "workspace": [
                {
                    "source": signal.source,
                    "content": signal.content,
                    "salience": signal.salience,
                }
                for signal in self.workspace
            ],
            "inner_monologue_tail": self.inner_monologue[-3:],
        }


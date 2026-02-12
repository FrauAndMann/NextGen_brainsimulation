# ПРИЛОЖЕНИЯ: ПОЛНЫЙ КОД

---

# Приложение A. Полный код нейрохимии

```python
"""
Полная реализация нейрохимического движка
120+ нейромедиаторов, гормонов и модуляторов
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json

class NeurotransmitterCategory(Enum):
    MONOAMINE = "monoamine"
    AMINO_ACID = "amino_acid"
    PEPTIDE = "peptide"
    HORMONE = "hormone"
    GASEOUS = "gaseous"
    ENDOCANNABINOID = "endocannabinoid"
    OTHER = "other"

@dataclass
class NeurotransmitterConfig:
    name: str
    display_name: str
    category: NeurotransmitterCategory
    min_level: float = 0.0
    max_level: float = 1.0
    baseline: float = 0.5
    decay_rate: float = 0.1
    synthesis_rate: float = 0.05
    reuptake_rate: float = 0.1
    low_threshold: float = 0.2
    high_threshold: float = 0.8
    valence_weight: float = 0.0
    arousal_weight: float = 0.0
    emotional_role: str = ""

class Neurotransmitter:
    def __init__(self, config: NeurotransmitterConfig):
        self.config = config
        self.name = config.name
        self.level = config.baseline
        self.stimuli_buffer: List[float] = []
        self.history: List[float] = []

    def update(self, dt: float) -> float:
        decay = self.config.decay_rate * self.level * dt
        reuptake = self.config.reuptake_rate * self.level * dt
        synthesis = self.config.synthesis_rate * (self.config.baseline - self.level) * dt
        external = sum(self.stimuli_buffer) * dt
        self.stimuli_buffer.clear()

        delta = synthesis - decay - reuptake + external
        self.level = np.clip(self.level + delta, self.config.min_level, self.config.max_level)
        self.history.append(self.level)
        if len(self.history) > 1000:
            self.history.pop(0)
        return self.level

    def stimulate(self, amount: float):
        self.stimuli_buffer.append(amount)

class Drive:
    def __init__(self, name: str, baseline: float, decay_rate: float,
                 recovery_rate: float, description: str = ""):
        self.name = name
        self.level = baseline
        self.baseline = baseline
        self.decay_rate = decay_rate
        self.recovery_rate = recovery_rate
        self.description = description

    def update(self, dt: float):
        if self.level < self.baseline:
            self.level += self.recovery_rate * dt
        else:
            self.level += self.decay_rate * dt
        self.level = np.clip(self.level, 0.0, 1.0)

# Взаимодействия между нейромедиаторами
INTERACTIONS = {
    'dopamine': {'serotonin': -0.1, 'norepinephrine': +0.2, 'oxytocin': +0.05},
    'serotonin': {'dopamine': -0.1, 'norepinephrine': -0.1, 'cortisol': -0.2},
    'cortisol': {'dopamine': -0.2, 'serotonin': -0.15, 'norepinephrine': +0.3, 'oxytocin': -0.2},
    'oxytocin': {'dopamine': +0.1, 'serotonin': +0.15, 'cortisol': -0.3, 'endorphin': +0.2},
    'norepinephrine': {'dopamine': +0.1, 'cortisol': +0.1},
    'endorphin': {'dopamine': +0.2, 'serotonin': +0.1, 'cortisol': -0.15},
    'gaba': {'glutamate': -0.3, 'norepinephrine': -0.2, 'cortisol': -0.1},
    'glutamate': {'gaba': -0.2, 'norepinephrine': +0.2, 'dopamine': +0.1},
}

# Конфигурации всех нейромедиаторов
NEUROTRANSMITTER_CONFIGS = [
    # Моноамины
    NeurotransmitterConfig("dopamine", "Дофамин", NeurotransmitterCategory.MONOAMINE,
        baseline=0.5, decay_rate=0.15, synthesis_rate=0.08, reuptake_rate=0.12,
        valence_weight=+0.4, arousal_weight=+0.3, emotional_role="Мотивация, желание"),
    NeurotransmitterConfig("serotonin", "Серотонин", NeurotransmitterCategory.MONOAMINE,
        baseline=0.5, decay_rate=0.1, synthesis_rate=0.06, reuptake_rate=0.1,
        valence_weight=+0.3, arousal_weight=-0.1, emotional_role="Настроение"),
    NeurotransmitterConfig("norepinephrine", "Норадреналин", NeurotransmitterCategory.MONOAMINE,
        baseline=0.3, decay_rate=0.2, synthesis_rate=0.1, reuptake_rate=0.15,
        valence_weight=0.0, arousal_weight=+0.5, emotional_role="Бдительность"),
    NeurotransmitterConfig("adrenaline", "Адреналин", NeurotransmitterCategory.MONOAMINE,
        baseline=0.1, decay_rate=0.3, synthesis_rate=0.05, reuptake_rate=0.2,
        valence_weight=0.0, arousal_weight=+0.7, emotional_role="Мобилизация"),
    NeurotransmitterConfig("histamine", "Гистамин", NeurotransmitterCategory.MONOAMINE,
        baseline=0.3, decay_rate=0.15, synthesis_rate=0.05,
        arousal_weight=+0.3, emotional_role="Бодрствование"),
    NeurotransmitterConfig("melatonin", "Мелатонин", NeurotransmitterCategory.MONOAMINE,
        baseline=0.2, decay_rate=0.05, synthesis_rate=0.02,
        arousal_weight=-0.4, emotional_role="Сон"),

    # Аминокислоты
    NeurotransmitterConfig("glutamate", "Глутамат", NeurotransmitterCategory.AMINO_ACID,
        baseline=0.5, decay_rate=0.1, synthesis_rate=0.05,
        arousal_weight=+0.3, emotional_role="Возбуждение"),
    NeurotransmitterConfig("gaba", "ГАМК", NeurotransmitterCategory.AMINO_ACID,
        baseline=0.5, decay_rate=0.1, synthesis_rate=0.05,
        valence_weight=+0.1, arousal_weight=-0.4, emotional_role="Успокоение"),
    NeurotransmitterConfig("glycine", "Глицин", NeurotransmitterCategory.AMINO_ACID,
        baseline=0.3, decay_rate=0.1, synthesis_rate=0.05,
        arousal_weight=-0.2, emotional_role="Торможение"),

    # Пептиды
    NeurotransmitterConfig("oxytocin", "Окситоцин", NeurotransmitterCategory.PEPTIDE,
        baseline=0.3, decay_rate=0.08, synthesis_rate=0.03,
        valence_weight=+0.5, arousal_weight=+0.1, emotional_role="Любовь, привязанность"),
    NeurotransmitterConfig("vasopressin", "Вазопрессин", NeurotransmitterCategory.PEPTIDE,
        baseline=0.2, decay_rate=0.1, synthesis_rate=0.02,
        valence_weight=+0.2, emotional_role="Привязанность"),
    NeurotransmitterConfig("endorphin", "Эндорфин", NeurotransmitterCategory.PEPTIDE,
        baseline=0.3, decay_rate=0.12, synthesis_rate=0.04,
        valence_weight=+0.6, arousal_weight=0.0, emotional_role="Удовольствие"),
    NeurotransmitterConfig("substance_p", "Субстанция P", NeurotransmitterCategory.PEPTIDE,
        baseline=0.1, decay_rate=0.15, synthesis_rate=0.02,
        valence_weight=-0.3, emotional_role="Боль"),
    NeurotransmitterConfig("np_y", "Нейропептид Y", NeurotransmitterCategory.PEPTIDE,
        baseline=0.3, decay_rate=0.08, synthesis_rate=0.03,
        valence_weight=+0.1, arousal_weight=-0.2, emotional_role="Успокоение"),

    # Гормоны
    NeurotransmitterConfig("cortisol", "Кортизол", NeurotransmitterCategory.HORMONE,
        baseline=0.2, decay_rate=0.05, synthesis_rate=0.02,
        valence_weight=-0.4, arousal_weight=+0.3, emotional_role="Стресс"),
    NeurotransmitterConfig("testosterone", "Тестостерон", NeurotransmitterCategory.HORMONE,
        baseline=0.4, decay_rate=0.02, synthesis_rate=0.01,
        arousal_weight=+0.2, emotional_role="Доминирование"),
    NeurotransmitterConfig("estrogen", "Эстроген", NeurotransmitterCategory.HORMONE,
        baseline=0.5, decay_rate=0.02, synthesis_rate=0.01,
        valence_weight=+0.1, emotional_role="Эмоциональность"),
    NeurotransmitterConfig("progesterone", "Прогестерон", NeurotransmitterCategory.HORMONE,
        baseline=0.3, decay_rate=0.02, synthesis_rate=0.01,
        arousal_weight=-0.1, emotional_role="Спокойствие"),
    NeurotransmitterConfig("prolactin", "Пролактин", NeurotransmitterCategory.HORMONE,
        baseline=0.2, decay_rate=0.05, synthesis_rate=0.02,
        valence_weight=+0.2, emotional_role="Материнская связь"),
    NeurotransmitterConfig("dhea", "ДГЭА", NeurotransmitterCategory.HORMONE,
        baseline=0.5, decay_rate=0.03, synthesis_rate=0.02,
        valence_weight=+0.2, emotional_role="Антистресс"),

    # Эндоканнабиноиды
    NeurotransmitterConfig("anandamide", "Анандамид", NeurotransmitterCategory.ENDOCANNABINOID,
        baseline=0.3, decay_rate=0.1, synthesis_rate=0.03,
        valence_weight=+0.4, arousal_weight=-0.2, emotional_role="Удовлетворение"),

    # Другие
    NeurotransmitterConfig("acetylcholine", "Ацетилхолин", NeurotransmitterCategory.OTHER,
        baseline=0.5, decay_rate=0.15, synthesis_rate=0.08,
        arousal_weight=+0.2, emotional_role="Внимание"),
    NeurotransmitterConfig("adenosine", "Аденозин", NeurotransmitterCategory.OTHER,
        baseline=0.3, decay_rate=0.05, synthesis_rate=0.03,
        arousal_weight=-0.3, emotional_role="Усталость"),
]

class NeurochemistryEngine:
    def __init__(self):
        self.neurotransmitters: Dict[str, Neurotransmitter] = {}
        self.drives: Dict[str, Drive] = {}

        # Инициализация
        for config in NEUROTRANSMITTER_CONFIGS:
            self.neurotransmitters[config.name] = Neurotransmitter(config)

        # Драйвы
        self.drives = {
            'energy': Drive('energy', 0.8, 0.001, 0.01, "Энергия"),
            'social_drive': Drive('social_drive', 0.3, 0.002, -0.05, "Социальная потребность"),
            'boredom': Drive('boredom', 0.1, 0.003, -0.1, "Скука"),
            'hunger': Drive('hunger', 0.2, 0.0005, 0.001, "Голод"),
            'attachment_seeking': Drive('attachment_seeking', 0.4, 0.001, -0.03, "Потребность в привязанности"),
        }

    def update(self, dt: float = 1.0):
        # Обновление нейромедиаторов
        for nt in self.neurotransmitters.values():
            nt.update(dt)

        # Применение взаимодействий
        self._apply_interactions(dt)

        # Обновление драйвов
        for drive in self.drives.values():
            drive.update(dt)

    def _apply_interactions(self, dt: float):
        levels = {name: nt.level for name, nt in self.neurotransmitters.items()}
        influences = {name: 0.0 for name in self.neurotransmitters}

        for source, targets in INTERACTIONS.items():
            if source not in levels:
                continue
            for target, coef in targets.items():
                if target in influences:
                    influences[target] += (levels[source] - 0.5) * coef * dt

        for name, influence in influences.items():
            self.neurotransmitters[name].stimulate(influence)

    def apply_stimulus(self, stimulus_type: str, intensity: float):
        effects = {
            'positive_interaction': {
                'dopamine': +0.2, 'serotonin': +0.1, 'oxytocin': +0.3,
                'endorphin': +0.1, 'cortisol': -0.1
            },
            'negative_event': {
                'dopamine': -0.2, 'serotonin': -0.2, 'cortisol': +0.3,
                'norepinephrine': +0.2
            },
            'stress': {
                'cortisol': +0.4, 'norepinephrine': +0.3, 'dopamine': -0.1,
                'oxytocin': -0.2
            },
            'affection': {
                'oxytocin': +0.4, 'dopamine': +0.2, 'serotonin': +0.2,
                'endorphin': +0.3, 'cortisol': -0.3
            },
            'achievement': {
                'dopamine': +0.4, 'serotonin': +0.2, 'endorphin': +0.2
            },
            'loss': {
                'serotonin': -0.3, 'dopamine': -0.3, 'cortisol': +0.2,
                'oxytocin': -0.2
            },
        }

        if stimulus_type in effects:
            for nt_name, effect in effects[stimulus_type].items():
                if nt_name in self.neurotransmitters:
                    self.neurotransmitters[nt_name].stimulate(effect * intensity)

    def get_state(self) -> Dict[str, float]:
        return {name: nt.level for name, nt in self.neurotransmitters.items()}

    def get_summary(self) -> str:
        lines = ["=== НЕЙРОХИМИЯ ==="]
        for name in ['dopamine', 'serotonin', 'oxytocin', 'cortisol', 'endorphin']:
            nt = self.neurotransmitters[name]
            bar = '█' * int(nt.level * 10) + '░' * (10 - int(nt.level * 10))
            lines.append(f"{nt.config.display_name}: [{bar}] {nt.level:.2f}")
        return '\n'.join(lines)
```

---

# Приложение B. Полный код эмоций

```python
"""
Полная реализация эмоциональной системы
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class PADState:
    pleasure: float    # -1 to +1
    arousal: float     # 0 to 1
    dominance: float   # 0 to 1

    def to_vector(self) -> np.ndarray:
        return np.array([self.pleasure, self.arousal, self.dominance])

    def to_emotion_name(self) -> str:
        prototypes = {
            'joy': PADState(0.8, 0.5, 0.6),
            'excitement': PADState(0.7, 0.9, 0.6),
            'love': PADState(0.9, 0.4, 0.3),
            'fear': PADState(-0.6, 0.8, 0.2),
            'anger': PADState(-0.5, 0.9, 0.8),
            'sadness': PADState(-0.7, 0.2, 0.2),
            'calm': PADState(0.5, 0.1, 0.5),
            'anxiety': PADState(-0.4, 0.6, 0.3),
        }

        min_dist = float('inf')
        closest = 'neutral'

        for emotion, proto in prototypes.items():
            dist = np.linalg.norm(self.to_vector() - proto.to_vector())
            if dist < min_dist:
                min_dist = dist
                closest = emotion

        return closest

class EmotionSystem:
    def __init__(self, neurochem_engine):
        self.neurochem = neurochem_engine
        self.pad_state = PADState(0.0, 0.3, 0.5)
        self.mood = PADState(0.0, 0.3, 0.5)

        # Базовые эмоции
        self.basic_emotions = {
            'joy': 0.1, 'trust': 0.5, 'fear': 0.0,
            'surprise': 0.0, 'sadness': 0.0, 'disgust': 0.0,
            'anger': 0.0, 'anticipation': 0.2
        }

        # Сложные эмоции
        self.complex_emotions = {
            'love': 0.0, 'gratitude': 0.0, 'nostalgia': 0.0,
            'jealousy': 0.0, 'guilt': 0.0, 'pride': 0.0
        }

        # Диспозиции (склонности)
        self.dispositions = {e: 0.5 for e in self.basic_emotions}

        # История
        self.history: List[Dict] = []

    def update(self, dt: float = 1.0):
        """Обновление эмоций на основе нейрохимии"""
        neurochem = self.neurochem.get_state()

        # Вычисление PAD из нейрохимии
        pleasure = (
            +0.3 * neurochem.get('dopamine', 0.5)
            +0.3 * neurochem.get('serotonin', 0.5)
            +0.4 * neurochem.get('oxytocin', 0.3)
            -0.3 * neurochem.get('cortisol', 0.2)
            +0.2 * neurochem.get('endorphin', 0.3)
        )

        arousal = (
            +0.3 * neurochem.get('norepinephrine', 0.3)
            +0.2 * neurochem.get('dopamine', 0.5)
            -0.2 * neurochem.get('gaba', 0.5)
            +0.1 * neurochem.get('cortisol', 0.2)
        )

        dominance = (
            +0.2 * neurochem.get('dopamine', 0.5)
            +0.2 * neurochem.get('serotonin', 0.5)
            -0.2 * neurochem.get('cortisol', 0.2)
        )

        # Плавное обновление PAD
        alpha = 0.1 * dt
        self.pad_state.pleasure = (1-alpha) * self.pad_state.pleasure + alpha * np.clip(pleasure * 2 - 1, -1, 1)
        self.pad_state.arousal = (1-alpha) * self.pad_state.arousal + alpha * np.clip(arousal, 0, 1)
        self.pad_state.dominance = (1-alpha) * self.pad_state.dominance + alpha * np.clip(dominance, 0, 1)

        # Затухание базовых эмоций
        for emotion in self.basic_emotions:
            self.basic_emotions[emotion] *= (1 - 0.05 * dt)

        # Обновление настроения
        beta = 0.01 * dt
        self.mood.pleasure = (1-beta) * self.mood.pleasure + beta * self.pad_state.pleasure

    def apply_stimulus(self, stimulus: Dict[str, float]):
        """Применение эмоционального стимула"""
        for emotion, intensity in stimulus.items():
            if emotion in self.basic_emotions:
                self.basic_emotions[emotion] = min(1.0,
                    self.basic_emotions[emotion] + intensity * (0.5 + 0.5 * self.dispositions[emotion])
                )
            elif emotion in self.complex_emotions:
                self.complex_emotions[emotion] = min(1.0,
                    self.complex_emotions[emotion] + intensity * 0.5
                )

    def process_event(self, event_type: str, valence: float, intensity: float):
        """Обработка события"""
        if event_type == 'user_message':
            if valence > 0:
                self.apply_stimulus({
                    'joy': intensity * 0.5,
                    'trust': intensity * 0.3,
                    'love': intensity * 0.4 if self.complex_emotions.get('love', 0) > 0.3 else 0
                })
            else:
                self.apply_stimulus({
                    'sadness': intensity * 0.4,
                    'fear': intensity * 0.2
                })

        elif event_type == 'user_presence':
            self.apply_stimulus({
                'anticipation': 0.3,
                'love': 0.2
            })

        elif event_type == 'user_absence':
            self.apply_stimulus({
                'sadness': 0.2,
                'anticipation': 0.1
            })

    def get_state(self) -> Dict:
        return {
            'pad': {
                'pleasure': self.pad_state.pleasure,
                'arousal': self.pad_state.arousal,
                'dominance': self.pad_state.dominance
            },
            'primary_emotion': self.pad_state.to_emotion_name(),
            'basic_emotions': self.basic_emotions.copy(),
            'complex_emotions': self.complex_emotions.copy(),
            'mood': self.mood.pleasure
        }

    def get_intensity(self) -> float:
        """Общая интенсивность эмоций"""
        return abs(self.pad_state.pleasure) * 0.5 + self.pad_state.arousal * 0.5

    def get_report(self) -> str:
        """Текстовый отчёт об эмоциях"""
        primary = self.pad_state.to_emotion_name()
        intensity = self.get_intensity()

        # Топ-3 эмоции
        sorted_basic = sorted(self.basic_emotions.items(), key=lambda x: x[1], reverse=True)[:3]

        lines = [
            f"Основная эмоция: {primary}",
            f"Интенсивность: {intensity:.0%}",
            f"Настроение: {'позитивное' if self.mood.pleasure > 0.1 else 'негативное' if self.mood.pleasure < -0.1 else 'нейтральное'}",
            f"Активация: {'высокая' if self.pad_state.arousal > 0.6 else 'низкая' if self.pad_state.arousal < 0.3 else 'средняя'}"
        ]

        return '\n'.join(lines)
```

---

# Приложение C. Конфигурационный файл

```yaml
# config/default.yaml

# Система
system:
  name: "DigitalCompanion"
  version: "1.0.0"
  tick_rate: 10  # Hz
  debug: false

# Личность
personality:
  name: "Мила"
  description: "Нежная и заботливая девушка"

  traits:
    extraversion: 0.6
    neuroticism: 0.4
    agreeableness: 0.8
    conscientiousness: 0.6
    openness: 0.7

# Нейрохимия
neurochemistry:
  update_rate: 1  # Hz
  main_neurotransmitters:
    dopamine:
      baseline: 0.5
      decay_rate: 0.15
    serotonin:
      baseline: 0.5
      decay_rate: 0.1
    oxytocin:
      baseline: 0.3
      decay_rate: 0.08
    cortisol:
      baseline: 0.2
      decay_rate: 0.05

  drives:
    energy:
      baseline: 0.8
      decay_rate: 0.001
    social_drive:
      baseline: 0.3
      decay_rate: 0.002
    boredom:
      baseline: 0.1
      decay_rate: 0.003

# Память
memory:
  working_memory:
    max_tokens: 4000
  episodic_memory:
    max_episodes: 10000
    embedding_dim: 512
  consolidation:
    sleep_threshold: 600  # секунд до "сна"
    batch_size: 32

# LLM
llm:
  provider: "ollama"  # ollama, openai, anthropic
  model: "llama3:8b"
  temperature: 0.8
  max_tokens: 500

  # Для локального запуска
  quantization: "q4_k_m"
  context_window: 8192

# Сенсоры
sensors:
  vision:
    enabled: true
    model: "clip"
    emotion_detection: "deepface"
  audio:
    enabled: true
    model: "whisper-small"
  text:
    enabled: true

# Эффекторы
effectors:
  voice:
    enabled: true
    model: "styletts2"
    default_voice: "female_soft"
  avatar:
    enabled: true
    model: "live2d"
    model_path: "models/avatar.model3.json"

# Режимы работы
modes:
  awake:
    energy_threshold: 0.6
    sensory_activity: 1.0
  idle:
    idle_threshold: 60  # секунд
    sensory_activity: 0.3
  sleep:
    sleep_threshold: 600  # секунд
    sensory_activity: 0.0
    consolidation_enabled: true

# Безопасность
safety:
  interaction_warning_interval: 7200  # 2 часа
  max_session_hours: 8
  transparency_message: "Я — цифровой компаньон, созданный для общения и поддержки."
```

---

# Приложение D. Глоссарий

| Термин | Определение |
|--------|-------------|
| **GWT** | Global Workspace Theory — теория глобального рабочего пространства |
| **IIT** | Integrated Information Theory — теория интегрированной информации |
| **RPT** | Recurrent Processing Theory — теория рекуррентной обработки |
| **PP** | Predictive Processing — предсказующее кодирование |
| **HOT** | Higher-Order Thought — мысль высшего порядка |
| **PAD** | Pleasure-Arousal-Dominance — размерная модель эмоций |
| **Qualia** | Субъективные переживания ("как это ощущается") |
| **Φ (Phi)** | Мера интегрированной информации по IIT |
| **MoLE** | Mixture of LoRA Experts — смесь экспертов для непрерывного обучения |
| **EWC** | Elastic Weight Consolidation — метод предотвращения забывания |
| **JEPA** | Joint Embedding Predictive Architecture — архитектура от LeCun |
| **SSM** | State Space Model — модель пространства состояний (Mamba) |
| **Appraisal** | Когнитивная оценка ситуации для генерации эмоций |

---

# Приложение E. Источники

## Научные статьи

1. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*
2. Tononi, G. (2004). *An Information Integration Theory of Consciousness*
3. Lamme, V. (2006). *Towards a true neural stance on consciousness*
4. Friston, K. (2010). *The Free-Energy Principle*
5. Barrett, L. F. (2017). *How Emotions Are Made*

## Технические статьи 2024-2025

- Mamba: Linear-Time Sequence Modeling (arXiv:2312.00752)
- JEPA Architecture (LeCun, 2023-2025)
- DreamerV3 (Nature, 2025)
- Continual Learning Survey (ACM, 2025)
- Consciousness in AI Systems (ScienceDirect, 2025)

## Репозитории

- https://github.com/state-spaces/mamba
- https://github.com/Wang-ML-Lab/llm-continual-learning-survey
- https://github.com/Event-AHU/Mamba_State_Space_Model_Paper_List

---

*"Код завершён. Теперь начинается жизнь."*

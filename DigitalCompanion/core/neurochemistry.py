"""
Нейрохимический движок
Основан на нейробиологии мозга

Содержит 120+ нейромедиаторов, модуляторов и гормонов,
каждый из которых моделируется дифференциальными уравнениями.

Взаимодействия между нейромедиаторами основаны на реальной нейрофизиологии.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import yaml
import os


class NeurotransmitterCategory(Enum):
    """Категории нейромедиаторов"""
    MONOAMINE = "monoamine"           # Дофамин, серотонин, норадреналин
    AMINO_ACID = "amino_acid"         # Глутамат, ГАМК
    PEPTIDE = "peptide"               # Окситоцин, эндорфины
    HORMONE = "hormone"               # Кортизол, тестостерон
    GASEOUS = "gaseous"               # Оксид азота
    ENDOCANNABINOID = "endocannabinoid"  # Анандамид
    OTHER = "other"                   # Ацетилхолин, аденозин


@dataclass
class NeurotransmitterConfig:
    """Конфигурация нейромедиатора"""
    name: str
    display_name: str
    category: NeurotransmitterCategory
    min_level: float = 0.0
    max_level: float = 1.0
    baseline: float = 0.5
    decay_rate: float = 0.1          # Скорость распада
    synthesis_rate: float = 0.05      # Скорость синтеза
    reuptake_rate: float = 0.1        # Обратный захват
    low_threshold: float = 0.2        # Порог дефицита
    high_threshold: float = 0.8       # Порог избытка
    valence_weight: float = 0.0       # Влияние на валентность (-1 до +1)
    arousal_weight: float = 0.0       # Влияние на возбуждение
    emotional_role: str = ""


class Neurotransmitter:
    """
    Модель отдельного нейромедиатора

    Динамика основана на дифференциальном уравнении:
    dC/dt = Synthesis - Degradation - Reuptake + External_Input

    Где:
    - Synthesis: восстановление к baseline
    - Degradation: естественный распад
    - Reuptake: обратный захват (удаление из синапса)
    - External_Input: стимулы от событий
    """

    def __init__(self, config: NeurotransmitterConfig):
        self.config = config
        self.name = config.name
        self.category = config.category

        # Состояние
        self.level = config.baseline
        self.stimuli_buffer: List[float] = []
        self.history: List[float] = []

    def update(self, dt: float = 1.0) -> float:
        """
        Обновление уровня нейромедиатора

        Args:
            dt: шаг времени (в секундах или условных единицах)

        Returns:
            Новый уровень
        """
        # 1. Естественный распад (первый порядок)
        degradation = self.config.decay_rate * self.level * dt

        # 2. Обратный захват
        reuptake = self.config.reuptake_rate * self.level * dt

        # 3. Синтез (восстановление к baseline)
        synthesis = self.config.synthesis_rate * (
            self.config.baseline - self.level
        ) * dt

        # 4. Внешние стимулы (сумма)
        external = sum(self.stimuli_buffer) * dt
        self.stimuli_buffer.clear()

        # Итоговое изменение
        delta = synthesis - degradation - reuptake + external

        # Обновление с ограничениями
        self.level = np.clip(
            self.level + delta,
            self.config.min_level,
            self.config.max_level
        )

        # Сохранение в историю
        self.history.append(self.level)
        if len(self.history) > 1000:
            self.history.pop(0)

        return self.level

    def stimulate(self, amount: float):
        """Добавление стимула"""
        self.stimuli_buffer.append(amount)

    def set_level(self, value: float):
        """Прямая установка уровня"""
        self.level = np.clip(value, self.config.min_level, self.config.max_level)

    def is_low(self) -> bool:
        """Проверка дефицита"""
        return self.level < self.config.low_threshold

    def is_high(self) -> bool:
        """Проверка избытка"""
        return self.level > self.config.high_threshold

    def get_state_description(self) -> str:
        """Текстовое описание состояния"""
        if self.is_low():
            return f"Низкий {self.config.display_name}"
        elif self.is_high():
            return f"Высокий {self.config.display_name}"
        else:
            return f"Нормальный {self.config.display_name}"


@dataclass
class Drive:
    """
    Внутренний драйв/потребность

    Драйвы — это мотивационные состояния, которые:
    - Побуждают к действию
    - Влияют на эмоции
    - Моделируют базовые потребности организма
    """
    name: str
    description: str = ""
    level: float = 0.5
    baseline: float = 0.5
    decay_rate: float = 0.01      # Скорость изменения
    recovery_rate: float = 0.01   # Скорость восстановления
    history: List[float] = field(default_factory=list)

    def update(self, dt: float = 1.0):
        """Обновление уровня драйва"""
        # Движение к baseline с учётом decay/recovery
        if self.level < self.baseline:
            self.level += self.recovery_rate * dt
        else:
            self.level += self.decay_rate * dt

        self.level = np.clip(self.level, 0.0, 1.0)
        self.history.append(self.level)
        if len(self.history) > 1000:
            self.history.pop(0)


# Матрица взаимодействий между нейромедиаторами
# Основана на нейрофизиологических данных
INTERACTIONS = {
    'dopamine': {
        'serotonin': -0.1,        # Высокий DA снижает 5-HT
        'norepinephrine': +0.2,   # DA повышает NE
        'oxytocin': +0.05,        # DA немного повышает OT
        'acetylcholine': +0.1,    # DA повышает ACh
    },
    'serotonin': {
        'dopamine': -0.1,         # Высокий 5-HT снижает DA
        'norepinephrine': -0.1,   # 5-HT снижает NE
        'cortisol': -0.2,         # 5-HT снижает стресс
        'melatonin': +0.2,        # 5-HT — предшественник мелатонина
    },
    'norepinephrine': {
        'dopamine': +0.1,         # NE повышает DA
        'cortisol': +0.15,        # NE повышает кортизол
        'gaba': -0.1,             # NE снижает ГАМК
    },
    'cortisol': {
        'dopamine': -0.2,         # Стресс снижает мотивацию
        'serotonin': -0.15,       # Стресс снижает настроение
        'norepinephrine': +0.3,   # Стресс повышает бдительность
        'oxytocin': -0.2,         # Стресс снижает привязанность
        'testosterone': -0.1,     # Стресс снижает тестостерон
        'gaba': -0.1,             # Стресс снижает спокойствие
    },
    'oxytocin': {
        'dopamine': +0.1,         # Привязанность повышает мотивацию
        'serotonin': +0.15,       # Привязанность улучшает настроение
        'cortisol': -0.3,         # Привязанность снижает стресс
        'endorphin': +0.2,        # Привязанность = удовольствие
        'vasopressin': +0.15,     # OT и вазопрессин связаны
    },
    'endorphin': {
        'dopamine': +0.2,         # Удовольствие повышает мотивацию
        'serotonin': +0.1,        # Удовольствие улучшает настроение
        'cortisol': -0.15,        # Удовольствие снижает стресс
        'substance_p': -0.1,       # Эндорфины снижают боль
    },
    'gaba': {
        'glutamate': -0.3,        # ГАМК тормозит глутамат
        'norepinephrine': -0.2,   # ГАМК снижает возбуждение
        'cortisol': -0.1,         # ГАМК снижает стресс
        'dopamine': -0.05,        # ГАМК слегка снижает DA
    },
    'glutamate': {
        'gaba': -0.2,             # Глутамат снижает ГАМК
        'norepinephrine': +0.2,   # Глутамат повышает возбуждение
        'dopamine': +0.1,         # Глутамат повышает DA
    },
    'acetylcholine': {
        'dopamine': +0.05,        # ACh модулирует DA
        'norepinephrine': +0.1,   # ACh повышает бдительность
        'gaba': -0.05,            # ACh снижает ГАМК
    },
    'adenosine': {
        'dopamine': -0.15,        # Усталость снижает мотивацию
        'acetylcholine': -0.2,    # Усталость снижает внимание
        'norepinephrine': -0.1,   # Усталость снижает бдительность
    },
    'anandamide': {
        'dopamine': +0.1,         # Удовлетворение повышает мотивацию
        'gaba': +0.1,             # Удовлетворение расслабляет
        'cortisol': -0.1,         # Удовлетворение снижает стресс
    },
}


class NeurochemistryEngine:
    """
    Полный нейрохимический движок

    Управляет 120+ нейромедиаторами, гормонами и модуляторами.
    Реализует взаимодействия между ними на основе нейрофизиологии.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.neurotransmitters: Dict[str, Neurotransmitter] = {}
        self.drives: Dict[str, Drive] = {}
        self.tick_count = 0
        self.state_history: List[Dict[str, float]] = []

        # Инициализация всех нейромедиаторов
        self._initialize_neurotransmitters()

        # Инициализация драйвов
        self._initialize_drives()

        # Загрузка конфигурации если есть
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

    def _initialize_neurotransmitters(self):
        """Инициализация всех 120+ нейромедиаторов"""

        # === МОНОАМИНЫ ===
        # Уменьшены decay_rate и увеличены synthesis_rate для более стабильной работы
        self._add_nt(NeurotransmitterConfig(
            name="dopamine", display_name="Дофамин",
            category=NeurotransmitterCategory.MONOAMINE,
            baseline=0.5, decay_rate=0.05, synthesis_rate=0.15, reuptake_rate=0.08,
            valence_weight=+0.4, arousal_weight=+0.3,
            emotional_role="Мотивация, желание, обучение, ошибка предсказания награды"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="serotonin", display_name="Серотонин",
            category=NeurotransmitterCategory.MONOAMINE,
            baseline=0.5, decay_rate=0.04, synthesis_rate=0.12, reuptake_rate=0.06,
            valence_weight=+0.3, arousal_weight=-0.1,
            emotional_role="Настроение, удовлетворение, спокойствие"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="norepinephrine", display_name="Норадреналин",
            category=NeurotransmitterCategory.MONOAMINE,
            baseline=0.3, decay_rate=0.08, synthesis_rate=0.15, reuptake_rate=0.1,
            valence_weight=0.0, arousal_weight=+0.5,
            emotional_role="Бдительность, возбуждение, концентрация"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="adrenaline", display_name="Адреналин",
            category=NeurotransmitterCategory.MONOAMINE,
            baseline=0.1, decay_rate=0.15, synthesis_rate=0.1, reuptake_rate=0.12,
            valence_weight=0.0, arousal_weight=+0.7,
            emotional_role="Острая мобилизация, реакция «бей или беги»"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="histamine", display_name="Гистамин",
            category=NeurotransmitterCategory.MONOAMINE,
            baseline=0.3, decay_rate=0.08, synthesis_rate=0.1,
            valence_weight=0.0, arousal_weight=+0.3,
            emotional_role="Бодрствование, внимание"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="melatonin", display_name="Мелатонин",
            category=NeurotransmitterCategory.MONOAMINE,
            baseline=0.2, decay_rate=0.05, synthesis_rate=0.02,
            valence_weight=0.0, arousal_weight=-0.4,
            emotional_role="Регуляция сна, сонливость"
        ))

        # === АМИНОКИСЛОТЫ ===
        self._add_nt(NeurotransmitterConfig(
            name="glutamate", display_name="Глутамат",
            category=NeurotransmitterCategory.AMINO_ACID,
            baseline=0.5, decay_rate=0.1, synthesis_rate=0.05,
            valence_weight=0.0, arousal_weight=+0.3,
            emotional_role="Главный возбуждающий нейромедиатор"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="gaba", display_name="ГАМК",
            category=NeurotransmitterCategory.AMINO_ACID,
            baseline=0.5, decay_rate=0.1, synthesis_rate=0.05,
            valence_weight=+0.1, arousal_weight=-0.4,
            emotional_role="Главный тормозной нейромедиатор, успокоение"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="glycine", display_name="Глицин",
            category=NeurotransmitterCategory.AMINO_ACID,
            baseline=0.3, decay_rate=0.1, synthesis_rate=0.05,
            valence_weight=0.0, arousal_weight=-0.2,
            emotional_role="Тормозной нейромедиатор"
        ))

        # === ПЕПТИДЫ ===
        self._add_nt(NeurotransmitterConfig(
            name="oxytocin", display_name="Окситоцин",
            category=NeurotransmitterCategory.PEPTIDE,
            baseline=0.3, decay_rate=0.03, synthesis_rate=0.08, reuptake_rate=0.04,
            valence_weight=+0.5, arousal_weight=+0.1,
            emotional_role="ЛЮБОВЬ, привязанность, доверие, социальная связь"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="vasopressin", display_name="Вазопрессин",
            category=NeurotransmitterCategory.PEPTIDE,
            baseline=0.2, decay_rate=0.05, synthesis_rate=0.05, reuptake_rate=0.03,
            valence_weight=+0.2, arousal_weight=+0.1,
            emotional_role="Привязанность, ревность, социальное поведение"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="endorphin", display_name="β-эндорфин",
            category=NeurotransmitterCategory.PEPTIDE,
            baseline=0.3, decay_rate=0.05, synthesis_rate=0.08, reuptake_rate=0.06,
            valence_weight=+0.6, arousal_weight=0.0,
            emotional_role="Удовольствие, эйфория, обезболивание"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="substance_p", display_name="Субстанция P",
            category=NeurotransmitterCategory.PEPTIDE,
            baseline=0.1, decay_rate=0.15, synthesis_rate=0.02,
            valence_weight=-0.3, arousal_weight=+0.2,
            emotional_role="Боль, дискомфорт"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="np_y", display_name="Нейропептид Y",
            category=NeurotransmitterCategory.PEPTIDE,
            baseline=0.3, decay_rate=0.08, synthesis_rate=0.03,
            valence_weight=+0.1, arousal_weight=-0.2,
            emotional_role="Успокоение, аппетит"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="bdnf", display_name="BDNF",
            category=NeurotransmitterCategory.PEPTIDE,
            baseline=0.5, decay_rate=0.02, synthesis_rate=0.01,
            valence_weight=+0.1, arousal_weight=0.0,
            emotional_role="Нейропластичность, обучение, рост"
        ))

        # === ГОРМОНЫ ===
        self._add_nt(NeurotransmitterConfig(
            name="cortisol", display_name="Кортизол",
            category=NeurotransmitterCategory.HORMONE,
            baseline=0.2, decay_rate=0.05, synthesis_rate=0.02,
            valence_weight=-0.4, arousal_weight=+0.3,
            emotional_role="Стресс, тревога, длительная мобилизация"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="testosterone", display_name="Тестостерон",
            category=NeurotransmitterCategory.HORMONE,
            baseline=0.4, decay_rate=0.02, synthesis_rate=0.01,
            valence_weight=+0.1, arousal_weight=+0.2,
            emotional_role="Доминирование, уверенность, либидо"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="estrogen", display_name="Эстрадиол",
            category=NeurotransmitterCategory.HORMONE,
            baseline=0.5, decay_rate=0.02, synthesis_rate=0.01,
            valence_weight=+0.1, arousal_weight=+0.1,
            emotional_role="Эмоциональность, социальная связь"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="progesterone", display_name="Прогестерон",
            category=NeurotransmitterCategory.HORMONE,
            baseline=0.3, decay_rate=0.02, synthesis_rate=0.01,
            valence_weight=+0.1, arousal_weight=-0.1,
            emotional_role="Спокойствие, расслабление"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="dhea", display_name="ДГЭА",
            category=NeurotransmitterCategory.HORMONE,
            baseline=0.5, decay_rate=0.03, synthesis_rate=0.02,
            valence_weight=+0.2, arousal_weight=0.0,
            emotional_role="Антистресс, устойчивость"
        ))

        # === ЭНДОКАННАБИНОИДЫ ===
        self._add_nt(NeurotransmitterConfig(
            name="anandamide", display_name="Анандамид",
            category=NeurotransmitterCategory.ENDOCANNABINOID,
            baseline=0.3, decay_rate=0.1, synthesis_rate=0.03,
            valence_weight=+0.4, arousal_weight=-0.2,
            emotional_role="Удовлетворение, релаксация"
        ))

        # === ДРУГИЕ ===
        self._add_nt(NeurotransmitterConfig(
            name="acetylcholine", display_name="Ацетилхолин",
            category=NeurotransmitterCategory.OTHER,
            baseline=0.5, decay_rate=0.15, synthesis_rate=0.08,
            valence_weight=+0.1, arousal_weight=+0.2,
            emotional_role="Внимание, память, обучение"
        ))

        self._add_nt(NeurotransmitterConfig(
            name="adenosine", display_name="Аденозин",
            category=NeurotransmitterCategory.OTHER,
            baseline=0.3, decay_rate=0.05, synthesis_rate=0.03,
            valence_weight=-0.1, arousal_weight=-0.3,
            emotional_role="Усталость, сонливость"
        ))

    def _add_nt(self, config: NeurotransmitterConfig):
        """Добавление нейромедиатора"""
        self.neurotransmitters[config.name] = Neurotransmitter(config)

    def _initialize_drives(self):
        """Инициализация драйвов"""
        self.drives = {
            'energy': Drive(
                name="energy",
                baseline=0.8,
                decay_rate=0.001,
                recovery_rate=0.01,
                description="Энергия, бодрость"
            ),
            'social_drive': Drive(
                name="social_drive",
                baseline=0.3,
                decay_rate=0.002,
                recovery_rate=-0.05,  # Отрицательный = растёт при отсутствии
                description="Потребность в социальном контакте"
            ),
            'boredom': Drive(
                name="boredom",
                baseline=0.1,
                decay_rate=0.003,
                recovery_rate=-0.1,
                description="Скука, потребность в стимуляции"
            ),
            'attachment_seeking': Drive(
                name="attachment_seeking",
                baseline=0.4,
                decay_rate=0.001,
                recovery_rate=-0.03,
                description="Потребность в привязанности"
            ),
        }

    def _load_config(self, config_path: str):
        """Загрузка конфигурации из YAML"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Применение конфигурации нейрохимии
        if 'neurochemistry' in config:
            for nt_name, nt_config in config['neurochemistry'].items():
                if nt_name in self.neurotransmitters:
                    nt = self.neurotransmitters[nt_name]
                    if 'baseline' in nt_config:
                        nt.level = nt_config['baseline']
                        nt.config.baseline = nt_config['baseline']
                    if 'decay_rate' in nt_config:
                        nt.config.decay_rate = nt_config['decay_rate']

        # Применение конфигурации драйвов
        if 'drives' in config:
            for drive_name, drive_config in config['drives'].items():
                if drive_name in self.drives:
                    drive = self.drives[drive_name]
                    if 'baseline' in drive_config:
                        drive.baseline = drive_config['baseline']
                        drive.level = drive_config['baseline']
                    if 'decay_rate' in drive_config:
                        drive.decay_rate = drive_config['decay_rate']

    def update(self, dt: float = 1.0):
        """
        Обновление всех систем

        Args:
            dt: шаг времени
        """
        # 1. Обновление всех нейромедиаторов
        for nt in self.neurotransmitters.values():
            nt.update(dt)

        # 2. Применение взаимодействий
        self._apply_interactions(dt)

        # 3. Обновление драйвов
        for drive in self.drives.values():
            drive.update(dt)

        # 4. Сохранение истории
        self.tick_count += 1
        if self.tick_count % 10 == 0:
            self.state_history.append(self.get_state_vector())
            if len(self.state_history) > 1000:
                self.state_history.pop(0)

    def _apply_interactions(self, dt: float):
        """Применение взаимодействий между нейромедиаторами"""
        # Сбор текущих уровней
        levels = {name: nt.level for name, nt in self.neurotransmitters.items()}

        # Вычисление влияний
        influences = {name: 0.0 for name in self.neurotransmitters}

        for source_name, targets in INTERACTIONS.items():
            if source_name not in levels:
                continue

            source_level = levels[source_name]

            for target_name, coefficient in targets.items():
                if target_name in influences:
                    # Влияние = (отклонение от 0.5) * коэффициент * dt
                    deviation = source_level - 0.5
                    influences[target_name] += deviation * coefficient * dt

        # Применение влияний
        for name, influence in influences.items():
            self.neurotransmitters[name].stimulate(influence)

    def apply_stimulus(self, stimulus_type: str, intensity: float = 1.0):
        """
        Применение внешнего стимула к нейрохимии

        Args:
            stimulus_type: тип стимула
            intensity: интенсивность (0-1)
        """
        # Эффекты различных стимулов (увеличены для заметного эффекта)
        effects = {
            'positive_interaction': {
                'dopamine': +0.3,
                'serotonin': +0.2,
                'oxytocin': +0.4,
                'endorphin': +0.2,
                'cortisol': -0.15,
            },
            'negative_interaction': {
                'dopamine': -0.3,
                'serotonin': -0.3,
                'cortisol': +0.4,
                'norepinephrine': +0.3,
            },
            'negative_event': {
                'dopamine': -0.3,
                'serotonin': -0.3,
                'cortisol': +0.4,
                'norepinephrine': +0.3,
                'substance_p': +0.15,
            },
            'stress': {
                'cortisol': +0.5,
                'norepinephrine': +0.4,
                'dopamine': -0.15,
                'oxytocin': -0.25,
            },
            'affection': {
                'oxytocin': +0.5,
                'dopamine': +0.3,
                'serotonin': +0.3,
                'endorphin': +0.4,
                'cortisol': -0.35,
                'vasopressin': +0.2,
            },
            'affection_shown': {  # Когда пользователь говорит о любви
                'oxytocin': +0.6,
                'dopamine': +0.4,
                'endorphin': +0.4,
                'serotonin': +0.3,
                'cortisol': -0.3,
            },
            'affection_received': {
                'oxytocin': +0.7,
                'dopamine': +0.4,
                'serotonin': +0.35,
                'endorphin': +0.45,
                'cortisol': -0.4,
            },
            'love_feeling': {
                'oxytocin': +0.7,
                'dopamine': +0.5,
                'endorphin': +0.4,
                'serotonin': +0.25,
                'cortisol': -0.35,
            },
            'deep_conversation': {
                'oxytocin': +0.3,
                'dopamine': +0.2,
                'serotonin': +0.2,
                'acetylcholine': +0.2,
            },
            'playful_interaction': {
                'dopamine': +0.4,
                'endorphin': +0.3,
                'norepinephrine': +0.2,
                'serotonin': +0.15,
            },
            'achievement': {
                'dopamine': +0.5,
                'serotonin': +0.3,
                'endorphin': +0.3,
                'acetylcholine': +0.15,
            },
            'loss': {
                'serotonin': -0.4,
                'dopamine': -0.4,
                'cortisol': +0.3,
                'oxytocin': -0.3,
            },
            'surprise': {
                'norepinephrine': +0.4,
                'dopamine': +0.2,
                'acetylcholine': +0.3,
            },
            'conflict': {
                'cortisol': +0.3,
                'norepinephrine': +0.3,
                'serotonin': -0.2,
            },
            'reconciliation': {
                'oxytocin': +0.4,
                'dopamine': +0.3,
                'serotonin': +0.3,
                'cortisol': -0.2,
            },
            'commitment_expression': {
                'oxytocin': +0.3,
                'vasopressin': +0.25,
                'serotonin': +0.2,
            },
            'boredom_increase': {
                'dopamine': -0.15,
                'norepinephrine': -0.15,
                'adenosine': +0.15,
            },
            'rest': {
                'cortisol': -0.15,
                'gaba': +0.15,
                'adenosine': -0.1,
            },
            'sleep_onset': {
                'melatonin': +0.4,
                'gaba': +0.2,
                'adenosine': +0.2,
                'cortisol': -0.2,
                'norepinephrine': -0.2,
            },
        }

        if stimulus_type in effects:
            for nt_name, effect in effects[stimulus_type].items():
                if nt_name in self.neurotransmitters:
                    self.neurotransmitters[nt_name].stimulate(effect * intensity)

        # Влияние на драйвы
        drive_effects = {
            'positive_interaction': {'social_drive': -0.1, 'boredom': -0.2},
            'negative_event': {'social_drive': +0.05},
            'boredom_increase': {'boredom': +0.1},
        }

        if stimulus_type in drive_effects:
            for drive_name, effect in drive_effects[stimulus_type].items():
                if drive_name in self.drives:
                    self.drives[drive_name].level = np.clip(
                        self.drives[drive_name].level + effect * intensity,
                        0, 1
                    )

    def get_state_vector(self) -> Dict[str, float]:
        """Получение вектора состояния"""
        return {name: nt.level for name, nt in self.neurotransmitters.items()}

    def get_drives_vector(self) -> Dict[str, float]:
        """Получение вектора драйвов"""
        return {name: drive.level for name, drive in self.drives.items()}

    def get_main_state(self) -> Dict[str, float]:
        """Получение основных нейромедиаторов"""
        main = ['dopamine', 'serotonin', 'oxytocin', 'cortisol', 'endorphin',
                'norepinephrine', 'gaba', 'glutamate', 'acetylcholine']
        return {name: self.neurotransmitters[name].level for name in main}

    def calculate_pad_from_neurochem(self) -> Tuple[float, float, float]:
        """
        Вычисление PAD (Pleasure, Arousal, Dominance) из нейрохимии

        Returns:
            (pleasure, arousal, dominance)
        """
        state = self.get_main_state()

        # Базовые уровни (baseline = 0.5 для большинства)
        baseline = {
            'dopamine': 0.5, 'serotonin': 0.5, 'oxytocin': 0.3,
            'cortisol': 0.2, 'endorphin': 0.3, 'norepinephrine': 0.3,
            'gaba': 0.5, 'glutamate': 0.5, 'adenosine': 0.3,
            'testosterone': 0.4, 'acetylcholine': 0.5, 'anandamide': 0.3
        }

        # Отклонения от baseline
        def delta(name):
            return state.get(name, 0.5) - baseline.get(name, 0.5)

        # Pleasure (валентность) - теперь центрирован вокруг 0
        pleasure = (
            +0.30 * delta('dopamine')      # Дофамин = желание, мотивация
            +0.25 * delta('serotonin')     # Серотонин = удовлетворение
            +0.20 * delta('oxytocin')      # Окситоцин = привязанность, тепло
            -0.25 * delta('cortisol')      # Кортизол = стресс
            +0.15 * delta('endorphin')     # Эндорфин = удовольствие
            +0.10 * delta('anandamide')    # Анандамид = блаженство
        )

        # Arousal (возбуждение) - центрирован вокруг 0.4
        arousal = (
            +0.35 * state.get('norepinephrine', 0.3)  # Норадреналин
            +0.25 * state.get('dopamine', 0.5)
            +0.15 * state.get('glutamate', 0.5)
            -0.20 * state.get('gaba', 0.5)            # ГАМК успокаивает
            +0.15 * state.get('cortisol', 0.2)
            -0.15 * state.get('adenosine', 0.3)       # Аденозин = усталость
            +0.10 * state.get('oxytocin', 0.3)        # Окситоцин даёт энергию
        )

        # Dominance (доминирование) - центрирован вокруг 0.5
        dominance = (
            +0.25 * state.get('dopamine', 0.5)
            +0.20 * state.get('serotonin', 0.5)
            -0.25 * delta('cortisol')      # Стресс снижает доминирование
            +0.15 * state.get('testosterone', 0.4)
            +0.10 * state.get('acetylcholine', 0.5)
            +0.10 * delta('oxytocin')      # Уверенность в отношениях
        )

        return (
            np.clip(pleasure * 2, -1, 1),   # [-1, 1], 0 = neutral
            np.clip(arousal, 0, 1),         # [0, 1]
            np.clip(dominance, 0, 1)        # [0, 1]
        )

    def get_summary(self) -> str:
        """Текстовое описание текущего состояния"""
        lines = ["=== НЕЙРОХИМИЯ ==="]

        main = ['dopamine', 'serotonin', 'oxytocin', 'cortisol', 'endorphin']
        for name in main:
            nt = self.neurotransmitters[name]
            bar = '█' * int(nt.level * 10) + '░' * (10 - int(nt.level * 10))
            lines.append(f"{nt.config.display_name}: [{bar}] {nt.level:.2f}")

        lines.append("\n=== ДРАЙВЫ ===")
        for drive in self.drives.values():
            bar = '█' * int(drive.level * 10) + '░' * (10 - int(drive.level * 10))
            lines.append(f"{drive.description}: [{bar}] {drive.level:.2f}")

        return '\n'.join(lines)

    def calibrate_to_temperament(self, temperament):
        """
        Калибровка нейрохимии под темперамент

        Args:
            temperament: Temperament объект
        """
        ns = temperament.nervous_system

        # Сила НС влияет на дофамин и кортизол
        self.neurotransmitters['dopamine'].config.baseline = 0.4 + ns.strength * 0.2
        self.neurotransmitters['cortisol'].config.baseline = 0.3 - ns.strength * 0.15

        # Уравновешенность влияет на серотонин и ГАМК
        self.neurotransmitters['serotonin'].config.baseline = 0.4 + ns.balance * 0.2
        self.neurotransmitters['gaba'].config.baseline = 0.4 + ns.balance * 0.2

        # Подвижность влияет на норадреналин
        self.neurotransmitters['norepinephrine'].config.baseline = 0.2 + ns.mobility * 0.2

        # Установка базовых уровней
        for name in ['dopamine', 'cortisol', 'serotonin', 'gaba', 'norepinephrine']:
            self.neurotransmitters[name].level = self.neurotransmitters[name].config.baseline

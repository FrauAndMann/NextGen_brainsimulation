"""
Система аватара для цифрового компаньона

Поддерживает:
- ASCII-арт аватары (без зависимостей)
- Эмоциональные выражения
- Анимации моргания и дыхания
- Подготовка для Live2D интеграции
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import time


class AvatarEmotion(Enum):
    """Эмоции аватара"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    LOVE = "love"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CALM = "calm"
    EXCITED = "excited"
    SLEEPY = "sleepy"
    WORRIED = "worried"


@dataclass
class AvatarState:
    """Состояние аватара"""
    emotion: AvatarEmotion = AvatarEmotion.NEUTRAL
    blink_state: bool = False          # Открыты/закрыты глаза
    mouth_open: float = 0.0             # 0-1, для речи
    blush_intensity: float = 0.0        # 0-1, румянец
    head_tilt: float = 0.0              # -1 до 1, наклон головы
    eye_sparkle: float = 0.5            # 0-1, блеск в глазах


class ASCIIAvatar:
    """
    ASCII-арт аватар

    Отображает эмоции через текстовую графику.
    Работает без внешних зависимостей.
    """

    # Базовые лица для разных эмоций
    FACES = {
        AvatarEmotion.NEUTRAL: [
            "    ∩∩    ",
            "   (・ω・)  ",
            "   _| ⊃_  ",
            "  (・＿・)  ",
        ],
        AvatarEmotion.HAPPY: [
            "    ∩∩    ",
            "   (◕‿◕)  ",
            "   _| ⊃_  ",
            "  (｡♥‿♥｡) ",
        ],
        AvatarEmotion.LOVE: [
            "    ∩∩    ",
            "   (♡▽♡)  ",
            "   _| ⊃_  ",
            "  (♥ω♥*)  ",
        ],
        AvatarEmotion.SAD: [
            "    ∩∩    ",
            "   (╥﹏╥)  ",
            "   _| ⊃_  ",
            "  (；´Д｀) ",
        ],
        AvatarEmotion.ANGRY: [
            "    ∩∩    ",
            "   (╬ಠ益ಠ) ",
            "   _| ⊃_  ",
            "  (ノಠ益ಠ)ノ",
        ],
        AvatarEmotion.SURPRISED: [
            "    ∩∩    ",
            "   (°o°)  ",
            "   _| ⊃_  ",
            "  (⊙_⊙)   ",
        ],
        AvatarEmotion.CALM: [
            "    ∩∩    ",
            "   (‾◡‾)  ",
            "   _| ⊃_  ",
            "  (─‿‿─)  ",
        ],
        AvatarEmotion.EXCITED: [
            "    ∩∩    ",
            "   (★ω★)  ",
            "   _| ⊃_  ",
            "  ヽ(>∀<☆)ノ",
        ],
        AvatarEmotion.SLEEPY: [
            "    ∩∩    ",
            "   (－ω－) ",
            "   _| ⊃_  ",
            "  (｡-ω-)zzZ",
        ],
        AvatarEmotion.WORRIED: [
            "    ∩∩    ",
            "   (・_・;) ",
            "   _| ⊃_  ",
            "  (；￣Д￣) ",
        ],
    }

    # Варианты с закрытыми глазами (моргание)
    FACES_BLINK = {
        AvatarEmotion.NEUTRAL: [
            "    ∩∩    ",
            "   (－ω－) ",
            "   _| ⊃_  ",
            "  (・＿・)  ",
        ],
        AvatarEmotion.HAPPY: [
            "    ∩∩    ",
            "   (－‿－) ",
            "   _| ⊃_  ",
            "  (｡♥‿♥｡) ",
        ],
        AvatarEmotion.LOVE: [
            "    ∩∩    ",
            "   (－ω－) ",
            "   _| ⊃_  ",
            "  (♥ω♥*)  ",
        ],
    }

    def __init__(self):
        self.state = AvatarState()
        self.last_blink_time = time.time()
        self.blink_interval = 3.0  # Секунды между морганиями
        self.blink_duration = 0.15

    def update(self, emotion: AvatarEmotion, blush: float = 0.0, dt: float = 0.1):
        """Обновление состояния аватара"""
        self.state.emotion = emotion
        self.state.blush_intensity = blush

        # Автоматическое моргание
        current_time = time.time()
        if current_time - self.last_blink_time > self.blink_interval:
            if not self.state.blink_state:
                self.state.blink_state = True
                self.last_blink_time = current_time
            elif current_time - self.last_blink_time > self.blink_duration:
                self.state.blink_state = False

    def render(self) -> str:
        """Рендеринг аватара в ASCII"""
        # Выбор лица с учётом моргания
        if self.state.blink_state and self.state.emotion in self.FACES_BLINK:
            face = self.FACES_BLINK[self.state.emotion]
        else:
            face = self.FACES.get(self.state.emotion, self.FACES[AvatarEmotion.NEUTRAL])

        # Добавление румянца
        if self.state.blush_intensity > 0.3:
            # Модифицируем лицо для отображения румянца
            face = [line.replace("(・", "(⁄⁄").replace("・)", "⁄⁄)") for line in face]

        return '\n'.join(face)

    def get_emotion_emoji(self) -> str:
        """Получение эмодзи для текущей эмоции"""
        emojis = {
            AvatarEmotion.NEUTRAL: "😐",
            AvatarEmotion.HAPPY: "😊",
            AvatarEmotion.LOVE: "😍",
            AvatarEmotion.SAD: "😢",
            AvatarEmotion.ANGRY: "😠",
            AvatarEmotion.SURPRISED: "😲",
            AvatarEmotion.CALM: "😌",
            AvatarEmotion.EXCITED: "🤩",
            AvatarEmotion.SLEEPY: "😴",
            AvatarEmotion.WORRIED: "😟",
        }
        return emojis.get(self.state.emotion, "😐")


class AvatarRenderer:
    """
    Рендерер аватара

    Управляет отображением аватара в разных форматах:
    - ASCII (консоль)
    - Unicode art (GUI)
    - SVG (для будущего расширения)
    """

    def __init__(self):
        self.ascii_avatar = ASCIIAvatar()
        self.animation_frame = 0

    def map_pad_to_emotion(
        self,
        pleasure: float,
        arousal: float,
        dominance: float,
        love_level: float = 0.0
    ) -> AvatarEmotion:
        """
        Маппинг PAD значений на эмоции аватара

        Args:
            pleasure: валентность (-1 до +1)
            arousal: возбуждение (0 до 1)
            dominance: доминирование (0 до 1)
            love_level: уровень любви (0 до 1)

        Returns:
            AvatarEmotion
        """
        # Любовь имеет приоритет
        if love_level > 0.5 and pleasure > 0:
            return AvatarEmotion.LOVE

        # Высокое удовольствие
        if pleasure > 0.4:
            if arousal > 0.6:
                return AvatarEmotion.EXCITED
            elif arousal > 0.3:
                return AvatarEmotion.HAPPY
            else:
                return AvatarEmotion.CALM

        # Низкое удовольствие
        if pleasure < -0.4:
            if arousal > 0.6:
                return AvatarEmotion.ANGRY if dominance > 0.5 else AvatarEmotion.WORRIED
            elif arousal > 0.3:
                return AvatarEmotion.WORRIED
            else:
                return AvatarEmotion.SAD

        # Нейтральная зона
        if arousal > 0.7:
            return AvatarEmotion.SURPRISED
        elif arousal < 0.2:
            return AvatarEmotion.SLEEPY

        return AvatarEmotion.NEUTRAL

    def update(
        self,
        pleasure: float,
        arousal: float,
        dominance: float,
        love_level: float = 0.0,
        dt: float = 0.1
    ):
        """Обновление состояния аватара"""
        emotion = self.map_pad_to_emotion(pleasure, arousal, dominance, love_level)

        # Румянец от любви или удовольствия
        blush = 0.0
        if love_level > 0.3:
            blush = love_level * 0.5
        elif pleasure > 0.3:
            blush = pleasure * 0.3

        self.ascii_avatar.update(emotion, blush, dt)
        self.animation_frame += 1

    def render_ascii(self) -> str:
        """Рендеринг в ASCII"""
        return self.ascii_avatar.render()

    def render_unicode(self) -> str:
        """Рендеринг в Unicode (для GUI)"""
        # Unicode-арт лица
        base_faces = {
            AvatarEmotion.NEUTRAL: """
    ╭───────╮
    │  ・ω・ │
    │   ▽   │
    ╰───────╯
""",
            AvatarEmotion.HAPPY: """
    ╭───────╮
    │  ◕‿◕  │
    │   ♡   │
    ╰───────╯
""",
            AvatarEmotion.LOVE: """
    ╭───────╮
    │  ♡▽♡  │
    │  ♥ω♥  │
    ╰───────╯
""",
            AvatarEmotion.SAD: """
    ╭───────╮
    │  ╥﹏╥  │
    │   ▽   │
    ╰───────╯
""",
            AvatarEmotion.ANGRY: """
    ╭───────╮
    │  ╬ಠ益ಠ │
    │   ─   │
    ╰───────╯
""",
            AvatarEmotion.EXCITED: """
    ╭───────╮
    │  ★ω★  │
    │   ◇   │
    ╰───────╯
""",
            AvatarEmotion.CALM: """
    ╭───────╮
    │  ‾◡‾  │
    │   ▽   │
    ╰───────╯
""",
            AvatarEmotion.SLEEPY: """
    ╭───────╮
    │  －ω－ │
    │   zZ  │
    ╰───────╯
""",
        }

        emotion = self.ascii_avatar.state.emotion
        return base_faces.get(emotion, base_faces[AvatarEmotion.NEUTRAL])

    def get_status_text(self) -> str:
        """Получение текстового описания статуса"""
        emotion = self.ascii_avatar.state.emotion
        emoji = self.ascii_avatar.get_emotion_emoji()

        descriptions = {
            AvatarEmotion.NEUTRAL: "нейтральное настроение",
            AvatarEmotion.HAPPY: "счастлива",
            AvatarEmotion.LOVE: "влюблена",
            AvatarEmotion.SAD: "грустит",
            AvatarEmotion.ANGRY: "сердится",
            AvatarEmotion.SURPRISED: "удивлена",
            AvatarEmotion.CALM: "спокойна",
            AvatarEmotion.EXCITED: "в восторге",
            AvatarEmotion.SLEEPY: "хочет спать",
            AvatarEmotion.WORRIED: "беспокоится",
        }

        return f"{emoji} {descriptions.get(emotion, '')}"


def create_avatar_display(width: int = 15, height: int = 8) -> str:
    """Создание рамки для аватара"""
    top = "╭" + "─" * (width - 2) + "╮"
    middle = "│" + " " * (width - 2) + "│"
    bottom = "╰" + "─" * (width - 2) + "╯"

    lines = [top] + [middle] * (height - 2) + [bottom]
    return '\n'.join(lines)

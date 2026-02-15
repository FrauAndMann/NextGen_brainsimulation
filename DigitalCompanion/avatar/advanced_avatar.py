"""
Advanced Avatar System - Продвинутая система аватара

Особенности:
- 12 базовых эмоций + смешанные состояния (бленды)
- Плавные переходы между эмоциями
- Микроэмоции (моргание, подёргивания)
- Дыхание и живые движения
- Реакция на речь в реальном времени
- Более детализированная графика

Автор: FrauAndMann
Версия: 2.0
"""

import tkinter as tk
from tkinter import Canvas
import math
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import colorsys


# === ЭМОЦИИ ===

class EmotionType(Enum):
    """Базовые эмоции (Ekman + расширения)"""
    # Базовые (Ekman)
    JOY = "joy"                 # Радость
    SADNESS = "sadness"         # Грусть
    ANGER = "anger"             # Гнев
    FEAR = "fear"               # Страх
    DISGUST = "disgust"         # Отвращение
    SURPRISE = "surprise"       # Удивление

    # Расширенные
    LOVE = "love"               # Любовь/нежность
    EXCITEMENT = "excitement"   # Восторг
    CALM = "calm"               # Спокойствие
    INTEREST = "interest"       # Интерес
    SHAME = "shame"             # Стыд
    CONTEMPT = "contempt"       # Презрение

    # Специальные
    NEUTRAL = "neutral"         # Нейтральное


# Параметры эмоций: (валентность, возбуждение, доминирование)
EMOTION_PAD = {
    EmotionType.JOY: (0.6, 0.5, 0.4),
    EmotionType.SADNESS: (-0.6, -0.3, -0.3),
    EmotionType.ANGER: (-0.5, 0.7, 0.5),
    EmotionType.FEAR: (-0.6, 0.5, -0.4),
    EmotionType.DISGUST: (-0.4, 0.2, 0.1),
    EmotionType.SURPRISE: (0.1, 0.8, -0.2),
    EmotionType.LOVE: (0.8, 0.4, -0.1),
    EmotionType.EXCITEMENT: (0.7, 0.8, 0.3),
    EmotionType.CALM: (0.3, -0.4, 0.2),
    EmotionType.INTEREST: (0.3, 0.3, 0.2),
    EmotionType.SHAME: (-0.4, 0.3, -0.5),
    EmotionType.CONTEMPT: (-0.3, 0.1, 0.5),
    EmotionType.NEUTRAL: (0.0, 0.0, 0.0),
}


@dataclass
class EmotionBlend:
    """Смешанная эмоция"""
    emotions: Dict[EmotionType, float]  # Эмоция -> вес (сумма = 1.0)

    def get_primary(self) -> Tuple[EmotionType, float]:
        """Получить доминирующую эмоцию"""
        if not self.emotions:
            return EmotionType.NEUTRAL, 1.0
        return max(self.emotions.items(), key=lambda x: x[1])

    @classmethod
    def from_pad(cls, valence: float, arousal: float, dominance: float = 0.0) -> 'EmotionBlend':
        """Создать бленд из PAD координат"""
        distances = {}
        for emotion, (v, a, d) in EMOTION_PAD.items():
            dist = math.sqrt((v - valence)**2 + (a - arousal)**2 + (d - dominance)**2)
            distances[emotion] = 1.0 / (dist + 0.1)  # Обратное расстояние

        # Нормализация
        total = sum(distances.values())
        emotions = {e: w / total for e, w in distances.items()}

        # Оставляем только значимые
        emotions = {e: w for e, w in emotions.items() if w > 0.1}

        return cls(emotions)


# === СОСТОЯНИЕ АВАТАРА ===

@dataclass
class AvatarFacialState:
    """Состояние лицевых параметров"""
    # Глаза
    eye_openness: float = 1.0          # 0-1 (закрыты - открыты)
    pupil_size: float = 0.5            # 0-1 (сужены - расширены)
    eye_gaze_x: float = 0.0            # -1 до 1 (влево - вправо)
    eye_gaze_y: float = 0.0            # -1 до 1 (вверх - вниз)
    eye_squint: float = 0.0            # 0-1 (прищур)

    # Брови
    eyebrow_raise: float = 0.0         # 0-1 (подняты)
    eyebrow_furrow: float = 0.0        # 0-1 (нахмурены)
    eyebrow_asymmetry: float = 0.0     # -1 до 1 (асимметрия)

    # Рот
    mouth_open: float = 0.0            # 0-1 (закрыт - открыт)
    mouth_curve: float = 0.0           # -1 до 1 (грусть - улыбка)
    mouth_width: float = 0.5           # 0-1 (узкий - широкий)
    mouth_pout: float = 0.0            # 0-1 (выпячены губы)

    # Щёки
    blush_intensity: float = 0.0       # 0-1 (румянец)

    # Нос
    nose_wrinkle: float = 0.0          # 0-1 (сморщен)

    # Живость
    breath_offset: float = 0.0         # Смещение от дыхания
    blink_timer: float = 0.0           # Таймер моргания
    is_blinking: bool = False
    speaking_intensity: float = 0.0    # Интенсивность говорения


@dataclass
class AvatarColors:
    """Цветовая схема аватара"""
    # Кожа
    skin_base: str = "#fce4d6"
    skin_shadow: str = "#e8c8b0"
    skin_highlight: str = "#fff0e6"

    # Волосы
    hair_base: str = "#4a3728"
    hair_highlight: str = "#5d4a3a"
    hair_shadow: str = "#2d1f15"

    # Глаза
    iris: str = "#3d5a80"
    iris_dark: str = "#1a2d40"
    pupil: str = "#0a0a0a"
    eye_white: str = "#f8f8f8"
    eye_shadow: str = "#d0d0d0"

    # Губы
    lip_base: str = "#d4787c"
    lip_dark: str = "#8b4557"
    lip_highlight: str = "#e8a0a4"

    # Румянец
    blush: str = "#ffb6b9"

    # Одежда
    dress_base: str = "#7b68ee"
    dress_shadow: str = "#5a4ab0"
    dress_highlight: str = "#9b88ff"

    # Фон
    background: str = "#1a1a2e"

    def get_variant(self, variant: str) -> 'AvatarColors':
        """Получить вариант цветовой схемы"""
        colors = AvatarColors()

        if variant == "warm":
            colors.skin_base = "#ffd5c8"
            colors.hair_base = "#8b4513"
            colors.iris = "#228b22"
        elif variant == "cool":
            colors.skin_base = "#f0e6d6"
            colors.hair_base = "#1a1a1a"
            colors.iris = "#4169e1"
        elif variant == "pink":
            colors.dress_base = "#ff69b4"
            colors.dress_shadow = "#c71585"

        return colors


# === ГЛАВНЫЙ КЛАСС АВАТАРА ===

class AdvancedAvatar:
    """
    Продвинутый аватар с реалистичными эмоциями

    Особенности:
    - 12 базовых эмоций + бленды
    - Плавные переходы
    - Микроэмоции
    - Реакция на речь
    """

    def __init__(self, name: str = "Лиза", width: int = 500, height: int = 600):
        self.name = name
        self.width = width
        self.height = height

        # Состояние
        self.facial = AvatarFacialState()
        self.target_facial = AvatarFacialState()
        self.colors = AvatarColors()

        # Эмоции
        self.current_emotion = EmotionBlend({EmotionType.NEUTRAL: 1.0})
        self.target_emotion = EmotionBlend({EmotionType.NEUTRAL: 1.0})

        # PAD состояние
        self.valence = 0.0
        self.arousal = 0.3
        self.dominance = 0.5
        self.attachment = 0.5

        # Анимация
        self.start_time = time.time()
        self.last_blink_time = time.time()
        self.blink_interval = 3.5
        self.speaking = False
        self.speaking_text = ""

        # Плавность
        self.transition_speed = 0.1

        # Tkinter
        self.root = None
        self.canvas = None
        self.text_label = None
        self.running = False

        # Callbacks
        self.on_click: Optional[Callable] = None

    # === УПРАВЛЕНИЕ ЭМОЦИЯМИ ===

    def set_emotion(self, emotion: EmotionType, intensity: float = 1.0):
        """Установить одну эмоцию"""
        self.target_emotion = EmotionBlend({emotion: intensity})
        self._update_target_facial()

    def set_emotion_blend(self, emotions: Dict[EmotionType, float]):
        """Установить смешанную эмоцию"""
        # Нормализация
        total = sum(emotions.values())
        if total > 0:
            emotions = {e: w / total for e, w in emotions.items()}
        self.target_emotion = EmotionBlend(emotions)
        self._update_target_facial()

    def set_pad(self, valence: float, arousal: float, dominance: float = None, attachment: float = None):
        """Установить состояние по PAD"""
        self.valence = max(-1, min(1, valence))
        self.arousal = max(0, min(1, arousal))
        if dominance is not None:
            self.dominance = max(0, min(1, dominance))
        if attachment is not None:
            self.attachment = max(0, min(1, attachment))

        self.target_emotion = EmotionBlend.from_pad(self.valence, self.arousal, self.dominance)
        self._update_target_facial()

    def _update_target_facial(self):
        """Обновить целевое состояние лица на основе эмоций"""
        tf = self.target_facial

        # Сброс
        tf.eye_openness = 1.0
        tf.eyebrow_raise = 0.0
        tf.eyebrow_furrow = 0.0
        tf.mouth_curve = 0.0
        tf.mouth_open = 0.0
        tf.blush_intensity = 0.0
        tf.pupil_size = 0.5

        for emotion, weight in self.target_emotion.emotions.items():
            if emotion == EmotionType.JOY:
                tf.mouth_curve += 0.8 * weight
                tf.eye_squint += 0.3 * weight
                tf.blush_intensity += 0.3 * weight

            elif emotion == EmotionType.SADNESS:
                tf.mouth_curve -= 0.5 * weight
                tf.eyebrow_raise += 0.3 * weight
                tf.eyebrow_furrow += 0.2 * weight

            elif emotion == EmotionType.ANGER:
                tf.eyebrow_furrow += 0.8 * weight
                tf.eye_squint += 0.4 * weight
                tf.mouth_curve -= 0.3 * weight
                tf.pupil_size += 0.2 * weight

            elif emotion == EmotionType.FEAR:
                tf.eye_openness += 0.3 * weight
                tf.eyebrow_raise += 0.6 * weight
                tf.pupil_size += 0.4 * weight

            elif emotion == EmotionType.SURPRISE:
                tf.eye_openness += 0.5 * weight
                tf.eyebrow_raise += 0.8 * weight
                tf.mouth_open += 0.3 * weight

            elif emotion == EmotionType.LOVE:
                tf.mouth_curve += 0.6 * weight
                tf.eye_squint += 0.4 * weight
                tf.blush_intensity += 0.7 * weight
                tf.pupil_size += 0.3 * weight

            elif emotion == EmotionType.EXCITEMENT:
                tf.eye_openness += 0.2 * weight
                tf.mouth_curve += 0.7 * weight
                tf.mouth_open += 0.2 * weight
                tf.blush_intensity += 0.4 * weight

            elif emotion == EmotionType.CALM:
                tf.eye_openness -= 0.1 * weight
                tf.mouth_curve += 0.2 * weight

            elif emotion == EmotionType.INTEREST:
                tf.eye_openness += 0.2 * weight
                tf.eyebrow_raise += 0.3 * weight

            elif emotion == EmotionType.SHAME:
                tf.eye_openness -= 0.3 * weight
                tf.eye_gaze_y -= 0.3 * weight
                tf.blush_intensity += 0.8 * weight

            elif emotion == EmotionType.DISGUST:
                tf.nose_wrinkle += 0.5 * weight
                tf.eyebrow_furrow += 0.3 * weight
                tf.mouth_curve -= 0.3 * weight

        # Модуляция от arousal
        tf.pupil_size = max(0.2, min(1.0, tf.pupil_size + (self.arousal - 0.5) * 0.3))

        # Модуляция от attachment
        tf.blush_intensity = min(1.0, tf.blush_intensity + self.attachment * 0.2)

    def set_speaking(self, speaking: bool, text: str = ""):
        """Установить режим говорения"""
        self.speaking = speaking
        self.speaking_text = text

    # === АНИМАЦИЯ ===

    def _update_animation(self, dt: float):
        """Обновление анимации"""
        now = time.time()

        # Дыхание
        breath_phase = (now - self.start_time) * 0.8
        self.facial.breath_offset = math.sin(breath_phase) * 3

        # Моргание
        if now - self.last_blink_time > self.blink_interval:
            self.facial.is_blinking = True
            self.last_blink_time = now
            # Случайный интервал
            self.blink_interval = 2.5 + (hash(now) % 30) / 10

        if self.facial.is_blinking:
            if now - self.last_blink_time > 0.15:
                self.facial.is_blinking = False

        # Микродвижения глаз (saccade)
        if not self.facial.is_blinking:
            saccade = math.sin(now * 0.3) * 0.05
            self.facial.eye_gaze_x = self.target_facial.eye_gaze_x + saccade

        # Говорение
        if self.speaking:
            self.facial.speaking_intensity = 0.5 + 0.5 * math.sin(now * 12)
            self.facial.mouth_open = max(
                self.target_facial.mouth_open,
                self.facial.speaking_intensity * 0.4
            )
        else:
            self.facial.speaking_intensity *= 0.9

        # Плавный переход к целевому состоянию
        self._interpolate_facial(dt)

        # Обновление текущей эмоции
        self.current_emotion = self.target_emotion

    def _interpolate_facial(self, dt: float):
        """Интерполяция к целевому состоянию"""
        speed = self.transition_speed * (1 + self.arousal)  # Быстрее при высоком arousal

        f = self.facial
        tf = self.target_facial

        # Моргание имеет приоритет
        if f.is_blinking:
            f.eye_openness = 0.1
        else:
            f.eye_openness = self._lerp(f.eye_openness, tf.eye_openness, speed)

        # Остальные параметры
        f.eyebrow_raise = self._lerp(f.eyebrow_raise, tf.eyebrow_raise, speed)
        f.eyebrow_furrow = self._lerp(f.eyebrow_furrow, tf.eyebrow_furrow, speed)
        f.eyebrow_asymmetry = self._lerp(f.eyebrow_asymmetry, tf.eyebrow_asymmetry, speed)
        f.eye_squint = self._lerp(f.eye_squint, tf.eye_squint, speed)
        f.eye_gaze_x = self._lerp(f.eye_gaze_x, tf.eye_gaze_x, speed * 2)
        f.eye_gaze_y = self._lerp(f.eye_gaze_y, tf.eye_gaze_y, speed * 2)
        f.pupil_size = self._lerp(f.pupil_size, tf.pupil_size, speed)

        if not self.speaking:
            f.mouth_open = self._lerp(f.mouth_open, tf.mouth_open, speed)

        f.mouth_curve = self._lerp(f.mouth_curve, tf.mouth_curve, speed)
        f.mouth_width = self._lerp(f.mouth_width, tf.mouth_width, speed)
        f.mouth_pout = self._lerp(f.mouth_pout, tf.mouth_pout, speed)
        f.blush_intensity = self._lerp(f.blush_intensity, tf.blush_intensity, speed * 0.5)
        f.nose_wrinkle = self._lerp(f.nose_wrinkle, tf.nose_wrinkle, speed)

    @staticmethod
    def _lerp(current: float, target: float, speed: float) -> float:
        """Линейная интерполяция"""
        return current + (target - current) * min(speed, 1.0)

    # === ОТРИСОВКА ===

    def create_window(self):
        """Создание окна"""
        self.root = tk.Tk()
        self.root.title(self.name)
        self.root.geometry(f"{self.width}x{self.height + 80}")
        self.root.configure(bg=self.colors.background)
        self.root.resizable(False, False)

        # Канвас
        self.canvas = Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=self.colors.background,
            highlightthickness=0
        )
        self.canvas.pack()

        # Текст
        self.text_frame = tk.Frame(self.root, bg=self.colors.background)
        self.text_frame.pack(fill=tk.X, padx=15, pady=5)

        self.name_label = tk.Label(
            self.text_frame,
            text=self.name,
            font=('Segoe UI', 14, 'bold'),
            fg='#ffffff',
            bg=self.colors.background
        )
        self.name_label.pack(anchor='w')

        self.text_label = tk.Label(
            self.text_frame,
            text="...",
            font=('Segoe UI', 11),
            fg='#e0e0e0',
            bg=self.colors.background,
            wraplength=self.width - 30,
            justify=tk.LEFT
        )
        self.text_label.pack(anchor='w', fill=tk.X)

        # События
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

    def _on_close(self):
        """Закрытие окна"""
        self.running = False
        self.root.destroy()

    def _on_canvas_click(self, event):
        """Клик по аватару"""
        if self.on_click:
            self.on_click(event.x, event.y)

    def draw(self):
        """Отрисовка аватара"""
        self.canvas.delete("all")

        cx = self.width // 2
        cy = self.height // 2 - 30

        # Смещение от дыхания
        breath = self.facial.breath_offset

        # Порядок отрисовки (сзади вперёд)
        self._draw_hair_back(cx, cy + breath)
        self._draw_neck(cx, cy + breath)
        self._draw_dress(cx, cy + breath)
        self._draw_ears(cx, cy + breath)
        self._draw_face(cx, cy + breath)
        self._draw_hair_front(cx, cy + breath)
        self._draw_eyebrows(cx, cy + breath)
        self._draw_eyes(cx, cy + breath)
        self._draw_nose(cx, cy + breath)
        self._draw_mouth(cx, cy + breath)
        self._draw_blush(cx, cy + breath)

    def _draw_face(self, cx, cy):
        """Рисование лица"""
        # Основа (овал)
        self._draw_oval(
            cx - 80, cy - 95, cx + 80, cy + 85,
            fill=self.colors.skin_base,
            outline=self.colors.skin_shadow,
            width=2
        )

        # Тени
        self._draw_oval(
            cx - 70, cy - 70, cx + 70, cy + 60,
            fill=self.colors.skin_highlight,
            outline='',
            stipple='gray25'
        )

    def _draw_ears(self, cx, cy):
        """Уши"""
        for side in [-1, 1]:
            ex = cx + side * 82
            self._draw_oval(
                ex - 12, cy - 30, ex + 12, cy + 10,
                fill=self.colors.skin_base,
                outline=self.colors.skin_shadow
            )

    def _draw_hair_back(self, cx, cy):
        """Волосы сзади"""
        # Длинные волосы
        points = [
            cx - 90, cy - 50,
            cx - 100, cy + 80,
            cx - 75, cy + 150,
            cx - 40, cy + 180,
            cx, cy + 190,
            cx + 40, cy + 180,
            cx + 75, cy + 150,
            cx + 100, cy + 80,
            cx + 90, cy - 50,
        ]
        self.canvas.create_polygon(points, fill=self.colors.hair_base, smooth=True)

        # Блики
        self._draw_curve(
            [cx - 85, cy + 20, cx - 70, cy + 100, cx - 50, cy + 150],
            fill=self.colors.hair_highlight,
            width=8,
            smooth=True
        )

    def _draw_hair_front(self, cx, cy):
        """Волосы спереди (чёлка)"""
        # Основная чёлка
        points = [
            cx - 85, cy - 50,
            cx - 70, cy - 90,
            cx - 35, cy - 105,
            cx, cy - 110,
            cx + 35, cy - 105,
            cx + 70, cy - 90,
            cx + 85, cy - 50,
            cx + 60, cy - 65,
            cx + 25, cy - 75,
            cx - 25, cy - 75,
            cx - 60, cy - 65,
        ]
        self.canvas.create_polygon(points, fill=self.colors.hair_base, smooth=True)

        # Пряди
        self._draw_curve(
            [cx - 60, cy - 70, cx - 50, cy - 30, cx - 55, cy + 10],
            fill=self.colors.hair_highlight,
            width=4,
            smooth=True
        )

    def _draw_eyes(self, cx, cy):
        """Глаза"""
        eye_y = cy - 25
        spacing = 32

        # Смещение взгляда
        gaze_x = self.facial.eye_gaze_x * 3
        gaze_y = self.facial.eye_gaze_y * 2

        for side in [-1, 1]:
            ex = cx + side * spacing

            # Веки (верхнее)
            eye_height = 22 * max(0.1, self.facial.eye_openness)
            squint = self.facial.eye_squint * 5

            # Белок
            self._draw_oval(
                ex - 18, eye_y - eye_height + squint,
                ex + 18, eye_y + eye_height * 0.7,
                fill=self.colors.eye_white,
                outline=self.colors.eye_shadow
            )

            if self.facial.eye_openness > 0.2:
                # Радужка
                iris_y = eye_y + 3 + gaze_y
                iris_size = 10
                self._draw_oval(
                    ex - iris_size + gaze_x, iris_y - iris_size,
                    ex + iris_size + gaze_x, iris_y + iris_size,
                    fill=self.colors.iris,
                    outline=self.colors.iris_dark,
                    width=1
                )

                # Зрачок
                pupil_size = 4 + self.facial.pupil_size * 2
                self._draw_oval(
                    ex - pupil_size + gaze_x, iris_y - pupil_size,
                    ex + pupil_size + gaze_x, iris_y + pupil_size,
                    fill=self.colors.pupil,
                    outline=''
                )

                # Блики
                self._draw_oval(
                    ex - 4 + gaze_x, iris_y - 6,
                    ex + 2 + gaze_x, iris_y - 1,
                    fill='#ffffff',
                    outline=''
                )
                self._draw_oval(
                    ex + 3 + gaze_x, iris_y + 2,
                    ex + 6 + gaze_x, iris_y + 5,
                    fill='#ffffff',
                    outline=''
                )

            # Тень от верхнего века
            if self.facial.eye_openness < 0.9:
                self._draw_arc(
                    ex - 18, eye_y - eye_height - 5,
                    ex + 18, eye_y + 5,
                    start=0, extent=180,
                    fill=self.colors.skin_shadow,
                    outline=''
                )

    def _draw_eyebrows(self, cx, cy):
        """Брови"""
        brow_y = cy - 60 - self.facial.eyebrow_raise * 8
        furrow = self.facial.eyebrow_furrow * 10

        # Угол от эмоции
        curve = self.facial.mouth_curve * 3

        for side in [-1, 1]:
            ex = cx + side * 35
            asym = self.facial.eyebrow_asymmetry * side * 5

            # Точки брови
            if side == -1:  # Левая
                points = [
                    ex - 25, brow_y + furrow + asym,
                    ex - 10, brow_y - 3 + curve * side + asym,
                    ex + 5, brow_y - 2 + asym,
                ]
            else:  # Правая
                points = [
                    ex + 25, brow_y + furrow - asym,
                    ex + 10, brow_y - 3 + curve * side - asym,
                    ex - 5, brow_y - 2 - asym,
                ]

            self.canvas.create_line(
                points,
                fill=self.colors.hair_shadow,
                width=4,
                smooth=True,
                capstyle=tk.ROUND
            )

    def _draw_nose(self, cx, cy):
        """Нос"""
        nose_y = cy + 10

        # Морщинки от эмоции
        wrinkle = self.facial.nose_wrinkle * 5

        # Основной контур
        self.canvas.create_line(
            cx, nose_y - 15,
            cx - 4, nose_y + 8 - wrinkle,
            cx + 4, nose_y + 8 - wrinkle,
            fill=self.colors.skin_shadow,
            width=2,
            smooth=True
        )

        # Ноздри
        self._draw_oval(
            cx - 12, nose_y + 3, cx - 4, nose_y + 10,
            fill=self.colors.skin_shadow,
            outline=''
        )
        self._draw_oval(
            cx + 4, nose_y + 3, cx + 12, nose_y + 10,
            fill=self.colors.skin_shadow,
            outline=''
        )

    def _draw_mouth(self, cx, cy):
        """Рот"""
        mouth_y = cy + 40
        mouth_width = 30 + self.facial.mouth_width * 10
        curve = self.facial.mouth_curve * 12
        open_amount = self.facial.mouth_open * 12

        # Верхняя губа
        upper_points = [
            cx - mouth_width, mouth_y - curve,
            cx - 15, mouth_y - 3,
            cx, mouth_y - 5 + curve * 0.3,
            cx + 15, mouth_y - 3,
            cx + mouth_width, mouth_y - curve,
        ]
        self.canvas.create_line(
            upper_points,
            fill=self.colors.lip_base,
            width=4,
            smooth=True
        )

        # Нижняя губа
        if open_amount > 1:
            lower_points = [
                cx - mouth_width * 0.7, mouth_y + open_amount,
                cx, mouth_y + open_amount + 3,
                cx + mouth_width * 0.7, mouth_y + open_amount,
            ]
            self.canvas.create_line(
                lower_points,
                fill=self.colors.lip_base,
                width=4,
                smooth=True
            )

            # Открытый рот
            self._draw_oval(
                cx - 10, mouth_y,
                cx + 10, mouth_y + open_amount,
                fill=self.colors.lip_dark,
                outline=''
            )

            # Зубы (если широко открыт)
            if open_amount > 8:
                self._draw_rectangle(
                    cx - 8, mouth_y + 2, cx + 8, mouth_y + open_amount - 2,
                    fill='#ffffff',
                    outline=''
                )
        else:
            # Закрытый рот - просто линия
            lower_points = [
                cx - mouth_width * 0.8, mouth_y + 1,
                cx, mouth_y + 2 - curve * 0.2,
                cx + mouth_width * 0.8, mouth_y + 1,
            ]
            self.canvas.create_line(
                lower_points,
                fill=self.colors.lip_dark,
                width=3,
                smooth=True
            )

    def _draw_blush(self, cx, cy):
        """Румянец"""
        if self.facial.blush_intensity < 0.1:
            return

        blush_y = cy + 15
        intensity = min(1.0, self.facial.blush_intensity)

        # Левый
        self._draw_oval(
            cx - 60, blush_y - 5,
            cx - 30, blush_y + 20,
            fill=self.colors.blush,
            outline='',
            stipple='gray50' if intensity < 0.5 else ''
        )

        # Правый
        self._draw_oval(
            cx + 30, blush_y - 5,
            cx + 60, blush_y + 20,
            fill=self.colors.blush,
            outline='',
            stipple='gray50' if intensity < 0.5 else ''
        )

    def _draw_neck(self, cx, cy):
        """Шея"""
        self._draw_rectangle(
            cx - 30, cy + 75,
            cx + 30, cy + 140,
            fill=self.colors.skin_base,
            outline=''
        )

    def _draw_dress(self, cx, cy):
        """Платье"""
        # Основное тело платья
        points = [
            cx - 45, cy + 120,
            cx - 70, cy + 220,
            cx + 70, cy + 220,
            cx + 45, cy + 120,
        ]
        self.canvas.create_polygon(points, fill=self.colors.dress_base, smooth=True)

        # Вырез
        self._draw_arc(
            cx - 35, cy + 105,
            cx + 35, cy + 145,
            start=0, extent=180,
            fill=self.colors.skin_base,
            outline=''
        )

        # Тень
        self.canvas.create_line(
            cx - 45, cy + 130, cx - 55, cy + 200,
            fill=self.colors.dress_shadow,
            width=3
        )

    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ РИСОВАНИЯ ===

    def _draw_oval(self, x1, y1, x2, y2, **kwargs):
        """Рисование овала"""
        self.canvas.create_oval(x1, y1, x2, y2, **kwargs)

    def _draw_rectangle(self, x1, y1, x2, y2, **kwargs):
        """Рисование прямоугольника"""
        self.canvas.create_rectangle(x1, y1, x2, y2, **kwargs)

    def _draw_arc(self, x1, y1, x2, y2, **kwargs):
        """Рисование дуги"""
        self.canvas.create_arc(x1, y1, x2, y2, style=tk.CHORD, **kwargs)

    def _draw_curve(self, points, **kwargs):
        """Рисование кривой"""
        self.canvas.create_line(points, **kwargs)

    def set_text(self, text: str):
        """Установка текста"""
        if self.text_label:
            self.text_label.config(text=text)

    # === ЗАПУСК ===

    def _animation_loop(self):
        """Цикл анимации"""
        if not self.running:
            return

        self._update_animation(0.033)
        self.draw()
        self.root.after(33, self._animation_loop)  # ~30 FPS

    def start(self):
        """Запуск аватара"""
        self.create_window()
        self.running = True
        self._animation_loop()
        self.root.mainloop()

    def start_async(self) -> threading.Thread:
        """Асинхронный запуск"""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Остановка"""
        self.running = False
        if self.root:
            self.root.quit()


# === ФАБРИЧНАЯ ФУНКЦИЯ ===

def create_advanced_avatar(name: str = "Лиза", **kwargs) -> AdvancedAvatar:
    """Создание продвинутого аватара"""
    return AdvancedAvatar(name=name, **kwargs)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    avatar = create_advanced_avatar("Лиза")

    # Демо-цикл эмоций
    emotions = [
        (EmotionType.JOY, 1.0),
        (EmotionType.SADNESS, 1.0),
        (EmotionType.LOVE, 1.0),
        (EmotionType.SURPRISE, 1.0),
        (EmotionType.INTEREST, 1.0),
    ]
    idx = [0]

    def cycle_emotion():
        if avatar.running:
            emotion, intensity = emotions[idx[0] % len(emotions)]
            avatar.set_emotion(emotion, intensity)
            avatar.set_text(f"Эмоция: {emotion.value}")
            idx[0] += 1
            avatar.root.after(3000, cycle_emotion)

    avatar.on_click = lambda x, y: cycle_emotion()
    avatar.set_text("Нажмите на аватар для смены эмоции")

    # Отложенный старт демо
    threading.Timer(2, cycle_emotion).start()

    avatar.start()

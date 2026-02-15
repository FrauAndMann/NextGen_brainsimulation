"""
Live2D-style Avatar - Качественный 2D аватар с эмоциями и lip-sync

Особенности:
- Live2D-подобная анимация
- Реальный lip-sync по visemes
- Плавные переходы эмоций
- Дыхание и моргание
"""

import tkinter as tk
from tkinter import Canvas
import math
import time
import threading
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import re


class VisemeType(Enum):
    """Viseme типы для lip-sync"""
    SILENT = "silent"      # Закрыт
    A = "A"               # А, Я
    E = "E"               # Э, Е, И
    O = "O"               # О, Ё
    U = "U"               # У, Ю
    L = "L"               # Л
    W = "W"               # В, Ф
    M = "M"               # М, П, Б
    TH = "TH"             # Т, Д, Н, С, З
    CH = "CH"             # Ч, Ш, Щ, Ж


@dataclass
class AvatarExpression:
    """Выражение лица"""
    # Глаза
    eye_openness: float = 1.0      # 0-1, 0 = закрыты
    eye_happy: float = 0.0         # 0-1, улыбка глазами
    eye_sad: float = 0.0           # 0-1, грустные глаза
    eye_surprised: float = 0.0     # 0-1, удивлённые глаза
    eye_angry: float = 0.0         # 0-1, злые глаза

    # Брови
    brow_raise: float = 0.0        # -1 до 1, подняты/опущены
    brow_angry: float = 0.0        # 0-1, нахмурены

    # Рот
    mouth_open: float = 0.0        # 0-1, открыт
    mouth_smile: float = 0.0       # 0-1, улыбка
    mouth_sad: float = 0.0         # 0-1, грусть
    mouth_pout: float = 0.0        # 0-1, губы дудочкой

    # Щёки
    blush: float = 0.0             # 0-1, румянец

    # Голова
    head_tilt: float = 0.0         # -1 до 1, наклон влево/вправо
    head_turn: float = 0.0         # -1 до 1, поворот
    head_nod: float = 0.0          # -1 до 1, наклон вперёд/назад


class Live2DAvatar:
    """
    Live2D-подобный аватар с полной анимацией

    Рисует анимированного персонажа с:
    - Плавной анимацией выражений
    - Lip-sync в реальном времени
    - Дыханием и морганием
    """

    # Цвета
    COLORS = {
        'bg_top': '#1a1a2e',
        'bg_bottom': '#16213e',
        'skin': '#fce4d6',
        'skin_shadow': '#e8c8b0',
        'skin_dark': '#d4a88e',
        'hair_main': '#2d1f1a',
        'hair_highlight': '#4a3728',
        'hair_shine': '#5d4a3a',
        'eye_white': '#ffffff',
        'eye_iris': '#3d5a80',
        'eye_iris_light': '#5d7a9f',
        'eye_pupil': '#1a1a2a',
        'eye_highlight': '#ffffff',
        'lip': '#d4787c',
        'lip_dark': '#b85c60',
        'blush': '#ffb6b9',
        'dress': '#7b68ee',
        'dress_shadow': '#5a4db8',
        'eyebrow': '#2d1f1a',
    }

    def __init__(self, name: str = "Лиза"):
        self.name = name
        self.expression = AvatarExpression()
        self.target_expression = AvatarExpression()

        # Состояние анимации
        self.blink_timer = 0
        self.is_blinking = False
        self.breath_phase = 0
        self.speaking = False
        self.viseme = VisemeType.SILENT
        self.viseme_intensity = 0.0

        # Текст и lip-sync
        self.current_text = ""
        self.text_progress = 0
        self.text_speed = 50  # символов в секунду

        # Время
        self.start_time = time.time()
        self.last_update = time.time()

        # GUI
        self.root = None
        self.canvas = None
        self.width = 500
        self.height = 600
        self.running = False

        # Callbacks
        self.on_text_complete = None

    def create_window(self, title: str = None):
        """Создание окна"""
        self.root = tk.Tk()
        self.root.title(title or self.name)
        self.root.geometry(f"{self.width}x{self.height + 80}")
        self.root.configure(bg=self.COLORS['bg_top'])
        self.root.resizable(False, False)

        # Канвас
        self.canvas = Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=self.COLORS['bg_top'],
            highlightthickness=0
        )
        self.canvas.pack()

        # Текстовое поле
        self.text_frame = tk.Frame(self.root, bg=self.COLORS['bg_top'])
        self.text_frame.pack(fill=tk.X, padx=20, pady=10)

        self.name_label = tk.Label(
            self.text_frame,
            text=self.name,
            font=('Segoe UI', 12, 'bold'),
            fg='#f48fb1',
            bg=self.COLORS['bg_top']
        )
        self.name_label.pack(anchor='w')

        self.text_label = tk.Label(
            self.text_frame,
            text="...",
            font=('Segoe UI', 11),
            fg='#e0e0e0',
            bg=self.COLORS['bg_top'],
            wraplength=self.width - 40,
            justify=tk.LEFT
        )
        self.text_label.pack(anchor='w', fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.running = False
        if self.root:
            self.root.destroy()

    def set_emotion(self, valence: float, arousal: float, attachment: float):
        """Установка эмоционального состояния"""
        self.target_expression = AvatarExpression()

        # Валентность -> выражение
        if valence > 0.5:
            self.target_expression.mouth_smile = min(1.0, valence)
            self.target_expression.eye_happy = valence * 0.5
        elif valence < -0.3:
            self.target_expression.mouth_sad = min(1.0, -valence)
            self.target_expression.eye_sad = -valence * 0.5
            self.target_expression.brow_raise = valence * 0.3

        # Возбуждение -> глаза и брови
        if arousal > 0.6:
            self.target_expression.eye_openness = min(1.0, 0.6 + arousal * 0.4)
            self.target_expression.brow_raise = arousal * 0.3
        elif arousal < 0.3:
            self.target_expression.eye_openness = 0.4 + arousal

        # Привязанность -> румянец
        if attachment > 0.5:
            self.target_expression.blush = (attachment - 0.5) * 2

    def speak(self, text: str):
        """Начать говорить текст"""
        self.current_text = text
        self.text_progress = 0
        self.speaking = True
        self.text_label.config(text="")

    def _update_animation(self, dt: float):
        """Обновление анимации"""
        now = time.time()

        # Дыхание
        self.breath_phase = (now - self.start_time) * 2

        # Моргание
        if not self.is_blinking:
            if now - self.blink_timer > 2.5 + (hash(str(now)) % 100) / 50:
                self.is_blinking = True
                self.blink_timer = now
        else:
            if now - self.blink_timer > 0.12:
                self.is_blinking = False

        # Плавная интерполяция к целевому выражению
        lerp_speed = 5.0 * dt

        self.expression.eye_openness = self._lerp(
            self.expression.eye_openness,
            0.1 if self.is_blinking else self.target_expression.eye_openness,
            lerp_speed * 3
        )
        self.expression.mouth_smile = self._lerp(
            self.expression.mouth_smile,
            self.target_expression.mouth_smile,
            lerp_speed
        )
        self.expression.mouth_sad = self._lerp(
            self.expression.mouth_sad,
            self.target_expression.mouth_sad,
            lerp_speed
        )
        self.expression.blush = self._lerp(
            self.expression.blush,
            self.target_expression.blush,
            lerp_speed
        )
        self.expression.brow_raise = self._lerp(
            self.expression.brow_raise,
            self.target_expression.brow_raise,
            lerp_speed
        )

        # Lip-sync при говорении
        if self.speaking and self.current_text:
            chars_to_show = int(self.text_progress)
            if chars_to_show < len(self.current_text):
                self.text_progress += self.text_speed * dt

                # Определяем viseme по текущему символу
                current_char = self.current_text[chars_to_show].lower() if chars_to_show < len(self.current_text) else ''
                self.viseme = self._char_to_viseme(current_char)
                self.viseme_intensity = 0.7 + 0.3 * math.sin(now * 20)

                # Обновляем текст
                visible_text = self.current_text[:chars_to_show + 1]
                self.text_label.config(text=visible_text)
            else:
                self.speaking = False
                self.viseme = VisemeType.SILENT
                self.viseme_intensity = 0
                self.text_label.config(text=self.current_text)
                if self.on_text_complete:
                    self.on_text_complete()

        # Затухание viseme
        if not self.speaking:
            self.viseme_intensity = max(0, self.viseme_intensity - dt * 5)

    def _lerp(self, a: float, b: float, t: float) -> float:
        """Линейная интерполяция"""
        return a + (b - a) * min(1.0, t)

    def _char_to_viseme(self, char: str) -> VisemeType:
        """Преобразование символа в viseme"""
        viseme_map = {
            'а': VisemeType.A, 'я': VisemeType.A,
            'э': VisemeType.E, 'е': VisemeType.E, 'и': VisemeType.E, 'ы': VisemeType.E,
            'о': VisemeType.O, 'ё': VisemeType.O,
            'у': VisemeType.U, 'ю': VisemeType.U,
            'л': VisemeType.L,
            'в': VisemeType.W, 'ф': VisemeType.W,
            'м': VisemeType.M, 'п': VisemeType.M, 'б': VisemeType.M,
            'т': VisemeType.TH, 'д': VisemeType.TH, 'н': VisemeType.TH,
            'с': VisemeType.TH, 'з': VisemeType.TH, 'ц': VisemeType.TH,
            'ч': VisemeType.CH, 'ш': VisemeType.CH, 'щ': VisemeType.CH, 'ж': VisemeType.CH,
            ' ': VisemeType.SILENT,
        }
        return viseme_map.get(char, VisemeType.TH)

    def draw(self):
        """Отрисовка аватара"""
        self.canvas.delete("all")

        # Градиентный фон
        self._draw_background()

        # Координаты центра
        cx, cy = self.width // 2, self.height // 2 - 30

        # Смещение от дыхания
        breath_offset = math.sin(self.breath_phase) * 3

        # Лёгкий наклон головы
        tilt = math.sin(self.breath_phase * 0.3) * 2

        # Рисуем части
        self._draw_hair_back(cx, cy + breath_offset, tilt)
        self._draw_body(cx, cy + breath_offset)
        self._draw_neck(cx, cy + breath_offset)
        self._draw_face(cx, cy + breath_offset, tilt)
        self._draw_hair_front(cx, cy + breath_offset, tilt)
        self._draw_eyes(cx, cy + breath_offset, tilt)
        self._draw_eyebrows(cx, cy + breath_offset, tilt)
        self._draw_nose(cx, cy + breath_offset)
        self._draw_mouth(cx, cy + breath_offset, tilt)
        self._draw_blush(cx, cy + breath_offset)

    def _draw_background(self):
        """Градиентный фон"""
        steps = 20
        for i in range(steps):
            y1 = i * self.height // steps
            y2 = (i + 1) * self.height // steps
            # Интерполяция цвета
            r1, g1, b1 = int(self.COLORS['bg_top'][1:3], 16), int(self.COLORS['bg_top'][3:5], 16), int(self.COLORS['bg_top'][5:7], 16)
            r2, g2, b2 = int(self.COLORS['bg_bottom'][1:3], 16), int(self.COLORS['bg_bottom'][3:5], 16), int(self.COLORS['bg_bottom'][5:7], 16)
            t = i / steps
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.canvas.create_rectangle(0, y1, self.width, y2, fill=color, outline='')

    def _draw_face(self, cx, cy, tilt):
        """Лицо"""
        # Основа лица - овал
        self.canvas.create_oval(
            cx - 75, cy - 90,
            cx + 75, cy + 80,
            fill=self.COLORS['skin'],
            outline=self.COLORS['skin_shadow'],
            width=2
        )

        # Тень под подбородком
        self.canvas.create_arc(
            cx - 60, cy + 40,
            cx + 60, cy + 90,
            start=200, extent=140,
            fill=self.COLORS['skin_shadow'],
            outline=''
        )

    def _draw_hair_back(self, cx, cy, tilt):
        """Волосы сзади"""
        # Длинные волосы
        points = []
        for i in range(20):
            angle = math.pi + (i / 19) * math.pi
            r = 95 + 20 * math.sin(angle * 2)
            x = cx + r * math.cos(angle) + tilt * 0.3
            y = cy + r * math.sin(angle) * 0.8 - 20
            points.extend([x, y])

        self.canvas.create_polygon(points, fill=self.COLORS['hair_main'], smooth=True)

    def _draw_hair_front(self, cx, cy, tilt):
        """Волосы спереди (чёлка)"""
        # Основа чёлки
        points = [
            cx - 80 + tilt, cy - 50,
            cx - 65 + tilt * 0.8, cy - 95,
            cx - 30 + tilt * 0.3, cy - 108,
            cx + tilt * 0.1, cy - 112,
            cx + 30 - tilt * 0.3, cy - 108,
            cx + 65 - tilt * 0.8, cy - 95,
            cx + 80 - tilt, cy - 50,
            cx + 55 - tilt * 0.5, cy - 65,
            cx + 25 - tilt * 0.2, cy - 75,
            cx - tilt * 0.1, cy - 78,
            cx - 25 + tilt * 0.2, cy - 75,
            cx - 55 + tilt * 0.5, cy - 65,
        ]
        self.canvas.create_polygon(points, fill=self.COLORS['hair_main'], smooth=True)

        # Блики на волосах
        self.canvas.create_line(
            cx - 40 + tilt * 0.3, cy - 85,
            cx - 20 + tilt * 0.2, cy - 95,
            cx - 10 + tilt * 0.1, cy - 85,
            fill=self.COLORS['hair_highlight'],
            width=8,
            smooth=True
        )

    def _draw_eyes(self, cx, cy, tilt):
        """Глаза с эмоциями"""
        eye_y = cy - 20
        eye_spacing = 30

        open_mult = self.expression.eye_openness
        happy_mult = self.expression.eye_happy

        for side in [-1, 1]:
            ex = cx + side * eye_spacing + tilt * 0.5

            # Белок
            eye_height = 22 * open_mult

            if happy_mult > 0.2 and open_mult > 0.5:
                # Счастливые глаза (изогнутые)
                self.canvas.create_arc(
                    ex - 18, eye_y - 12,
                    ex + 18, eye_y + 12,
                    start=0, extent=180,
                    fill='white',
                    outline=self.COLORS['eye_white']
                )
            else:
                # Обычные глаза
                self.canvas.create_oval(
                    ex - 16, eye_y - eye_height,
                    ex + 16, eye_y + eye_height * 0.7,
                    fill='white',
                    outline='#c0c0c0'
                )

            if open_mult > 0.25:
                # Радужка
                iris_y = eye_y + 2
                iris_size = 10

                self.canvas.create_oval(
                    ex - iris_size, iris_y - iris_size,
                    ex + iris_size, iris_y + iris_size,
                    fill=self.COLORS['eye_iris'],
                    outline=''
                )

                # Зрачок (сужается при возбуждении)
                pupil_size = 5
                self.canvas.create_oval(
                    ex - pupil_size, iris_y - pupil_size,
                    ex + pupil_size, iris_y + pupil_size,
                    fill=self.COLORS['eye_pupil'],
                    outline=''
                )

                # Блики
                self.canvas.create_oval(
                    ex - 4, iris_y - 6,
                    ex + 2, iris_y - 1,
                    fill='white',
                    outline=''
                )
                self.canvas.create_oval(
                    ex + 3, iris_y + 2,
                    ex + 6, iris_y + 5,
                    fill='white',
                    outline=''
                )

            # Ресницы
            if open_mult > 0.5:
                for i in range(5):
                    angle = -30 + i * 15
                    rad = math.radians(angle)
                    lx = ex + 16 * math.cos(rad)
                    ly = eye_y - eye_height * math.sin(rad)
                    self.canvas.create_line(
                        lx, ly,
                        lx + 4 * math.cos(rad), ly - 5,
                        fill=self.COLORS['hair_main'],
                        width=2
                    )

    def _draw_eyebrows(self, cx, cy, tilt):
        """Брови"""
        brow_y = cy - 55 - self.expression.brow_raise * 8

        # Форма бровей зависит от эмоции
        curve = self.expression.mouth_smile * 3 - self.expression.mouth_sad * 3

        for side in [-1, 1]:
            bx = cx + side * 32 + tilt * 0.3

            # Левая бровь (side=-1) или правая (side=1)
            points = [
                bx - side * 18, brow_y + curve * side,
                bx - side * 5, brow_y - 4,
                bx + side * 10, brow_y - 2 + curve * -side,
            ]

            self.canvas.create_line(
                points,
                fill=self.COLORS['eyebrow'],
                width=4,
                smooth=True
            )

    def _draw_nose(self, cx, cy):
        """Нос"""
        nose_y = cy + 8

        # Простая линия носа
        self.canvas.create_line(
            cx, nose_y - 12,
            cx - 4, nose_y + 8,
            cx + 4, nose_y + 8,
            fill=self.COLORS['skin_dark'],
            width=2,
            smooth=True
        )

        # Ноздри
        self.canvas.create_arc(
            cx - 8, nose_y + 4,
            cx + 8, nose_y + 12,
            start=0, extent=180,
            fill=self.COLORS['skin_shadow'],
            outline=''
        )

    def _draw_mouth(self, cx, cy, tilt):
        """Рот с lip-sync"""
        mouth_y = cy + 40

        # Открытие рта от viseme
        viseme_open = {
            VisemeType.SILENT: 0.0,
            VisemeType.A: 0.8,
            VisemeType.E: 0.4,
            VisemeType.O: 0.7,
            VisemeType.U: 0.5,
            VisemeType.L: 0.3,
            VisemeType.W: 0.2,
            VisemeType.M: 0.0,
            VisemeType.TH: 0.35,
            VisemeType.CH: 0.25,
        }

        target_open = viseme_open.get(self.viseme, 0.3) * self.viseme_intensity
        mouth_open = target_open * 12

        # Ширина и форма от эмоции
        mouth_width = 28 + self.expression.mouth_smile * 10
        smile_curve = self.expression.mouth_smile * 12
        sad_curve = self.expression.mouth_sad * 8

        if self.expression.mouth_smile > 0.3:
            # Улыбка
            self.canvas.create_line(
                cx - mouth_width, mouth_y + smile_curve,
                cx - 10, mouth_y - 3,
                cx + 10, mouth_y - 3,
                cx + mouth_width, mouth_y + smile_curve,
                fill=self.COLORS['lip'],
                width=4,
                smooth=True
            )

            if mouth_open > 3:
                # Открытая улыбка
                self.canvas.create_line(
                    cx - mouth_width * 0.6, mouth_y + mouth_open,
                    cx, mouth_y + mouth_open + 3,
                    cx + mouth_width * 0.6, mouth_y + mouth_open,
                    fill=self.COLORS['lip'],
                    width=4,
                    smooth=True
                )

                # Зубы
                self.canvas.create_rectangle(
                    cx - 12, mouth_y + 2,
                    cx + 12, mouth_y + mouth_open - 2,
                    fill='white',
                    outline=''
                )

        elif self.expression.mouth_sad > 0.3:
            # Грусть
            self.canvas.create_line(
                cx - mouth_width * 0.8, mouth_y - sad_curve,
                cx, mouth_y + 2,
                cx + mouth_width * 0.8, mouth_y - sad_curve,
                fill=self.COLORS['lip'],
                width=3,
                smooth=True
            )
        else:
            # Нейтральный рот
            if mouth_open > 2:
                # Открыт
                # Верхняя губа
                self.canvas.create_line(
                    cx - 20, mouth_y - 2,
                    cx - 8, mouth_y - 4,
                    cx + 8, mouth_y - 4,
                    cx + 20, mouth_y - 2,
                    fill=self.COLORS['lip'],
                    width=3,
                    smooth=True
                )

                # Нижняя губа
                self.canvas.create_line(
                    cx - 18, mouth_y + mouth_open,
                    cx, mouth_y + mouth_open + 4,
                    cx + 18, mouth_y + mouth_open,
                    fill=self.COLORS['lip'],
                    width=3,
                    smooth=True
                )

                # Внутренность рта
                self.canvas.create_oval(
                    cx - 10, mouth_y,
                    cx + 10, mouth_y + mouth_open,
                    fill='#8b4557',
                    outline=''
                )
            else:
                # Закрыт
                self.canvas.create_line(
                    cx - 18, mouth_y,
                    cx - 6, mouth_y - 2,
                    cx + 6, mouth_y - 2,
                    cx + 18, mouth_y,
                    fill=self.COLORS['lip'],
                    width=3,
                    smooth=True
                )

    def _draw_blush(self, cx, cy):
        """Румянец"""
        if self.expression.blush > 0.1:
            alpha = min(1, self.expression.blush)

            for side in [-1, 1]:
                bx = cx + side * 50
                by = cy + 15

                # Создаём эффект румянца несколькими овалами
                for i in range(3):
                    size = 15 + i * 5
                    self.canvas.create_oval(
                        bx - size, by - size * 0.6,
                        bx + size, by + size * 0.6,
                        fill=self.COLORS['blush'],
                        outline='',
                        stipple='gray50' if alpha < 0.5 else ''
                    )

    def _draw_neck(self, cx, cy):
        """Шея"""
        self.canvas.create_rectangle(
            cx - 28, cy + 70,
            cx + 28, cy + 130,
            fill=self.COLORS['skin'],
            outline=''
        )

        # Тень на шее
        self.canvas.create_polygon(
            cx - 28, cy + 70,
            cx + 28, cy + 70,
            cx + 20, cy + 90,
            cx - 20, cy + 90,
            fill=self.COLORS['skin_shadow'],
            smooth=True
        )

    def _draw_body(self, cx, cy):
        """Тело/одежда"""
        # Платье
        points = [
            cx - 45, cy + 115,
            cx - 80, cy + 250,
            cx + 80, cy + 250,
            cx + 45, cy + 115,
        ]
        self.canvas.create_polygon(points, fill=self.COLORS['dress'], smooth=True)

        # Тень на платье
        self.canvas.create_polygon(
            cx + 10, cy + 120,
            cx + 70, cy + 250,
            cx + 80, cy + 250,
            cx + 45, cy + 115,
            fill=self.COLORS['dress_shadow'],
            smooth=True
        )

        # Вырез
        self.canvas.create_arc(
            cx - 35, cy + 95,
            cx + 35, cy + 145,
            start=0, extent=180,
            fill=self.COLORS['skin'],
            outline=''
        )

    def animation_loop(self):
        """Цикл анимации"""
        if not self.running:
            return

        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        self._update_animation(dt)
        self.draw()

        self.root.after(33, self.animation_loop)  # ~30 FPS

    def start(self):
        """Запуск аватара"""
        self.create_window()
        self.running = True
        self.last_update = time.time()
        self.animation_loop()
        self.root.mainloop()

    def start_async(self) -> threading.Thread:
        """Асинхронный запуск"""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def set_text(self, text: str):
        """Установка текста без анимации"""
        self.text_label.config(text=text)


# Утилиты для конвертации эмоций
def emotion_to_expression(emotion: str, intensity: float = 0.5) -> AvatarExpression:
    """Конвертация имени эмоции в выражение"""
    expr = AvatarExpression()

    emotion_map = {
        'happy': lambda e, i: setattr(e, 'mouth_smile', i),
        'sad': lambda e, i: (setattr(e, 'mouth_sad', i), setattr(e, 'eye_sad', i * 0.5)),
        'angry': lambda e, i: (setattr(e, 'brow_angry', i), setattr(e, 'eye_angry', i * 0.3)),
        'surprised': lambda e, i: (setattr(e, 'eye_surprised', i), setattr(e, 'eye_openness', 1.0)),
        'love': lambda e, i: (setattr(e, 'blush', i), setattr(e, 'mouth_smile', i * 0.5)),
        'neutral': lambda e, i: None,
    }

    if emotion in emotion_map:
        emotion_map[emotion](expr, intensity)

    return expr

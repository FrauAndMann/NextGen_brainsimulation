"""
Avatar Module - Визуальный аватар для ANIMA

Красивый анимированный аватар с эмоциями:
- Выражение лица меняется по эмоциям
- Анимация дыхания
- Реакция на речь
"""

import tkinter as tk
from tkinter import ttk, font
from typing import Optional, Tuple
import math
import time
import threading


class AvatarState:
    """Состояние аватара"""
    def __init__(self):
        # Эмоциональное состояние
        self.valence: float = 0.0      # -1 до 1
        self.arousal: float = 0.3      # 0 до 1
        self.attachment: float = 0.5   # 0 до 1

        # Анимация
        self.blink_timer: float = 0
        self.is_blinking: bool = False
        self.is_speaking: bool = False
        self.speak_amplitude: float = 0

        # Визуальные параметры
        self.eye_openness: float = 1.0
        self.mouth_openness: float = 0.0
        self.eyebrow_raise: float = 0.0


class AvatarGUI:
    """
    Графический интерфейс аватара

    Рисует анимированного персонажа с эмоциями.
    """

    # Цвета
    COLORS = {
        'bg': '#1a1a2e',
        'skin': '#fce4d6',
        'skin_shadow': '#e8c8b0',
        'hair': '#4a3728',
        'hair_highlight': '#5d4a3a',
        'eye': '#3d5a80',
        'eye_highlight': '#ffffff',
        'lip': '#d4787c',
        'blush': '#ffb6b9',
        'dress': '#7b68ee',
    }

    def __init__(self, name: str = "Лиза"):
        self.name = name
        self.state = AvatarState()
        self.running = False

        # Окно
        self.root = None
        self.canvas = None

        # Размеры
        self.width = 400
        self.height = 500

        # Время для анимации
        self.start_time = time.time()

    def create_window(self):
        """Создание окна"""
        self.root = tk.Tk()
        self.root.title(self.name)
        self.root.geometry(f"{self.width}x{self.height + 100}")
        self.root.configure(bg=self.COLORS['bg'])
        self.root.resizable(False, False)

        # Канвас для аватара
        self.canvas = tk.Canvas(
            self.root,
            width=self.width,
            height=self.height,
            bg=self.COLORS['bg'],
            highlightthickness=0
        )
        self.canvas.pack()

        # Поле для текста
        self.text_frame = tk.Frame(self.root, bg=self.COLORS['bg'])
        self.text_frame.pack(fill=tk.X, padx=10, pady=5)

        self.name_label = tk.Label(
            self.text_frame,
            text=self.name,
            font=('Segoe UI', 12, 'bold'),
            fg='#ffffff',
            bg=self.COLORS['bg']
        )
        self.name_label.pack(anchor='w')

        self.text_label = tk.Label(
            self.text_frame,
            text="...",
            font=('Segoe UI', 11),
            fg='#e0e0e0',
            bg=self.COLORS['bg'],
            wraplength=self.width - 20,
            justify=tk.LEFT
        )
        self.text_label.pack(anchor='w', fill=tk.X)

        # Привязка закрытия
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        """Обработка закрытия"""
        self.running = False
        self.root.destroy()

    def update_state(self, valence: float = None, arousal: float = None,
                     attachment: float = None, speaking: bool = None):
        """Обновление состояния"""
        if valence is not None:
            self.state.valence = max(-1, min(1, valence))
        if arousal is not None:
            self.state.arousal = max(0, min(1, arousal))
        if attachment is not None:
            self.state.attachment = max(0, min(1, attachment))
        if speaking is not None:
            self.state.is_speaking = speaking

    def set_text(self, text: str):
        """Установка текста"""
        if self.text_label:
            self.text_label.config(text=text)

    def draw_avatar(self):
        """Отрисовка аватара"""
        self.canvas.delete("all")

        cx, cy = self.width // 2, self.height // 2 - 20

        # Анимация дыхания
        breath_offset = math.sin((time.time() - self.start_time) * 2) * 3

        # Моргание
        if time.time() - self.state.blink_timer > 3 + hash(time.time()) % 3:
            self.state.is_blinking = True
            self.state.blink_timer = time.time()
        elif time.time() - self.state.blink_timer > 0.15:
            self.state.is_blinking = False

        if self.state.is_blinking:
            self.state.eye_openness = 0.1
        else:
            self.state.eye_openness = 1.0

        # Рот при говорении
        if self.state.is_speaking:
            self.state.speak_amplitude = 0.3 + 0.2 * math.sin(time.time() * 15)
        else:
            self.state.speak_amplitude = max(0, self.state.speak_amplitude - 0.1)

        # Брови от эмоции
        self.state.eyebrow_raise = (self.state.valence + 0.5) * 5

        # === РИСОВАНИЕ ===

        # Волосы (сзади)
        self._draw_hair_back(cx, cy + breath_offset)

        # Шея
        self._draw_neck(cx, cy + breath_offset)

        # Платье
        self._draw_dress(cx, cy + breath_offset)

        # Лицо
        self._draw_face(cx, cy + breath_offset)

        # Волосы (спереди)
        self._draw_hair_front(cx, cy + breath_offset)

        # Глаза
        self._draw_eyes(cx, cy + breath_offset)

        # Брови
        self._draw_eyebrows(cx, cy + breath_offset)

        # Нос
        self._draw_nose(cx, cy + breath_offset)

        # Рот
        self._draw_mouth(cx, cy + breath_offset)

        # Румянец
        if self.state.valence > 0.2:
            self._draw_blush(cx, cy + breath_offset)

    def _draw_face(self, cx, cy):
        """Рисование лица"""
        # Основа лица (овал)
        self.canvas.create_oval(
            cx - 70, cy - 85, cx + 70, cy + 75,
            fill=self.COLORS['skin'],
            outline=self.COLORS['skin_shadow'],
            width=2
        )

    def _draw_hair_back(self, cx, cy):
        """Волосы сзади"""
        # Длинные волосы
        points = [
            cx - 80, cy - 40,
            cx - 90, cy + 60,
            cx - 70, cy + 120,
            cx - 40, cy + 150,
            cx, cy + 160,
            cx + 40, cy + 150,
            cx + 70, cy + 120,
            cx + 90, cy + 60,
            cx + 80, cy - 40,
        ]
        self.canvas.create_polygon(points, fill=self.COLORS['hair'], smooth=True)

    def _draw_hair_front(self, cx, cy):
        """Волосы спереди (чёлка)"""
        # Чёлка
        points = [
            cx - 75, cy - 45,
            cx - 60, cy - 80,
            cx - 30, cy - 95,
            cx, cy - 100,
            cx + 30, cy - 95,
            cx + 60, cy - 80,
            cx + 75, cy - 45,
            cx + 50, cy - 60,
            cx + 20, cy - 70,
            cx - 20, cy - 70,
            cx - 50, cy - 60,
        ]
        self.canvas.create_polygon(points, fill=self.COLORS['hair'], smooth=True)

    def _draw_eyes(self, cx, cy):
        """Глаза"""
        eye_y = cy - 20
        eye_spacing = 28

        for side in [-1, 1]:
            ex = cx + side * eye_spacing

            # Белок глаза
            eye_height = 20 * self.state.eye_openness
            self.canvas.create_oval(
                ex - 15, eye_y - eye_height,
                ex + 15, eye_y + eye_height * 0.8,
                fill='white',
                outline='#d0d0d0'
            )

            if self.state.eye_openness > 0.3:
                # Радужка
                iris_y = eye_y + 2
                self.canvas.create_oval(
                    ex - 8, iris_y - 8,
                    ex + 8, iris_y + 8,
                    fill=self.COLORS['eye'],
                    outline=''
                )

                # Зрачок
                pupil_size = 4 - self.state.arousal * 2  # Сужается при возбуждении
                self.canvas.create_oval(
                    ex - pupil_size, iris_y - pupil_size,
                    ex + pupil_size, iris_y + pupil_size,
                    fill='#000000',
                    outline=''
                )

                # Блик
                self.canvas.create_oval(
                    ex - 3, iris_y - 5,
                    ex + 2, iris_y - 1,
                    fill='white',
                    outline=''
                )

    def _draw_eyebrows(self, cx, cy):
        """Брови"""
        brow_y = cy - 50 - self.state.eyebrow_raise

        # Левая бровь
        brow_angle = self.state.valence * 5  # Поднимается при радости
        self.canvas.create_line(
            cx - 45, brow_y + brow_angle,
            cx - 15, brow_y - 3,
            cx - 10, brow_y - 2,
            fill=self.COLORS['hair'],
            width=3,
            smooth=True
        )

        # Правая бровь
        self.canvas.create_line(
            cx + 45, brow_y + brow_angle,
            cx + 15, brow_y - 3,
            cx + 10, brow_y - 2,
            fill=self.COLORS['hair'],
            width=3,
            smooth=True
        )

    def _draw_nose(self, cx, cy):
        """Нос"""
        nose_y = cy + 5
        # Простой нос
        self.canvas.create_line(
            cx, nose_y - 10,
            cx - 3, nose_y + 8,
            cx + 3, nose_y + 8,
            fill=self.COLORS['skin_shadow'],
            width=2,
            smooth=True
        )

    def _draw_mouth(self, cx, cy):
        """Рот"""
        mouth_y = cy + 35
        mouth_width = 25 + self.state.attachment * 5  # Шире при привязанности

        # Открытие рта
        mouth_open = self.state.speak_amplitude * 8

        # Изгиб губ от эмоции
        curve = self.state.valence * 8

        # Верхняя губа
        self.canvas.create_line(
            cx - mouth_width, mouth_y - curve,
            cx - 10, mouth_y - 2,
            cx + 10, mouth_y - 2,
            cx + mouth_width, mouth_y - curve,
            fill=self.COLORS['lip'],
            width=3,
            smooth=True
        )

        # Нижняя губа (если открыт рот)
        if mouth_open > 1:
            self.canvas.create_line(
                cx - mouth_width * 0.7, mouth_y + mouth_open,
                cx, mouth_y + mouth_open + 2,
                cx + mouth_width * 0.7, mouth_y + mouth_open,
                fill=self.COLORS['lip'],
                width=3,
                smooth=True
            )

            # Рот внутри (темный)
            self.canvas.create_oval(
                cx - 8, mouth_y,
                cx + 8, mouth_y + mouth_open,
                fill='#8b4557',
                outline=''
            )

    def _draw_blush(self, cx, cy):
        """Румянец"""
        blush_intensity = min(1, self.state.valence)

        # Левый румянец
        self.canvas.create_oval(
            cx - 55, cy + 10,
            cx - 30, cy + 30,
            fill=self.COLORS['blush'],
            outline='',
            stipple='gray50' if blush_intensity < 0.5 else ''
        )

        # Правый румянец
        self.canvas.create_oval(
            cx + 30, cy + 10,
            cx + 55, cy + 30,
            fill=self.COLORS['blush'],
            outline='',
            stipple='gray50' if blush_intensity < 0.5 else ''
        )

    def _draw_neck(self, cx, cy):
        """Шея"""
        self.canvas.create_rectangle(
            cx - 25, cy + 65,
            cx + 25, cy + 120,
            fill=self.COLORS['skin'],
            outline=''
        )

    def _draw_dress(self, cx, cy):
        """Платье/одежда"""
        # Простое платье
        points = [
            cx - 40, cy + 100,
            cx - 60, cy + 200,
            cx + 60, cy + 200,
            cx + 40, cy + 100,
        ]
        self.canvas.create_polygon(points, fill=self.COLORS['dress'], smooth=True)

        # Вырез
        self.canvas.create_arc(
            cx - 30, cy + 90,
            cx + 30, cy + 130,
            start=0, extent=180,
            fill=self.COLORS['skin'],
            outline=''
        )

    def animation_loop(self):
        """Цикл анимации"""
        if not self.running:
            return

        self.draw_avatar()
        self.root.after(33, self.animation_loop)  # ~30 FPS

    def start(self):
        """Запуск аватара"""
        self.create_window()
        self.running = True
        self.animation_loop()
        self.root.mainloop()

    def start_async(self):
        """Асинхронный запуск в отдельном потоке"""
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread


def create_avatar_window(name: str = "Лиза", on_close=None) -> AvatarGUI:
    """Создание окна аватара"""
    avatar = AvatarGUI(name)
    return avatar

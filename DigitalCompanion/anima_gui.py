"""
ANIMA Full GUI - Полноценный интерфейс с аватаром

Красивое окно с:
- Анимированным аватаром
- Распознаванием речи
- Вебкамерой для эмоций
- Чистым интерфейсом без технических деталей
"""

import sys
import os
import time
import threading
import queue

# Фикс кодировки Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ANIMA Core
from core.anima import AnimaSystem, AnimaConfig
from core.llm_effector import LLMEffector, LLMConfig
from core.affective_prompting import create_asp
from core.will_engine import INTENT_REGISTRY

# Avatar
from avatar.gui_avatar import AvatarGUI


@dataclass
class SensorData:
    """Данные от сенсоров"""
    text: str = ""
    voice_valence: float = 0.0
    voice_arousal: float = 0.3
    face_valence: float = 0.0
    face_arousal: float = 0.3
    face_detected: bool = False
    timestamp: float = 0.0


class AnimaFullGUI:
    """
    Полноценный GUI для ANIMA

    Объединяет:
    - Аватар с эмоциями
    - Распознавание речи
    - Детекцию эмоций с камеры
    - Красивый чат-интерфейс
    """

    def __init__(self, name: str = "Лиза"):
        self.name = name
        self.running = False

        # ANIMA система
        self.anima = AnimaSystem(AnimaConfig(name=name))
        self.llm = LLMEffector(LLMConfig())

        # Сенсоры (опционально)
        self.stt = None  # Speech to text
        self.vision = None  # Камера

        # Аватар
        self.avatar = None
        self.avatar_thread = None

        # Состояние
        self.sensor_data = SensorData()
        self.message_queue = queue.Queue()

        # Контекст разговора
        self.conversation_context = []
        self.max_context = 20

        # GUI
        self.root = None
        self.chat_display = None
        self.input_field = None
        self.status_label = None

        # Флаг говорения
        self.is_speaking = False

    def _init_sensors(self):
        """Инициализация сенсоров"""
        # STT
        try:
            from sensors.speech import SpeechToText, STTProvider, check_stt_availability
            stt_status = check_stt_availability()
            if stt_status['recommended']:
                provider = STTProvider.WHISPER if stt_status['whisper'] else STTProvider.GOOGLE
                self.stt = SpeechToText(provider)
                print(f"[Sensor] STT: {stt_status['recommended']}")
            else:
                print("[Sensor] STT недоступен")
        except Exception as e:
            print(f"[Sensor] Ошибка STT: {e}")

        # Vision
        try:
            from sensors.vision import VisionSensor, check_vision_availability
            vision_status = check_vision_availability()
            if vision_status['camera']:
                self.vision = VisionSensor()
                print(f"[Sensor] Vision: камера {'+' if vision_status['camera'] else '-'}")
            else:
                print("[Sensor] Камера недоступна")
        except Exception as e:
            print(f"[Sensor] Ошибка Vision: {e}")

    def create_gui(self):
        """Создание GUI"""
        self.root = tk.Tk()
        self.root.title(f"ANIMA - {self.name}")
        self.root.geometry("900x700")
        self.root.configure(bg='#1a1a2e')

        # Стили
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='white')
        style.configure('TButton', padding=10)

        # Главный контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Левая панель - аватар
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Кнопки управления
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.voice_btn = ttk.Button(
            control_frame,
            text="🎤 Говорить",
            command=self._on_voice_button
        )
        self.voice_btn.pack(side=tk.LEFT, padx=5)

        self.camera_btn = ttk.Button(
            control_frame,
            text="📷 Камера",
            command=self._toggle_camera
        )
        self.camera_btn.pack(side=tk.LEFT, padx=5)

        # Статус
        self.status_label = ttk.Label(
            left_frame,
            text="● Готова к общению",
            font=('Segoe UI', 10)
        )
        self.status_label.pack(pady=5)

        # Текст аватара (большой)
        self.avatar_text = tk.Label(
            left_frame,
            text="...",
            font=('Segoe UI', 14),
            bg='#1a1a2e',
            fg='#e0e0e0',
            wraplength=350,
            justify=tk.LEFT
        )
        self.avatar_text.pack(pady=20, fill=tk.X)

        # Правая панель - чат
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Заголовок
        title_label = ttk.Label(
            right_frame,
            text=f"💬 Чат с {self.name}",
            font=('Segoe UI', 14, 'bold')
        )
        title_label.pack(anchor='w', pady=(0, 10))

        # История чата
        chat_frame = ttk.Frame(right_frame)
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            bg='#16213e',
            fg='#e0e0e0',
            insertbackground='white',
            selectbackground='#4a69bd',
            relief=tk.FLAT,
            padx=15,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Теги для стилей сообщений
        self.chat_display.tag_configure('user', foreground='#4fc3f7', font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('anima', foreground='#f48fb1', font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('text', foreground='#e0e0e0')

        # Поле ввода
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=(10, 0))

        self.input_field = ttk.Entry(
            input_frame,
            font=('Segoe UI', 11)
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_field.bind('<Return>', self._on_send_message)

        send_btn = ttk.Button(
            input_frame,
            text="Отправить",
            command=self._on_send_message
        )
        send_btn.pack(side=tk.RIGHT)

        # Приветственное сообщение
        self._add_chat_message(self.name, "Привет! Я тебя слушаю 😊")

        # Обработчик закрытия
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _add_chat_message(self, sender: str, text: str):
        """Добавление сообщения в чат"""
        self.chat_display.config(state=tk.NORMAL)

        # Имя отправителя
        tag = 'user' if sender != self.name else 'anima'
        self.chat_display.insert(tk.END, f"{sender}: ", tag)

        # Текст сообщения
        self.chat_display.insert(tk.END, f"{text}\n\n", 'text')

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _on_send_message(self, event=None):
        """Обработка отправки сообщения"""
        text = self.input_field.get().strip()
        if not text:
            return

        self.input_field.delete(0, tk.END)

        # Добавляем в чат
        self._add_chat_message("Ты", text)

        # Обрабатываем в отдельном потоке
        threading.Thread(
            target=self._process_message,
            args=(text,),
            daemon=True
        ).start()

    def _process_message(self, text: str):
        """Обработка сообщения (в отдельном потоке)"""
        # Обновляем статус
        self._update_status("● Думаю...")

        # Добавляем в контекст
        self.conversation_context.append(f"User: {text}")

        # Анализируем текст
        valence, intensity = self._analyze_input(text)

        # Инъекция стимула
        self.anima.s_core.inject_stimulus('affection_shown', intensity=intensity, valence=valence)

        # Тики
        for _ in range(10):
            self.anima.s_core.tick()

        # Получаем состояние
        S = self.anima.s_core.S.to_array()
        snapshot = self.anima.get_state_snapshot()
        s_core = snapshot.get('s_core', {})

        # Обновляем аватар
        self._update_avatar_state(S[0], S[1], S[3])

        # Выбираем интент
        action = self.anima.will_engine.select_action(
            S, s_core.get('tension', 0), S[5], temperature_override=0.3
        )

        # Генерируем ответ
        context = "\n".join(self.conversation_context[-10:])
        asp = create_asp(
            S, self.anima.s_core.M, s_core.get('tension', 0),
            action.intent.value, INTENT_REGISTRY[action.intent].name,
            action.confidence, action.constraints
        )

        response, meta = self.llm.generate(asp, context)

        if not response:
            response = "..."

        # Добавляем ответ в контекст
        self.conversation_context.append(f"{self.name}: {response}")
        if len(self.conversation_context) > self.max_context:
            self.conversation_context = self.conversation_context[-self.max_context:]

        # Обновляем UI (в главном потоке)
        self.root.after(0, lambda: self._add_chat_message(self.name, response))
        self.root.after(0, lambda: self._set_avatar_text(response))
        self.root.after(0, lambda: self._update_status("● Готова к общению"))

    def _analyze_input(self, text: str) -> tuple:
        """Анализ входного текста"""
        text_lower = text.lower()

        positive = ['люблю', 'рад', 'счастлив', 'прекрасно', 'отлично', 'класс',
                   'спасибо', 'обнимаю', 'целую', 'скучал', 'хорошо', 'привет']
        negative = ['ненавижу', 'плохо', 'ужасно', 'грустно', 'обидно', 'злюсь',
                   'устал', 'надоело', 'отстань', 'замолчи']

        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)

        valence = (pos_count - neg_count) * 0.3
        valence = max(-1.0, min(1.0, valence))

        intensity = min(1.0, len(text) / 100 + text.count('!') * 0.1)

        return valence, max(0.3, intensity)

    def _on_voice_button(self):
        """Обработка кнопки голоса"""
        if self.stt is None:
            messagebox.showinfo("Информация", "Распознавание речи недоступно.\n\nУстановите:\npip install openai-whisper\nили\npip install SpeechRecognition")
            return

        def listen():
            self._update_status("● Слушаю...")
            self.voice_btn.config(state=tk.DISABLED)

            result = self.stt.listen_from_microphone(timeout=10)

            self.voice_btn.config(state=tk.NORMAL)

            if result.text:
                self.root.after(0, lambda: self.input_field.insert(0, result.text))
                self.root.after(0, lambda: self._update_status("● Готова к общению"))
            else:
                self.root.after(0, lambda: self._update_status("● Не расслышала..."))

        threading.Thread(target=listen, daemon=True).start()

    def _toggle_camera(self):
        """Переключение камеры"""
        if self.vision is None:
            messagebox.showinfo("Информация", "Камера недоступна.\n\nУстановите:\npip install opencv-python")
            return

        if self.vision._is_running:
            self.vision.stop_continuous()
            self._update_status("● Камера выключена")
            self.camera_btn.config(text="📷 Камера")
        else:
            self.vision.start_continuous(self._on_face_detected, interval=0.2)
            self._update_status("● Камера активна")
            self.camera_btn.config(text="📷 Выкл")

    def _on_face_detected(self, analysis):
        """Обработка обнаруженного лица"""
        self.sensor_data.face_valence = analysis.valence
        self.sensor_data.face_arousal = analysis.arousal
        self.sensor_data.face_detected = True

    def _update_status(self, text: str):
        """Обновление статуса"""
        if self.status_label:
            self.root.after(0, lambda: self.status_label.config(text=text))

    def _set_avatar_text(self, text: str):
        """Установка текста аватара"""
        if self.avatar_text:
            self.root.after(0, lambda: self.avatar_text.config(text=text))

    def _update_avatar_state(self, valence: float, arousal: float, attachment: float):
        """Обновление состояния аватара"""
        if self.avatar:
            self.avatar.update_state(valence, arousal, attachment)

    def _start_anima(self):
        """Запуск ANIMA"""
        self.anima.start()
        self._update_status("● Готова к общению")

    def _on_close(self):
        """Закрытие приложения"""
        self.running = False

        # Останавливаем ANIMA
        self.anima.stop()

        # Останавливаем камеру
        if self.vision:
            self.vision.release()

        # Закрываем окно
        if self.root:
            self.root.destroy()

    def run(self):
        """Запуск приложения"""
        # Инициализация сенсоров
        self._init_sensors()

        # Создание GUI
        self.create_gui()

        # Запуск ANIMA
        self._start_anima()

        # Создание аватара в отдельном окне
        self.avatar = AvatarGUI(self.name)
        self.avatar_thread = self.avatar.start_async()

        # Ждём создания окна аватара
        time.sleep(0.5)

        # Главный цикл
        self.running = True
        self.root.mainloop()


def main():
    """Точка входа"""
    import argparse

    parser = argparse.ArgumentParser(description='ANIMA Full GUI')
    parser.add_argument('--name', '-n', default='Лиза', help='Имя')

    args = parser.parse_args()

    app = AnimaFullGUI(name=args.name)
    app.run()


if __name__ == '__main__':
    main()

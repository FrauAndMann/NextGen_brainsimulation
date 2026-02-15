"""
ANIMA Full Experience - Полноценная система с GLM-5 и Live2D аватаром

Интеграция:
- GLM-5 (без рассуждений) - качественный русский
- Live2D-подобный аватар с lip-sync
- Распознавание эмоций с камеры
- Анализ речи и интонации
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
from typing import Optional
import numpy as np

# ANIMA Core
from core.anima import AnimaSystem, AnimaConfig
from core.affective_prompting import create_asp
from core.will_engine import INTENT_REGISTRY

# GLM-5 Effector
from core.llm_glm5 import GLM5Effector, GLM5Config

# Live2D Avatar
from avatar.live2d_avatar import Live2DAvatar


class AnimaFullExperience:
    """
    Полноценная система ANIMA с:
    - GLM-5 для генерации
    - Live2D аватаром
    - Восприятием через камеру и микрофон
    """

    def __init__(self, name: str = "Лиза", glm_api_key: str = None):
        self.name = name
        self.running = False

        # === ЯДРО ANIMA ===
        self.anima = AnimaSystem(AnimaConfig(name=name))

        # === GLM-5 ===
        glm_config = GLM5Config()
        if glm_api_key:
            glm_config.api_key = glm_api_key
        self.llm = GLM5Effector(glm_config)

        # === Сенсоры (опционально) ===
        self.vision = None
        self.stt = None

        # === Аватар ===
        self.avatar = None
        self.avatar_window = None

        # === Состояние ===
        self.conversation_context = []
        self.max_context = 20
        self.is_speaking = False

        # === GUI ===
        self.root = None
        self.chat_display = None
        self.input_field = None
        self.status_label = None

        # Очередь для обновления UI
        self.ui_queue = queue.Queue()

    def _init_sensors(self):
        """Инициализация сенсоров"""
        # Vision
        try:
            from sensors.vision import VisionSensor, check_vision_availability
            vision_status = check_vision_availability()
            if vision_status['camera']:
                self.vision = VisionSensor()
                print(f"[Sensor] Камера: доступна")
        except Exception as e:
            print(f"[Sensor] Vision: {e}")

        # STT
        try:
            from sensors.speech import SpeechToText, STTProvider, check_stt_availability
            stt_status = check_stt_availability()
            if stt_status['recommended']:
                provider = STTProvider.WHISPER if stt_status['whisper'] else STTProvider.GOOGLE
                self.stt = SpeechToText(provider)
                print(f"[Sensor] STT: {stt_status['recommended']}")
        except Exception as e:
            print(f"[Sensor] STT: {e}")

    def create_gui(self):
        """Создание главного GUI"""
        self.root = tk.Tk()
        self.root.title(f"ANIMA - {self.name}")
        self.root.geometry("1000x750")
        self.root.configure(bg='#1a1a2e')

        # Стили
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='white')
        style.configure('TButton', padding=10)
        style.configure('TEntry', fieldbackground='#16213e', foreground='white')

        # Главный контейнер
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === ЛЕВАЯ ПАНЕЛЬ - Аватар ===
        left_frame = ttk.Frame(main_frame, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Информация об API
        api_frame = ttk.Frame(left_frame)
        api_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(api_frame, text="GLM-5 API Key:", font=('Segoe UI', 9)).pack(anchor='w')
        self.api_entry = ttk.Entry(api_frame, show='*', width=35)
        self.api_entry.pack(fill=tk.X, pady=(2, 5))

        connect_btn = ttk.Button(api_frame, text="Подключить", command=self._connect_glm)
        connect_btn.pack(fill=tk.X)

        # Статус
        self.status_label = ttk.Label(
            left_frame,
            text="● Введите API ключ",
            font=('Segoe UI', 10)
        )
        self.status_label.pack(pady=10)

        # Кнопки управления
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.voice_btn = ttk.Button(
            control_frame,
            text="🎤 Голос",
            command=self._on_voice_button,
            width=12
        )
        self.voice_btn.pack(side=tk.LEFT, padx=2)

        self.camera_btn = ttk.Button(
            control_frame,
            text="📷 Камера",
            command=self._toggle_camera,
            width=12
        )
        self.camera_btn.pack(side=tk.LEFT, padx=2)

        self.avatar_btn = ttk.Button(
            control_frame,
            text="👩 Аватар",
            command=self._show_avatar,
            width=12
        )
        self.avatar_btn.pack(side=tk.LEFT, padx=2)

        # Текст аватара
        self.avatar_text = tk.Label(
            left_frame,
            text="...",
            font=('Segoe UI', 12),
            bg='#1a1a2e',
            fg='#e0e0e0',
            wraplength=320,
            justify=tk.LEFT
        )
        self.avatar_text.pack(pady=20, fill=tk.X)

        # === ПРАВАЯ ПАНЕЛЬ - Чат ===
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Заголовок
        title = ttk.Label(
            right_frame,
            text=f"💬 Чат с {self.name}",
            font=('Segoe UI', 14, 'bold')
        )
        title.pack(anchor='w', pady=(0, 10))

        # История чата
        self.chat_display = scrolledtext.ScrolledText(
            right_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            bg='#16213e',
            fg='#e0e0e0',
            insertbackground='white',
            relief=tk.FLAT,
            padx=15,
            pady=10
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)

        # Теги стилей
        self.chat_display.tag_configure('user', foreground='#4fc3f7', font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('anima', foreground='#f48fb1', font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('text', foreground='#e0e0e0')

        # Поле ввода
        input_frame = ttk.Frame(right_frame)
        input_frame.pack(fill=tk.X, pady=(10, 0))

        self.input_field = ttk.Entry(input_frame, font=('Segoe UI', 11))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_field.bind('<Return>', self._on_send)

        send_btn = ttk.Button(input_frame, text="→", command=self._on_send, width=3)
        send_btn.pack(side=tk.RIGHT)

        # Приветствие
        self._add_message(self.name, "Привет! Я тебя слушаю 😊")

        # Закрытие
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Запуск обработки очереди UI
        self._process_ui_queue()

    def _connect_glm(self):
        """Подключение к GLM-5"""
        api_key = self.api_entry.get().strip()
        if not api_key:
            messagebox.showwarning("Внимание", "Введите API ключ от Z.AI\n\nПолучить: https://open.bigmodel.cn/")
            return

        self.llm.config.api_key = api_key

        # Тестовое подключение
        self._update_status("● Проверка подключения...")

        def test():
            available, msg = self.llm.check_availability()
            self.root.after(0, lambda: self._update_status(f"● {msg}"))
            if available:
                self.root.after(0, lambda: self._add_message(self.name, "Подключилась! Теперь можем общаться ✨"))

        threading.Thread(target=test, daemon=True).start()

    def _update_status(self, text: str):
        """Обновление статуса"""
        self.status_label.config(text=text)

    def _add_message(self, sender: str, text: str):
        """Добавление сообщения в чат"""
        self.chat_display.config(state=tk.NORMAL)

        tag = 'user' if sender != self.name else 'anima'
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        self.chat_display.insert(tk.END, f"{text}\n\n", 'text')

        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def _on_send(self, event=None):
        """Отправка сообщения"""
        text = self.input_field.get().strip()
        if not text:
            return

        self.input_field.delete(0, tk.END)
        self._add_message("Ты", text)

        threading.Thread(target=self._process, args=(text,), daemon=True).start()

    def _process(self, text: str):
        """Обработка сообщения"""
        self.root.after(0, lambda: self._update_status("● Думаю..."))

        # Анализ текста
        valence, intensity = self._analyze_text(text)

        # Инъекция стимула
        self.anima.s_core.inject_stimulus('affection_shown', intensity=intensity, valence=valence)

        # Обработка
        for _ in range(10):
            self.anima.s_core.tick()

        # Состояние
        S = self.anima.s_core.S.to_array()
        snapshot = self.anima.get_state_snapshot()
        s_core = snapshot.get('s_core', {})

        # Обновляем аватар
        if self.avatar:
            self.root.after(0, lambda: self.avatar.set_emotion(S[0], S[1], S[3]))

        # Проверка API ключа
        if not self.llm.config.api_key:
            self.root.after(0, lambda: self._add_message(self.name, "Сначала подключи GLM-5 через API ключ :)"))
            self.root.after(0, lambda: self._update_status("● Нужен API ключ"))
            return

        # Выбор интента
        action = self.anima.will_engine.select_action(
            S, s_core.get('tension', 0), S[5], temperature_override=0.3
        )

        # Контекст
        context = "\n".join(self.conversation_context[-10:])

        # ASP
        asp = create_asp(
            S, self.anima.s_core.M, s_core.get('tension', 0),
            action.intent.value, INTENT_REGISTRY[action.intent].name,
            action.confidence, action.constraints
        )

        # Генерация через GLM-5
        response, meta = self.llm.generate(asp, context)

        if not response:
            response = "..."

        # Контекст
        self.conversation_context.append(f"User: {text}")
        self.conversation_context.append(f"{self.name}: {response}")
        if len(self.conversation_context) > self.max_context:
            self.conversation_context = self.conversation_context[-self.max_context:]

        # Обновляем UI
        self.root.after(0, lambda: self._add_message(self.name, response))
        self.root.after(0, lambda: self._set_avatar_text(response))
        self.root.after(0, lambda: self._update_status("● Готова к общению"))

    def _analyze_text(self, text: str) -> tuple:
        """Анализ текста"""
        text_lower = text.lower()

        positive = ['люблю', 'рад', 'счастлив', 'прекрасно', 'отлично', 'класс', 'спасибо',
                   'обнимаю', 'целую', 'скучал', 'хорошо', 'привет', 'красивая', 'милая']
        negative = ['ненавижу', 'плохо', 'ужасно', 'грустно', 'обидно', 'злюсь', 'устал',
                   'надоело', 'отстань', 'замолчи', 'дура', 'глупая']

        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)

        valence = max(-1, min(1, (pos - neg) * 0.3))
        intensity = min(1.0, len(text) / 100 + text.count('!') * 0.1)

        return valence, max(0.3, intensity)

    def _set_avatar_text(self, text: str):
        """Текст для аватара"""
        self.avatar_text.config(text=text)

        if self.avatar and self.avatar_window:
            self.avatar.speak(text)

    def _on_voice_button(self):
        """Кнопка голоса"""
        if not self.stt:
            messagebox.showinfo("Голос", "Распознавание речи недоступно.\n\npip install SpeechRecognition pyaudio")
            return

        def listen():
            self.root.after(0, lambda: self._update_status("● Слушаю..."))
            result = self.stt.listen_from_microphone(timeout=10)
            self.root.after(0, lambda: self._update_status("● Готова"))

            if result.text:
                self.root.after(0, lambda: self.input_field.insert(0, result.text))

        threading.Thread(target=listen, daemon=True).start()

    def _toggle_camera(self):
        """Камера"""
        if not self.vision:
            messagebox.showinfo("Камера", "Камера недоступна.\n\npip install opencv-python")
            return

        if self.vision._is_running:
            self.vision.stop_continuous()
            self._update_status("● Камера выкл")
        else:
            self.vision.start_continuous(self._on_face, interval=0.2)
            self._update_status("● Камера вкл")

    def _on_face(self, analysis):
        """Обработка лица"""
        # Инъекция эмоции в систему
        if analysis.detected:
            self.anima.s_core.inject_stimulus(
                'presence', intensity=0.3, valence=analysis.valence
            )

    def _show_avatar(self):
        """Показать аватар"""
        if self.avatar and self.avatar_window:
            self.avatar_window.lift()
            return

        self.avatar = Live2DAvatar(self.name)
        self.avatar_window = self.avatar.root

        # Запуск в отдельном потоке
        threading.Thread(target=self.avatar.start, daemon=True).start()

    def _process_ui_queue(self):
        """Обработка очереди UI"""
        try:
            while True:
                task = self.ui_queue.get_nowait()
                task()
        except queue.Empty:
            pass

        self.root.after(100, self._process_ui_queue)

    def _start_anima(self):
        """Запуск ANIMA"""
        self.anima.start()

    def _on_close(self):
        """Закрытие"""
        self.running = False
        self.anima.stop()
        if self.vision:
            self.vision.release()
        if self.root:
            self.root.destroy()

    def run(self):
        """Запуск"""
        self._init_sensors()
        self.create_gui()
        self._start_anima()
        self.running = True
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='Лиза')
    parser.add_argument('--api-key', '-k', default=None)
    args = parser.parse_args()

    app = AnimaFullExperience(name=args.name, glm_api_key=args.api_key)
    app.run()


if __name__ == '__main__':
    main()

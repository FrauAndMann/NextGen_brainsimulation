"""
ANIMA Pro - Полноценная система с GLM-4.7 и MuseTalk аватаром

Реалистичный цифровой компаньон с:
- GLM-4.7 для генерации текста
- MuseTalk для реалистичного аватара с lip-sync
- Восприятие через камеру и микрофон

Запуск:
    python anima_pro.py

Требования:
    - API ключ Z.AI (https://open.bigmodel.cn/)
    - NVIDIA GPU 4GB+ (для MuseTalk)
    - Или используйте режим без MuseTalk
"""

import sys
import os
import time
import threading
import queue

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Optional, Tuple
from pathlib import Path
import subprocess

# ANIMA Core
from core.anima import AnimaSystem, AnimaConfig
from core.affective_prompting import create_asp
from core.will_engine import INTENT_REGISTRY

# GLM-4.7
from core.llm_glm4 import GLM4Effector, GLMConfig


class AnimaPro:
    """
    ANIMA Pro - Полноценная система

    Режимы:
    1. Полный - с MuseTalk аватаром (требует GPU)
    2. Упрощённый - с базовым аватаром
    """

    def __init__(self, name: str = "Лиза"):
        self.name = name
        self.running = False

        # Core
        self.anima = AnimaSystem(AnimaConfig(name=name))
        self.llm = GLM4Effector(GLMConfig(model="glm-4.7-flash"))

        # MuseTalk (опционально)
        self.musetalk = None
        self.use_musetalk = False

        # Sensors
        self.vision = None
        self.stt = None

        # Context
        self.context = []
        self.max_context = 20

        # GUI
        self.root = None

        # Video display
        self.video_label = None
        self.current_frame = None

    def check_musetalk(self) -> Tuple[bool, str]:
        """Проверка MuseTalk"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA недоступна"

            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram < 4:
                return False, f"VRAM: {vram:.1f}GB (нужно 4GB+)"

            # Проверка MuseTalk
            try:
                from avatar.musetalk_avatar import MuseTalkAvatar
                return True, "MuseTalk доступен"
            except ImportError:
                return False, "MuseTalk не установлен"

        except ImportError:
            return False, "PyTorch не установлен"

    def init_sensors(self):
        """Инициализация сенсоров"""
        try:
            from sensors.vision import VisionSensor
            self.vision = VisionSensor()
            print("[OK] Камера")
        except Exception as e:
            print(f"[--] Камера: {e}")

        try:
            from sensors.speech import SpeechToText
            self.stt = SpeechToText()
            print("[OK] STT")
        except Exception as e:
            print(f"[--] STT: {e}")

    def create_gui(self):
        """Создание GUI"""
        self.root = tk.Tk()
        self.root.title(f"ANIMA Pro - {self.name}")
        self.root.geometry("1200x800")
        self.root.configure(bg='#0d0d1a')

        # Styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#0d0d1a')
        style.configure('TLabel', background='#0d0d1a', foreground='#e0e0e0')
        style.configure('TButton', padding=10)
        style.configure('TEntry', fieldbackground='#1a1a2e', foreground='white')

        # Main container
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # === LEFT PANEL - Avatar ===
        left = ttk.Frame(main, width=450)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left.pack_propagate(False)

        # API Key
        api_frame = ttk.Frame(left)
        api_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(api_frame, text="🔑 Z.AI API Key:", font=('Segoe UI', 10)).pack(anchor='w')
        self.api_entry = ttk.Entry(api_frame, show='•', width=40)
        self.api_entry.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(api_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Подключить", command=self._connect).pack(side=tk.LEFT, padx=(0, 5))

        # Status
        self.status = ttk.Label(left, text="● Введите API ключ", font=('Segoe UI', 11))
        self.status.pack(pady=10)

        # Avatar display
        avatar_frame = ttk.Frame(left)
        avatar_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.avatar_label = tk.Label(
            avatar_frame,
            text=f"👩\n{self.name}",
            font=('Segoe UI', 48),
            bg='#1a1a2e',
            fg='#f48fb1',
            width=20,
            height=8
        )
        self.avatar_label.pack(fill=tk.BOTH, expand=True)

        # Response text
        self.response_text = tk.Label(
            left,
            text="...",
            font=('Segoe UI', 13),
            bg='#0d0d1a',
            fg='#e0e0e0',
            wraplength=400,
            justify=tk.LEFT
        )
        self.response_text.pack(pady=10, fill=tk.X)

        # Controls
        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=10)

        self.voice_btn = ttk.Button(controls, text="🎤", width=5, command=self._voice)
        self.voice_btn.pack(side=tk.LEFT, padx=2)

        self.camera_btn = ttk.Button(controls, text="📷", width=5, command=self._camera)
        self.camera_btn.pack(side=tk.LEFT, padx=2)

        # === RIGHT PANEL - Chat ===
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(
            right,
            text=f"💬 Чат с {self.name}",
            font=('Segoe UI', 16, 'bold')
        ).pack(anchor='w', pady=(0, 10))

        # Chat
        self.chat = scrolledtext.ScrolledText(
            right,
            wrap=tk.WORD,
            font=('Segoe UI', 12),
            bg='#1a1a2e',
            fg='#e0e0e0',
            insertbackground='white',
            relief=tk.FLAT,
            padx=15,
            pady=10
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.config(state=tk.DISABLED)

        self.chat.tag_configure('user', foreground='#4fc3f7', font=('Segoe UI', 12, 'bold'))
        self.chat.tag_configure('anima', foreground='#f48fb1', font=('Segoe UI', 12, 'bold'))

        # Input
        input_frame = ttk.Frame(right)
        input_frame.pack(fill=tk.X, pady=(15, 0))

        self.input = ttk.Entry(input_frame, font=('Segoe UI', 12))
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input.bind('<Return>', self._send)

        ttk.Button(input_frame, text="→", command=self._send, width=3).pack(side=tk.RIGHT)

        # Welcome
        self._add_chat(self.name, "Привет! ✨")

        # Close
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _connect(self):
        """Подключение к GLM-4.7"""
        key = self.api_entry.get().strip()
        if not key:
            messagebox.showwarning("API Key", "Введите ключ от Z.AI\n\nПолучить: https://open.bigmodel.cn/")
            return

        self.llm.config.api_key = key
        self._update_status("● Проверка...")

        def check():
            ok, msg = self.llm.check_availability()
            self.root.after(0, lambda: self._update_status(f"● {msg}"))
            if ok:
                self.root.after(0, lambda: self._add_chat(self.name, "Подключилась! Можем общаться 💬"))

        threading.Thread(target=check, daemon=True).start()

    def _update_status(self, text: str):
        self.status.config(text=text)

    def _update_avatar(self, valence: float, arousal: float):
        """Обновление аватара"""
        # Эмодзи в зависимости от эмоции
        if valence > 0.5:
            emoji = "😊"
        elif valence > 0.2:
            emoji = "🙂"
        elif valence < -0.4:
            emoji = "😢"
        elif valence < -0.1:
            emoji = "😕"
        else:
            emoji = "😐"

        if arousal > 0.7:
            emoji = "😮" if valence > 0 else "😤"

        self.avatar_label.config(text=f"{emoji}\n{self.name}")

    def _add_chat(self, sender: str, text: str):
        self.chat.config(state=tk.NORMAL)
        tag = 'user' if sender != self.name else 'anima'
        self.chat.insert(tk.END, f"{sender}: ", tag)
        self.chat.insert(tk.END, f"{text}\n\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)

    def _send(self, event=None):
        text = self.input.get().strip()
        if not text:
            return

        self.input.delete(0, tk.END)
        self._add_chat("Ты", text)
        threading.Thread(target=self._process, args=(text,), daemon=True).start()

    def _process(self, text: str):
        self.root.after(0, lambda: self._update_status("● Думаю..."))

        # Analyze
        valence, intensity = self._analyze(text)

        # Inject
        self.anima.s_core.inject_stimulus('affection_shown', intensity, valence)
        for _ in range(10):
            self.anima.s_core.tick()

        S = self.anima.s_core.S.to_array()
        snapshot = self.anima.get_state_snapshot()
        s_core = snapshot.get('s_core', {})

        # Update avatar
        self.root.after(0, lambda: self._update_avatar(S[0], S[1]))

        if not self.llm.config.api_key:
            self.root.after(0, lambda: self._add_chat(self.name, "Сначала подключи GLM-4.7 :)"))
            self.root.after(0, lambda: self._update_status("● Нужен API ключ"))
            return

        # Select intent
        action = self.anima.will_engine.select_action(S, s_core.get('tension', 0), S[5], 0.3)

        # Generate
        context = "\n".join(self.context[-10:])
        asp = create_asp(S, self.anima.s_core.M, s_core.get('tension', 0),
                        action.intent.value, INTENT_REGISTRY[action.intent].name,
                        action.confidence, action.constraints)

        response, meta = self.llm.generate(asp, context)

        if not response:
            response = "..."

        # Update context
        self.context.append(f"User: {text}")
        self.context.append(f"{self.name}: {response}")
        if len(self.context) > self.max_context:
            self.context = self.context[-self.max_context:]

        # Update UI
        self.root.after(0, lambda: self._add_chat(self.name, response))
        self.root.after(0, lambda: self.response_text.config(text=response))
        self.root.after(0, lambda: self._update_status("● Готова"))

    def _analyze(self, text: str) -> Tuple[float, float]:
        text_lower = text.lower()
        pos = ['люблю', 'рад', 'счастлив', 'отлично', 'класс', 'спасибо', 'привет', 'красивая']
        neg = ['ненавижу', 'плохо', 'ужасно', 'грустно', 'злюсь', 'отстань']

        valence = (sum(1 for w in pos if w in text_lower) - sum(1 for w in neg if w in text_lower)) * 0.3
        intensity = min(1.0, len(text) / 100 + text.count('!') * 0.1)

        return max(-1, min(1, valence)), max(0.3, intensity)

    def _voice(self):
        if not self.stt:
            messagebox.showinfo("Голос", "pip install SpeechRecognition pyaudio")
            return

        def listen():
            self.root.after(0, lambda: self._update_status("● Слушаю..."))
            result = self.stt.listen_from_microphone(timeout=10)
            self.root.after(0, lambda: self._update_status("● Готова"))
            if result.text:
                self.root.after(0, lambda: self.input.insert(0, result.text))

        threading.Thread(target=listen, daemon=True).start()

    def _camera(self):
        if not self.vision:
            messagebox.showinfo("Камера", "pip install opencv-python")
            return

        if self.vision._is_running:
            self.vision.stop_continuous()
            self._update_status("● Камера выкл")
        else:
            self.vision.start_continuous(lambda a: None, 0.2)
            self._update_status("● Камера вкл")

    def _close(self):
        self.running = False
        self.anima.stop()
        if self.vision:
            self.vision.release()
        self.root.destroy()

    def run(self):
        self.init_sensors()
        self.create_gui()
        self.anima.start()
        self.running = True
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='Лиза')
    args = parser.parse_args()

    app = AnimaPro(name=args.name)
    app.run()


if __name__ == '__main__':
    main()

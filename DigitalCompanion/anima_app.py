"""
ANIMA Application - Современный GUI для цифрового компаньона

Особенности:
- Современный дизайн (CustomTkinter или Tkinter)
- Интеграция с UnifiedAnima
- Продвинутый аватар с эмоциями
- Голосовой ввод/вывод
- Визуализация внутреннего состояния

Автор: FrauAndMann
Версия: 2.0
"""

import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Callable

# Попытка импорта CustomTkinter
try:
    import customtkinter as ctk
    HAS_CTK = True
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    HAS_CTK = False
    import tkinter as tk
    from tkinter import ttk

# Импорты системы
from unified_anima import UnifiedAnima, AnimaConfig
from avatar.advanced_avatar import AdvancedAvatar, EmotionType


class AnimaApp:
    """
    Современное приложение ANIMA

    GUI с интеграцией всех систем компаньона.
    """

    def __init__(self, config: AnimaConfig = None):
        self.config = config or AnimaConfig()

        # ANIMA система
        self.anima: Optional[UnifiedAnima] = None

        # Аватар
        self.avatar: Optional[AdvancedAvatar] = None
        self.avatar_window = None

        # Состояние UI
        self.is_listening = False
        self.is_speaking = False

        # Создание окна
        self._create_window()

        # Callbacks
        self._setup_callbacks()

    def _create_window(self):
        """Создание главного окна"""
        if HAS_CTK:
            self._create_ctk_window()
        else:
            self._create_tk_window()

    def _create_ctk_window(self):
        """Создание окна с CustomTkinter"""
        self.root = ctk.CTk()
        self.root.title(f"{self.config.name} - ANIMA")
        self.root.geometry("900x700")
        self.root.configure(fg_color="#1a1a2e")

        # Главный контейнер
        self.main_frame = ctk.CTkFrame(self.root, fg_color="#1a1a2e")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Верхняя панель - статус
        self._create_status_panel_ctk()

        # Центральная область - чат
        self._create_chat_panel_ctk()

        # Нижняя панель - ввод
        self._create_input_panel_ctk()

        # Боковая панель - состояние
        self._create_state_panel_ctk()

    def _create_status_panel_ctk(self):
        """Панель статуса (CustomTkinter)"""
        self.status_frame = ctk.CTkFrame(self.main_frame, height=50, fg_color="#16213e")
        self.status_frame.pack(fill="x", pady=(0, 10))

        # Имя
        self.name_label = ctk.CTkLabel(
            self.status_frame,
            text=self.config.name,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#e94560"
        )
        self.name_label.pack(side="left", padx=20, pady=10)

        # Статус
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Инициализация...",
            font=ctk.CTkFont(size=12),
            text_color="#a0a0a0"
        )
        self.status_label.pack(side="left", padx=10, pady=10)

        # Кнопки управления
        self.btn_frame = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        self.btn_frame.pack(side="right", padx=10)

        self.avatar_btn = ctk.CTkButton(
            self.btn_frame,
            text="Аватар",
            width=80,
            command=self._toggle_avatar
        )
        self.avatar_btn.pack(side="left", padx=5)

        self.voice_btn = ctk.CTkButton(
            self.btn_frame,
            text="Голос",
            width=80,
            command=self._toggle_voice
        )
        self.voice_btn.pack(side="left", padx=5)

        self.save_btn = ctk.CTkButton(
            self.btn_frame,
            text="Сохранить",
            width=80,
            command=self._save_state
        )
        self.save_btn.pack(side="left", padx=5)

    def _create_chat_panel_ctk(self):
        """Панель чата (CustomTkinter)"""
        self.chat_frame = ctk.CTkFrame(self.main_frame, fg_color="#0f0f23")
        self.chat_frame.pack(fill="both", expand=True, pady=5)

        # Текстовое поле чата
        self.chat_text = ctk.CTkTextbox(
            self.chat_frame,
            font=ctk.CTkFont(size=13),
            text_color="#e0e0e0",
            fg_color="#0f0f23",
            wrap="word"
        )
        self.chat_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Теги для стилей
        self.chat_text._textbox.tag_configure("user", foreground="#4fc3f7")
        self.chat_text._textbox.tag_configure("anima", foreground="#f48fb1")
        self.chat_text._textbox.tag_configure("system", foreground="#808080")
        self.chat_text._textbox.tag_configure("time", foreground="#606060")

    def _create_input_panel_ctk(self):
        """Панель ввода (CustomTkinter)"""
        self.input_frame = ctk.CTkFrame(self.main_frame, height=60, fg_color="#16213e")
        self.input_frame.pack(fill="x", pady=(10, 0))

        # Поле ввода
        self.input_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text="Напишите сообщение...",
            font=ctk.CTkFont(size=14),
            height=40
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.input_entry.bind("<Return>", self._on_send)

        # Кнопка отправки
        self.send_btn = ctk.CTkButton(
            self.input_frame,
            text="Отправить",
            width=100,
            command=self._on_send
        )
        self.send_btn.pack(side="right", padx=10, pady=10)

        # Кнопка микрофона
        self.mic_btn = ctk.CTkButton(
            self.input_frame,
            text="🎤",
            width=50,
            command=self._on_mic
        )
        self.mic_btn.pack(side="right", padx=5, pady=10)

    def _create_state_panel_ctk(self):
        """Панель состояния (CustomTkinter)"""
        self.state_frame = ctk.CTkFrame(self.main_frame, width=200, fg_color="#16213e")
        self.state_frame.pack(side="right", fill="y", padx=(10, 0))

        # Заголовок
        ctk.CTkLabel(
            self.state_frame,
            text="Внутреннее состояние",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#e94560"
        ).pack(pady=10)

        # Эмоциональные индикаторы
        self.emotion_bars = {}

        emotions = [
            ("Валентность", "valence", "#4caf50"),
            ("Возбуждение", "arousal", "#ff9800"),
            ("Доминирование", "dominance", "#2196f3"),
            ("Привязанность", "attachment", "#e91e63"),
            ("Энергия", "energy", "#9c27b0"),
        ]

        for name, key, color in emotions:
            frame = ctk.CTkFrame(self.state_frame, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=5)

            ctk.CTkLabel(
                frame,
                text=name,
                font=ctk.CTkFont(size=11),
                width=80,
                anchor="w"
            ).pack(side="left")

            bar = ctk.CTkProgressBar(frame, width=100)
            bar.set(0.5)
            bar.pack(side="right")
            self.emotion_bars[key] = bar

        # Текущая эмоция
        self.emotion_label = ctk.CTkLabel(
            self.state_frame,
            text="Эмоция: нейтральная",
            font=ctk.CTkFont(size=12),
            text_color="#a0a0a0"
        )
        self.emotion_label.pack(pady=20)

        # Режим
        self.mode_label = ctk.CTkLabel(
            self.state_frame,
            text="Режим: AWAKE",
            font=ctk.CTkFont(size=12),
            text_color="#607d8b"
        )
        self.mode_label.pack(pady=5)

    # === TKINTER FALLBACK ===

    def _create_tk_window(self):
        """Создание окна с обычным Tkinter"""
        self.root = tk.Tk()
        self.root.title(f"{self.config.name} - ANIMA")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")

        # Стиль
        style = ttk.Style()
        style.theme_use('clam')

        # Главный контейнер
        self.main_frame = tk.Frame(self.root, bg="#1a1a2e")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Создаём панели
        self._create_status_panel_tk()
        self._create_chat_panel_tk()
        self._create_input_panel_tk()
        self._create_state_panel_tk()

    def _create_status_panel_tk(self):
        """Панель статуса (Tkinter)"""
        self.status_frame = tk.Frame(self.main_frame, bg="#16213e", height=50)
        self.status_frame.pack(fill="x", pady=(0, 10))
        self.status_frame.pack_propagate(False)

        self.name_label = tk.Label(
            self.status_frame,
            text=self.config.name,
            font=('Segoe UI', 20, 'bold'),
            fg="#e94560",
            bg="#16213e"
        )
        self.name_label.pack(side="left", padx=20, pady=10)

        self.status_label = tk.Label(
            self.status_frame,
            text="Инициализация...",
            font=('Segoe UI', 12),
            fg="#a0a0a0",
            bg="#16213e"
        )
        self.status_label.pack(side="left", padx=10, pady=10)

        btn_frame = tk.Frame(self.status_frame, bg="#16213e")
        btn_frame.pack(side="right", padx=10)

        for text, cmd in [("Аватар", self._toggle_avatar), ("Сохранить", self._save_state)]:
            btn = tk.Button(
                btn_frame,
                text=text,
                command=cmd,
                bg="#e94560",
                fg="white",
                font=('Segoe UI', 10),
                width=10
            )
            btn.pack(side="left", padx=5)

    def _create_chat_panel_tk(self):
        """Панель чата (Tkinter)"""
        self.chat_frame = tk.Frame(self.main_frame, bg="#0f0f23")
        self.chat_frame.pack(fill="both", expand=True, pady=5, side="left")

        # Scrollbar
        scrollbar = tk.Scrollbar(self.chat_frame)
        scrollbar.pack(side="right", fill="y")

        self.chat_text = tk.Text(
            self.chat_frame,
            font=('Segoe UI', 13),
            fg="#e0e0e0",
            bg="#0f0f23",
            wrap="word",
            yscrollcommand=scrollbar.set
        )
        self.chat_text.pack(fill="both", expand=True, padx=10, pady=10)
        scrollbar.config(command=self.chat_text.yview)

        # Теги
        self.chat_text.tag_configure("user", foreground="#4fc3f7")
        self.chat_text.tag_configure("anima", foreground="#f48fb1")
        self.chat_text.tag_configure("system", foreground="#808080")
        self.chat_text.tag_configure("time", foreground="#606060")

    def _create_input_panel_tk(self):
        """Панель ввода (Tkinter)"""
        self.input_frame = tk.Frame(self.main_frame, bg="#16213e", height=60)
        self.input_frame.pack(fill="x", pady=(10, 0), side="bottom")
        self.input_frame.pack_propagate(False)

        self.input_entry = tk.Entry(
            self.input_frame,
            font=('Segoe UI', 14),
            bg="#0f0f23",
            fg="#e0e0e0",
            insertbackground="#e0e0e0"
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=10, pady=15)
        self.input_entry.bind("<Return>", self._on_send)

        send_btn = tk.Button(
            self.input_frame,
            text="Отправить",
            command=self._on_send,
            bg="#e94560",
            fg="white",
            font=('Segoe UI', 11)
        )
        send_btn.pack(side="right", padx=10, pady=15)

    def _create_state_panel_tk(self):
        """Панель состояния (Tkinter)"""
        self.state_frame = tk.Frame(self.main_frame, bg="#16213e", width=200)
        self.state_frame.pack(side="right", fill="y", padx=(10, 0))

        tk.Label(
            self.state_frame,
            text="Внутреннее состояние",
            font=('Segoe UI', 14, 'bold'),
            fg="#e94560",
            bg="#16213e"
        ).pack(pady=10)

        self.emotion_label = tk.Label(
            self.state_frame,
            text="Эмоция: нейтральная",
            font=('Segoe UI', 12),
            fg="#a0a0a0",
            bg="#16213e"
        )
        self.emotion_label.pack(pady=20)

    # === CALLBACKS ===

    def _setup_callbacks(self):
        """Настройка callbacks"""
        if self.anima:
            self.anima.on_response = self._on_anima_response
            self.anima.on_state_change = self._on_state_change

    def _on_send(self, event=None):
        """Отправка сообщения"""
        if HAS_CTK:
            text = self.input_entry.get().strip()
            self.input_entry.delete(0, "end")
        else:
            text = self.input_entry.get().strip()
            self.input_entry.delete(0, "end")

        if not text:
            return

        # Добавление в чат
        self._add_message("user", text)

        # Обработка ANIMA
        if self.anima:
            threading.Thread(
                target=self._process_message,
                args=(text,),
                daemon=True
            ).start()

    def _process_message(self, text: str):
        """Обработка сообщения в отдельном потоке"""
        try:
            response = self.anima.process_input(text)
            self.root.after(0, lambda: self._add_message("anima", response))

            # Обновление аватара
            if self.avatar and self.avatar.running:
                self.avatar.set_pad(
                    self.anima.current_valence,
                    self.anima.current_arousal
                )
                if response:
                    self.avatar.set_speaking(True, response)
                    # Остановка говорения через время
                    delay = max(2, len(response) / 15)
                    threading.Timer(delay, lambda: self.avatar.set_speaking(False)).start()

        except Exception as e:
            self.root.after(0, lambda: self._add_message("system", f"Ошибка: {e}"))

    def _add_message(self, role: str, text: str):
        """Добавление сообщения в чат"""
        timestamp = datetime.now().strftime("%H:%M")

        self.chat_text.configure(state="normal")

        # Время
        self.chat_text.insert("end", f"[{timestamp}] ", "time")

        # Имя отправителя
        if role == "user":
            self.chat_text.insert("end", "Вы: ", "user")
        elif role == "anima":
            self.chat_text.insert("end", f"{self.config.name}: ", "anima")
        else:
            self.chat_text.insert("end", "Система: ", "system")

        # Текст
        self.chat_text.insert("end", f"{text}\n\n")

        self.chat_text.configure(state="disabled")
        self.chat_text.see("end")

    def _on_mic(self):
        """Обработка микрофона"""
        self._add_message("system", "Голосовой ввод в разработке...")

    def _toggle_avatar(self):
        """Переключение аватара"""
        if self.avatar and self.avatar.running:
            self.avatar.stop()
            self._add_message("system", "Аватар закрыт")
        else:
            self._show_avatar()

    def _show_avatar(self):
        """Показать аватар"""
        if not self.avatar:
            self.avatar = AdvancedAvatar(name=self.config.name)
            self.avatar.start_async()
            self._add_message("system", "Аватар открыт")
        else:
            self.avatar.start_async()

    def _toggle_voice(self):
        """Переключение голоса"""
        self.config.enable_tts = not self.config.enable_tts
        status = "включён" if self.config.enable_tts else "выключен"
        self._add_message("system", f"TTS {status}")

    def _save_state(self):
        """Сохранение состояния"""
        if self.anima:
            self.anima.save_state()
            self._add_message("system", "Состояние сохранено")

    def _on_anima_response(self, response: str):
        """Callback при ответе ANIMA"""
        self.root.after(0, lambda: self._add_message("anima", response))

    def _on_state_change(self, state: dict):
        """Callback при изменении состояния"""
        self.root.after(0, lambda: self._update_state_display(state))

    def _update_state_display(self, state: dict):
        """Обновление отображения состояния"""
        if HAS_CTK:
            # Обновление баров
            s_core = state.get('s_core', {})
            S = s_core.get('S', [0.5] * 6)

            if len(S) >= 6:
                # Нормализация для отображения
                valence = (S[0] + 1) / 2  # -1..1 -> 0..1
                self.emotion_bars['valence'].set(valence)
                self.emotion_bars['arousal'].set(S[1])
                self.emotion_bars['dominance'].set(S[2])
                self.emotion_bars['attachment'].set(S[3])
                self.emotion_bars['energy'].set(S[5])

            # Эмоция
            emotion = state.get('emotion', 'neutral')
            self.emotion_label.configure(text=f"Эмоция: {emotion}")

            # Режим
            mode = state.get('mode', 'AWAKE')
            self.mode_label.configure(text=f"Режим: {mode}")

        # Статус
        mode = state.get('mode', 'AWAKE')
        tick = state.get('tick', 0)
        self.status_label.configure(text=f"Режим: {mode} | Тиков: {tick}")

    # === ЗАПУСК ===

    def start(self):
        """Запуск приложения"""
        # Инициализация ANIMA
        self._init_anima()

        # Запуск GUI
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _init_anima(self):
        """Инициализация ANIMA"""
        self._add_message("system", "Инициализация ANIMA...")

        try:
            # Проверка Ollama
            from core.llm_effector import check_ollama_available
            available, msg = check_ollama_available(self.config.llm_model)

            if not available:
                self._add_message("system", f"Ошибка: {msg}")
                return

            self._add_message("system", f"LLM: {msg}")

            # Создание ANIMA
            self.anima = UnifiedAnima(self.config)
            self.anima.start()

            # Настройка callbacks
            self.anima.on_response = self._on_anima_response
            self.anima.on_state_change = self._on_state_change

            self._add_message("system", f"{self.config.name} готова к общению!")
            self.status_label.configure(text="Онлайн")

        except Exception as e:
            self._add_message("system", f"Ошибка инициализации: {e}")

    def _on_close(self):
        """Закрытие приложения"""
        if self.anima:
            self.anima.save_state()
            self.anima.stop()

        if self.avatar:
            self.avatar.stop()

        self.root.destroy()


# === ТОЧКА ВХОДА ===

def main():
    """Главная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='ANIMA Application')
    parser.add_argument('--name', default='Лиза', help='Имя компаньона')
    parser.add_argument('--model', default='dolphin-mistral:7b', help='Модель LLM')
    parser.add_argument('--no-tts', action='store_true', help='Отключить TTS')
    parser.add_argument('--temperament', default='melancholic',
                       choices=['sanguine', 'choleric', 'phlegmatic', 'melancholic'])
    args = parser.parse_args()

    config = AnimaConfig(
        name=args.name,
        llm_model=args.model,
        enable_tts=not args.no_tts,
        temperament_type=args.temperament
    )

    app = AnimaApp(config)
    app.start()


if __name__ == "__main__":
    main()

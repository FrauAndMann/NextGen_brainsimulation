"""
Улучшенный графический интерфейс с аватаром

Особенности:
- Большой аватар с эмоциями
- Анимация моргания
- Чат с историей
- Панель состояния
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font
import threading
import queue
from typing import Optional
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.companion import DigitalCompanion
from core.llm_interface import LLMInterface, LLMConfig, check_ollama_available
from effectors.tts import TTSEngine, TTSProvider, check_tts_availability
from effectors.avatar import AvatarRenderer, AvatarEmotion


class CompanionGUIv2:
    """
    Улучшенный графический интерфейс компаньона
    """

    COLORS = {
        'bg_dark': '#1a1a2e',
        'bg_medium': '#16213e',
        'bg_light': '#0f3460',
        'accent': '#e94560',
        'accent_light': '#ff6b9d',
        'text': '#eaeaea',
        'text_dim': '#888888',
        'love': '#ff6b9d',
        'user': '#4fc3f7',
        'companion': '#f48fb1',
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Лиза ♡ Цифровой компаньон")
        self.root.geometry("1000x750")
        self.root.configure(bg=self.COLORS['bg_dark'])
        self.root.minsize(800, 600)

        # Компоненты
        self.companion = None
        self.llm = None
        self.tts = None
        self.avatar = AvatarRenderer()
        self.message_queue = queue.Queue()

        # Состояние UI
        self.is_typing = False
        self.blink_job = None

        # Создание интерфейса
        self._create_styles()
        self._create_widgets()

        # Инициализация
        self._start_initialization()

        # Запуск анимации
        self._animate_avatar()

    def _create_styles(self):
        """Создание стилей"""
        style = ttk.Style()
        style.theme_use('clam')

        # Фреймы
        style.configure('Dark.TFrame', background=self.COLORS['bg_dark'])
        style.configure('Medium.TFrame', background=self.COLORS['bg_medium'])

        # Метки
        style.configure('Dark.TLabel',
                       background=self.COLORS['bg_dark'],
                       foreground=self.COLORS['text'],
                       font=('Segoe UI', 10))
        style.configure('Title.TLabel',
                       background=self.COLORS['bg_dark'],
                       foreground=self.COLORS['accent_light'],
                       font=('Segoe UI', 16, 'bold'))
        style.configure('Subtitle.TLabel',
                       background=self.COLORS['bg_dark'],
                       foreground=self.COLORS['text_dim'],
                       font=('Segoe UI', 9))
        style.configure('Emotion.TLabel',
                       background=self.COLORS['bg_medium'],
                       foreground=self.COLORS['accent_light'],
                       font=('Segoe UI', 12, 'bold'))

        # Кнопки
        style.configure('Accent.TButton',
                       background=self.COLORS['accent'],
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       padding=(20, 10))
        style.map('Accent.TButton',
                 background=[('active', self.COLORS['accent_light'])])

        # Entry
        style.configure('Dark.TEntry',
                       fieldbackground=self.COLORS['bg_medium'],
                       foreground=self.COLORS['text'])

    def _create_widgets(self):
        """Создание виджетов"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root, style='Dark.TFrame', padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === ВЕРХНЯЯ ПАНЕЛЬ ===
        header_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 15))

        # Заголовок
        title_frame = ttk.Frame(header_frame, style='Dark.TFrame')
        title_frame.pack(side=tk.LEFT)

        self.title_label = ttk.Label(
            title_frame,
            text="♡ Лиза",
            style='Title.TLabel'
        )
        self.title_label.pack(anchor=tk.W)

        self.subtitle_label = ttk.Label(
            title_frame,
            text="Цифровой компаньон",
            style='Subtitle.TLabel'
        )
        self.subtitle_label.pack(anchor=tk.W)

        # Статус
        status_frame = ttk.Frame(header_frame, style='Dark.TFrame')
        status_frame.pack(side=tk.RIGHT)

        self.status_label = ttk.Label(
            status_frame,
            text="● Загрузка...",
            style='Subtitle.TLabel'
        )
        self.status_label.pack(anchor=tk.E)

        # === ЦЕНТРАЛЬНАЯ ОБЛАСТЬ ===
        center_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        center_frame.pack(fill=tk.BOTH, expand=True)

        # Левая панель - Аватар
        left_panel = ttk.Frame(center_frame, style='Dark.TFrame', width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)

        # Фрейм аватара
        avatar_outer = tk.Frame(
            left_panel,
            bg=self.COLORS['bg_medium'],
            relief=tk.FLAT,
            bd=0
        )
        avatar_outer.pack(fill=tk.BOTH, expand=True)

        # Аватар (текстовый)
        self.avatar_label = tk.Label(
            avatar_outer,
            text="",
            font=('Consolas', 12),
            bg=self.COLORS['bg_medium'],
            fg=self.COLORS['text'],
            justify=tk.CENTER
        )
        self.avatar_label.pack(expand=True, pady=20)

        # Эмоция под аватаром
        self.emotion_label = ttk.Label(
            left_panel,
            text="😐 нейтрально",
            style='Emotion.TLabel'
        )
        self.emotion_label.pack(pady=(0, 10))

        # Информация о состоянии
        info_frame = tk.Frame(left_panel, bg=self.COLORS['bg_medium'])
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # Любовь
        self.love_frame = tk.Frame(info_frame, bg=self.COLORS['bg_medium'])
        self.love_frame.pack(fill=tk.X, pady=5, padx=10)

        tk.Label(
            self.love_frame,
            text="♡ Любовь:",
            bg=self.COLORS['bg_medium'],
            fg=self.COLORS['love'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT)

        self.love_progress = ttk.Progressbar(
            self.love_frame,
            length=100,
            mode='determinate'
        )
        self.love_progress.pack(side=tk.LEFT, padx=10)

        self.love_value = tk.Label(
            self.love_frame,
            text="0%",
            bg=self.COLORS['bg_medium'],
            fg=self.COLORS['text'],
            font=('Segoe UI', 9)
        )
        self.love_value.pack(side=tk.LEFT)

        # Нейрохимия
        neuro_frame = tk.Frame(left_panel, bg=self.COLORS['bg_medium'])
        neuro_frame.pack(fill=tk.X, padx=5, pady=5)

        self.neuro_bars = {}
        for nt_name, color in [('Окситоцин', '#ff9eb5'),
                               ('Дофамин', '#81d4fa'),
                               ('Серотонин', '#a5d6a7'),
                               ('Кортизол', '#ef9a9a')]:
            frame = tk.Frame(neuro_frame, bg=self.COLORS['bg_medium'])
            frame.pack(fill=tk.X, pady=2, padx=10)

            tk.Label(
                frame,
                text=f"{nt_name}:",
                bg=self.COLORS['bg_medium'],
                fg=color,
                font=('Segoe UI', 9),
                width=10,
                anchor='w'
            ).pack(side=tk.LEFT)

            progress = ttk.Progressbar(frame, length=80, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5)

            value = tk.Label(
                frame,
                text="0%",
                bg=self.COLORS['bg_medium'],
                fg=self.COLORS['text_dim'],
                font=('Segoe UI', 8)
            )
            value.pack(side=tk.LEFT)

            self.neuro_bars[nt_name] = (progress, value)

        # Правая панель - Чат
        right_panel = ttk.Frame(center_frame, style='Dark.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # История чата
        chat_frame = tk.Frame(right_panel, bg=self.COLORS['bg_medium'])
        chat_frame.pack(fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            bg=self.COLORS['bg_medium'],
            fg=self.COLORS['text'],
            insertbackground=self.COLORS['text'],
            selectbackground=self.COLORS['bg_light'],
            relief=tk.FLAT,
            padx=15,
            pady=15
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.configure(state='disabled')

        # Теги для чата
        self.chat_display.tag_configure('user',
                                        foreground=self.COLORS['user'],
                                        font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('companion',
                                        foreground=self.COLORS['companion'],
                                        font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('system',
                                        foreground=self.COLORS['text_dim'],
                                        font=('Segoe UI', 9, 'italic'))
        self.chat_display.tag_configure('message',
                                        foreground=self.COLORS['text'],
                                        font=('Segoe UI', 11))

        # Поле ввода
        input_frame = ttk.Frame(right_panel, style='Dark.TFrame')
        input_frame.pack(fill=tk.X, pady=(15, 0))

        self.message_entry = tk.Entry(
            input_frame,
            font=('Segoe UI', 11),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            insertbackground=self.COLORS['text'],
            relief=tk.FLAT
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=8)
        self.message_entry.bind('<Return>', self._send_message)
        self.message_entry.bind('<KeyRelease>', self._on_typing)

        self.send_button = tk.Button(
            input_frame,
            text="♡ Отправить",
            font=('Segoe UI', 10, 'bold'),
            bg=self.COLORS['accent'],
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            command=self._send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        # === НИЖНЯЯ ПАНЕЛЬ ===
        footer_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        footer_frame.pack(fill=tk.X, pady=(15, 0))

        # Кнопки управления
        tk.Button(
            footer_frame,
            text="📊 Статус",
            font=('Segoe UI', 9),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self._show_status
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            footer_frame,
            text="💕 Отношения",
            font=('Segoe UI', 9),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self._show_relationship
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            footer_frame,
            text="💾 Сохранить",
            font=('Segoe UI', 9),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            relief=tk.FLAT,
            padx=15,
            pady=5,
            command=self._save_state
        ).pack(side=tk.LEFT, padx=5)

    def _start_initialization(self):
        """Запуск инициализации"""
        def init():
            try:
                save_file = "saves/liza_state.json"
                os.makedirs("saves", exist_ok=True)

                if os.path.exists(save_file):
                    self.companion = DigitalCompanion.load_state(save_file)
                    self._log_message(f"Загружена {self.companion.name}", 'system')
                else:
                    self.companion = DigitalCompanion(name="Лиза", temperament_type="sanguine")
                    self._log_message("Лиза создана!", 'system')

                # LLM
                available, _ = check_ollama_available("llama3.2")
                if available:
                    self.llm = LLMInterface(LLMConfig(provider="ollama", model="llama3.2"))
                    self._log_message("LLM подключена", 'system')
                else:
                    self._log_message("LLM недоступна - демо-режим", 'system')

                # TTS
                tts_status = check_tts_availability()
                if tts_status['recommended']:
                    provider = TTSProvider.EDGE_TTS if tts_status['edge_tts'] else TTSProvider.PYTTSX3
                    self.tts = TTSEngine(provider)

                self._update_status("● Онлайн")
                self._add_welcome_message()

            except Exception as e:
                self._log_message(f"Ошибка: {e}", 'system')
                self._update_status("● Ошибка")

        threading.Thread(target=init, daemon=True).start()
        self._update_ui()

    def _add_welcome_message(self):
        """Добавление приветственного сообщения"""
        if self.companion:
            love = 0
            if hasattr(self.companion, 'relationship'):
                love = self.companion.relationship.love.get_total_love()

            if love > 0.3:
                msg = "Привет, любимый! Рада тебя видеть! ♡"
            else:
                msg = "Привет! Я Лиза, твой цифровой компаньон. Рада познакомиться!"

            self._log_message(msg, 'companion')

            if self.tts:
                self.tts.speak(msg, emotion='happy', pleasure=0.5, arousal=0.4)

    def _update_ui(self):
        """Обновление UI"""
        try:
            while True:
                msg, tag = self.message_queue.get_nowait()
                self._add_to_chat(msg, tag)
        except queue.Empty:
            pass

        if self.companion:
            self._update_display()

        self.root.after(300, self._update_ui)

    def _animate_avatar(self):
        """Анимация аватара"""
        if self.companion:
            # Обновление аватара
            emotion = self.avatar.map_pad_to_emotion(
                self.companion.emotion.pleasure,
                self.companion.emotion.arousal,
                self.companion.emotion.dominance,
                self.companion.relationship.love.get_total_love() if hasattr(self.companion, 'relationship') else 0
            )
            self.avatar.update(
                self.companion.emotion.pleasure,
                self.companion.emotion.arousal,
                self.companion.emotion.dominance,
                self.companion.relationship.love.get_total_love() if hasattr(self.companion, 'relationship') else 0
            )

            # Рендеринг
            self.avatar_label.configure(text=self.avatar.render_unicode())
            self.emotion_label.configure(text=self.avatar.get_status_text())

        self.root.after(500, self._animate_avatar)

    def _log_message(self, message: str, tag: str = ''):
        self.message_queue.put((message, tag))

    def _add_to_chat(self, message: str, tag: str = ''):
        self.chat_display.configure(state='normal')

        timestamp = datetime.now().strftime("%H:%M")

        if tag == 'user':
            self.chat_display.insert(tk.END, f"\n[{timestamp}] ", 'system')
            self.chat_display.insert(tk.END, "Вы:\n", 'user')
            self.chat_display.insert(tk.END, f"{message}\n", 'message')
        elif tag == 'companion':
            self.chat_display.insert(tk.END, f"\n[{timestamp}] ", 'system')
            self.chat_display.insert(tk.END, "Лиза:\n", 'companion')
            self.chat_display.insert(tk.END, f"{message}\n", 'message')
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] {message}\n", 'system')

        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')

    def _send_message(self, event=None):
        message = self.message_entry.get().strip()
        if not message or not self.companion:
            return

        self.message_entry.delete(0, tk.END)
        self._add_to_chat(message, 'user')

        def process():
            interaction_type = self._classify_interaction(message)
            valence = self._estimate_valence(message)

            self.companion.process_interaction(
                interaction_type=interaction_type,
                content=message,
                valence=valence,
                intensity=0.6
            )

            for _ in range(15):
                self.companion.tick(dt=0.1)

            if self.llm:
                context = self.companion.memory.get_recent_context(n=5)
                response, _ = self.llm.generate_response(
                    user_message=message,
                    companion=self.companion,
                    memory_context=context
                )
                self._log_message(response, 'companion')

                if self.tts:
                    self.tts.speak(
                        text=response,
                        emotion=self.companion.emotion.primary_emotion,
                        pleasure=self.companion.emotion.pleasure,
                        arousal=self.companion.emotion.arousal
                    )
            else:
                responses = {
                    'love': "♡ Я тоже тебя люблю!",
                    'happy': "Мне так хорошо с тобой!",
                    'calm': "Мне нравится наш разговор...",
                }
                emotion = self.companion.emotion.primary_emotion
                self._log_message(responses.get(emotion, "Интересно..."), 'companion')

        threading.Thread(target=process, daemon=True).start()

    def _classify_interaction(self, message: str) -> str:
        ml = message.lower()
        if any(w in ml for w in ['люблю', 'обожаю', 'нравишься']):
            return 'affection_shown'
        if any(w in ml for w in ['плохо', 'грустно', 'злюсь']):
            return 'negative_interaction'
        return 'positive_interaction'

    def _estimate_valence(self, message: str) -> float:
        ml = message.lower()
        if any(w in ml for w in ['люблю', 'класс', 'супер']):
            return 0.5
        if any(w in ml for w in ['плохо', 'грустно']):
            return -0.5
        return 0.0

    def _on_typing(self, event=None):
        pass

    def _update_display(self):
        if not self.companion:
            return

        state = self.companion.get_state()
        neuro = state['neurochemistry']

        # Любовь
        if hasattr(self.companion, 'relationship'):
            love = self.companion.relationship.love.get_total_love()
            self.love_progress['value'] = love * 100
            self.love_value.configure(text=f"{love:.0%}")

        # Нейрохимия
        nt_map = {
            'Окситоцин': 'oxytocin',
            'Дофамин': 'dopamine',
            'Серотонин': 'serotonin',
            'Кортизол': 'cortisol',
        }

        for name, key in nt_map.items():
            value = neuro.get(key, 0.5)
            bar, label = self.neuro_bars[name]
            bar['value'] = value * 100
            label.configure(text=f"{value:.0%}")

    def _update_status(self, text: str):
        self.status_label.configure(text=text)

    def _show_status(self):
        if self.companion:
            window = tk.Toplevel(self.root)
            window.title("Статус Лизы")
            window.geometry("500x600")
            window.configure(bg=self.COLORS['bg_dark'])

            text = scrolledtext.ScrolledText(
                window,
                wrap=tk.WORD,
                font=('Consolas', 10),
                bg=self.COLORS['bg_medium'],
                fg=self.COLORS['text'],
                padx=15,
                pady=15
            )
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert(tk.END, self.companion.get_report())
            text.configure(state='disabled')

    def _show_relationship(self):
        if self.companion and hasattr(self.companion, 'relationship'):
            window = tk.Toplevel(self.root)
            window.title("Отношения ♡")
            window.geometry("400x500")
            window.configure(bg=self.COLORS['bg_dark'])

            text = scrolledtext.ScrolledText(
                window,
                wrap=tk.WORD,
                font=('Consolas', 10),
                bg=self.COLORS['bg_medium'],
                fg=self.COLORS['text'],
                padx=15,
                pady=15
            )
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert(tk.END, self.companion.relationship.get_relationship_report())
            text.configure(state='disabled')

    def _save_state(self):
        if self.companion:
            os.makedirs("saves", exist_ok=True)
            self.companion.save_state("saves/liza_state.json")
            messagebox.showinfo("Сохранение", "Состояние сохранено! ♡")

    def run(self):
        self.root.mainloop()


def main():
    app = CompanionGUIv2()
    app.run()


if __name__ == "__main__":
    main()

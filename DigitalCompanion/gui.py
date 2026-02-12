"""
Графический интерфейс для цифрового компаньона

Простой GUI на Tkinter с:
- Чатом
- Отображением состояния
- Аватаром
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
from typing import Optional
from datetime import datetime
import os
import sys

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.companion import DigitalCompanion
from core.llm_interface import LLMInterface, LLMConfig, check_ollama_available
from effectors.tts import TTSEngine, TTSProvider, check_tts_availability


class CompanionGUI:
    """
    Графический интерфейс компаньона
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Лиза - Цифровой компаньон")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')

        # Компоненты
        self.companion = None
        self.llm = None
        self.tts = None
        self.message_queue = queue.Queue()

        # Стили
        self._setup_styles()

        # Создание виджетов
        self._create_widgets()

        # Инициализация в отдельном потоке
        self._start_initialization()

    def _setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()
        style.theme_use('clam')

        # Цветовая схема
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'), foreground='#ff6b9d')
        style.configure('Status.TLabel', font=('Segoe UI', 9), foreground='#888888')
        style.configure('Emotion.TLabel', font=('Segoe UI', 11), foreground='#ff9eb5')

    def _create_widgets(self):
        """Создание виджетов"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Верхняя панель - заголовок и статус
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        self.title_label = ttk.Label(
            header_frame,
            text="Лиза",
            style='Title.TLabel'
        )
        self.title_label.pack(side=tk.LEFT)

        self.status_label = ttk.Label(
            header_frame,
            text="Загрузка...",
            style='Status.TLabel'
        )
        self.status_label.pack(side=tk.RIGHT)

        # Центральная область
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(fill=tk.BOTH, expand=True)

        # Левая панель - чат
        chat_frame = ttk.Frame(center_frame)
        chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # История чата
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Segoe UI', 11),
            bg='#1e1e1e',
            fg='#ffffff',
            insertbackground='#ffffff',
            selectbackground='#404040',
            height=20
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.configure(state='disabled')

        # Настройка тегов для разных типов сообщений
        self.chat_display.tag_configure('user', foreground='#6bb3f8')
        self.chat_display.tag_configure('companion', foreground='#ff9eb5')
        self.chat_display.tag_configure('system', foreground='#888888')

        # Поле ввода
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, pady=(10, 0))

        self.message_entry = ttk.Entry(
            input_frame,
            font=('Segoe UI', 11)
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.message_entry.bind('<Return>', self._send_message)

        self.send_button = ttk.Button(
            input_frame,
            text="Отправить",
            command=self._send_message
        )
        self.send_button.pack(side=tk.RIGHT)

        # Правая панель - информация
        info_frame = ttk.Frame(center_frame, width=250)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        info_frame.pack_propagate(False)

        # Аватар (простой текстовый)
        self.avatar_frame = tk.Frame(
            info_frame,
            bg='#1e1e1e',
            width=200,
            height=200
        )
        self.avatar_frame.pack(pady=(0, 10))
        self.avatar_frame.pack_propagate(False)

        self.avatar_label = tk.Label(
            self.avatar_frame,
            text="♡",
            font=('Segoe UI', 72),
            bg='#1e1e1e',
            fg='#ff6b9d'
        )
        self.avatar_label.pack(expand=True)

        # Текущее состояние
        state_frame = ttk.LabelFrame(info_frame, text="Состояние", padding="10")
        state_frame.pack(fill=tk.X, pady=(0, 10))

        self.emotion_label = ttk.Label(
            state_frame,
            text="Эмоция: —",
            style='Emotion.TLabel'
        )
        self.emotion_label.pack(anchor=tk.W)

        self.mood_label = ttk.Label(
            state_frame,
            text="Настроение: —"
        )
        self.mood_label.pack(anchor=tk.W)

        self.love_label = ttk.Label(
            state_frame,
            text="Любовь: —"
        )
        self.love_label.pack(anchor=tk.W)

        # Нейрохимия
        neuro_frame = ttk.LabelFrame(info_frame, text="Нейрохимия", padding="10")
        neuro_frame.pack(fill=tk.X, pady=(0, 10))

        self.neuro_labels = {}
        for nt in ['Окситоцин', 'Дофамин', 'Серотонин', 'Кортизол']:
            frame = ttk.Frame(neuro_frame)
            frame.pack(fill=tk.X, pady=2)

            ttk.Label(frame, text=f"{nt}:", width=10).pack(side=tk.LEFT)

            progress = ttk.Progressbar(frame, length=100, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5)
            self.neuro_labels[nt] = progress

            value_label = ttk.Label(frame, text="0%", width=5)
            value_label.pack(side=tk.LEFT)
            self.neuro_labels[f"{nt}_value"] = value_label

        # Кнопки
        buttons_frame = ttk.Frame(info_frame)
        buttons_frame.pack(fill=tk.X)

        ttk.Button(
            buttons_frame,
            text="Полный статус",
            command=self._show_status
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            buttons_frame,
            text="Сохранить",
            command=self._save_state
        ).pack(fill=tk.X, pady=2)

    def _start_initialization(self):
        """Запуск инициализации в отдельном потоке"""
        def init():
            try:
                # Создание компаньона
                save_file = "saves/liza_state.json"
                if os.path.exists(save_file):
                    self.companion = DigitalCompanion.load_state(save_file)
                    self._log_message(f"Загружена {self.companion.name}", 'system')
                else:
                    self.companion = DigitalCompanion(name="Лиза", temperament_type="sanguine")
                    self._log_message("Лиза создана!", 'system')

                # Инициализация LLM
                available, _ = check_ollama_available("llama3.2")
                if available:
                    self.llm = LLMInterface(LLMConfig(provider="ollama", model="llama3.2"))
                    self._log_message("LLM готова", 'system')
                else:
                    self._log_message("LLLM недоступна - демо-режим", 'system')

                # Инициализация TTS
                tts_status = check_tts_availability()
                if tts_status['recommended']:
                    provider = TTSProvider.EDGE_TTS if tts_status['edge_tts'] else TTSProvider.PYTTSX3
                    self.tts = TTSEngine(provider)
                    self._log_message("TTS готов", 'system')

                self._update_status("Готова к общению")
                self._update_display()

            except Exception as e:
                self._log_message(f"Ошибка инициализации: {e}", 'system')
                self._update_status("Ошибка")

        thread = threading.Thread(target=init, daemon=True)
        thread.start()

        # Запуск обновления UI
        self._update_ui()

    def _log_message(self, message: str, tag: str = ''):
        """Добавление сообщения в очередь"""
        self.message_queue.put((message, tag))

    def _update_ui(self):
        """Обновление UI из очереди"""
        try:
            while True:
                message, tag = self.message_queue.get_nowait()
                self._add_to_chat(message, tag)
        except queue.Empty:
            pass

        # Обновление состояния
        if self.companion:
            self._update_display()

        # Повторный вызов
        self.root.after(500, self._update_ui)

    def _add_to_chat(self, message: str, tag: str = ''):
        """Добавление сообщения в чат"""
        self.chat_display.configure(state='normal')

        timestamp = datetime.now().strftime("%H:%M")
        prefix = ""
        if tag == 'user':
            prefix = f"[{timestamp}] Вы: "
        elif tag == 'companion':
            prefix = f"[{timestamp}] Лиза: "
        else:
            prefix = f"[{timestamp}] "

        self.chat_display.insert(tk.END, prefix, tag)
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.see(tk.END)
        self.chat_display.configure(state='disabled')

    def _send_message(self, event=None):
        """Отправка сообщения"""
        message = self.message_entry.get().strip()
        if not message:
            return

        self.message_entry.delete(0, tk.END)

        # Отображение сообщения пользователя
        self._add_to_chat(message, 'user')

        # Обработка в отдельном потоке
        def process():
            # Классификация взаимодействия
            interaction_type = self._classify_interaction(message)
            valence = self._estimate_valence(message)

            # Обработка
            self.companion.process_interaction(
                interaction_type=interaction_type,
                content=message,
                valence=valence,
                intensity=0.6
            )

            # Тики
            for _ in range(10):
                self.companion.tick(dt=0.1)

            # Генерация ответа
            if self.llm:
                context = self.companion.memory.get_recent_context(n=5)
                response, _ = self.llm.generate_response(
                    user_message=message,
                    companion=self.companion,
                    memory_context=context
                )

                self._log_message(response, 'companion')

                # TTS
                if self.tts:
                    self.tts.speak(
                        text=response,
                        emotion=self.companion.emotion.primary_emotion,
                        pleasure=self.companion.emotion.pleasure,
                        arousal=self.companion.emotion.arousal
                    )
            else:
                # Демо-ответ
                demo_responses = {
                    'love': "Я тоже тебя люблю... ♡",
                    'joy': "Как здорово! Я так рада!",
                    'happiness': "Мне так хорошо с тобой!",
                    'calm': "Мне нравится общаться с тобой...",
                    'neutral': "Интересно...",
                }
                emotion = self.companion.emotion.primary_emotion
                response = demo_responses.get(emotion, "Я тебя слушаю...")
                self._log_message(response, 'companion')

        threading.Thread(target=process, daemon=True).start()

    def _classify_interaction(self, message: str) -> str:
        """Классификация типа взаимодействия"""
        message_lower = message.lower()
        if any(w in message_lower for w in ['люблю', 'обожаю', 'нравишься']):
            return 'affection_shown'
        if any(w in message_lower for w in ['плохо', 'грустно', 'злюсь']):
            return 'negative_interaction'
        if any(w in message_lower for w in ['хаха', 'lol', 'прикол']):
            return 'playful_interaction'
        return 'positive_interaction'

    def _estimate_valence(self, message: str) -> float:
        """Оценка валентности"""
        message_lower = message.lower()
        positive = ['люблю', 'класс', 'здорово', 'супер', 'отлично']
        negative = ['плохо', 'грустно', 'обидно', 'злюсь']
        if any(w in message_lower for w in positive):
            return 0.5
        if any(w in message_lower for w in negative):
            return -0.5
        return 0.0

    def _update_display(self):
        """Обновление отображения состояния"""
        if not self.companion:
            return

        state = self.companion.get_state()
        emotion = state['emotion']
        neuro = state['neurochemistry']

        # Эмоция
        emotion_names = {
            'love': 'Любовь ♡',
            'joy': 'Радость',
            'happiness': 'Счастье',
            'sadness': 'Грусть',
            'anger': 'Злость',
            'calm': 'Спокойствие',
            'neutral': 'Нейтрально',
        }
        self.emotion_label.configure(
            text=f"Эмоция: {emotion_names.get(emotion['primary'], emotion['primary'])}"
        )

        # Настроение
        mood = emotion['mood']
        mood_text = "отличное" if mood > 0.3 else "хорошее" if mood > 0 else "нормальное"
        self.mood_label.configure(text=f"Настроение: {mood_text}")

        # Любовь
        if hasattr(self.companion, 'relationship'):
            love = self.companion.relationship.love.get_total_love()
            self.love_label.configure(text=f"Любовь: {love:.0%}")

        # Нейрохимия
        nt_map = {
            'Окситоцин': 'oxytocin',
            'Дофамин': 'dopamine',
            'Серотонин': 'serotonin',
            'Кортизол': 'cortisol',
        }

        for name, key in nt_map.items():
            value = neuro.get(key, 0.5)
            self.neuro_labels[name]['value'] = int(value * 100)
            self.neuro_labels[f"{name}_value"].configure(text=f"{value:.0%}")

    def _update_status(self, text: str):
        """Обновление строки статуса"""
        self.status_label.configure(text=text)

    def _show_status(self):
        """Показ полного статуса"""
        if self.companion:
            report = self.companion.get_report()
            # Создание нового окна
            window = tk.Toplevel(self.root)
            window.title("Полный статус")
            window.geometry("500x600")

            text = scrolledtext.ScrolledText(
                window,
                wrap=tk.WORD,
                font=('Consolas', 10),
                bg='#1e1e1e',
                fg='#ffffff'
            )
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text.insert(tk.END, report)
            text.configure(state='disabled')

    def _save_state(self):
        """Сохранение состояния"""
        if self.companion:
            os.makedirs("saves", exist_ok=True)
            self.companion.save_state("saves/liza_state.json")
            messagebox.showinfo("Сохранение", "Состояние сохранено!")

    def run(self):
        """Запуск GUI"""
        self.root.mainloop()


def main():
    """Точка входа"""
    app = CompanionGUI()
    app.run()


if __name__ == "__main__":
    main()

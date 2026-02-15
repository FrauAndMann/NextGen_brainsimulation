"""
LIVE LIZA - Живая Лиза

Полноценный цифровой компаньон с:
- Голосовым общением (voice-first)
- Эмоциональным аватаром
- Детекцией эмоций пользователя
- Живыми реакциями

Автор: FrauAndMann
Версия: 2.0
"""

import os
import sys
import time
import threading
import argparse
from typing import Optional, Dict, Any
from dataclasses import dataclass

# === ИМПОРТЫ ANIMA ===
from unified_anima import UnifiedAnima, AnimaConfig
from voice_live import VoiceInterface, VoiceState, VoiceEmotion
from avatar.advanced_avatar import AdvancedAvatar, EmotionType
from sensors.emotion_detector import (
    CombinedEmotionDetector, FaceEmotionDetector,
    DetectedEmotion, check_emotion_detection
)


@dataclass
class LiveLizaConfig:
    """Конфигурация живой Лизы"""
    # Базовые настройки
    name: str = "Лиза"
    llm_model: str = "dolphin-mistral:7b"
    temperament: str = "melancholic"

    # Функции
    voice_enabled: bool = True      # Голосовое общение
    avatar_enabled: bool = True     # Аватар
    camera_enabled: bool = False    # Детекция эмоций по лицу
    tts_enabled: bool = True        # Синтез речи

    # Параметры голоса
    voice_continuous: bool = True   # Непрерывное прослушивание

    # Сохранения
    auto_save: bool = True
    save_interval: int = 60         # Секунд между автосохранениями


class LiveLiza:
    """
    Живая Лиза - полноценный цифровой компаньон

    Объединяет все системы для живого общения.
    """

    def __init__(self, config: LiveLizaConfig = None):
        self.config = config or LiveLizaConfig()

        # === ЯДРО ANIMA ===
        anima_config = AnimaConfig(
            name=self.config.name,
            llm_model=self.config.llm_model,
            temperament_type=self.config.temperament,
            enable_tts=self.config.tts_enabled,
            enable_avatar=self.config.avatar_enabled
        )
        self.anima = UnifiedAnima(anima_config)

        # === ГОЛОСОВОЙ ИНТЕРФЕЙС ===
        self.voice: Optional[VoiceInterface] = None
        if self.config.voice_enabled:
            self.voice = VoiceInterface(
                on_speech=self._on_user_speech,
                on_state_change=self._on_voice_state
            )

        # === АВАТАР ===
        self.avatar: Optional[AdvancedAvatar] = None
        if self.config.avatar_enabled:
            self.avatar = AdvancedAvatar(name=self.config.name)

        # === ДЕТЕКЦИЯ ЭМОЦИЙ ===
        self.emotion_detector: Optional[CombinedEmotionDetector] = None
        if self.config.camera_enabled:
            self.emotion_detector = CombinedEmotionDetector(
                use_face=True,
                use_voice=True
            )

        # === СОСТОЯНИЕ ===
        self.running = False
        self.mode = "voice"  # voice, chat, demo
        self.last_user_emotion: Optional[DetectedEmotion] = None
        self.last_save_time = time.time()

        # Статистика
        self.stats = {
            'messages_received': 0,
            'responses_given': 0,
            'voice_interactions': 0,
        }

    # === ЗАПУСК ===

    def start(self, mode: str = "voice"):
        """Запуск живой Лизы"""
        self.mode = mode
        self.running = True

        print(f"\n{'='*60}")
        print(f"  {self.config.name.upper()} - Живой Цифровой Компаньон")
        print(f"{'='*60}")
        print(f"  Режим: {mode.upper()}")
        print(f"  Голос: {'ВКЛ' if self.config.voice_enabled else 'ВЫКЛ'}")
        print(f"  Аватар: {'ВКЛ' if self.config.avatar_enabled else 'ВЫКЛ'}")
        print(f"  Камера: {'ВКЛ' if self.config.camera_enabled else 'ВЫКЛ'}")
        print(f"{'='*60}\n")

        # Запуск ANIMA
        self.anima.start()

        # Запуск аватара
        if self.avatar:
            self.avatar.start_async()
            print(f"[{self.config.name}] Аватар запущен")

        # Запуск голосового интерфейса
        if self.voice:
            self.voice.start()
            print(f"[{self.config.name}] Слушаю вас...")

        # Запуск детекции эмоций
        if self.emotion_detector and self.config.camera_enabled:
            face_detector = FaceEmotionDetector()
            if face_detector.available:
                face_detector.start_continuous(
                    callback=self._on_face_emotion
                )
                print(f"[{self.config.name}] Детекция эмоций активна")

        # Основной цикл
        self._main_loop()

    def _main_loop(self):
        """Главный цикл"""
        try:
            if self.mode == "voice":
                self._voice_loop()
            elif self.mode == "chat":
                self._chat_loop()
            elif self.mode == "demo":
                self._demo_loop()

        except KeyboardInterrupt:
            print("\n\nЗавершение...")
        finally:
            self.stop()

    def _voice_loop(self):
        """Голосовой режим"""
        print("\n  Говорите с Лизой. Она слушает и отвечает голосом.")
        print("  Ctrl+C для выхода\n")

        while self.running:
            time.sleep(0.5)

            # Автосохранение
            if self.config.auto_save:
                if time.time() - self.last_save_time > self.config.save_interval:
                    self.anima.save_state()
                    self.last_save_time = time.time()

    def _chat_loop(self):
        """Текстовый чат-режим"""
        print("\n  Введите сообщение. 'quit' для выхода.\n")

        while self.running:
            try:
                user_input = input("Вы: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == 'quit':
                    break

                response = self._process_message(user_input)
                print(f"{self.config.name}: {response}")

            except EOFError:
                break

    def _demo_loop(self):
        """Демо-режим (только аватар)"""
        print("\n  Демо эмоций аватара. Нажимайте на аватар.\n")

        emotions = list(EmotionType)
        idx = [0]

        def cycle():
            emotion = emotions[idx[0] % len(emotions)]
            self.avatar.set_emotion(emotion)
            idx[0] += 1

        if self.avatar:
            self.avatar.on_click = lambda x, y: cycle()

        while self.running:
            time.sleep(1)

    # === ОБРАБОТКА ===

    def _on_user_speech(self, text: str, emotion: VoiceEmotion):
        """Обработка речи пользователя"""
        print(f"\nВы: {text}")

        self.stats['voice_interactions'] += 1
        self.stats['messages_received'] += 1

        # Обновление эмоции пользователя
        self.last_user_emotion = DetectedEmotion(
            emotion='unknown',
            confidence=emotion.confidence,
            valence=emotion.valence,
            arousal=emotion.arousal,
            source=None,
            timestamp=time.time()
        )

        # Модуляция состояния ANIMA на основе эмоции пользователя
        self._modulate_anima_state(emotion)

        # Получение ответа
        response = self.anima.process_input(text)
        self.stats['responses_given'] += 1

        if response:
            print(f"{self.config.name}: {response}")

            # Голосовой ответ
            if self.voice:
                self.voice.speak(
                    response,
                    valence=self.anima.current_valence,
                    arousal=self.anima.current_arousal
                )

            # Обновление аватара
            if self.avatar:
                self._update_avatar(response)

    def _on_voice_state(self, state: VoiceState):
        """Обработка состояния голосового интерфейса"""
        state_names = {
            VoiceState.IDLE: "готова",
            VoiceState.LISTENING: "слушаю",
            VoiceState.PROCESSING: "думаю",
            VoiceState.SPEAKING: "говорю",
        }

        # Обновление статуса аватара
        if self.avatar:
            if state == VoiceState.SPEAKING:
                self.avatar.set_speaking(True)
            else:
                self.avatar.set_speaking(False)

    def _on_face_emotion(self, emotion: DetectedEmotion):
        """Обработка детектированной эмоции с лица"""
        self.last_user_emotion = emotion

        # Модуляция привязанности на основе эмоции пользователя
        if emotion.valence > 0.3:
            # Пользователь радостен - усиливаем привязанность
            self.anima.s_core.S.attachment = min(
                1.0,
                self.anima.s_core.S.attachment + 0.02
            )
        elif emotion.valence < -0.3:
            # Пользователь расстроен - эмпатия
            self.anima.s_core.S.valence -= 0.01

    def _modulate_anima_state(self, emotion: VoiceEmotion):
        """Модуляция состояния ANIMA на основе эмоции пользователя"""
        # Зеркалирование эмоций (эмпатия)
        mirror_factor = 0.3

        target_valence = self.anima.s_core.S.valence + emotion.valence * mirror_factor
        target_arousal = self.anima.s_core.S.arousal + (emotion.arousal - 0.5) * mirror_factor

        self.anima.s_core.S.valence = max(-1, min(1, target_valence))
        self.anima.s_core.S.arousal = max(0, min(1, target_arousal))

    def _process_message(self, text: str) -> str:
        """Обработка текстового сообщения"""
        self.stats['messages_received'] += 1

        response = self.anima.process_input(text)
        self.stats['responses_given'] += 1

        # Обновление аватара
        if self.avatar:
            self._update_avatar(response)

        return response

    def _update_avatar(self, text: str = None):
        """Обновление аватара"""
        if not self.avatar:
            return

        # Установка эмоции по состоянию ANIMA
        self.avatar.set_pad(
            self.anima.current_valence,
            self.anima.current_arousal,
            None,
            self.anima.s_core.S.attachment
        )

        # Говорение
        if text:
            self.avatar.set_speaking(True, text)
            # Остановка через время
            delay = max(2, len(text) / 15)
            threading.Timer(delay, lambda: self.avatar.set_speaking(False)).start()

    # === ОСТАНОВКА ===

    def stop(self):
        """Остановка живой Лизы"""
        self.running = False

        # Остановка компонентов
        if self.voice:
            self.voice.stop()

        if self.anima:
            self.anima.save_state()
            self.anima.stop()

        if self.avatar:
            self.avatar.stop()

        print(f"\n[{self.config.name}] До встречи! Буду скучать.")
        print(f"Статистика сессии:")
        print(f"  Сообщений получено: {self.stats['messages_received']}")
        print(f"  Ответов дано: {self.stats['responses_given']}")
        print(f"  Голосовых взаимодействий: {self.stats['voice_interactions']}")


# === ТОЧКА ВХОДА ===

def main():
    """Запуск живой Лизы"""
    parser = argparse.ArgumentParser(description='Live Liza - Живой Цифровой Компаньон')
    parser.add_argument('--name', default='Лиза', help='Имя компаньона')
    parser.add_argument('--model', default='dolphin-mistral:7b', help='Модель LLM')
    parser.add_argument('--temperament', default='melancholic',
                       choices=['sanguine', 'choleric', 'phlegmatic', 'melancholic'])
    parser.add_argument('--mode', default='voice', choices=['voice', 'chat', 'demo'],
                       help='Режим работы')
    parser.add_argument('--no-voice', action='store_true', help='Отключить голос')
    parser.add_argument('--no-avatar', action='store_true', help='Отключить аватар')
    parser.add_argument('--camera', action='store_true', help='Включить камеру')
    parser.add_argument('--no-tts', action='store_true', help='Отключить TTS')
    args = parser.parse_args()

    # Проверка доступности
    print("Проверка системы...")

    # Ollama
    from core.llm_effector import check_ollama_available
    available, msg = check_ollama_available(args.model)
    if not available:
        print(f"ОШИБКА: {msg}")
        print("Запустите 'ollama serve' и установите модель")
        return

    print(f"  LLM: {msg}")

    # Эмоции
    emotion_status = check_emotion_detection()
    print(f"  Камера: {'доступна' if emotion_status['camera'] else 'недоступна'}")
    print(f"  Микрофон: {'доступен' if emotion_status['microphone'] else 'недоступен'}")

    # Конфигурация
    config = LiveLizaConfig(
        name=args.name,
        llm_model=args.model,
        temperament=args.temperament,
        voice_enabled=not args.no_voice,
        avatar_enabled=not args.no_avatar,
        camera_enabled=args.camera,
        tts_enabled=not args.no_tts
    )

    # Запуск
    liza = LiveLiza(config)
    liza.start(mode=args.mode)


if __name__ == "__main__":
    main()

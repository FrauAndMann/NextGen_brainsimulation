"""
Voice Interface - Продвинутая голосовая система для ANIMA

Особенности:
- Непрерывное прослушивание с VAD (Voice Activity Detection)
- Эмоциональный анализ голоса в реальном времени
- Живой TTS с интонацией и паузами
- Голосовой режим "voice-first" (без текстового ввода)
- Прерывание ответа голосом пользователя

Автор: FrauAndMann
Версия: 2.0
"""

import os
import sys
import time
import threading
import queue
import asyncio
from typing import Optional, Callable, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Импорты ANIMA
from sensors.speech import SpeechToText, VoiceAnalysis, STTProvider, check_stt_availability
from effectors.tts import TTSEngine, TTSProvider


class VoiceState(Enum):
    """Состояния голосового интерфейса"""
    IDLE = "idle"               # Ожидание
    LISTENING = "listening"     # Слушает пользователя
    PROCESSING = "processing"   # Обрабатывает
    SPEAKING = "speaking"       # Говорит
    INTERRUPTED = "interrupted" # Прерван пользователем


@dataclass
class VoiceEmotion:
    """Эмоциональный анализ голоса"""
    valence: float = 0.0        # -1 до 1 (негатив - позитив)
    arousal: float = 0.3        # 0 до 1 (спокойствие - возбуждение)
    dominance: float = 0.5      # 0 до 1 (подчинение - доминирование)
    confidence: float = 0.5     # Уверенность в анализе

    # Дополнительные параметры
    speech_rate: float = 1.0    # Скорость речи
    volume: float = 0.5         # Громкость
    pitch_mean: float = 0.5     # Средняя высота голоса
    pitch_variance: float = 0.3 # Вариативность высоты


@dataclass
class EmotionalTTSConfig:
    """Конфигурация эмоционального TTS"""
    base_rate: float = 1.0
    pitch_adjust: float = 0
    volume: float = 0.9
    pause_between_sentences: float = 0.3
    emphasis_words: list = field(default_factory=list)


class VoiceActivityDetector:
    """
    Детектор голосовой активности (VAD)

    Определяет, когда пользователь говорит.
    """

    def __init__(self, energy_threshold: float = 0.02, hangover_frames: int = 10):
        self.energy_threshold = energy_threshold
        self.hangover_frames = hangover_frames
        self.hangover_count = 0
        self.is_speaking = False
        self.silence_frames = 0
        self.min_speech_frames = 5
        self.speech_frame_count = 0

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Обработать аудиокадр

        Returns:
            True если обнаружена речь
        """
        energy = np.sqrt(np.mean(audio_frame ** 2))

        if energy > self.energy_threshold:
            self.speech_frame_count += 1
            self.hangover_count = self.hangover_frames
            self.silence_frames = 0

            if self.speech_frame_count >= self.min_speech_frames:
                self.is_speaking = True
        else:
            self.hangover_count -= 1
            if self.hangover_count <= 0:
                self.is_speaking = False
                self.speech_frame_count = 0
            self.silence_frames += 1

        return self.is_speaking

    def reset(self):
        """Сброс детектора"""
        self.hangover_count = 0
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frame_count = 0


class EmotionalTTS:
    """
    Эмоциональный синтез речи

    Делает голос живым с интонацией и паузами.
    """

    # Эмоциональные пресеты для TTS
    EMOTION_PRESETS = {
        'joy': EmotionalTTSConfig(
            base_rate=1.1,
            pitch_adjust=10,
            volume=0.95,
            pause_between_sentences=0.2,
            emphasis_words=['!', 'здорово', 'класс', 'супер', 'обожаю']
        ),
        'love': EmotionalTTSConfig(
            base_rate=0.95,
            pitch_adjust=5,
            volume=0.85,
            pause_between_sentences=0.4,
            emphasis_words=['люблю', 'родной', 'милый', 'нежно']
        ),
        'sadness': EmotionalTTSConfig(
            base_rate=0.85,
            pitch_adjust=-10,
            volume=0.7,
            pause_between_sentences=0.5,
            emphasis_words=['...', 'грустно', 'жаль']
        ),
        'anger': EmotionalTTSConfig(
            base_rate=1.15,
            pitch_adjust=-5,
            volume=1.0,
            pause_between_sentences=0.15,
            emphasis_words=['!']
        ),
        'excitement': EmotionalTTSConfig(
            base_rate=1.2,
            pitch_adjust=15,
            volume=1.0,
            pause_between_sentences=0.15,
            emphasis_words=['!', 'вау', 'ого', 'нереально']
        ),
        'calm': EmotionalTTSConfig(
            base_rate=0.9,
            pitch_adjust=0,
            volume=0.75,
            pause_between_sentences=0.4,
            emphasis_words=[]
        ),
        'interest': EmotionalTTSConfig(
            base_rate=1.0,
            pitch_adjust=5,
            volume=0.85,
            pause_between_sentences=0.3,
            emphasis_words=['?', 'интересно', 'любопытно']
        ),
        'neutral': EmotionalTTSConfig(
            base_rate=1.0,
            pitch_adjust=0,
            volume=0.85,
            pause_between_sentences=0.3,
            emphasis_words=[]
        ),
    }

    def __init__(self):
        self.tts = TTSEngine(provider=TTSProvider.EDGE_TTS)
        self.current_emotion = 'neutral'

    def speak(self, text: str, emotion: str = None, valence: float = 0.0,
              arousal: float = 0.3, on_complete: Callable = None) -> bool:
        """
        Произнести текст с эмоциональной интонацией

        Args:
            text: Текст для произнесения
            emotion: Название эмоции
            valence: Валентность (-1 до 1)
            arousal: Возбуждение (0 до 1)
            on_complete: Callback при завершении
        """
        if not text.strip():
            return False

        # Определение эмоции если не указана
        if emotion is None:
            emotion = self._infer_emotion(valence, arousal)

        self.current_emotion = emotion

        # Получение пресета
        preset = self.EMOTION_PRESETS.get(emotion, self.EMOTION_PRESETS['neutral'])

        # Модификация текста для выразительности
        processed_text = self._process_text(text, preset)

        # Разбивка на предложения для естественных пауз
        sentences = self._split_sentences(processed_text)

        def speak_thread():
            try:
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        # Определение скорости и тона для предложения
                        rate = preset.base_rate
                        pitch = preset.pitch_adjust

                        # Модификация на основе позиции в тексте
                        if i == 0:
                            rate *= 1.02  # Первое предложение чуть быстрее
                        elif i == len(sentences) - 1:
                            rate *= 0.98  # Последнее чуть медленнее

                        self.tts.speak(
                            sentence,
                            emotion=emotion,
                            pleasure=valence,
                            arousal=arousal,
                            blocking=True
                        )

                        # Пауза между предложениями
                        if i < len(sentences) - 1:
                            time.sleep(preset.pause_between_sentences)

                if on_complete:
                    on_complete()

            except Exception as e:
                print(f"[TTS] Ошибка: {e}")

        # Запуск в отдельном потоке
        threading.Thread(target=speak_thread, daemon=True).start()
        return True

    def _infer_emotion(self, valence: float, arousal: float) -> str:
        """Определение эмоции по PAD"""
        if valence > 0.4:
            if arousal > 0.6:
                return 'excitement'
            elif arousal < 0.3:
                return 'calm'
            else:
                return 'joy'
        elif valence < -0.4:
            if arousal > 0.6:
                return 'anger'
            else:
                return 'sadness'
        else:
            if arousal > 0.5:
                return 'interest'
            else:
                return 'neutral'

    def _process_text(self, text: str, preset: EmotionalTTSConfig) -> str:
        """Обработка текста для выразительности"""
        # Добавление пауз
        text = text.replace('...', ' ... ')
        text = text.replace('—', ' ... ')

        return text.strip()

    def _split_sentences(self, text: str) -> list:
        """Разбивка на предложения"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def stop(self):
        """Остановить воспроизведение"""
        # Edge-TTS не поддерживает прямую остановку
        pass


class VoiceInterface:
    """
    Голосовой интерфейс ANIMA

    Объединяет распознавание и синтез речи с эмоциональностью.
    """

    def __init__(
        self,
        on_speech: Callable[[str, VoiceEmotion], None] = None,
        on_state_change: Callable[[VoiceState], None] = None,
        continuous: bool = True
    ):
        """
        Args:
            on_speech: Callback при распознавании речи
            on_state_change: Callback при смене состояния
            continuous: Непрерывное прослушивание
        """
        self.on_speech = on_speech
        self.on_state_change = on_state_change
        self.continuous = continuous

        # Компоненты - автоматически выбираем лучший провайдер
        self.stt = SpeechToText()  # Автовыбор: Whisper (если есть ffmpeg) -> Google
        self.tts = EmotionalTTS()
        self.vad = VoiceActivityDetector()

        # Состояние
        self.state = VoiceState.IDLE
        self.running = False
        self._listen_thread = None
        self._audio_queue = queue.Queue()

        # Параметры аудио
        self.sample_rate = 16000
        self.channels = 1

        # Буфер для накопления речи
        self._speech_buffer = []
        self._max_buffer_seconds = 30

    def start(self):
        """Запуск голосового интерфейса"""
        if self.running:
            return

        self.running = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        print("[Voice] Голосовой интерфейс запущен")

    def stop(self):
        """Остановка голосового интерфейса"""
        self.running = False
        if self._listen_thread:
            self._listen_thread.join(timeout=2)
        print("[Voice] Голосовой интерфейс остановлен")

    def _set_state(self, new_state: VoiceState):
        """Установка нового состояния"""
        if self.state != new_state:
            self.state = new_state
            if self.on_state_change:
                self.on_state_change(new_state)

    def _listen_loop(self):
        """Главный цикл прослушивания"""
        try:
            import sounddevice as sd
        except ImportError:
            print("[Voice] sounddevice не установлен: pip install sounddevice")
            return

        print("[Voice] Начинаю слушать...")

        audio_buffer = []
        is_recording = False
        silence_frames = 0
        max_silence = int(self.sample_rate * 1.5 / 512)  # 1.5 секунды тишины

        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_buffer, is_recording, silence_frames

            if self.state == VoiceState.SPEAKING:
                # Не слушаем пока говорим
                return

            frame = indata[:, 0]
            has_speech = self.vad.process_frame(frame)

            if has_speech:
                if not is_recording:
                    self._set_state(VoiceState.LISTENING)
                    is_recording = True
                    print("[Voice] Слушаю...", end="", flush=True)

                audio_buffer.append(frame.copy())
                silence_frames = 0

            elif is_recording:
                silence_frames += 1
                audio_buffer.append(frame.copy())

                if silence_frames >= max_silence:
                    # Конец фразы
                    print(" обрабатываю...", flush=True)
                    self._process_speech(audio_buffer)
                    audio_buffer = []
                    is_recording = False
                    silence_frames = 0
                    self.vad.reset()

        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.float32,
                blocksize=512
            ):
                while self.running:
                    time.sleep(0.1)

        except Exception as e:
            print(f"[Voice] Ошибка: {e}")

    def _process_speech(self, audio_buffer: list):
        """Обработка распознанной речи"""
        self._set_state(VoiceState.PROCESSING)

        try:
            # Объединение буфера
            audio_data = np.concatenate(audio_buffer)

            # Распознавание (Google работает напрямую с numpy array)
            result = self.stt.transcribe(audio_data=audio_data)

            if result.text.strip():
                # Анализ эмоций
                emotion = VoiceEmotion(
                    valence=result.valence,
                    arousal=result.arousal,
                    confidence=result.confidence,
                    speech_rate=result.speech_rate if hasattr(result, 'speech_rate') else 1.0,
                    volume=result.volume if hasattr(result, 'volume') else 0.5
                )

                print(f"[Voice] Распознано: {result.text}")

                # Callback
                if self.on_speech:
                    self.on_speech(result.text, emotion)

        except Exception as e:
            print(f"[Voice] Ошибка обработки: {e}")

        finally:
            self._set_state(VoiceState.IDLE)

    def speak(self, text: str, valence: float = 0.0, arousal: float = 0.3,
              emotion: str = None, on_complete: Callable = None):
        """
        Произнести текст

        Args:
            text: Текст для произнесения
            valence: Эмоциональная валентность
            arousal: Уровень возбуждения
            emotion: Название эмоции
            on_complete: Callback при завершении
        """
        self._set_state(VoiceState.SPEAKING)

        def complete():
            self._set_state(VoiceState.IDLE)
            if on_complete:
                on_complete()

        self.tts.speak(text, emotion, valence, arousal, complete)

    def is_speaking(self) -> bool:
        """Проверка, говорит ли система"""
        return self.state == VoiceState.SPEAKING

    def is_listening(self) -> bool:
        """Проверка, слушает ли система"""
        return self.state == VoiceState.LISTENING


# === ГОЛОСОВОЙ РЕЖИМ ANIMA ===

class VoiceAnima:
    """
    ANIMA в голосовом режиме

    Полностью голосовое общение без текстового ввода.
    """

    def __init__(self, unified_anima):
        """
        Args:
            unified_anima: Экземпляр UnifiedAnima
        """
        self.anima = unified_anima

        # Голосовой интерфейс
        self.voice = VoiceInterface(
            on_speech=self._on_speech,
            on_state_change=self._on_voice_state
        )

        # Состояние
        self.running = False
        self.last_response = ""

    def _on_speech(self, text: str, emotion: VoiceEmotion):
        """Обработка речи пользователя"""
        print(f"\nВы: {text}")

        # Передача эмоций в ANIMA
        self.anima.current_valence = emotion.valence
        self.anima.current_arousal = emotion.arousal

        # Получение ответа
        response = self.anima.process_input(text)

        if response:
            print(f"{self.anima.name}: {response}")
            self.last_response = response

            # Голосовой ответ
            self.voice.speak(
                response,
                valence=self.anima.current_valence,
                arousal=self.anima.current_arousal
            )

    def _on_voice_state(self, state: VoiceState):
        """Обработка смены состояния голосового интерфейса"""
        state_names = {
            VoiceState.IDLE: "готова",
            VoiceState.LISTENING: "слушаю",
            VoiceState.PROCESSING: "думаю",
            VoiceState.SPEAKING: "говорю",
        }
        print(f"[Voice] Состояние: {state_names.get(state, state.value)}")

    def start(self):
        """Запуск голосового режима"""
        print(f"\n{'='*50}")
        print(f"  {self.anima.name} - Голосовой режим")
        print(f"{'='*50}")
        print("  Говорите - я слушаю и отвечаю голосом")
        print("  Нажмите Ctrl+C для выхода")
        print(f"{'='*50}\n")

        self.running = True
        self.anima.start()
        self.voice.start()

        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\nЗавершение...")
            self.stop()

    def stop(self):
        """Остановка голосового режима"""
        self.running = False
        self.voice.stop()
        self.anima.stop()
        self.anima.save_state()
        print(f"\n{self.anima.name}: До встречи! Буду скучать.")


# === ТОЧКА ВХОДА ===

def main():
    """Запуск голосового режима"""
    import argparse

    parser = argparse.ArgumentParser(description='ANIMA Voice Mode')
    parser.add_argument('--name', default='Лиза', help='Имя компаньона')
    parser.add_argument('--model', default='dolphin-mistral:7b', help='Модель LLM')
    parser.add_argument('--temperament', default='melancholic',
                       choices=['sanguine', 'choleric', 'phlegmatic', 'melancholic'])
    args = parser.parse_args()

    # Импорт UnifiedAnima
    from unified_anima import UnifiedAnima, AnimaConfig

    # Конфигурация
    config = AnimaConfig(
        name=args.name,
        llm_model=args.model,
        temperament_type=args.temperament,
        enable_tts=True,
        enable_avatar=False  # В голосовом режиме аватар опционален
    )

    # Проверка Ollama
    from core.llm_effector import check_ollama_available
    available, msg = check_ollama_available(config.llm_model)
    if not available:
        print(f"Ошибка: {msg}")
        return

    # Создание ANIMA
    anima = UnifiedAnima(config)

    # Голосовой режим
    voice_anima = VoiceAnima(anima)
    voice_anima.start()


if __name__ == "__main__":
    main()

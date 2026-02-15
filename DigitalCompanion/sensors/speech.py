"""
Speech Recognition Module - Распознавание речи для ANIMA

Поддерживает:
- Whisper (локальный, высокое качество)
- SpeechRecognition (Google, работает онлайн)
- Анализ интонации и эмоций из голоса
- sounddevice (альтернатива PyAudio)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time
import tempfile
import wave
import os


class STTProvider(Enum):
    """Провайдеры распознавания речи"""
    WHISPER = "whisper"       # Локальный Whisper
    GOOGLE = "google"         # Google Speech Recognition
    VOSK = "vosk"            # Vosk (офлайн)


@dataclass
class VoiceAnalysis:
    """Анализ голоса"""
    text: str                          # Распознанный текст
    valence: float = 0.0               # Эмоциональная валентность (-1 до 1)
    arousal: float = 0.3               # Уровень возбуждения (0 до 1)
    confidence: float = 0.5            # Уверенность распознавания
    speech_rate: float = 1.0           # Скорость речи
    volume: float = 0.5                # Громкость
    pitch_variation: float = 0.5       # Вариация высоты голоса


class SpeechToText:
    """
    Модуль распознавания речи

    Поддерживает несколько провайдеров с автоматическим выбором.
    """

    def __init__(self, provider: STTProvider = STTProvider.WHISPER):
        self.provider = provider
        self._whisper_model = None
        self._recognizer = None
        self._vosk_model = None

        # Очередь для асинхронной обработки
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._processing = False

        self._initialize_provider()

    def _initialize_provider(self):
        """Инициализация провайдера"""
        if self.provider == STTProvider.WHISPER:
            try:
                import whisper
                # Используем базовую модель (можно заменить на 'small', 'medium')
                self._whisper_model = whisper.load_model("base")
                print("[STT] Whisper инициализирован (base model)")
            except ImportError:
                print("[STT] Whisper не установлен. Установите: pip install openai-whisper")
                self.provider = STTProvider.GOOGLE

        if self.provider == STTProvider.GOOGLE:
            try:
                import speech_recognition as sr
                self._recognizer = sr.Recognizer()
                print("[STT] SpeechRecognition готов (Google)")
            except ImportError:
                print("[STT] SpeechRecognition не установлен: pip install SpeechRecognition")
                self.provider = None

        if self.provider == STTProvider.VOSK:
            try:
                from vosk import Model, KaldiRecognizer
                # Нужно скачать модель отдельно
                self._vosk_model = Model("models/vosk-model-ru")
                print("[STT] Vosk инициализирован")
            except Exception as e:
                print(f"[STT] Vosk недоступен: {e}")
                self.provider = None

    def transcribe(self, audio_data: bytes = None, audio_file: str = None) -> VoiceAnalysis:
        """
        Транскрибация аудио в текст

        Args:
            audio_data: Сырые аудиоданные
            audio_file: Путь к аудиофайлу

        Returns:
            VoiceAnalysis с текстом и анализом эмоций
        """
        if self.provider == STTProvider.WHISPER and self._whisper_model:
            return self._transcribe_whisper(audio_file)

        elif self.provider == STTProvider.GOOGLE:
            return self._transcribe_google(audio_data)

        return VoiceAnalysis(text="", confidence=0)

    def _transcribe_whisper(self, audio_file: str) -> VoiceAnalysis:
        """Транскрибация через Whisper"""
        try:
            # Транскрибация
            result = self._whisper_model.transcribe(
                audio_file,
                language="ru",
                task="transcribe"
            )

            text = result.get("text", "").strip()

            # Анализ эмоций (упрощённый на основе текста)
            valence, arousal = self._analyze_text_emotion(text)

            return VoiceAnalysis(
                text=text,
                valence=valence,
                arousal=arousal,
                confidence=0.9  # Whisper обычно точен
            )

        except Exception as e:
            print(f"[STT] Ошибка Whisper: {e}")
            return VoiceAnalysis(text="", confidence=0)

    def _transcribe_google(self, audio_data: bytes) -> VoiceAnalysis:
        """Транскрибация через Google"""
        try:
            import speech_recognition as sr

            # Конвертация в нужный формат
            audio = sr.AudioData(audio_data, 16000, 2)
            text = self._recognizer.recognize_google(audio, language="ru-RU")

            valence, arousal = self._analyze_text_emotion(text)

            return VoiceAnalysis(
                text=text,
                valence=valence,
                arousal=arousal,
                confidence=0.8
            )

        except sr.UnknownValueError:
            return VoiceAnalysis(text="", confidence=0)
        except Exception as e:
            print(f"[STT] Ошибка Google: {e}")
            return VoiceAnalysis(text="", confidence=0)

    def _analyze_text_emotion(self, text: str) -> Tuple[float, float]:
        """Анализ эмоций в тексте"""
        text_lower = text.lower()

        # Позитивные маркеры
        positive = ['люблю', 'рад', 'счастлив', 'прекрасно', 'отлично', 'класс',
                   'супер', 'обожаю', 'здорово', 'замечательно', 'привет', 'спасибо']
        # Негативные маркеры
        negative = ['ненавижу', 'плохо', 'ужасно', 'грустно', 'обидно', 'злюсь',
                   'бесит', 'устал', 'надоело', 'скучно', 'пошёл', 'отстань']
        # Возбуждающие маркеры
        exciting = ['!', '?', 'вау', 'ого', 'нереально', 'срочно', 'быстро',
                   'сейчас', 'немедленно', 'очень', 'сильно']
        # Спокойные маркеры
        calm = ['...', 'спокойно', 'тихо', 'медленно', 'не торопись', 'ладно']

        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        exc_count = sum(1 for w in exciting if w in text_lower)
        calm_count = sum(1 for w in calm if w in text_lower)

        valence = (pos_count - neg_count) * 0.2
        valence = max(-1, min(1, valence))

        arousal = 0.3 + (exc_count - calm_count) * 0.1
        arousal = max(0, min(1, arousal))

        return valence, arousal

    def listen_from_microphone(self, timeout: int = 5) -> VoiceAnalysis:
        """
        Слушать с микрофона (использует sounddevice)

        Args:
            timeout: Таймаут ожидания в секундах

        Returns:
            VoiceAnalysis с распознанным текстом
        """
        try:
            import sounddevice as sd

            SAMPLE_RATE = 16000
            CHANNELS = 1

            print("[STT] Слушаю...", end="", flush=True)

            # Запись аудио с автоматическим определением тишины
            audio_chunks = []
            silence_threshold = 0.01
            silence_duration = 0
            max_silence = 1.5  # секунд тишины до остановки
            recording = False
            start_time = time.time()

            def audio_callback(indata, frames, time_info, status):
                nonlocal audio_chunks, recording, silence_duration, start_time
                volume_norm = np.linalg.norm(indata) / frames

                if volume_norm > silence_threshold:
                    if not recording:
                        recording = True
                        print(" [запись] ", end="", flush=True)
                    audio_chunks.append(indata.copy())
                    silence_duration = 0
                elif recording:
                    silence_duration += frames / SAMPLE_RATE
                    audio_chunks.append(indata.copy())  # Записываем даже тишину для естественности

            # Начинаем прослушивание
            with sd.InputStream(callback=audio_callback,
                               channels=CHANNELS,
                               samplerate=SAMPLE_RATE,
                               dtype=np.float32):
                while time.time() - start_time < timeout:
                    if recording and silence_duration >= max_silence:
                        break
                    time.sleep(0.05)

            if not audio_chunks:
                print(" не услышал")
                return VoiceAnalysis(text="", confidence=0)

            print(" обработка...", end="", flush=True)

            # Объединяем чанки
            audio_data = np.concatenate(audio_chunks, axis=0)

            # Конвертируем в int16 для WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())

            # Распознаём
            if self.provider == STTProvider.WHISPER and self._whisper_model:
                result = self._transcribe_whisper(temp_path)
            else:
                # Для Google используем байты
                result = self._transcribe_google(audio_int16.tobytes())

            # Удаляем временный файл
            try:
                os.unlink(temp_path)
            except:
                pass

            print(f" готово!")
            return result

        except ImportError:
            print("[STT] sounddevice не установлен: pip install sounddevice")
            return VoiceAnalysis(text="", confidence=0)
        except Exception as e:
            print(f" ошибка: {e}")
            import traceback
            traceback.print_exc()
            return VoiceAnalysis(text="", confidence=0)

    def is_available(self) -> bool:
        """Проверка доступности"""
        return self.provider is not None


def check_stt_availability() -> Dict[str, bool]:
    """Проверка доступности STT провайдеров"""
    result = {
        'whisper': False,
        'google': False,
        'vosk': False,
        'microphone': False,
        'recommended': None
    }

    # Проверка Whisper
    try:
        import whisper
        result['whisper'] = True
        result['recommended'] = 'whisper'
    except ImportError:
        pass

    # Проверка Google Speech Recognition
    try:
        import speech_recognition
        result['google'] = True
        if result['recommended'] is None:
            result['recommended'] = 'google'
    except ImportError:
        pass

    # Проверка Vosk
    try:
        from vosk import Model
        result['vosk'] = True
    except ImportError:
        pass

    # Проверка микрофона через sounddevice
    try:
        import sounddevice as sd
        # Проверяем, есть ли доступные устройства ввода
        devices = sd.query_devices()
        has_input = any(d['max_input_channels'] > 0 for d in devices)
        if has_input:
            result['microphone'] = True
    except:
        pass

    return result

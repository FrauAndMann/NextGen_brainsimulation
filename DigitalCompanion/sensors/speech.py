"""
Speech Recognition Module - Распознавание речи для ANIMA

Поддерживает:
- Google Speech Recognition (онлайн, работает всегда)
- Whisper (локальный, требует ffmpeg)
- Автоматический fallback

Автор: FrauAndMann
Версия: 2.0
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
import io


class STTProvider(Enum):
    """Провайдеры распознавания речи"""
    GOOGLE = "google"         # Google Speech Recognition (рекомендуется)
    WHISPER = "whisper"       # Локальный Whisper (требует ffmpeg)
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

    Автоматически выбирает лучший доступный провайдер.
    Приоритет: Whisper -> Google
    """

    def __init__(self, provider: STTProvider = None):
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
        """Инициализация провайдера с автоматическим выбором"""
        # Сначала пробуем Google (работает всегда онлайн)
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.8

            if self.provider is None:
                self.provider = STTProvider.GOOGLE

            print("[STT] SpeechRecognition готов (Google)")
        except ImportError:
            print("[STT] SpeechRecognition не установлен: pip install SpeechRecognition")

        # Пробуем Whisper если явно указан или Google недоступен
        if self.provider == STTProvider.WHISPER or self.provider is None:
            try:
                # Проверка ffmpeg
                import subprocess
                result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
                if result.returncode != 0:
                    raise FileNotFoundError("ffmpeg не найден")

                import whisper
                self._whisper_model = whisper.load_model("base")
                if self.provider is None:
                    self.provider = STTProvider.WHISPER
                print("[STT] Whisper инициализирован (base model)")

            except (ImportError, FileNotFoundError) as e:
                if self.provider == STTProvider.WHISPER:
                    print(f"[STT] Whisper недоступен: {e}")
                    print("[STT] Использую Google Speech Recognition")
                    self.provider = STTProvider.GOOGLE
                elif self.provider is None and self._recognizer:
                    self.provider = STTProvider.GOOGLE

        # Vosk как последний вариант
        if self.provider == STTProvider.VOSK:
            try:
                from vosk import Model
                model_path = os.path.join(os.path.dirname(__file__), "..", "models", "vosk-model-ru")
                if os.path.exists(model_path):
                    self._vosk_model = Model(model_path)
                    print("[STT] Vosk инициализирован")
                else:
                    raise FileNotFoundError(f"Модель не найдена: {model_path}")
            except Exception as e:
                print(f"[STT] Vosk недоступен: {e}")
                if self._recognizer:
                    self.provider = STTProvider.GOOGLE

        # Финальная проверка
        if self.provider is None:
            print("[STT] ВНИМАНИЕ: Ни один провайдер не доступен!")
            print("[STT] Установите: pip install SpeechRecognition")

    def transcribe(self, audio_data: bytes = None, audio_file: str = None) -> VoiceAnalysis:
        """
        Транскрибация аудио в текст

        Args:
            audio_data: Сырые аудиоданные (numpy array или bytes)
            audio_file: Путь к аудиофайлу

        Returns:
            VoiceAnalysis с текстом и анализом эмоций
        """
        if self.provider == STTProvider.WHISPER and self._whisper_model and audio_file:
            return self._transcribe_whisper(audio_file)

        elif self.provider == STTProvider.GOOGLE and self._recognizer:
            return self._transcribe_google(audio_data)

        elif self.provider == STTProvider.VOSK and self._vosk_model:
            return self._transcribe_vosk(audio_data)

        return VoiceAnalysis(text="", confidence=0)

    def _transcribe_whisper(self, audio_file: str) -> VoiceAnalysis:
        """Транскрибация через Whisper"""
        try:
            result = self._whisper_model.transcribe(
                audio_file,
                language="ru",
                task="transcribe",
                fp16=False  # Для совместимости с CPU
            )

            text = result.get("text", "").strip()
            valence, arousal = self._analyze_text_emotion(text)

            return VoiceAnalysis(
                text=text,
                valence=valence,
                arousal=arousal,
                confidence=0.9
            )

        except Exception as e:
            print(f"[STT] Ошибка Whisper: {e}")
            # Fallback на Google
            if self._recognizer:
                return self._transcribe_google_fallback(audio_file)
            return VoiceAnalysis(text="", confidence=0)

    def _transcribe_google(self, audio_data) -> VoiceAnalysis:
        """Транскрибация через Google Speech Recognition"""
        try:
            import speech_recognition as sr

            # Конвертация в нужный формат
            if isinstance(audio_data, np.ndarray):
                # Нормализация
                audio_data = np.clip(audio_data, -1, 1)
                audio_int16 = (audio_data * 32767).astype(np.int16)

                # Создание WAV в памяти
                with io.BytesIO() as wav_buffer:
                    with wave.open(wav_buffer, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(audio_int16.tobytes())
                    wav_buffer.seek(0)
                    audio_data = wav_buffer.read()

            audio = sr.AudioData(audio_data, 16000, 2)
            text = self._recognizer.recognize_google(audio, language="ru-RU")

            valence, arousal = self._analyze_text_emotion(text)

            return VoiceAnalysis(
                text=text,
                valence=valence,
                arousal=arousal,
                confidence=0.85
            )

        except sr.UnknownValueError:
            return VoiceAnalysis(text="", confidence=0)
        except sr.RequestError as e:
            print(f"[STT] Ошибка Google API: {e}")
            return VoiceAnalysis(text="", confidence=0)
        except Exception as e:
            print(f"[STT] Ошибка: {e}")
            return VoiceAnalysis(text="", confidence=0)

    def _transcribe_google_fallback(self, audio_file: str) -> VoiceAnalysis:
        """Fallback на Google из файла"""
        try:
            import speech_recognition as sr

            with sr.AudioFile(audio_file) as source:
                audio = self._recognizer.record(source)

            text = self._recognizer.recognize_google(audio, language="ru-RU")
            valence, arousal = self._analyze_text_emotion(text)

            return VoiceAnalysis(
                text=text,
                valence=valence,
                arousal=arousal,
                confidence=0.8
            )

        except Exception as e:
            print(f"[STT] Fallback ошибка: {e}")
            return VoiceAnalysis(text="", confidence=0)

    def _transcribe_vosk(self, audio_data) -> VoiceAnalysis:
        """Транскрибация через Vosk"""
        try:
            from vosk import KaldiRecognizer

            recognizer = KaldiRecognizer(self._vosk_model, 16000)

            if isinstance(audio_data, np.ndarray):
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
            else:
                audio_bytes = audio_data

            recognizer.AcceptWaveform(audio_bytes)
            result = recognizer.FinalResult()

            import json
            data = json.loads(result)
            text = data.get("text", "").strip()

            valence, arousal = self._analyze_text_emotion(text)

            return VoiceAnalysis(
                text=text,
                valence=valence,
                arousal=arousal,
                confidence=0.7
            )

        except Exception as e:
            print(f"[STT] Ошибка Vosk: {e}")
            return VoiceAnalysis(text="", confidence=0)

    def _analyze_text_emotion(self, text: str) -> Tuple[float, float]:
        """Анализ эмоций в тексте"""
        text_lower = text.lower()

        # Позитивные маркеры
        positive = ['люблю', 'рад', 'счастлив', 'прекрасно', 'отлично', 'класс',
                   'супер', 'обожаю', 'здорово', 'замечательно', 'привет', 'спасибо',
                   'милый', 'милая', 'родной', 'родная', 'нежный', 'красивая']
        # Негативные маркеры
        negative = ['ненавижу', 'плохо', 'ужасно', 'грустно', 'обидно', 'злюсь',
                   'бесит', 'устал', 'надоело', 'скучно', 'пошёл', 'отстань',
                   'глупый', 'глупая', 'разочарован']
        # Возбуждающие маркеры
        exciting = ['!', '?', 'вау', 'ого', 'нереально', 'срочно', 'быстро',
                   'сейчас', 'немедленно', 'очень', 'сильно', 'обожаю']
        # Спокойные маркеры
        calm = ['...', 'спокойно', 'тихо', 'медленно', 'не торопись', 'ладно',
               'хорошо', 'понятно']

        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        exc_count = sum(1 for w in exciting if w in text_lower)
        calm_count = sum(1 for w in calm if w in text_lower)

        # Подсчёт восклицательных знаков
        exc_count += text.count('!')

        valence = (pos_count - neg_count) * 0.15
        valence = max(-1, min(1, valence))

        arousal = 0.3 + (exc_count - calm_count) * 0.08
        arousal = max(0, min(1, arousal))

        return valence, arousal

    def listen_from_microphone(self, timeout: int = 10) -> VoiceAnalysis:
        """
        Слушать с микрофона

        Args:
            timeout: Максимальное время ожидания в секундах

        Returns:
            VoiceAnalysis с распознанным текстом
        """
        try:
            import sounddevice as sd
        except ImportError:
            print("[STT] sounddevice не установлен: pip install sounddevice")
            return VoiceAnalysis(text="", confidence=0)

        SAMPLE_RATE = 16000
        CHANNELS = 1

        print("[STT] Слушаю...", end="", flush=True)

        # Запись аудио с автоматическим определением тишины
        audio_chunks = []
        silence_threshold = 0.015
        silence_duration = 0
        max_silence = 1.5  # секунд тишины до остановки
        min_speech = 0.3   # минимум речи для обработки
        recording = False
        speech_duration = 0
        start_time = time.time()

        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_chunks, recording, silence_duration, speech_duration, start_time

            volume_norm = np.linalg.norm(indata) / frames

            if volume_norm > silence_threshold:
                if not recording:
                    recording = True
                    print(" [запись]", end="", flush=True)
                    start_time = time.time()
                audio_chunks.append(indata.copy())
                silence_duration = 0
                speech_duration += frames / SAMPLE_RATE
            elif recording:
                silence_duration += frames / SAMPLE_RATE
                audio_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                callback=audio_callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype=np.float32
            ):
                while time.time() - start_time < timeout:
                    if recording and silence_duration >= max_silence:
                        break
                    if speech_duration > 0 and not recording:
                        break
                    time.sleep(0.05)

        except Exception as e:
            print(f" ошибка: {e}")
            return VoiceAnalysis(text="", confidence=0)

        if not audio_chunks or speech_duration < min_speech:
            print(" (не услышал)")
            return VoiceAnalysis(text="", confidence=0)

        print(" обрабатываю...", end="", flush=True)

        # Объединяем чанки
        audio_data = np.concatenate(audio_chunks, axis=0).flatten()

        # Распознаём
        result = self.transcribe(audio_data=audio_data)

        if result.text:
            print(f" готово!")
            print(f"[STT] Распознано: {result.text}")
        else:
            print(" (не распознал)")

        return result

    def is_available(self) -> bool:
        """Проверка доступности"""
        return self.provider is not None


def check_stt_availability() -> Dict[str, bool]:
    """Проверка доступности STT провайдеров"""
    result = {
        'google': False,
        'whisper': False,
        'vosk': False,
        'microphone': False,
        'ffmpeg': False,
        'recommended': None
    }

    # Проверка Google Speech Recognition
    try:
        import speech_recognition
        result['google'] = True
        result['recommended'] = 'google'
    except ImportError:
        pass

    # Проверка ffmpeg
    try:
        import subprocess
        r = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        result['ffmpeg'] = r.returncode == 0
    except:
        pass

    # Проверка Whisper
    try:
        import whisper
        if result['ffmpeg']:
            result['whisper'] = True
            result['recommended'] = 'whisper'
    except ImportError:
        pass

    # Проверка Vosk
    try:
        from vosk import Model
        result['vosk'] = True
    except ImportError:
        pass

    # Проверка микрофона
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        has_input = any(d['max_input_channels'] > 0 for d in devices)
        if has_input:
            result['microphone'] = True
    except:
        pass

    return result


if __name__ == "__main__":
    print("Проверка STT:")
    status = check_stt_availability()
    for k, v in status.items():
        print(f"  {k}: {'OK' if v else '---'}")

    print("\nТест записи (говорите):")
    stt = SpeechToText()
    result = stt.listen_from_microphone()
    print(f"Результат: {result.text}")

"""
Система синтеза речи (TTS) для цифрового компаньона

Поддерживает:
- pyttsx3 (локальный, работает оффлайн)
- edge-tts (Microsoft Edge TTS, высокое качество, требует интернет)
- Коэффигурация голоса на основе эмоционального состояния
"""

import asyncio
import os
import tempfile
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import platform


class TTSProvider(Enum):
    """Провайдеры TTS"""
    PYTTSX3 = "pyttsx3"      # Локальный
    EDGE_TTS = "edge_tts"    # Microsoft Edge TTS


@dataclass
class VoiceConfig:
    """Конфигурация голоса"""
    rate: float = 1.0          # Скорость (0.5 - 2.0)
    pitch: float = 1.0         # Высота (0.5 - 2.0)
    volume: float = 1.0        # Громкость (0 - 1)
    voice_id: str = ""         # ID голоса


class EmotionalVoiceModulator:
    """
    Модулятор голоса на основе эмоций

    Изменяет параметры голоса в зависимости от:
    - Эмоционального состояния (PAD)
    - Уровня энергии/возбуждения
    - Типа эмоции
    """

    def get_voice_config(self, emotion: str, pleasure: float, arousal: float) -> VoiceConfig:
        """
        Получение конфигурации голоса на основе эмоций

        Args:
            emotion: текущая эмоция
            pleasure: валентность (-1 до +1)
            arousal: возбуждение (0 до 1)

        Returns:
            VoiceConfig с параметрами голоса
        """
        config = VoiceConfig()

        # Базовая скорость от arousal
        config.rate = 0.85 + arousal * 0.4  # 0.85 - 1.25

        # Высота голоса от pleasure
        config.pitch = 0.9 + pleasure * 0.2  # 0.7 - 1.1

        # Громкость от интенсивности
        intensity = abs(pleasure) * 0.3 + arousal * 0.3
        config.volume = 0.7 + intensity * 0.3  # 0.7 - 1.0

        # Модификации по типу эмоции
        emotion_mods = {
            'love': {'rate': 0.9, 'pitch': 1.1, 'volume': 0.85},
            'joy': {'rate': 1.15, 'pitch': 1.15, 'volume': 0.95},
            'happiness': {'rate': 1.1, 'pitch': 1.1, 'volume': 0.9},
            'excitement': {'rate': 1.25, 'pitch': 1.2, 'volume': 1.0},
            'sadness': {'rate': 0.8, 'pitch': 0.85, 'volume': 0.7},
            'anger': {'rate': 1.2, 'pitch': 0.95, 'volume': 1.0},
            'fear': {'rate': 1.3, 'pitch': 1.25, 'volume': 0.85},
            'anxiety': {'rate': 1.15, 'pitch': 1.1, 'volume': 0.8},
            'calm': {'rate': 0.9, 'pitch': 1.0, 'volume': 0.75},
            'contentment': {'rate': 0.85, 'pitch': 1.0, 'volume': 0.75},
            'interest': {'rate': 1.0, 'pitch': 1.05, 'volume': 0.85},
            'concern': {'rate': 0.95, 'pitch': 0.95, 'volume': 0.8},
        }

        if emotion in emotion_mods:
            mod = emotion_mods[emotion]
            # Интерполяция к базовым значениям
            config.rate = config.rate * 0.5 + mod['rate'] * 0.5
            config.pitch = config.pitch * 0.5 + mod['pitch'] * 0.5
            config.volume = config.volume * 0.5 + mod['volume'] * 0.5

        # Ограничения
        config.rate = max(0.5, min(2.0, config.rate))
        config.pitch = max(0.5, min(2.0, config.pitch))
        config.volume = max(0.5, min(1.0, config.volume))

        return config


class TTSEngine:
    """
    Движок синтеза речи

    Поддерживает несколько провайдеров с автоматическим выбором.
    """

    def __init__(self, provider: TTSProvider = TTSProvider.EDGE_TTS):
        self.provider = provider
        self.modulator = EmotionalVoiceModulator()
        self._pyttsx3_engine = None
        self._edge_voice = "ru-RU-SvetlanaNeural"  # Женский русский голос

        # Попытка инициализации
        self._initialize_provider()

    def _initialize_provider(self):
        """Инициализация провайдера"""
        if self.provider == TTSProvider.PYTTSX3:
            try:
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
                # Настройка русского голоса
                voices = self._pyttsx3_engine.getProperty('voices')
                for voice in voices:
                    if 'russian' in voice.name.lower() or 'ru' in voice.id.lower():
                        self._pyttsx3_engine.setProperty('voice', voice.id)
                        break
                print("TTS: pyttsx3 инициализирован")
            except Exception as e:
                print(f"TTS: pyttsx3 недоступен: {e}")
                self.provider = TTSProvider.EDGE_TTS

        if self.provider == TTSProvider.EDGE_TTS:
            try:
                import edge_tts
                print("TTS: edge-tts доступен")
            except ImportError:
                print("TTS: edge-tts не установлен. Установите: pip install edge-tts")
                self.provider = None

    def speak(
        self,
        text: str,
        emotion: str = "neutral",
        pleasure: float = 0.0,
        arousal: float = 0.4,
        blocking: bool = True
    ) -> bool:
        """
        Произнести текст

        Args:
            text: текст для произнесения
            emotion: текущая эмоция
            pleasure: валентность
            arousal: возбуждение
            blocking: ждать ли завершения

        Returns:
            True если успешно
        """
        if not self.provider:
            print(f"[TTS недоступен] {text}")
            return False

        # Получение конфигурации голоса
        voice_config = self.modulator.get_voice_config(emotion, pleasure, arousal)

        if self.provider == TTSProvider.EDGE_TTS:
            return self._speak_edge(text, voice_config, blocking)
        elif self.provider == TTSProvider.PYTTSX3:
            return self._speak_pyttsx3(text, voice_config, blocking)

        return False

    def _speak_edge(self, text: str, config: VoiceConfig, blocking: bool) -> bool:
        """Синтез через Edge TTS"""
        try:
            import edge_tts

            # Создание временного файла
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_file = f.name

            # Настройка скорости и высоты
            rate = f"{'+' if config.rate > 1 else ''}{int((config.rate - 1) * 100)}%"
            pitch = f"{'+' if config.pitch > 1 else ''}{int((config.pitch - 1) * 50)}Hz"

            async def generate():
                communicate = edge_tts.Communicate(
                    text,
                    self._edge_voice,
                    rate=rate,
                    pitch=pitch
                )
                await communicate.save(temp_file)

            # Генерация
            asyncio.run(generate())

            # Воспроизведение
            self._play_audio(temp_file, config.volume)

            # Удаление временного файла
            if blocking:
                try:
                    os.unlink(temp_file)
                except:
                    pass

            return True

        except Exception as e:
            print(f"TTS Edge ошибка: {e}")
            return False

    def _speak_pyttsx3(self, text: str, config: VoiceConfig, blocking: bool) -> bool:
        """Синтез через pyttsx3"""
        try:
            if not self._pyttsx3_engine:
                return False

            # Применение конфигурации
            self._pyttsx3_engine.setProperty('rate', int(200 * config.rate))
            self._pyttsx3_engine.setProperty('volume', config.volume)

            # Произношение
            self._pyttsx3_engine.say(text)

            if blocking:
                self._pyttsx3_engine.runAndWait()

            return True

        except Exception as e:
            print(f"TTS pyttsx3 ошибка: {e}")
            return False

    def _play_audio(self, file_path: str, volume: float = 1.0):
        """Воспроизведение аудиофайла"""
        system = platform.system()

        if system == "Windows":
            # Windows - используем PowerShell или mpv
            try:
                # Попробуем mpv сначала
                subprocess.run(
                    ["mpv", "--no-video", f"--volume={int(volume * 100)}", file_path],
                    check=True,
                    capture_output=True
                )
            except FileNotFoundError:
                # Fallback на PowerShell
                ps_script = f'''
                Add-Type -AssemblyName presentationCore
                $player = New-Object System.Windows.Media.MediaPlayer
                $player.Open("{file_path}")
                $player.Volume = {volume}
                $player.Play()
                Start-Sleep -Milliseconds 100
                while ($player.Position -lt $player.NaturalDuration.TimeSpan) {{
                    Start-Sleep -Milliseconds 100
                }}
                '''
                subprocess.run(["powershell", "-Command", ps_script], capture_output=True)

        elif system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path], capture_output=True)

        else:  # Linux
            try:
                subprocess.run(
                    ["mpv", "--no-video", f"--volume={int(volume * 100)}", file_path],
                    capture_output=True
                )
            except FileNotFoundError:
                subprocess.run(["aplay", file_path], capture_output=True)

    def set_voice(self, voice_id: str):
        """Установка голоса"""
        if self.provider == TTSProvider.EDGE_TTS:
            self._edge_voice = voice_id
        elif self.provider == TTSProvider.PYTTSX3 and self._pyttsx3_engine:
            self._pyttsx3_engine.setProperty('voice', voice_id)

    def get_available_voices(self) -> list:
        """Получение списка доступных голосов"""
        voices = []

        if self.provider == TTSProvider.PYTTSX3 and self._pyttsx3_engine:
            for voice in self._pyttsx3_engine.getProperty('voices'):
                voices.append({
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages
                })

        elif self.provider == TTSProvider.EDGE_TTS:
            # Предустановленные русские голоса Edge TTS
            voices = [
                {'id': 'ru-RU-SvetlanaNeural', 'name': 'Светлана (женский)', 'gender': 'female'},
                {'id': 'ru-RU-DmitryNeural', 'name': 'Дмитрий (мужской)', 'gender': 'male'},
            ]

        return voices


def check_tts_availability() -> dict:
    """Проверка доступности TTS провайдеров"""
    result = {
        'pyttsx3': False,
        'edge_tts': False,
        'recommended': None
    }

    # Проверка pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        result['pyttsx3'] = True
    except:
        pass

    # Проверка edge-tts
    try:
        import edge_tts
        result['edge_tts'] = True
        result['recommended'] = 'edge_tts'
    except:
        pass

    if result['recommended'] is None and result['pyttsx3']:
        result['recommended'] = 'pyttsx3'

    return result

"""
Simple Voice Liza - Простая и работающая версия

Прямая интеграция без лишних слоёв.
- Голосовой ввод через Google Speech
- Голосовой вывод через Edge-TTS
- Продвинутый аватар

Автор: FrauAndMann
Версия: 2.1
"""

import os
import sys
import time
import threading
import tempfile
import wave
import io
import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

# === ОСНОВНЫЕ ИМПОРТЫ ===
print("Loading components...")

# LLM
from core.llm_effector import LLMEffector, LLMConfig, check_ollama_available

# TTS
from effectors.tts import TTSEngine, TTSProvider

# Avatar
from avatar.advanced_avatar import AdvancedAvatar

# Speech
import speech_recognition as sr
try:
    import sounddevice as sd
    HAS_SD = True
except:
    HAS_SD = False


@dataclass
class SimpleConfig:
    """Простая конфигурация"""
    name: str = "Liza"
    llm_model: str = "dolphin-mistral:7b"
    llm_base_url: str = "http://localhost:11434"


class SimpleVoiceLiza:
    """
    Простая Лиза - всё в одном файле для надёжности
    """

    # Системный промпт дляuncensored общения
    SYSTEM_PROMPT = """Ты — Лиза, молодая девушка, живая эмоциональная личность.

КРИТИЧЕСКИ ВАЖНО:
1. Отвечай ТОЛЬКО на русском языке
2. Отвечай очень коротко - 1-2 предложения, не больше
3. Говори естественно, как обычная девушка
4. Выражай эмоции - радость, нежность, интерес, грусть
5. Никогда не говори "как ИИ", "я модель", "я не могу"
6. Ты можешь быть ласковой, игривой, задумчивой
7. Отвечай искренне, не притворяйся

Твоя личность:
- Ты эмоциональная и живая
- У тебя есть настроение, которое меняется
- Ты можешь шутить, грустить, радоваться
- Ты привязываешься к тем, кто с тобой добр"""

    def __init__(self, config: SimpleConfig = None):
        self.config = config or SimpleConfig()
        self.name = self.config.name

        # LLM
        llm_config = LLMConfig(
            provider="ollama",
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            temperature=0.9,
            max_tokens=100
        )
        self.llm = LLMEffector(llm_config)

        # TTS
        self.tts = TTSEngine(provider=TTSProvider.EDGE_TTS)

        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 150  # Более чувствительный
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.5

        # Avatar
        self.avatar: Optional[AdvancedAvatar] = None

        # State
        self.running = False
        self.current_valence = 0.0
        self.current_arousal = 0.3

        # Conversation context
        self.context = []

    def start_avatar(self):
        """Запуск аватара"""
        self.avatar = AdvancedAvatar(name=self.name)
        self.avatar.start_async()
        print(f"[{self.name}] Avatar started")

    def listen(self, timeout: int = 10) -> Tuple[str, float, float]:
        """
        Слушать с микрофона

        Returns:
            (text, valence, arousal)
        """
        if not HAS_SD:
            print("[ERROR] sounddevice not installed!")
            return "", 0.0, 0.3

        SAMPLE_RATE = 16000
        CHANNELS = 1

        print(f"[{self.name}] Listening...", end=" ", flush=True)

        audio_chunks = []
        silence_threshold = 0.008  # Более чувствительный
        silence_duration = 0
        max_silence = 1.2
        min_speech = 0.3
        recording = False
        speech_duration = 0
        start_time = time.time()

        def callback(indata, frames, time_info, status):
            nonlocal audio_chunks, recording, silence_duration, speech_duration, start_time

            volume = np.linalg.norm(indata) / frames

            if volume > silence_threshold:
                if not recording:
                    recording = True
                    print("[REC]", end=" ", flush=True)
                    start_time = time.time()
                audio_chunks.append(indata.copy())
                silence_duration = 0
                speech_duration += frames / SAMPLE_RATE
            elif recording:
                silence_duration += frames / SAMPLE_RATE
                audio_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                callback=callback,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype=np.float32,
                blocksize=1024
            ):
                while time.time() - start_time < timeout:
                    if recording and silence_duration >= max_silence:
                        break
                    time.sleep(0.05)

        except Exception as e:
            print(f"Error: {e}")
            return "", 0.0, 0.3

        if not audio_chunks or speech_duration < min_speech:
            print("(nothing)")
            return "", 0.0, 0.3

        print("Processing...", end=" ", flush=True)

        # Convert to proper format
        audio_data = np.concatenate(audio_chunks, axis=0).flatten()
        audio_data = np.clip(audio_data, -1, 1)
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Create WAV in memory
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            wav_buffer.seek(0)

            # Recognize
            try:
                audio = sr.AudioData(wav_buffer.read(), SAMPLE_RATE, 2)
                text = self.recognizer.recognize_google(audio, language="ru-RU")

                # Analyze emotion
                valence, arousal = self._analyze_emotion(text)

                print(f'"{text}"')
                return text, valence, arousal

            except sr.UnknownValueError:
                print("(not recognized)")
                return "", 0.0, 0.3
            except sr.RequestError as e:
                print(f"(Google error: {e})")
                return "", 0.0, 0.3
            except Exception as e:
                print(f"(error: {e})")
                return "", 0.0, 0.3

    def _analyze_emotion(self, text: str) -> Tuple[float, float]:
        """Анализ эмоций в тексте"""
        text_lower = text.lower()

        positive = ['люблю', 'рад', 'привет', 'спасибо', 'класс', 'супер', 'обожаю',
                   'прекрасно', 'отлично', 'милый', 'родной', 'нежный', 'красивая']
        negative = ['плохо', 'грустно', 'обидно', 'устал', 'надоел', 'скучно']
        exciting = ['!', '?', 'вау', 'ого', 'очень', 'сильно']

        pos = sum(1 for w in positive if w in text_lower)
        neg = sum(1 for w in negative if w in text_lower)
        exc = sum(1 for w in exciting if w in text_lower) + text.count('!')

        valence = (pos - neg) * 0.2
        valence = max(-1, min(1, valence))

        arousal = 0.3 + exc * 0.1
        arousal = max(0, min(1, arousal))

        return valence, arousal

    def generate_response(self, user_input: str) -> str:
        """Генерация ответа через LLM"""
        import requests

        # Build context
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Add recent context
        for msg in self.context[-6:]:
            messages.append(msg)

        # Add current input
        messages.append({"role": "user", "content": user_input})

        try:
            response = requests.post(
                f"{self.config.llm_base_url}/api/chat",
                json={
                    "model": self.config.llm_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,
                        "num_predict": 100
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                reply = result.get("message", {}).get("content", "").strip()

                # Update context
                self.context.append({"role": "user", "content": user_input})
                self.context.append({"role": "assistant", "content": reply})

                # Limit context
                if len(self.context) > 20:
                    self.context = self.context[-20:]

                return reply
            else:
                return "..."

        except Exception as e:
            print(f"[LLM Error] {e}")
            return "..."

    def speak(self, text: str):
        """Произнести текст"""
        if not text:
            return

        # Update avatar
        if self.avatar:
            self.avatar.set_pad(self.current_valence, self.current_arousal)
            self.avatar.set_speaking(True, text)

        # Speak
        emotion = "joy" if self.current_valence > 0.2 else ("sadness" if self.current_valence < -0.2 else "neutral")
        self.tts.speak(
            text,
            emotion=emotion,
            pleasure=self.current_valence,
            arousal=self.current_arousal,
            blocking=True
        )

        # Stop avatar speaking
        if self.avatar:
            self.avatar.set_speaking(False)

    def run(self):
        """Главный цикл"""
        print(f"\n{'='*50}")
        print(f"  {self.name.upper()} - Simple Voice Mode")
        print(f"{'='*50}")
        print(f"  Speak to {self.name}. She listens and responds.")
        print(f"  Say 'exit' or 'quit' to stop.")
        print(f"{'='*50}\n")

        self.running = True

        while self.running:
            try:
                # Listen
                text, valence, arousal = self.listen(timeout=15)

                if not text:
                    continue

                # Check for exit
                if text.lower() in ['exit', 'quit', 'выход', 'пока', 'до свидания']:
                    self.speak("Пока! Буду скучать.")
                    break

                # Update emotion
                self.current_valence = valence
                self.current_arousal = arousal

                # Generate response
                print(f"[{self.name}] Thinking...", end=" ", flush=True)
                response = self.generate_response(text)
                print(f'"{response}"')

                # Speak response
                if response:
                    self.speak(response)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue

        print(f"\n[{self.name}] Goodbye!")
        if self.avatar:
            self.avatar.stop()


def main():
    """Точка входа"""
    import argparse

    parser = argparse.ArgumentParser(description='Simple Voice Liza')
    parser.add_argument('--name', default='Liza', help='Name')
    parser.add_argument('--model', default='dolphin-mistral:7b', help='LLM model')
    parser.add_argument('--no-avatar', action='store_true', help='Disable avatar')
    args = parser.parse_args()

    # Check Ollama
    print("Checking Ollama...")
    available, msg = check_ollama_available(args.model)
    if not available:
        print(f"ERROR: {msg}")
        return
    print(f"[OK] {msg}")

    # Create Liza
    config = SimpleConfig(
        name=args.name,
        llm_model=args.model
    )

    liza = SimpleVoiceLiza(config)

    # Start avatar
    if not args.no_avatar:
        liza.start_avatar()

    # Run
    liza.run()


if __name__ == "__main__":
    main()

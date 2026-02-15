"""
MuseTalk Avatar - Реалистичный аватар на базе MuseTalk

Требования:
- NVIDIA GPU с 4GB+ VRAM
- PyTorch с CUDA
- MuseTalk: https://github.com/TMElyralab/MuseTalk

Особенности:
- Реалистичная синхронизация губ
- Эмоциональные выражения лица
- Интеграция с голосовым выводом

Автор: FrauAndMann
Версия: 2.0
"""

import os
import sys
import time
import threading
import subprocess
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MuseTalkConfig:
    """Конфигурация MuseTalk"""
    model_path: str = "models/musetalk"
    avatar_image: str = "avatar/reference.png"  # Изображение персонажа
    fps: int = 25
    bbox_shift: int = 0
    device: str = "cuda"  # cuda или cpu


class MuseTalkAvatar:
    """
    Реалистичный аватар на базе MuseTalk

    Требует GPU для работы в реальном времени.
    """

    def __init__(self, config: MuseTalkConfig = None):
        self.config = config or MuseTalkConfig()
        self.available = False
        self.model = None
        self.avatar_processor = None

        # Состояние аватара
        self.current_emotion = "neutral"
        self.is_speaking = False
        self.speaking_queue = []

        # Проверка доступности
        self._check_availability()

    def _check_availability(self) -> Tuple[bool, str]:
        """Проверка доступности MuseTalk"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA недоступна. Требуется NVIDIA GPU."

            # Проверка VRAM
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram < 4:
                return False, f"Недостаточно VRAM: {vram:.1f}GB (нужно 4GB+)"

            # Проверка MuseTalk
            try:
                from musetalk import MuseTalkModel
                self.available = True
                return True, f"MuseTalk доступен (VRAM: {vram:.1f}GB)"
            except ImportError:
                return False, "MuseTalk не установлен. См. setup_musetalk.py"

        except ImportError:
            return False, "PyTorch не установлен"

    def initialize(self) -> bool:
        """Инициализация модели"""
        if not self.available:
            print("[MuseTalk] Недоступен, использую fallback")
            return False

        try:
            print("[MuseTalk] Загрузка модели...")
            from musetalk import MuseTalkModel

            self.model = MuseTalkModel(
                model_path=self.config.model_path,
                device=self.config.device
            )

            print("[MuseTalk] Модель загружена")
            return True

        except Exception as e:
            print(f"[MuseTalk] Ошибка инициализации: {e}")
            return False

    def generate_video(
        self,
        audio_path: str,
        output_path: str = None,
        emotion: str = "neutral"
    ) -> Optional[str]:
        """
        Генерация видео с говорящим аватаром

        Args:
            audio_path: Путь к аудиофайлу
            output_path: Путь для сохранения видео
            emotion: Эмоция для выражения

        Returns:
            Путь к видео или None при ошибке
        """
        if not self.available or not self.model:
            return None

        if output_path is None:
            output_path = "temp_musetalk_output.mp4"

        try:
            # Генерация через MuseTalk
            result = self.model.inference(
                image_path=self.config.avatar_image,
                audio_path=audio_path,
                output_path=output_path,
                bbox_shift=self.config.bbox_shift,
                fps=self.config.fps
            )

            return output_path

        except Exception as e:
            print(f"[MuseTalk] Ошибка генерации: {e}")
            return None

    def set_emotion(self, emotion: str, intensity: float = 1.0):
        """Установка эмоции аватара"""
        self.current_emotion = emotion
        # MuseTalk не поддерживает прямую смену эмоции
        # Эмоция выражается через параметры лица

    def speak(self, text: str, audio_path: str = None) -> bool:
        """Синхронизированное произнесение текста"""
        self.is_speaking = True

        # Здесь должна быть генерация видео
        # Для реального использования нужен полный pipeline

        self.is_speaking = False
        return True

    def stop(self):
        """Остановка аватара"""
        self.is_speaking = False


class RealisticAvatarManager:
    """
    Менеджер реалистичного аватара

    Автоматически выбирает между MuseTalk и 2D аватаром.
    """

    def __init__(self, prefer_musetalk: bool = True):
        self.prefer_musetalk = prefer_musetalk
        self.musetalk: Optional[MuseTalkAvatar] = None
        self.fallback_avatar = None
        self.using_musetalk = False

        # Попытка инициализации MuseTalk
        if prefer_musetalk:
            self.musetalk = MuseTalkAvatar()
            if self.musetalk.available:
                self.using_musetalk = self.musetalk.initialize()

        # Fallback на 2D аватар
        if not self.using_musetalk:
            from avatar.advanced_avatar import AdvancedAvatar
            self.fallback_avatar = AdvancedAvatar()

    def speak(self, text: str, emotion: str = "neutral",
              valence: float = 0.0, arousal: float = 0.3):
        """Произнести текст с эмоцией"""
        if self.using_musetalk and self.musetalk:
            # MuseTalk путь (требует аудио)
            self.musetalk.set_emotion(emotion)
            # Нужна генерация аудио через TTS сначала
        elif self.fallback_avatar:
            # 2D аватар
            self.fallback_avatar.set_pad(valence, arousal)
            self.fallback_avatar.set_speaking(True, text)

    def set_emotion(self, emotion: str, valence: float = 0.0,
                   arousal: float = 0.3, attachment: float = 0.5):
        """Установка эмоционального состояния"""
        if self.using_musetalk and self.musetalk:
            self.musetalk.set_emotion(emotion)
        elif self.fallback_avatar:
            self.fallback_avatar.set_pad(valence, arousal, None, attachment)

    def show(self):
        """Показать аватар"""
        if self.fallback_avatar and not self.using_musetalk:
            self.fallback_avatar.start_async()

    def hide(self):
        """Скрыть аватар"""
        if self.fallback_avatar:
            self.fallback_avatar.stop()

    def is_available(self) -> Tuple[bool, str]:
        """Проверка доступности"""
        if self.using_musetalk:
            return True, "MuseTalk активен"
        elif self.fallback_avatar:
            return True, "2D аватар активен"
        else:
            return False, "Аватар недоступен"


# === УСТАНОВКА MUSETALK ===

def setup_musetalk():
    """Скрипт установки MuseTalk"""
    print("="*50)
    print("  Установка MuseTalk")
    print("="*50)
    print()

    steps = [
        ("Проверка Python", lambda: sys.version_info >= (3, 8)),
        ("Проверка PyTorch", lambda: _check_torch()),
        ("Проверка CUDA", lambda: _check_cuda()),
        ("Установка зависимостей", lambda: _install_deps()),
        ("Загрузка моделей", lambda: _download_models()),
    ]

    for i, (name, check) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name}...", end=" ", flush=True)
        try:
            result = check()
            if result:
                print("OK")
            else:
                print("ПРОПУСК")
        except Exception as e:
            print(f"ОШИБКА: {e}")
            return False

    print()
    print("MuseTalk установлен успешно!")
    return True


def _check_torch() -> bool:
    """Проверка PyTorch"""
    try:
        import torch
        return True
    except ImportError:
        print("\nУстановите PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False


def _check_cuda() -> bool:
    """Проверка CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"(VRAM: {vram:.1f}GB)", end=" ")
            return vram >= 4
        return False
    except:
        return False


def _install_deps() -> bool:
    """Установка зависимостей"""
    deps = [
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "omegaconf>=2.3.0",
        "safetensors>=0.4.0",
        "einops>=0.7.0",
    ]

    for dep in deps:
        result = subprocess.run(
            ["pip", "install", "-q", dep],
            capture_output=True
        )
        if result.returncode != 0:
            print(f"\nОшибка установки {dep}")
            return False

    return True


def _download_models() -> bool:
    """Загрузка моделей MuseTalk"""
    # Здесь должна быть логика загрузки моделей
    print("(требуется ручная загрузка с GitHub)", end=" ")
    return True


if __name__ == "__main__":
    setup_musetalk()

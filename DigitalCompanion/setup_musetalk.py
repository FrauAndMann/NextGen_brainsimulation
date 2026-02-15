"""
MuseTalk Avatar - Реалистичный 3D аватар с lip-sync

MuseTalk 1.5 - реальное время, высокое качество (30fps+)
Работает с любым изображением как базой для аватара.

Установка:
1. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
2. pip install diffusers transformers accelerate opencv-python
3. pip install soundfile librosa omegaconf safetensors
4. Скачать веса моделей (см. download_models())

Использование:
    from avatar.musetalk_avatar import MuseTalkAvatar

    avatar = MuseTalkAvatar()
    avatar.load_image("avatar_base.png")
    avatar.speak("Привет! Как дела?", audio_file="greeting.wav")
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import time


class MuseTalkSetup:
    """Установка и настройка MuseTalk"""

    REQUIREMENTS = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "opencv-python>=4.8.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "omegaconf>=2.3.0",
        "safetensors>=0.4.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
    ]

    @staticmethod
    def check_cuda():
        """Проверка CUDA"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return True, f"CUDA OK: {gpu_name} ({vram:.1f}GB)"
            return False, "CUDA недоступна"
        except ImportError:
            return False, "PyTorch не установлен"

    @staticmethod
    def install_dependencies():
        """Установка зависимостей"""
        print("Установка PyTorch с CUDA...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True)

        print("Установка остальных зависимостей...")
        for req in MuseTalkSetup.REQUIREMENTS[3:]:  # Пропускаем torch
            subprocess.run([sys.executable, "-m", "pip", "install", req], check=True)

        print("✓ Зависимости установлены")

    @staticmethod
    def download_models(models_dir: str = "models/musetalk"):
        """Скачать модели MuseTalk"""
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)

        # URLs для скачивания
        models = {
            "musetalk/musetalk.json": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/musetalk/musetalk.json",
            "musetalk/pytorch_model.bin": "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/musetalk/pytorch_model.bin",
            "sd-vae-ft-mse": "https://huggingface.co/stabilityai/sd-vae-ft-mse",
            "whisper-tiny": "https://huggingface.co/openai/whisper-tiny",
        }

        print("Для полной работы нужно скачать модели:")
        print("1. Клонировать репозиторий:")
        print("   git clone https://github.com/TMElyralab/MuseTalk.git")
        print("2. Скачать веса:")
        print("   cd MuseTalk")
        print("   python -c \"from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='.')\"")


class MuseTalkAvatar:
    """
    Реалистичный аватар с lip-sync на базе MuseTalk

    Требования:
    - NVIDIA GPU с 4GB+ VRAM
    - CUDA 11.8+
    - PyTorch 2.0+
    """

    def __init__(self, models_dir: str = "models/musetalk"):
        self.models_dir = Path(models_dir)
        self.pipeline = None
        self.source_image = None
        self.source_features = None
        self.is_initialized = False

    def initialize(self) -> Tuple[bool, str]:
        """Инициализация модели"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA недоступна. Нужна NVIDIA GPU."

            # Проверка VRAM
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram < 4:
                return False, f"Мало VRAM: {vram:.1f}GB (нужно 4GB+)"

            print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.1f}GB)")

            # Импорт MuseTalk
            try:
                from musetalk import MuseTalkPipeline
                self.pipeline = MuseTalkPipeline.from_pretrained(str(self.models_dir))
                self.pipeline.to("cuda")
                self.is_initialized = True
                return True, "MuseTalk инициализирован"
            except ImportError:
                return False, "MuseTalk не установлен. Запустите setup_musetalk.py"

        except Exception as e:
            return False, f"Ошибка инициализации: {e}"

    def load_avatar_image(self, image_path: str) -> bool:
        """Загрузка базового изображения аватара"""
        if not self.is_initialized:
            print("Сначала инициализируйте модель")
            return False

        try:
            from PIL import Image
            self.source_image = Image.open(image_path).convert("RGB")
            self.source_features = self.pipeline.prepare_source(self.source_image)
            return True
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return False

    def generate(
        self,
        audio_path: str,
        output_path: str = "output.mp4",
        fps: int = 25,
        bbox_shift: int = 0
    ) -> Tuple[bool, str]:
        """
        Генерация видео с lip-sync

        Args:
            audio_path: Путь к аудиофайлу
            output_path: Путь для сохранения видео
            fps: Кадров в секунду
            bbox_shift: Смещение области рта (-10 до 10)

        Returns:
            (success, message_or_path)
        """
        if not self.is_initialized or self.source_image is None:
            return False, "Загрузите изображение аватара"

        try:
            result = self.pipeline.generate(
                source_image=self.source_image,
                source_features=self.source_features,
                audio_path=audio_path,
                fps=fps,
                bbox_shift=bbox_shift
            )

            # Сохранение
            result.save(output_path)
            return True, output_path

        except Exception as e:
            return False, f"Ошибка генерации: {e}"

    def generate_realtime(self, audio_chunks, fps: int = 25):
        """Генерация в реальном времени"""
        if not self.is_initialized:
            return None

        for chunk in audio_chunks:
            frame = self.pipeline.generate_frame(chunk, fps=fps)
            yield frame


class SimpleAvatar:
    """
    Упрощённый аватар без MuseTalk (fallback)

    Использует готовые видео/изображения с простой анимацией.
    """

    def __init__(self, assets_dir: str = "assets/avatar"):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Состояния аватара (изображения или видео)
        self.states = {
            'idle': None,
            'talking': None,
            'happy': None,
            'sad': None,
            'thinking': None,
        }

    def load_state(self, state: str, path: str):
        """Загрузить изображение/видео для состояния"""
        if state in self.states:
            self.states[state] = path

    def get_state_media(self, emotion: str = 'idle') -> Optional[str]:
        """Получить медиа для состояния"""
        return self.states.get(emotion, self.states.get('idle'))


def check_system_requirements() -> dict:
    """Проверка системных требований"""
    results = {}

    # Python
    results['python'] = f"{sys.version_info.major}.{sys.version_info.minor}"

    # PyTorch & CUDA
    try:
        import torch
        results['pytorch'] = torch.__version__
        results['cuda'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results['gpu'] = torch.cuda.get_device_name(0)
            results['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except ImportError:
        results['pytorch'] = None

    # FFmpeg
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        results['ffmpeg'] = result.returncode == 0
    except:
        results['ffmpeg'] = False

    return results


def print_system_report():
    """Вывод отчёта о системе"""
    print("=" * 50)
    print("СИСТЕМНЫЕ ТРЕБОВАНИЯ ДЛЯ MUSEATALK")
    print("=" * 50)

    req = check_system_requirements()

    print(f"\nPython: {req.get('python', 'N/A')} {'✓' if req.get('python') else '✗'}")
    print(f"PyTorch: {req.get('pytorch', 'Не установлен')} {'✓' if req.get('pytorch') else '✗'}")
    print(f"CUDA: {'Доступна' if req.get('cuda') else 'Недоступна'} {'✓' if req.get('cuda') else '✗'}")

    if req.get('gpu'):
        print(f"GPU: {req['gpu']}")
        vram = req.get('vram_gb', 0)
        vram_ok = vram >= 4
        print(f"VRAM: {vram:.1f} GB {'✓' if vram_ok else '✗ (нужно 4GB+)'}")

    print(f"FFmpeg: {'Установлен' if req.get('ffmpeg') else 'Не установлен'} {'✓' if req.get('ffmpeg') else '✗'}")

    print("\n" + "=" * 50)

    if req.get('cuda') and req.get('vram_gb', 0) >= 4:
        print("✓ Система ГОТОВА для MuseTalk")
    else:
        print("✗ Система НЕ ГОТОВА")
        print("\nДля установки выполните:")
        print("1. Установите CUDA 11.8+")
        print("2. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. pip install -r requirements_musetalk.txt")


if __name__ == "__main__":
    print_system_report()

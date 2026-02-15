"""
Emotion Detector - Детекция эмоций пользователя

Поддерживает:
- DeepFace (анализ лица с камеры)
- Анализ голоса (интонация, темп)
- Комбинированный анализ (face + voice + text)

Автор: FrauAndMann
Версия: 2.0
"""

import os
import time
import threading
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

# DeepFace может быть недоступен
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False


class EmotionSource(Enum):
    """Источник детекции эмоций"""
    FACE = "face"       # По лицу (камера)
    VOICE = "voice"     # По голосу
    TEXT = "text"       # По тексту
    COMBINED = "combined"  # Комбинированный


@dataclass
class DetectedEmotion:
    """Детектированная эмоция"""
    emotion: str           # Название эмоции
    confidence: float      # Уверенность (0-1)
    valence: float         # Валентность (-1 до 1)
    arousal: float         # Возбуждение (0-1)
    source: EmotionSource  # Источник
    timestamp: float       # Время детекции

    # Все вероятности эмоций
    all_emotions: Dict[str, float] = None


class FaceEmotionDetector:
    """
    Детекция эмоций по лицу через DeepFace

    Требует: pip install deepface
    """

    # Соответствие эмоций DeepFace -> PAD
    EMOTION_TO_PAD = {
        'happy':    (0.7, 0.5, 0.3),
        'sad':      (-0.6, -0.2, -0.3),
        'angry':    (-0.5, 0.7, 0.5),
        'fear':     (-0.6, 0.6, -0.4),
        'surprise': (0.2, 0.8, -0.2),
        'disgust':  (-0.4, 0.2, 0.1),
        'neutral':  (0.0, 0.0, 0.0),
    }

    def __init__(self):
        self.available = HAS_DEEPFACE
        self.last_detection: Optional[DetectedEmotion] = None
        self._running = False
        self._thread = None

    def detect_from_frame(self, frame: np.ndarray) -> Optional[DetectedEmotion]:
        """
        Детекция эмоции из кадра

        Args:
            frame: Изображение (numpy array, RGB или BGR)

        Returns:
            DetectedEmotion или None
        """
        if not self.available:
            return None

        try:
            # Анализ через DeepFace
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'neutral')

            # PAD конвертация
            valence, arousal, _ = self.EMOTION_TO_PAD.get(
                dominant, (0.0, 0.0, 0.0)
            )

            self.last_detection = DetectedEmotion(
                emotion=dominant,
                confidence=emotions.get(dominant, 0.5) / 100,
                valence=valence,
                arousal=arousal,
                source=EmotionSource.FACE,
                timestamp=time.time(),
                all_emotions=emotions
            )

            return self.last_detection

        except Exception as e:
            print(f"[FaceEmotion] Ошибка: {e}")
            return None

    def start_continuous(self, camera_index: int = 0,
                        callback: callable = None,
                        fps: int = 10):
        """
        Запуск непрерывной детекции

        Args:
            camera_index: Индекс камеры
            callback: Callback при детекции
            fps: Частота анализа
        """
        if not self.available:
            print("[FaceEmotion] DeepFace недоступен")
            return

        self._running = True

        def loop():
            try:
                import cv2
                cap = cv2.VideoCapture(camera_index)

                interval = 1.0 / fps

                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # BGR -> RGB для DeepFace
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    emotion = self.detect_from_frame(rgb_frame)

                    if emotion and callback:
                        callback(emotion)

                    time.sleep(interval)

                cap.release()

            except Exception as e:
                print(f"[FaceEmotion] Ошибка: {e}")

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_continuous(self):
        """Остановка непрерывной детекции"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


class VoiceEmotionDetector:
    """
    Детекция эмоций по голосу

    Анализирует интонацию, темп, громкость.
    """

    # Акустические признаки эмоций
    EMOTION_ACOUSTICS = {
        'joy': {
            'pitch_var': 0.7,   # Высокая вариативность
            'energy': 0.7,      # Высокая энергия
            'tempo': 1.2,       # Быстрый темп
        },
        'sadness': {
            'pitch_var': 0.2,
            'energy': 0.3,
            'tempo': 0.8,
        },
        'anger': {
            'pitch_var': 0.5,
            'energy': 0.9,
            'tempo': 1.3,
        },
        'fear': {
            'pitch_var': 0.6,
            'energy': 0.6,
            'tempo': 1.1,
        },
        'calm': {
            'pitch_var': 0.3,
            'energy': 0.4,
            'tempo': 1.0,
        },
    }

    def __init__(self):
        self.last_detection: Optional[DetectedEmotion] = None

    def detect_from_audio(self, audio_data: np.ndarray,
                         sample_rate: int = 16000) -> Optional[DetectedEmotion]:
        """
        Детекция эмоции из аудио

        Args:
            audio_data: Аудиоданные (numpy array)
            sample_rate: Частота дискретизации

        Returns:
            DetectedEmotion или None
        """
        try:
            # Извлечение признаков
            features = self._extract_features(audio_data, sample_rate)

            # Классификация эмоции
            emotion, confidence = self._classify_emotion(features)

            # PAD конвертация
            valence, arousal = self._emotion_to_pad(emotion)

            self.last_detection = DetectedEmotion(
                emotion=emotion,
                confidence=confidence,
                valence=valence,
                arousal=arousal,
                source=EmotionSource.VOICE,
                timestamp=time.time()
            )

            return self.last_detection

        except Exception as e:
            print(f"[VoiceEmotion] Ошибка: {e}")
            return None

    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Извлечение акустических признаков"""
        features = {}

        # Энергия
        features['energy'] = np.sqrt(np.mean(audio ** 2))

        # Нормализация энергии
        features['energy'] = min(1.0, features['energy'] * 10)

        # Вариативность (как proxy для pitch variance)
        features['pitch_var'] = np.std(audio) / (np.mean(np.abs(audio)) + 1e-6)
        features['pitch_var'] = min(1.0, features['pitch_var'])

        # Темп (через zero-crossing rate)
        zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        features['tempo'] = 0.5 + zcr * 2  # Нормализация
        features['tempo'] = min(2.0, max(0.5, features['tempo']))

        return features

    def _classify_emotion(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Классификация эмоции по признакам"""
        best_emotion = 'calm'
        best_score = -1

        for emotion, acoustics in self.EMOTION_ACOUSTICS.items():
            # Сходство с акустическим профилем
            score = 0
            for key, target in acoustics.items():
                if key in features:
                    # Чем ближе к целевому, тем выше оценка
                    diff = abs(features[key] - target)
                    score += 1 - diff

            if score > best_score:
                best_score = score
                best_emotion = emotion

        confidence = best_score / len(self.EMOTION_ACOUSTICS)
        return best_emotion, min(1.0, confidence)

    def _emotion_to_pad(self, emotion: str) -> Tuple[float, float]:
        """Конвертация эмоции в PAD"""
        pad_map = {
            'joy': (0.7, 0.5),
            'sadness': (-0.6, 0.2),
            'anger': (-0.5, 0.7),
            'fear': (-0.6, 0.6),
            'calm': (0.0, 0.3),
        }
        return pad_map.get(emotion, (0.0, 0.3))


class CombinedEmotionDetector:
    """
    Комбинированный детектор эмоций

    Объединяет данные из нескольких источников.
    """

    def __init__(self, use_face: bool = True, use_voice: bool = True):
        self.face_detector = FaceEmotionDetector() if use_face else None
        self.voice_detector = VoiceEmotionDetector() if use_voice else None

        self.last_detection: Optional[DetectedEmotion] = None
        self._face_emotion: Optional[DetectedEmotion] = None
        self._voice_emotion: Optional[DetectedEmotion] = None

        # Веса источников
        self.weights = {
            EmotionSource.FACE: 0.5,
            EmotionSource.VOICE: 0.3,
            EmotionSource.TEXT: 0.2,
        }

    def update_face(self, emotion: DetectedEmotion):
        """Обновление эмоции с лица"""
        self._face_emotion = emotion
        self._combine()

    def update_voice(self, emotion: DetectedEmotion):
        """Обновление эмоции с голоса"""
        self._voice_emotion = emotion
        self._combine()

    def _combine(self):
        """Комбинирование эмоций из разных источников"""
        emotions = []

        if self._face_emotion:
            emotions.append((self._face_emotion, self.weights[EmotionSource.FACE]))
        if self._voice_emotion:
            emotions.append((self._voice_emotion, self.weights[EmotionSource.VOICE]))

        if not emotions:
            return

        # Взвешенное усреднение PAD
        total_weight = sum(w for _, w in emotions)

        valence = sum(e.valence * w for e, w in emotions) / total_weight
        arousal = sum(e.arousal * w for e, w in emotions) / total_weight
        confidence = sum(e.confidence * w for e, w in emotions) / total_weight

        # Определение доминирующей эмоции
        emotion_name = self._pad_to_emotion(valence, arousal)

        self.last_detection = DetectedEmotion(
            emotion=emotion_name,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            source=EmotionSource.COMBINED,
            timestamp=time.time()
        )

    def _pad_to_emotion(self, valence: float, arousal: float) -> str:
        """Конвертация PAD в название эмоции"""
        if valence > 0.3:
            if arousal > 0.5:
                return 'joy'
            else:
                return 'calm'
        elif valence < -0.3:
            if arousal > 0.5:
                return 'anger'
            else:
                return 'sadness'
        else:
            if arousal > 0.5:
                return 'surprise'
            else:
                return 'neutral'

    def get_current_emotion(self) -> Optional[DetectedEmotion]:
        """Получить текущую эмоцию"""
        return self.last_detection


# === ПРОВЕРКА ДОСТУПНОСТИ ===

def check_emotion_detection() -> Dict[str, bool]:
    """Проверка доступности детекции эмоций"""
    result = {
        'deepface': HAS_DEEPFACE,
        'camera': False,
        'microphone': False,
    }

    # Проверка камеры
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            result['camera'] = True
            cap.release()
    except:
        pass

    # Проверка микрофона
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        result['microphone'] = any(d['max_input_channels'] > 0 for d in devices)
    except:
        pass

    return result


if __name__ == "__main__":
    print("Проверка детекции эмоций:")
    print(check_emotion_detection())

    if HAS_DEEPFACE:
        print("\nDeepFace доступен. Запуск теста с камеры...")
        detector = FaceEmotionDetector()

        def on_emotion(emotion: DetectedEmotion):
            print(f"Эмоция: {emotion.emotion} ({emotion.confidence:.0%})")
            print(f"  Валентность: {emotion.valence:.2f}")
            print(f"  Возбуждение: {emotion.arousal:.2f}")

        detector.start_continuous(callback=on_emotion)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            detector.stop_continuous()

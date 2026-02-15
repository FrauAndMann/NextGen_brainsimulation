"""
Vision Module - Компьютерное зрение для ANIMA

Поддерживает:
- Захват с вебкамеры
- Обнаружение лица
- Распознавание эмоций
- Детекция взгляда
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time


class EmotionType(Enum):
    """Типы эмоций"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    CONTEMPT = "contempt"


@dataclass
class FaceAnalysis:
    """Анализ лица"""
    detected: bool = False
    emotion: EmotionType = EmotionType.NEUTRAL
    emotion_confidence: float = 0.0

    # Аффективные параметры
    valence: float = 0.0      # -1 (негатив) до 1 (позитив)
    arousal: float = 0.3      # 0 (спокойно) до 1 (возбуждено)

    # Позиция и размеры
    face_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

    # Дополнительные признаки
    looking_at_camera: bool = False
    blink_count: int = 0
    smile_detected: bool = False


class VisionSensor:
    """
    Модуль компьютерного зрения

    Обнаруживает лицо, эмоции и взгляд пользователя.
    """

    def __init__(self):
        self._camera = None
        self._face_detector = None
        self._emotion_detector = None
        self._smile_cascade = None
        self._eye_cascade = None
        self._is_running = False

        # Кэш последнего анализа
        self._last_analysis: Optional[FaceAnalysis] = None
        self._analysis_time = 0

        self._initialize()

    def _initialize(self):
        """Инициализация компонентов"""
        # Инициализация камеры
        try:
            import cv2
            self._camera = cv2.VideoCapture(0)
            if self._camera.isOpened():
                print("[Vision] Камера инициализирована")
            else:
                print("[Vision] Камера недоступна")
                self._camera = None
        except ImportError:
            print("[Vision] OpenCV не установлен: pip install opencv-python")
            return

        # Инициализация детектора лиц (OpenCV Haar Cascade)
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_detector = cv2.CascadeClassifier(cascade_path)
            print("[Vision] Детектор лиц загружен")
        except Exception as e:
            print(f"[Vision] Ошибка загрузки детектора лиц: {e}")

        # Инициализация дополнительных каскадов для анализа эмоций
        try:
            import cv2
            self._smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
            self._eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            # Используем встроенный простой детектор эмоций
            self._emotion_detector = 'builtin'
            print("[Vision] Простой детектор эмоций активен")
        except Exception as e:
            print(f"[Vision] Каскады для эмоций: {e}")

        # Инициализация продвинутого детектора эмоций (если доступен)
        try:
            from deepface import DeepFace
            self._emotion_detector = 'deepface'
            print("[Vision] DeepFace доступен для эмоций")
        except ImportError:
            try:
                from fer import FER
                self._emotion_detector = FER(mtcnn=True)
                print("[Vision] FER доступен для эмоций")
            except ImportError:
                pass  # Используем встроенный простой детектор

    def capture_frame(self) -> Optional[np.ndarray]:
        """Захват кадра с камеры"""
        if self._camera is None or not self._camera.isOpened():
            return None

        ret, frame = self._camera.read()
        if ret:
            return frame
        return None

    def analyze_frame(self, frame: np.ndarray = None) -> FaceAnalysis:
        """
        Анализ кадра

        Args:
            frame: Кадр для анализа (если None, захватывается с камеры)

        Returns:
            FaceAnalysis с результатами
        """
        import cv2

        if frame is None:
            frame = self.capture_frame()

        if frame is None:
            return FaceAnalysis(detected=False)

        analysis = FaceAnalysis()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лица
        if self._face_detector is not None:
            faces = self._face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) > 0:
                analysis.detected = True
                # Берём самое большое лицо
                face = max(faces, key=lambda f: f[2] * f[3])
                analysis.face_rect = tuple(face)

                # Анализ эмоций
                if self._emotion_detector is not None:
                    emotion, confidence, valence, arousal = self._detect_emotion(frame, face)
                    analysis.emotion = emotion
                    analysis.emotion_confidence = confidence
                    analysis.valence = valence
                    analysis.arousal = arousal

                # Детекция улыбки
                x, y, w, h = face
                roi_gray = gray[y:y+h, x:x+w]
                if hasattr(self, '_smile_cascade') and self._smile_cascade is not None:
                    smiles = self._smile_cascade.detectMultiScale(roi_gray, 1.7, 22)
                    analysis.smile_detected = len(smiles) > 0
                    if analysis.smile_detected:
                        analysis.valence = max(analysis.valence, 0.3)

        self._last_analysis = analysis
        self._analysis_time = time.time()

        return analysis

    def _detect_emotion(self, frame: np.ndarray, face: Tuple) -> Tuple[EmotionType, float, float, float]:
        """Детекция эмоции"""
        import cv2

        x, y, w, h = face
        face_img = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+h, x:x+w]

        # Значения по умолчанию
        emotion = EmotionType.NEUTRAL
        confidence = 0.5
        valence = 0.0
        arousal = 0.3

        try:
            # Встроенный простой детектор эмоций на основе Haar Cascades
            if isinstance(self._emotion_detector, str) and self._emotion_detector == 'builtin':
                # Детекция улыбки
                smiles = self._smile_cascade.detectMultiScale(roi_gray, 1.7, 22)

                # Детекция глаз
                eyes = self._eye_cascade.detectMultiScale(roi_gray, 1.1, 10)

                smile_detected = len(smiles) > 0
                eyes_detected = len(eyes)

                if smile_detected:
                    # Улыбка = счастье
                    emotion = EmotionType.HAPPY
                    confidence = 0.7
                    valence = 0.6 + 0.2 * len(smiles)  # Больше улыбок = больше радости
                    arousal = 0.5
                elif eyes_detected < 2:
                    # Мало или нет глаз = возможно закрыты (усталость/грусть)
                    emotion = EmotionType.SAD
                    confidence = 0.4
                    valence = -0.3
                    arousal = 0.2
                else:
                    # Нейтральное состояние
                    emotion = EmotionType.NEUTRAL
                    confidence = 0.5
                    valence = 0.0
                    arousal = 0.3

                # Анализ размера лица (приближение/удаление может указывать на интерес)
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area

                if face_ratio > 0.3:  # Лицо близко к камере
                    arousal = min(1.0, arousal + 0.2)  # Больше вовлечённости

            elif isinstance(self._emotion_detector, str) and self._emotion_detector == 'deepface':
                from deepface import DeepFace
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                if result:
                    emotions = result[0]['emotion']
                    dominant = result[0]['dominant_emotion']
                    emotion = EmotionType(dominant.lower())
                    confidence = emotions.get(dominant, 50) / 100
                    valence, arousal = self._emotion_to_va(dominant.lower())

            elif hasattr(self._emotion_detector, 'detect_emotions'):
                # FER
                result = self._emotion_detector.detect_emotions(frame)
                if result:
                    emotions = result[0]['emotions']
                    dominant = max(emotions, key=emotions.get)
                    emotion = EmotionType(dominant)
                    confidence = emotions[dominant]
                    valence, arousal = self._emotion_to_va(dominant)

        except Exception as e:
            print(f"[Vision] Ошибка анализа эмоций: {e}")

        return emotion, confidence, valence, arousal

    def _emotion_to_va(self, emotion: str) -> Tuple[float, float]:
        """Конвертация эмоции в валентность/возбуждение"""
        # V-A маппинг для различных эмоций
        mapping = {
            'happy': (0.8, 0.5),
            'sad': (-0.7, 0.2),
            'angry': (-0.5, 0.8),
            'surprised': (0.2, 0.9),
            'fearful': (-0.6, 0.7),
            'disgusted': (-0.6, 0.4),
            'contempt': (-0.4, 0.3),
            'neutral': (0.0, 0.3),
        }
        return mapping.get(emotion.lower(), (0.0, 0.3))

    def get_last_analysis(self) -> Optional[FaceAnalysis]:
        """Получить последний анализ"""
        if time.time() - self._analysis_time < 2:  # Актуален 2 секунды
            return self._last_analysis
        return None

    def is_available(self) -> bool:
        """Проверка доступности"""
        return self._camera is not None and self._camera.isOpened()

    def start_continuous(self, callback=None, interval: float = 0.1):
        """Запуск непрерывного анализа"""
        import threading

        self._is_running = True

        def _loop():
            while self._is_running:
                analysis = self.analyze_frame()
                if callback and analysis.detected:
                    callback(analysis)
                time.sleep(interval)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()

    def stop_continuous(self):
        """Остановка непрерывного анализа"""
        self._is_running = False

    def release(self):
        """Освобождение ресурсов"""
        self._is_running = False
        if self._camera is not None:
            self._camera.release()


def check_vision_availability() -> Dict[str, bool]:
    """Проверка доступности компонентов зрения"""
    result = {
        'opencv': False,
        'camera': False,
        'builtin_emotion': False,
        'deepface': False,
        'fer': False,
        'recommended': None
    }

    # OpenCV
    try:
        import cv2
        result['opencv'] = True
    except ImportError:
        pass

    # Камера
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        result['camera'] = cap.isOpened()
        cap.release()
    except:
        pass

    # Встроенный детектор эмоций (на основе Haar Cascades)
    try:
        import cv2
        smile_cascade = cv2.data.haarcascades + 'haarcascade_smile.xml'
        eye_cascade = cv2.data.haarcascades + 'haarcascade_eye.xml'
        if cv2.CascadeClassifier(smile_cascade).load(smile_cascade):
            result['builtin_emotion'] = True
            result['recommended'] = 'builtin'
    except:
        pass

    # DeepFace
    try:
        from deepface import DeepFace
        result['deepface'] = True
        result['recommended'] = 'deepface'
    except ImportError:
        pass

    # FER
    try:
        from fer import FER
        result['fer'] = True
        if result['recommended'] is None or result['recommended'] == 'builtin':
            result['recommended'] = 'fer'
    except ImportError:
        pass

    return result

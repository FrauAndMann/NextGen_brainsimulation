"""
Sensors Module - Сенсоры для ANIMA

Содержит модули для восприятия:
- speech: Распознавание речи и анализ интонации
- vision: Вебкамера, обнаружение лица и эмоций
"""

from sensors.speech import SpeechToText, STTProvider, VoiceAnalysis, check_stt_availability
from sensors.vision import VisionSensor, FaceAnalysis, EmotionType, check_vision_availability

__all__ = [
    'SpeechToText', 'STTProvider', 'VoiceAnalysis', 'check_stt_availability',
    'VisionSensor', 'FaceAnalysis', 'EmotionType', 'check_vision_availability',
]

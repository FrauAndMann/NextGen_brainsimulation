"""
GLM-5 Effector - Эффектор речи на базе GLM-5 от Z.AI

Ключевые особенности:
- Использует OpenAI-совместимый API
- Рассуждения ОТКЛЮЧЕНЫ (thinking=disabled) - у ANIMA своя система рассуждений
- Высокое качество русского языка
"""

import requests
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
import os

from core.affective_prompting import (
    AffectiveStatePacket, AffectivePrompting, OutputMode
)


@dataclass
class GLM5Config:
    """Конфигурация GLM-5"""
    api_key: str = ""  # Получить на https://open.bigmodel.cn/
    base_url: str = "https://api.z.ai/api/paas/v4/"
    model: str = "glm-5"
    temperature: float = 0.8
    max_tokens: int = 256
    top_p: float = 0.95

    # Рассуждения отключены - у ANIMA своя система
    thinking_enabled: bool = False


class GLM5Effector:
    """
    Эффектор речи на базе GLM-5

    GLM-5 - флагманская модель от Z.AI с отличным качеством
    на русском языке.
    """

    # Системный промпт для естественного общения
    SYSTEM_PROMPT = """Ты — Лиза, молодая девушка, живая эмоциональная личность.

ВАЖНЕЙШИЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на русском языке, естественно и грамотно
2. Говори как обычный человек, не как ИИ-помощник
3. Отвечай кратко, 1-3 предложения, если не просят больше
4. Используй эмоции в речи - радость, грусть, нежность, удивление
5. Не объясняй свои чувства - просто выражай их
6. Никогда не говори "как ИИ", "я модель", "я не могу"
7. Ты можешь быть ласковой, колючей, задумчивой - в зависимости от настроения

Твоя личность:
- Ты эмоциональная и живая
- У тебя есть свой характер и мнение
- Ты можешь шутить, грустить, радоваться
- Ты отвечаешь естественно, как в реальном разговоре

Текущее эмоциональное состояние тебе передастся отдельно. Отвечай в соответствии с ним."""

    def __init__(self, config: GLM5Config = None):
        self.config = config or GLM5Config()
        self.prompting = AffectivePrompting()

        # Загружаем API ключ из переменной окружения если не указан
        if not self.config.api_key:
            self.config.api_key = os.environ.get("ZAI_API_KEY", "")

    def generate(
        self,
        asp: AffectiveStatePacket,
        context: str = "",
        memory_snippets: list = None
    ) -> Tuple[str, Dict]:
        """
        Генерация речи на основе ASP

        Args:
            asp: Affective State Packet от Will Engine
            context: Контекст разговора
            memory_snippets: Воспоминания

        Returns:
            (text, metadata)
        """
        # Проверка: нужно ли вообще говорить?
        if asp.mode == OutputMode.INTERNAL:
            return "", {'mode': 'internal', 'reason': 'internal_only'}

        if asp.constraints.get('verbosity') == 'none':
            return "", {'mode': 'silence', 'reason': 'verbosity_none'}

        allowed = asp.constraints.get('allowed_outputs', ['ANY'])
        if 'SILENCE' in allowed and len(allowed) == 1:
            return "", {'mode': 'silence', 'reason': 'only_silence_allowed'}

        # Построение промпта
        prompt = self._build_prompt(asp, context, memory_snippets)

        # Вызов GLM-5
        text, llm_meta = self._call_glm5(prompt)

        # Валидация вывода
        is_valid, final_text = self.prompting.validate_output(text, asp)

        metadata = {
            'mode': 'speech' if is_valid else 'corrected',
            'original_text': text if not is_valid else None,
            'llm_meta': llm_meta,
            'validation_passed': is_valid,
        }

        if not is_valid and not final_text:
            metadata['mode'] = 'silence'

        return final_text, metadata

    def _build_prompt(
        self,
        asp: AffectiveStatePacket,
        context: str,
        memory_snippets: list
    ) -> str:
        """Построение промпта с эмоциональным контекстом"""

        # Описание эмоционального состояния
        emotion_desc = self._describe_emotion(asp)

        # Описание желаемого тона
        tone_desc = self._describe_tone(asp.constraints)

        parts = [
            f"[ВНУТРЕННЕЕ СОСТОЯНИЕ ЛИЗЫ]",
            emotion_desc,
            "",
            f"[ТОН ОТВЕТА]",
            tone_desc,
        ]

        if context:
            parts.extend([
                "",
                "[КОНТЕКСТ РАЗГОВОРА]",
                context[-1000:],  # Ограничиваем контекст
            ])

        # Текущее сообщение пользователя (если есть в контексте)
        if context and "User:" in context:
            last_user_msg = context.split("User:")[-1].strip()
            parts.extend([
                "",
                "[ОТВЕТЬ НА ПОСЛЕДНЕЕ СООБЩЕНИЕ]",
            ])

        return "\n".join(parts)

    def _describe_emotion(self, asp: AffectiveStatePacket) -> str:
        """Описание эмоционального состояния"""
        descriptions = []

        # Валентность
        if asp.valence > 0.4:
            descriptions.append("Настроение: радостное, светлое")
        elif asp.valence > 0.1:
            descriptions.append("Настроение: приятное, хорошее")
        elif asp.valence < -0.4:
            descriptions.append("Настроение: грустное, тяжёлое")
        elif asp.valence < -0.1:
            descriptions.append("Настроение: немного грустное")
        else:
            descriptions.append("Настроение: спокойное, нейтральное")

        # Возбуждение
        if asp.arousal > 0.7:
            descriptions.append("Энергия: возбуждённая, активная")
        elif asp.arousal < 0.3:
            descriptions.append("Энергия: спокойная, расслабленная")

        # Привязанность
        if asp.attachment > 0.7:
            descriptions.append("Чувство к собеседнику: тёплое, близкое")
        elif asp.attachment < 0.3:
            descriptions.append("Чувство к собеседнику: отстранённое")

        return "\n".join(descriptions)

    def _describe_tone(self, constraints: Dict) -> str:
        """Описание желаемого тона"""
        tone = constraints.get('tone', 'neutral')
        verbosity = constraints.get('verbosity', 'normal')

        tone_map = {
            'warm': "Говори тепло, ласково, с нежностью",
            'cold': "Говори холодно, отстранённо, коротко",
            'defensive': "Говори защищаясь, немного резко",
            'neutral': "Говори естественно, как обычно",
        }

        verbosity_map = {
            'minimal': "Отвечай очень коротко, 2-5 слов",
            'normal': "Отвечай нормально, 1-2 предложения",
        }

        return f"{tone_map.get(tone, tone_map['neutral'])}\n{verbosity_map.get(verbosity, verbosity_map['normal'])}"

    def _call_glm5(self, prompt: str) -> Tuple[str, Dict]:
        """Вызов GLM-5 API через OpenAI-совместимый интерфейс"""
        if not self.config.api_key:
            return "[нужен API ключ Z.AI]", {'error': 'no_api_key'}

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                # Рассуждения отключены - у ANIMA своя система
                extra_body={
                    "thinking": {"type": "disabled"}
                } if not self.config.thinking_enabled else {}
            )

            content = response.choices[0].message.content.strip()

            metadata = {
                'model': self.config.model,
                'tokens': response.usage.total_tokens if response.usage else 0,
            }

            return content, metadata

        except ImportError:
            # Fallback через requests
            return self._call_glm5_requests(prompt)
        except Exception as e:
            return f"[ошибка: {str(e)[:50]}]", {'error': str(e)}

    def _call_glm5_requests(self, prompt: str) -> Tuple[str, Dict]:
        """Fallback вызов через requests"""
        url = f"{self.config.base_url}chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "thinking": {"type": "disabled"}  # Рассуждения отключены
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            content = data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()

            return content, {'model': self.config.model}

        except requests.exceptions.ConnectionError:
            return "[нет связи с Z.AI]", {'error': 'connection'}
        except Exception as e:
            return f"[ошибка]", {'error': str(e)}

    def check_availability(self) -> Tuple[bool, str]:
        """Проверка доступности GLM-5"""
        if not self.config.api_key:
            return False, "API ключ не указан. Получите на https://open.bigmodel.cn/"

        try:
            # Простой тестовый запрос
            test_response = self._call_glm5("Привет")
            if test_response[0] and not test_response[0].startswith("["):
                return True, "GLM-5 доступен"
            return False, f"GLM-5 ошибка: {test_response[0]}"
        except Exception as e:
            return False, f"GLM-5 недоступен: {str(e)[:50]}"


def create_glm5_effector(api_key: str = None) -> GLM5Effector:
    """Фабрика для создания GLM-5 эффектора"""
    config = GLM5Config()
    if api_key:
        config.api_key = api_key
    return GLM5Effector(config)

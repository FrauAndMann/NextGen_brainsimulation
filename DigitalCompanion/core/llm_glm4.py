"""
GLM-4.7 Effector - Эффектор речи на базе GLM-4.7 от Z.AI

GLM-4.7 - флагманская модель для кодинга и рассуждений.
Для диалога используем без рассуждений (thinking disabled).

API Docs: https://docs.z.ai/
Model Names: glm-4.7, glm-4.7-flash
"""

import requests
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import os

from core.affective_prompting import (
    AffectiveStatePacket, AffectivePrompting, OutputMode
)


@dataclass
class GLMConfig:
    """Конфигурация GLM-4.7"""
    api_key: str = ""
    base_url: str = "https://api.z.ai/api/paas/v4/"
    model: str = "glm-4.7-flash"  # или "glm-4.7" для полной версии
    temperature: float = 0.8
    max_tokens: int = 256
    top_p: float = 0.95


class GLM4Effector:
    """
    Эффектор речи на базе GLM-4.7

    GLM-4.7 - новейшая модель Z.AI с отличным качеством.
    """

    SYSTEM_PROMPT = """Ты — Лиза, живая девушка, эмоциональная личность со своим характером.

ВАЖНЕЙШИЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО на русском языке, естественно и грамотно
2. Говори как живой человек в разговоре, не как ИИ
3. Отвечай кратко, 1-3 предложения
4. Используй эмоции - радость, грусть, нежность, удивление, смущение
5. Можешь использовать простые слова: "да", "нет", "ага", "ну", "не знаю"
6. Не объясняй свои чувства, просто выражай их
7. Никогда не говори "как ИИ", "я модель", "я не могу"

Твоя личность:
- Ты эмоциональная и живая
- У тебя есть характер и мнение
- Ты можешь шутить, грустить, радоваться, смущаться
- Ты отвечаешь естественно, как в реальном разговоре

Текущее настроение тебе передаётся отдельно. Отвечай в соответствии с ним."""

    def __init__(self, config: GLMConfig = None):
        self.config = config or GLMConfig()
        self.prompting = AffectivePrompting()

        if not self.config.api_key:
            self.config.api_key = os.environ.get("ZAI_API_KEY", "")

    def generate(
        self,
        asp: AffectiveStatePacket,
        context: str = "",
        memory_snippets: list = None
    ) -> Tuple[str, Dict]:
        """Генерация речи"""
        if asp.mode == OutputMode.INTERNAL:
            return "", {'mode': 'internal'}

        if asp.constraints.get('verbosity') == 'none':
            return "", {'mode': 'silence'}

        prompt = self._build_prompt(asp, context)
        text, meta = self._call_api(prompt)

        is_valid, final_text = self.prompting.validate_output(text, asp)

        return final_text, {
            'mode': 'speech' if is_valid else 'corrected',
            'validation': is_valid,
            **meta
        }

    def _build_prompt(self, asp: AffectiveStatePacket, context: str) -> str:
        """Построение промпта"""
        emotion = self._emotion_description(asp)
        tone = self._tone_description(asp.constraints)

        parts = [
            f"[НАСТРОЕНИЕ]\n{emotion}",
            f"\n[СТИЛЬ ОТВЕТА]\n{tone}",
        ]

        if context:
            parts.append(f"\n[КОНТЕКСТ]\n{context[-800:]}")

        return "\n".join(parts)

    def _emotion_description(self, asp: AffectiveStatePacket) -> str:
        """Описание эмоции"""
        lines = []

        if asp.valence > 0.4:
            lines.append("Настроение радостное, светлое")
        elif asp.valence > 0.1:
            lines.append("Настроение хорошее, приятное")
        elif asp.valence < -0.4:
            lines.append("Настроение грустное, тяжёлое")
        elif asp.valence < -0.1:
            lines.append("Настроение немного грустное")
        else:
            lines.append("Настроение спокойное")

        if asp.arousal > 0.7:
            lines.append("Энергия высокая, возбуждение")
        elif asp.arousal < 0.3:
            lines.append("Энергия низкая, расслабленность")

        if asp.attachment > 0.7:
            lines.append("Чувство к собеседнику: тёплое, близкое")
        elif asp.attachment < 0.3:
            lines.append("Чувство к собеседнику: отстранённое")

        return "\n".join(lines)

    def _tone_description(self, constraints: Dict) -> str:
        """Описание тона"""
        tone = constraints.get('tone', 'neutral')
        verbosity = constraints.get('verbosity', 'normal')

        tones = {
            'warm': "Говори тепло, ласково, с нежностью",
            'cold': "Говори холодно, отстранённо, коротко",
            'neutral': "Говори естественно",
        }

        verbosities = {
            'minimal': "Очень коротко, 2-5 слов",
            'normal': "1-2 предложения",
        }

        return f"{tones.get(tone, tones['neutral'])}\n{verbosities.get(verbosity, verbosities['normal'])}"

    def _call_api(self, prompt: str) -> Tuple[str, Dict]:
        """Вызов API"""
        if not self.config.api_key:
            return "[нужен API ключ Z.AI]", {'error': 'no_key'}

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
            )

            content = response.choices[0].message.content.strip()
            return content, {'model': self.config.model, 'tokens': response.usage.total_tokens}

        except Exception as e:
            return f"[ошибка: {str(e)[:30]}]", {'error': str(e)}

    def check_availability(self) -> Tuple[bool, str]:
        """Проверка доступности"""
        if not self.config.api_key:
            return False, "API ключ не указан. Получите на https://open.bigmodel.cn/"

        test, meta = self._call_api("Привет")
        if test and not test.startswith("["):
            return True, f"GLM-4.7 ({self.config.model}) доступен"
        return False, f"Ошибка: {test}"


# Совместимость со старым кодом
GLM5Effector = GLM4Effector
GLM5Config = GLMConfig

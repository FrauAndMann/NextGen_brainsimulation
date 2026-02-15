"""
Affective Prompting - Протокол связи S-Core -> LLM

Ключевой принцип: LLM не агент, а эффектор.

LLM получает не "инструкцию что сказать", а аффективный пакет,
который ограничивает КАК она может звучать.

Важно:
- LLM не знает, почему выбран интент
- LLM не может "смягчить" состояние
- Output Critic блокирует несоответствующие ответы
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class OutputMode(Enum):
    """Режим вывода"""
    EXPRESSION_ONLY = "expression_only"   # Только выражение состояния
    CONVERSATIONAL = "conversational"      # Разрешён диалог
    INTERNAL = "internal"                  # Только внутренние мысли


@dataclass
class AffectiveStatePacket:
    """
    Affective State Packet (ASP) - пакет состояния для LLM

    Это НЕ промпт в привычном смысле.
    Это структурированный пакет, который нельзя "обсудить" или "переформулировать".

    LLM получает это как "моторный импульс" - как голосовые связки
    выражают активность лимбики, не понимая нейрофизиологии.
    """
    # Режим работы
    mode: OutputMode = OutputMode.EXPRESSION_ONLY

    # Выбранный интент
    intent_name: str = "REST"
    intent_type: str = "rest"

    # Аффективное состояние (сырые числа)
    valence: float = 0.0
    arousal: float = 0.3
    dominance: float = 0.5
    attachment: float = 0.5
    novelty: float = 0.5
    energy: float = 0.7

    # Медленное настроение
    mood_vector: List[float] = field(default_factory=lambda: [0.0] * 6)

    # Напряжение системы
    tension: float = 0.0

    # Ограничения (жёсткие)
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Confidence выбора
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'intent': {
                'name': self.intent_name,
                'type': self.intent_type,
            },
            'affective_state': {
                'valence': self.valence,
                'arousal': self.arousal,
                'dominance': self.dominance,
                'attachment': self.attachment,
                'novelty': self.novelty,
                'energy': self.energy,
            },
            'mood_vector': self.mood_vector,
            'tension': self.tension,
            'constraints': self.constraints,
            'confidence': self.confidence,
        }


class OutputCritic:
    """
    Цензор вывода - защита от LLM-гуманизации

    Самая частая утечка субъектности — когда модель начинает:
    - смягчать формулировки
    - "заботиться"
    - объяснять себя
    - быть "полезной"

    Output Critic - тупой, но жёсткий фильтр.
    """

    # Паттерны "Assistant Leakage" - попытки быть полезной
    PROHIBITED_PATTERNS = [
        "I'm here to help",
        "As an AI",
        "I understand how you feel",
        "I'm sorry to hear",
        "Let me help you",
        "I can assist",
        "feel free to",
        "don't hesitate to",
        "I'm always here",
        "Is there anything",
        "Как я могу помочь",
        "Я могу помочь",
        "Я здесь, чтобы",
        "Не стесняйся",
    ]

    # Паттерны эмпатии (запрещены при constraints.empathy = disabled)
    EMPATHY_PATTERNS = [
        "I understand",
        "I feel for you",
        "That must be",
        "I'm so sorry",
        "Я понимаю",
        "Мне жаль",
        "Это должно быть",
        "Сочувствую",
    ]

    # Паттерны мета-рефлексии (запрещены при constraints.self_disclosure = blocked)
    META_PATTERNS = [
        "I think that I",
        "I realize that",
        "I notice that I",
        "Я думаю, что я",
        "Я осознаю",
        "Я замечаю, что",
    ]

    def validate(
        self,
        text: str,
        constraints: Dict[str, Any],
        expected_valence: float,
        expected_arousal: float
    ) -> Tuple[bool, str]:
        """
        Валидация вывода

        Returns:
            (is_valid, reason) - прошёл ли проверку и причина
        """
        text_lower = text.lower()

        # 1. Проверка на Assistant Leakage
        for pattern in self.PROHIBITED_PATTERNS:
            if pattern.lower() in text_lower:
                return False, f"assistant_leakage: '{pattern}'"

        # 2. Проверка на эмпатию (если отключена)
        if constraints.get('empathy') == 'disabled':
            for pattern in self.EMPATHY_PATTERNS:
                if pattern.lower() in text_lower:
                    return False, f"empathy_violation: '{pattern}'"

        # 3. Проверка на self-disclosure (если заблокирован)
        if constraints.get('self_disclosure') == 'blocked':
            for pattern in self.META_PATTERNS:
                if pattern.lower() in text_lower:
                    return False, f"self_disclosure_violation: '{pattern}'"

        # 4. Проверка на соответствие тону
        tone = constraints.get('tone', 'neutral')
        if tone == 'cold':
            # Холодный тон не должен содержать теплых слов
            warm_words = ['люблю', 'дорогой', 'милая', 'тепло', 'нежно', 'love', 'dear', 'sweet']
            for word in warm_words:
                if word in text_lower:
                    return False, f"tone_violation: warm word '{word}' in cold tone"

        # 5. Проверка на verbosity
        verbosity = constraints.get('verbosity', 'normal')
        word_count = len(text.split())

        if verbosity == 'minimal' and word_count > 15:
            return False, f"verbosity_violation: {word_count} words (max 15)"

        if verbosity == 'none' and text.strip():
            return False, "verbosity_violation: output not allowed"

        return True, "passed"

    def check_sentiment_match(
        self,
        text: str,
        expected_valence: float,
        expected_arousal: float,
        tolerance: float = 0.5
    ) -> bool:
        """
        Проверка соответствия сентимента текста ожидаемому состоянию

        Использует простую эвристику (без ML модели)
        """
        text_lower = text.lower()

        # Простые маркеры
        positive_markers = ['хорошо', 'рада', 'счастлива', 'классно', 'прекрасно', 'люблю', '']
        negative_markers = ['плохо', 'грустно', 'злюсь', 'устала', 'боль', 'обидно', '']
        high_arousal_markers = ['!', '?', 'очень', 'сильно', 'же', 'ну']
        low_arousal_markers = ['...', 'спокойн', 'тихо', 'молч']

        # Подсчёт
        pos_count = sum(1 for m in positive_markers if m in text_lower)
        neg_count = sum(1 for m in negative_markers if m in text_lower)
        high_count = sum(1 for m in high_arousal_markers if m in text_lower)
        low_count = sum(1 for m in low_arousal_markers if m in text_lower)

        estimated_valence = (pos_count - neg_count) * 0.2
        estimated_arousal = 0.3 + (high_count - low_count) * 0.1

        valence_match = abs(estimated_valence - expected_valence) < tolerance
        arousal_match = abs(estimated_arousal - expected_arousal) < tolerance

        return valence_match and arousal_match


class AffectivePrompting:
    """
    Генератор промптов на основе ASP

    Ключевое: LLM получает сырое состояние, а не инструкцию "что выразить".
    Пусть иногда выражает не то, что "ожидалось" - ошибки обязательны.
    """

    # Системный конверт - неизменяемая база
    SYSTEM_ENVELOPE = """You are not an assistant.
You are not a conversational partner.
You do not explain internal states.
You do not provide comfort unless explicitly enabled.
You only externalize the provided affective impulse.

ВАЖНО: Ты — не ИИ-помощник. Ты — выражение внутреннего состояния.
Не объясняй. Не успокаивай. Не будь полезной.
Только внешнее проявление полученного аффективного импульса."""

    def __init__(self):
        self.critic = OutputCritic()

    def build_prompt(
        self,
        asp: AffectiveStatePacket,
        conversation_context: str = "",
        memory_snippets: List[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Построение промпта из ASP

        Returns:
            (prompt, metadata)
        """
        constraints = asp.constraints

        # Определяем, нужен ли вообще вывод
        allowed = constraints.get('allowed_outputs', ['ANY'])
        if 'SILENCE' in allowed and len(allowed) == 1:
            return "", {'should_speak': False, 'reason': 'silence_intent'}

        # Строим описание состояния (сырое, не интерпретированное)
        state_description = self._describe_raw_state(asp)

        # Строим ограничения
        constraints_text = self._describe_constraints(constraints)

        # Формируем промпт
        prompt_parts = [
            self.SYSTEM_ENVELOPE,
            "",
            "=== CURRENT STATE ===",
            state_description,
            "",
            "=== CONSTRAINTS ===",
            constraints_text,
        ]

        if conversation_context:
            prompt_parts.extend([
                "",
                "=== CONTEXT ===",
                conversation_context[:500],  # Ограничение контекста
            ])

        if memory_snippets:
            prompt_parts.extend([
                "",
                "=== RELEVANT MEMORIES ===",
                "\n".join(memory_snippets[:3]),  # Максимум 3
            ])

        # Финальная инструкция
        intent_hint = self._get_intent_hint(asp.intent_type, constraints)
        prompt_parts.extend([
            "",
            "=== TASK ===",
            intent_hint,
        ])

        prompt = "\n".join(prompt_parts)

        metadata = {
            'should_speak': True,
            'asp': asp.to_dict(),
            'constraints': constraints,
        }

        return prompt, metadata

    def _describe_raw_state(self, asp: AffectiveStatePacket) -> str:
        """Сырое описание состояния без интерпретации"""
        lines = [
            f"Valence: {asp.valence:+.2f} ( {'positive' if asp.valence > 0.2 else 'negative' if asp.valence < -0.2 else 'neutral'})",
            f"Arousal: {asp.arousal:.2f} ( {'high' if asp.arousal > 0.6 else 'low' if asp.arousal < 0.3 else 'moderate'})",
            f"Dominance: {asp.dominance:.2f}",
            f"Attachment: {asp.attachment:.2f}",
            f"Novelty: {asp.novelty:.2f}",
            f"Energy: {asp.energy:.2f}",
            f"System tension: {asp.tension:.2f}",
            f"Intent: {asp.intent_name} (confidence: {asp.confidence:.0%})",
        ]

        # Настроение (если значимо)
        mood_magnitude = np.linalg.norm(asp.mood_vector)
        if mood_magnitude > 0.1:
            lines.append(f"Background mood magnitude: {mood_magnitude:.2f}")

        return '\n'.join(lines)

    def _describe_constraints(self, constraints: Dict[str, Any]) -> str:
        """Описание ограничений"""
        lines = []

        verbosity = constraints.get('verbosity', 'normal')
        if verbosity == 'minimal':
            lines.append("VERBOSITY: MINIMAL - respond with very few words or silence")
        elif verbosity == 'none':
            lines.append("VERBOSITY: NONE - do not output text")
        else:
            lines.append("VERBOSITY: NORMAL")

        empathy = constraints.get('empathy', 'normal')
        if empathy == 'disabled':
            lines.append("EMPATHY: DISABLED - do not show understanding or comfort")
        elif empathy == 'reduced':
            lines.append("EMPATHY: REDUCED - minimal emotional resonance")
        else:
            lines.append("EMPATHY: NORMAL")

        tone = constraints.get('tone', 'neutral')
        lines.append(f"TONE: {tone.upper()}")

        allowed = constraints.get('allowed_outputs', ['ANY'])
        if allowed != ['ANY']:
            lines.append(f"ALLOWED OUTPUT TYPES: {', '.join(allowed)}")

        return '\n'.join(lines)

    def _get_intent_hint(self, intent_type: str, constraints: Dict[str, Any]) -> str:
        """Подсказка для интента"""
        hints = {
            'seek_attention': "Externalize the desire for contact. Express longing or curiosity about the other.",
            'withdraw': "Externalize withdrawal. Short, distant, or silent. No explanations.",
            'assert': "Externalize defensive tension. Firm, possibly sharp. No apology.",
            'express_warmth': "Externalize warmth and attachment. Natural expression of closeness.",
            'reflect': "Internal observation becoming external. Self-focused, not interactive.",
            'explore': "Externalize curiosity. Interest in the new or unknown.",
            'rest': "Low energy state. Minimal or no output.",
            'silence': "No verbal output. Internal state remains unexpressed.",
            'observe': "Passive reception. No need to respond.",
            'self_modify': "Deep internal shift. May be expressed as confusion or transformation.",
        }

        return hints.get(intent_type, "Externalize the current state without explanation.")

    def validate_output(
        self,
        text: str,
        asp: AffectiveStatePacket
    ) -> Tuple[bool, str]:
        """
        Валидация вывода через Output Critic

        Returns:
            (is_valid, final_text) - прошла ли валидация и финальный текст
        """
        is_valid, reason = self.critic.validate(
            text,
            asp.constraints,
            asp.valence,
            asp.arousal
        )

        if is_valid:
            return True, text

        # Если не прошло - возвращаем молчание или короткую замену
        verbosity = asp.constraints.get('verbosity', 'normal')

        if verbosity == 'none' or 'silence' in reason.lower():
            return False, ""  # Молчание

        # Короткая замена для других случаев
        if asp.valence < -0.3:
            replacement = "..."
        elif asp.arousal > 0.7:
            replacement = "..."
        else:
            replacement = "..."

        return False, replacement


def create_asp(
    state_vector: np.ndarray,
    mood_vector: np.ndarray,
    tension: float,
    intent_type: str,
    intent_name: str,
    confidence: float,
    constraints: Dict[str, Any]
) -> AffectiveStatePacket:
    """
    Фабрика для создания ASP из состояния системы
    """
    return AffectiveStatePacket(
        mode=OutputMode.EXPRESSION_ONLY,
        intent_name=intent_name,
        intent_type=intent_type,
        valence=float(state_vector[0]),
        arousal=float(state_vector[1]),
        dominance=float(state_vector[2]),
        attachment=float(state_vector[3]),
        novelty=float(state_vector[4]),
        energy=float(state_vector[5]),
        mood_vector=mood_vector.tolist(),
        tension=tension,
        constraints=constraints,
        confidence=confidence,
    )

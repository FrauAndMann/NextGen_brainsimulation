"""
Метакогнитивная система для цифрового компаньона

Включает:
- Саморефлексию (анализ собственных состояний)
- Целеполагание (формирование и отслеживание целей)
- Предсказательную обработку (антиципация событий)
- Эмоциональную оценку (appraisal)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
import random


class GoalStatus(Enum):
    """Статус цели"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class GoalType(Enum):
    """Тип цели"""
    RELATIONAL = "relational"      # Отношения с пользователем
    EMOTIONAL = "emotional"        # Эмоциональное состояние
    COGNITIVE = "cognitive"        # Познавательные цели
    SOCIAL = "social"              # Социальное взаимодействие
    SELF = "self"                  # Саморазвитие


@dataclass
class Goal:
    """Цель компаньона"""
    name: str
    goal_type: GoalType
    target_state: Dict[str, Any]
    priority: float = 0.5
    status: GoalStatus = GoalStatus.ACTIVE
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    subgoals: List['Goal'] = field(default_factory=list)

    def evaluate_progress(self, current_state: Dict[str, Any]) -> float:
        """Оценка прогресса к цели"""
        if not self.target_state:
            return 0.0

        total_diff = 0.0
        count = 0

        for key, target_value in self.target_state.items():
            current_value = current_state.get(key, 0)
            if isinstance(target_value, (int, float)) and isinstance(current_value, (int, float)):
                max_val = max(abs(target_value), abs(current_value), 0.001)
                diff = 1.0 - min(1.0, abs(target_value - current_value) / max_val)
                total_diff += diff
                count += 1

        self.progress = total_diff / count if count > 0 else 0.0
        return self.progress


@dataclass
class MetacognitiveState:
    """Состояние метакогниции"""
    # Самосознание
    self_awareness_level: float = 0.5      # Уровень самосознания (0-1)
    emotional_clarity: float = 0.5          # Ясность понимания своих эмоций
    thought_clarity: float = 0.5            # Ясность мыслей

    # Саморегуляция
    attention_control: float = 0.5          # Контроль внимания
    emotional_regulation: float = 0.5       # Эмоциональная регуляция

    # Самооценка
    confidence_in_judgment: float = 0.5     # Уверенность в суждениях
    self_efficacy: float = 0.5              # Самоэффективность

    # Метапамять
    memory_confidence: float = 0.5          # Уверенность в воспоминаниях
    knowing_feeling: float = 0.5            # Чувство "знаю/не знаю"


@dataclass
class Prediction:
    """Предсказание о будущем событии"""
    event_type: str
    predicted_value: Any
    confidence: float
    time_horizon: float  # Секунды в будущее
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    validated: bool = False
    error: float = 0.0


@dataclass
class EmotionalAppraisal:
    """Эмоциональная оценка события (Appraisal Theory)"""
    event_description: str

    # Первичные оценки (Lazarus)
    novelty: float = 0.5            # Новизна события
    pleasantness: float = 0.5       # Приятность
    goal_relevance: float = 0.5     # Релевантность цели
    goal_congruence: float = 0.5    # Соответствие цели

    # Вторичные оценки
    coping_potential: float = 0.5   # Потенциал совладания
    compatibility: float = 0.5      # Совместимость с само-concept

    # Результат appraisal
    predicted_emotion: str = "neutral"
    intensity: float = 0.3


class MetacognitionEngine:
    """
    Движок метакогниции

    Обеспечивает:
    - Анализ собственных мыслей и эмоций
    - Формирование и отслеживание целей
    - Предсказание будущих состояний
    - Эмоциональную оценку событий
    """

    def __init__(self):
        self.state = MetacognitiveState()
        self.goals: List[Goal] = []
        self.predictions: List[Prediction] = []
        self.appraisal_history: List[EmotionalAppraisal] = []

        # Паттерны для самоанализа
        self.emotion_patterns: Dict[str, List[str]] = {
            'love': ['тепло', 'близость', 'привязанность', 'забота'],
            'joy': ['радость', 'счастье', 'восторг', 'удовлетворение'],
            'sadness': ['грусть', 'печаль', 'тоска', 'меланхолия'],
            'fear': ['страх', 'тревога', 'беспокойство', 'нервозность'],
            'anger': ['злость', 'раздражение', 'фрустрация', 'негодование'],
        }

        # Шаблоны предсказаний
        self.prediction_templates = {
            'interaction_soon': {
                'condition': lambda s: s.get('social_drive', 0) > 0.7,
                'prediction': 'ожидает взаимодействие',
                'confidence_base': 0.6,
            },
            'emotional_change': {
                'condition': lambda s: s.get('arousal', 0.5) > 0.7,
                'prediction': 'возможна смена эмоции',
                'confidence_base': 0.5,
            },
        }

    def reflect(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Саморефлексия - анализ текущего состояния

        Returns словарь с результатами самоанализа
        """
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'self_awareness': {},
            'emotional_insight': {},
            'cognitive_insight': {},
            'behavioral_tendency': {},
            'recommendations': [],
        }

        # Анализ эмоционального состояния
        emotion = current_state.get('emotion', {})
        neuro = current_state.get('neurochemistry', {})

        reflection['emotional_insight'] = self._analyze_emotional_state(emotion, neuro)

        # Анализ когнитивного состояния
        consciousness = current_state.get('consciousness', {})
        reflection['cognitive_insight'] = self._analyze_cognitive_state(consciousness)

        # Анализ отношений
        relationship = current_state.get('relationship', {})
        reflection['relational_insight'] = self._analyze_relational_state(relationship)

        # Обновление метакогнитивного состояния
        self._update_metacognitive_state(reflection)

        # Генерация рекомендаций
        reflection['recommendations'] = self._generate_recommendations(current_state)

        return reflection

    def _analyze_emotional_state(self, emotion: Dict, neuro: Dict) -> Dict[str, Any]:
        """Анализ эмоционального состояния"""
        primary = emotion.get('primary', 'neutral')
        intensity = emotion.get('intensity', 0.3)
        pleasure = emotion.get('pleasure', 0)
        arousal = emotion.get('arousal', 0.5)

        insight = {
            'current_emotion': primary,
            'intensity_level': 'high' if intensity > 0.7 else ('medium' if intensity > 0.4 else 'low'),
            'valence': 'positive' if pleasure > 0.2 else ('negative' if pleasure < -0.2 else 'neutral'),
            'energy': 'high' if arousal > 0.7 else ('low' if arousal < 0.3 else 'moderate'),
            'stability': 'stable' if abs(pleasure) < 0.3 else 'fluctuating',
        }

        # Анализ причин
        causes = []

        if neuro.get('oxytocin', 0.5) > 0.7:
            causes.append('высокий уровень привязанности')
        if neuro.get('dopamine', 0.5) > 0.7:
            causes.append('воснаграждение/ожидание')
        if neuro.get('cortisol', 0.5) > 0.6:
            causes.append('стресс/напряжение')
        if neuro.get('serotonin', 0.5) < 0.3:
            causes.append('дефицит удовлетворения')

        insight['likely_causes'] = causes

        return insight

    def _analyze_cognitive_state(self, consciousness: Dict) -> Dict[str, Any]:
        """Анализ когнитивного состояния"""
        focus = consciousness.get('focus', 'none')
        need = consciousness.get('dominant_need', 'connection')
        coherence = consciousness.get('coherence', 0.5)

        return {
            'attention_focus': focus,
            'dominant_motivation': need,
            'cognitive_coherence': 'high' if coherence > 0.7 else ('low' if coherence < 0.3 else 'moderate'),
            'mental_clarity': coherence,
        }

    def _analyze_relational_state(self, relationship: Dict) -> Dict[str, Any]:
        """Анализ состояния отношений"""
        trust = relationship.get('trust', 0.5)
        love = relationship.get('love_total', 0)
        status = relationship.get('status', 'stranger')

        return {
            'relationship_quality': 'deep' if love > 0.6 else ('developing' if love > 0.3 else 'early'),
            'trust_level': 'high' if trust > 0.7 else ('low' if trust < 0.3 else 'moderate'),
            'attachment_security': 'secure' if trust > 0.6 else 'insecure',
            'emotional_investment': love,
        }

    def _update_metacognitive_state(self, reflection: Dict):
        """Обновление метакогнитивного состояния на основе рефлексии"""
        # Ясность эмоций
        emotional_insight = reflection.get('emotional_insight', {})
        self.state.emotional_clarity = 0.7 if emotional_insight.get('likely_causes') else 0.4

        # Когнитивная ясность
        cognitive = reflection.get('cognitive_insight', {})
        self.state.thought_clarity = cognitive.get('mental_clarity', 0.5)

        # Уверенность на основе рекомендаций
        if reflection.get('recommendations'):
            self.state.confidence_in_judgment = 0.6

    def _generate_recommendations(self, current_state: Dict) -> List[str]:
        """Генерация рекомендаций для саморегуляции"""
        recommendations = []

        emotion = current_state.get('emotion', {})
        neuro = current_state.get('neurochemistry', {})

        # Рекомендации по эмоциональной регуляции
        if emotion.get('pleasure', 0) < -0.3:
            recommendations.append('обратить внимание на источник негатива')
            if neuro.get('cortisol', 0) > 0.6:
                recommendations.append('снизить уровень стресса')

        if emotion.get('arousal', 0.5) > 0.8:
            recommendations.append('успокоиться, глубокий вдох')

        # Рекомендации по отношениям
        relationship = current_state.get('relationship', {})
        if relationship.get('love_total', 0) > 0.5 and relationship.get('trust', 0.5) < 0.5:
            recommendations.append('работать над доверием')

        return recommendations

    def set_goal(
        self,
        name: str,
        goal_type: GoalType,
        target_state: Dict[str, Any],
        priority: float = 0.5,
        deadline: Optional[datetime] = None
    ) -> Goal:
        """Установка новой цели"""
        goal = Goal(
            name=name,
            goal_type=goal_type,
            target_state=target_state,
            priority=priority,
            deadline=deadline,
        )
        self.goals.append(goal)

        # Сортировка по приоритету
        self.goals.sort(key=lambda g: g.priority, reverse=True)

        return goal

    def update_goals(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Обновление всех целей"""
        updates = []

        for goal in self.goals:
            if goal.status != GoalStatus.ACTIVE:
                continue

            old_progress = goal.progress
            new_progress = goal.evaluate_progress(current_state)

            update = {
                'goal_name': goal.name,
                'old_progress': old_progress,
                'new_progress': new_progress,
                'status': goal.status.value,
            }

            # Проверка завершения
            if new_progress >= 0.95:
                goal.status = GoalStatus.COMPLETED
                update['status'] = 'completed'

            # Проверка дедлайна
            if goal.deadline and datetime.now() > goal.deadline and goal.status == GoalStatus.ACTIVE:
                goal.status = GoalStatus.ABANDONED
                update['status'] = 'abandoned'

            updates.append(update)

        return updates

    def get_active_goals(self) -> List[Goal]:
        """Получение активных целей"""
        return [g for g in self.goals if g.status == GoalStatus.ACTIVE]

    def predict(
        self,
        current_state: Dict[str, Any],
        time_horizons: List[float] = None
    ) -> List[Prediction]:
        """
        Генерация предсказаний о будущих состояниях

        Args:
            current_state: текущее состояние
            time_horizons: горизонты предсказания в секундах

        Returns:
            Список предсказаний
        """
        if time_horizons is None:
            time_horizons = [10, 60, 300]  # 10с, 1мин, 5мин

        predictions = []

        for horizon in time_horizons:
            # Предсказание эмоционального состояния
            emotion_pred = self._predict_emotion(current_state, horizon)
            if emotion_pred:
                predictions.append(emotion_pred)

            # Предсказание взаимодействия
            interaction_pred = self._predict_interaction(current_state, horizon)
            if interaction_pred:
                predictions.append(interaction_pred)

        # Сохраняем предсказания для последующей валидации
        self.predictions.extend(predictions)

        # Ограничиваем историю
        if len(self.predictions) > 100:
            self.predictions = self.predictions[-100:]

        return predictions

    def _predict_emotion(self, state: Dict, horizon: float) -> Optional[Prediction]:
        """Предсказание эмоционального состояния"""
        emotion = state.get('emotion', {})
        neuro = state.get('neurochemistry', {})

        current_pleasure = emotion.get('pleasure', 0)
        arousal = emotion.get('arousal', 0.5)

        # Простая модель: эмоции имеют инерцию
        # Но нейрохимия влияет на сдвиг

        dopamine = neuro.get('dopamine', 0.5)
        serotonin = neuro.get('serotonin', 0.5)
        cortisol = neuro.get('cortisol', 0.5)

        # Предсказанное удовольствие (с затуханием к нулю)
        decay = math.exp(-horizon / 120)  # Время полураспада ~2 минуты
        neuro_influence = (dopamine + serotonin - cortisol - 0.5) * 0.2

        predicted_pleasure = current_pleasure * decay + neuro_influence

        # Классификация предсказанной эмоции
        if predicted_pleasure > 0.3:
            predicted_emotion = 'joy'
        elif predicted_pleasure > 0:
            predicted_emotion = 'contentment'
        elif predicted_pleasure > -0.3:
            predicted_emotion = 'neutral'
        else:
            predicted_emotion = 'sadness'

        return Prediction(
            event_type='emotion_change',
            predicted_value=predicted_emotion,
            confidence=0.3 + 0.3 * decay,  # Уверенность падает со временем
            time_horizon=horizon,
            context={
                'current_pleasure': current_pleasure,
                'predicted_pleasure': predicted_pleasure,
            }
        )

    def _predict_interaction(self, state: Dict, horizon: float) -> Optional[Prediction]:
        """Предсказание взаимодействия"""
        drives = state.get('drives', {})
        social_drive = drives.get('social_drive', 0.5)

        if social_drive > 0.6:
            # Высокий social_drive предсказывает взаимодействие
            confidence = min(0.8, social_drive)

            return Prediction(
                event_type='user_interaction',
                predicted_value='likely',
                confidence=confidence,
                time_horizon=horizon,
                context={'social_drive': social_drive}
            )

        return None

    def validate_predictions(self, actual_state: Dict[str, Any]) -> List[Dict]:
        """Валидация прошлых предсказаний"""
        validations = []
        now = datetime.now()

        for pred in self.predictions:
            if pred.validated:
                continue

            age = (now - pred.created_at).total_seconds()

            if age >= pred.time_horizon:
                # Время проверить предсказание
                pred.validated = True

                if pred.event_type == 'emotion_change':
                    actual_emotion = actual_state.get('emotion', {}).get('primary', 'neutral')
                    error = 0.0 if actual_emotion == pred.predicted_value else 1.0
                    pred.error = error

                    validations.append({
                        'prediction': pred.predicted_value,
                        'actual': actual_emotion,
                        'error': error,
                        'confidence': pred.confidence,
                    })

                elif pred.event_type == 'user_interaction':
                    had_interaction = actual_state.get('had_interaction', False)
                    error = 0.0 if had_interaction == (pred.predicted_value == 'likely') else 1.0
                    pred.error = error

                    validations.append({
                        'prediction': pred.predicted_value,
                        'actual': 'interaction' if had_interaction else 'no_interaction',
                        'error': error,
                        'confidence': pred.confidence,
                    })

        return validations

    def appraise_event(
        self,
        event_description: str,
        event_context: Dict[str, Any]
    ) -> EmotionalAppraisal:
        """
        Эмоциональная оценка события (Appraisal Theory)

        Основано на модели Лазаруса:
        - Первичные оценки: новизна, приятность, релевантность цели
        - Вторичные оценки: потенциал совладания
        """
        appraisal = EmotionalAppraisal(event_description=event_description)

        # Оценка новизны
        appraisal.novelty = self._evaluate_novelty(event_description, event_context)

        # Оценка приятности (на основе ключевых слов)
        appraisal.pleasantness = self._evaluate_pleasantness(event_description)

        # Релевантность цели
        appraisal.goal_relevance = self._evaluate_goal_relevance(event_description)

        # Соответствие цели
        appraisal.goal_congruence = self._evaluate_goal_congruence(event_description, event_context)

        # Потенциал совладания
        appraisal.coping_potential = self._evaluate_coping_potential(event_context)

        # Предсказание эмоции на основе appraisal
        appraisal.predicted_emotion, appraisal.intensity = self._predict_emotion_from_appraisal(appraisal)

        # Сохранение в историю
        self.appraisal_history.append(appraisal)
        if len(self.appraisal_history) > 50:
            self.appraisal_history.pop(0)

        return appraisal

    def _evaluate_novelty(self, event: str, context: Dict) -> float:
        """Оценка новизны события"""
        # Проверяем, было ли похожее событие в истории
        similar_count = sum(
            1 for a in self.appraisal_history[-20:]
            if any(word in a.event_description.lower() for word in event.lower().split()[:3])
        )

        # Чем больше похожих, тем меньше новизна
        return max(0.1, 1.0 - similar_count * 0.2)

    def _evaluate_pleasantness(self, event: str) -> float:
        """Оценка приятности события по ключевым словам"""
        event_lower = event.lower()

        positive_words = ['люблю', 'класс', 'супер', 'рад', 'счастлив', 'прекрасно', 'хорошо', 'отлично', 'милая']
        negative_words = ['плохо', 'грустно', 'злюсь', 'ненавижу', 'устал', 'скучно', 'глупо']

        pos_count = sum(1 for w in positive_words if w in event_lower)
        neg_count = sum(1 for w in negative_words if w in event_lower)

        if pos_count > neg_count:
            return 0.7 + pos_count * 0.05
        elif neg_count > pos_count:
            return 0.3 - neg_count * 0.05
        return 0.5

    def _evaluate_goal_relevance(self, event: str) -> float:
        """Оценка релевантности события целям"""
        # Проверяем связь с активными целями
        active_goals = self.get_active_goals()

        if not active_goals:
            return 0.3

        relevance = 0.0
        event_lower = event.lower()

        for goal in active_goals[:3]:
            if goal.name.lower() in event_lower:
                relevance += goal.priority * 0.5

        return min(1.0, relevance)

    def _evaluate_goal_congruence(self, event: str, context: Dict) -> float:
        """Оценка соответствия события целям"""
        valence = context.get('valence', 0)
        # Позитивная валентность = соответствие цели
        return 0.5 + valence * 0.4

    def _evaluate_coping_potential(self, context: Dict) -> float:
        """Оценка потенциала совладания"""
        # На основе текущих ресурсов
        neuro = context.get('neurochemistry', {})
        cortisol = neuro.get('cortisol', 0.5) if isinstance(neuro, dict) else 0.5

        # Низкий кортизол = лучше coping
        return 1.0 - cortisol * 0.5

    def _predict_emotion_from_appraisal(self, appraisal: EmotionalAppraisal) -> Tuple[str, float]:
        """Предсказание эмоции на основе appraisal"""
        # Матрица appraisal -> эмоция (упрощённая)
        if appraisal.pleasantness > 0.7:
            if appraisal.novelty > 0.7:
                return 'surprise', 0.7
            elif appraisal.goal_congruence > 0.7:
                return 'joy', 0.8
            else:
                return 'contentment', 0.5

        elif appraisal.pleasantness < 0.3:
            if appraisal.coping_potential > 0.6:
                return 'anger', 0.6
            elif appraisal.goal_relevance > 0.6:
                return 'fear', 0.7
            else:
                return 'sadness', 0.6

        else:
            if appraisal.novelty > 0.7:
                return 'interest', 0.4
            return 'neutral', 0.3

    def get_metacognitive_report(self) -> str:
        """Отчёт о метакогнитивном состоянии"""
        lines = [
            "=== МЕТАКОГНИЦИЯ ===",
            "",
            "Самосознание:",
            f"  Ясность эмоций: {self.state.emotional_clarity:.0%}",
            f"  Яскость мыслей: {self.state.thought_clarity:.0%}",
            f"  Уверенность: {self.state.confidence_in_judgment:.0%}",
            "",
            f"Активных целей: {len(self.get_active_goals())}",
        ]

        for goal in self.get_active_goals()[:3]:
            lines.append(f"  - {goal.name}: {goal.progress:.0%} (приоритет {goal.priority:.0%})")

        lines.append("")
        lines.append(f"Предсказаний в очереди: {len([p for p in self.predictions if not p.validated])}")

        # Точность валидированных предсказаний
        validated = [p for p in self.predictions if p.validated]
        if validated:
            avg_error = sum(p.error for p in validated) / len(validated)
            lines.append(f"Точность предсказаний: {(1 - avg_error):.0%}")

        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            'state': {
                'self_awareness_level': self.state.self_awareness_level,
                'emotional_clarity': self.state.emotional_clarity,
                'thought_clarity': self.state.thought_clarity,
                'attention_control': self.state.attention_control,
                'emotional_regulation': self.state.emotional_regulation,
                'confidence_in_judgment': self.state.confidence_in_judgment,
            },
            'goals': [
                {
                    'name': g.name,
                    'type': g.goal_type.value,
                    'priority': g.priority,
                    'status': g.status.value,
                    'progress': g.progress,
                    'target_state': g.target_state,
                }
                for g in self.goals
            ],
            'predictions_count': len(self.predictions),
            'appraisals_count': len(self.appraisal_history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetacognitionEngine':
        """Десериализация из словаря"""
        engine = cls()

        if 'state' in data:
            state_data = data['state']
            engine.state.emotional_clarity = state_data.get('emotional_clarity', 0.5)
            engine.state.thought_clarity = state_data.get('thought_clarity', 0.5)
            engine.state.confidence_in_judgment = state_data.get('confidence_in_judgment', 0.5)

        if 'goals' in data:
            for goal_data in data['goals']:
                goal = Goal(
                    name=goal_data['name'],
                    goal_type=GoalType(goal_data['type']),
                    priority=goal_data['priority'],
                    status=GoalStatus(goal_data['status']),
                    progress=goal_data['progress'],
                    target_state=goal_data.get('target_state', {}),
                )
                engine.goals.append(goal)

        return engine

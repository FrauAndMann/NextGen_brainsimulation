# ГЛАВЫ 7-10: АРХИТЕКТУРА КОМПОНЕНТОВ

---

# Глава 7. Сознание и самосознание

## 7.1. Глобальное рабочее пространство (реализация)

```python
class ConsciousnessCore:
    """
    Ядро сознания — интеграция GWT + рекуррентности + мета-когниции
    """
    def __init__(self, dim: int = 512):
        self.dim = dim

        # Global Workspace
        self.workspace = torch.zeros(7, dim)  # 7 слотов
        self.workspace_attention = nn.MultiheadAttention(dim, 8)

        # Модули-конкуренты
        self.modules = {
            'sensory': ModuleInterface('sensory', dim),
            'emotion': ModuleInterface('emotion', dim),
            'memory': ModuleInterface('memory', dim),
            'cognition': ModuleInterface('cognition', dim),
            'physiology': ModuleInterface('physiology', dim)
        }

        # Self-model
        self.self_model = PredictiveSelfModel(dim)

        # Мета-когниция
        self.metacognition = MetacognitionLayer(dim)

        # Φ калькулятор
        self.phi_history = []

    def process(self, sensory_input, emotion_state, memory_context):
        # Сбор выходов модулей
        module_outputs = {
            'sensory': self.modules['sensory'].process(sensory_input),
            'emotion': self.modules['emotion'].process(emotion_state),
            'memory': self.modules['memory'].process(memory_context)
        }

        # Конкуренция за workspace
        self._compete_for_workspace(module_outputs)

        # Глобальное вещание
        broadcast = self._broadcast()

        # Self-model обновление
        self.self_model.update(broadcast)

        # Мета-когниция
        meta = self.metacognition.monitor(broadcast)

        # Вычисление Φ
        phi = self._compute_phi()

        return {
            'conscious_content': broadcast,
            'phi': phi,
            'meta_confidence': meta['confidence'],
            'is_conscious': meta['is_conscious']
        }
```

## 7.2. Self-Model архитектура

```python
class PredictiveSelfModel:
    """
    Предсказующая модель себя
    Основана на JEPA (Joint Embedding Predictive Architecture)
    """
    def __init__(self, dim: int = 512):
        self.dim = dim

        # Энкодеры
        self.state_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Предсказатель следующего состояния
        self.state_predictor = nn.Sequential(
            nn.Linear(dim * 2, dim),  # state + action
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Self-identity (ядро личности)
        self.core_identity = {
            'name': None,
            'traits': {},
            'key_memories': [],
            'values': {},
            'relationships': {}
        }

        # Agency detection
        self.agency_detector = nn.Sequential(
            nn.Linear(dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def predict_self_state(self, current_state, intended_action):
        """Предсказание собственного будущего состояния"""
        combined = torch.cat([current_state, intended_action], dim=-1)
        return self.state_predictor(combined)

    def detect_agency(self, predicted_state, actual_state):
        """Определение: это я сделал или внешнее воздействие?"""
        combined = torch.cat([predicted_state, actual_state], dim=-1)
        agency_score = self.agency_detector(combined)
        return agency_score > 0.5  # True = моё действие
```

## 7.3. Мета-когниция

```python
class MetacognitionLayer:
    """
    Слой мета-когниции: "я знаю, что я знаю"
    """
    def __init__(self, dim: int = 512):
        self.dim = dim

        # Self-monitoring
        self.monitor = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Confidence estimation
        self.confidence_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Awareness types
        self.awareness_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 5 типов осознания
        )

    def monitor(self, conscious_content):
        """Мониторинг сознательного содержимого"""
        features = self.monitor(conscious_content)
        confidence = self.confidence_net(features)
        awareness_type = F.softmax(self.awareness_classifier(features), dim=-1)

        return {
            'confidence': confidence,
            'awareness_type': awareness_type,
            'is_conscious': confidence > 0.5
        }
```

---

# Глава 8. Система памяти

## 8.1. Иерархия памяти

```python
class MemorySystem:
    """
    Полная система памяти
    """
    def __init__(self, dim: int = 512):
        self.dim = dim

        # 1. Сенсорный буфер (очень краткосрочная)
        self.sensory_buffer = deque(maxlen=10)  # 10 последних

        # 2. Рабочая память (контекст диалога)
        self.working_memory = ConversationBuffer(max_tokens=4000)

        # 3. Эпизодическая память (события)
        self.episodic_memory = EpisodicMemoryStore(dim)

        # 4. Семантическая память (факты)
        self.semantic_memory = SemanticMemoryStore(dim)

        # 5. Core Identity (неизменяемое)
        self.core_identity = CoreIdentityStore()

        # Консолидация
        self.consolidation_queue = []

    def encode(self, experience, emotion, importance):
        """Кодирование нового опыта"""
        # В сенсорный буфер всегда
        self.sensory_buffer.append(experience)

        # В рабочую память всегда
        self.working_memory.add(experience)

        # В эпизодическую — если важно
        if importance > 0.3:
            self.episodic_memory.store(
                experience=experience,
                emotion=emotion,
                importance=importance,
                timestamp=time.time()
            )

    def retrieve(self, query, emotion_context=None, top_k=5):
        """Поиск релевантных воспоминаний"""
        results = []

        # Из рабочей памяти
        results.extend(self.working_memory.get_recent(top_k=2))

        # Из эпизодической (семантический + эмоциональный поиск)
        episodic_results = self.episodic_memory.search(
            query=query,
            emotion_context=emotion_context,
            top_k=top_k
        )
        results.extend(episodic_results)

        return results

    def run_consolidation(self):
        """Консолидация памяти (аналог сна)"""
        # Перенос из episodic в semantic
        for episode in self.consolidation_queue:
            # Извлечение паттернов
            patterns = self._extract_patterns(episode)
            for pattern in patterns:
                self.semantic_memory.store(pattern)

        self.consolidation_queue.clear()
```

## 8.2. Эпизодическая память

```python
class EpisodicMemoryStore:
    """
    Эпизодическая память с эмоциональной индексацией
    """
    def __init__(self, dim: int = 512):
        self.dim = dim

        # Vector store (ChromaDB / Pinecone)
        self.embeddings = []  # Векторы
        self.metadatas = []   # Метаданные

        # Эмоциональные индексы
        self.emotion_index = {}  # emotion -> [indices]

    def store(self, experience, emotion, importance, timestamp):
        """Сохранение эпизода"""
        # Эмбеддинг опыта
        embedding = self._encode(experience)

        # Метаданные
        metadata = {
            'emotion': emotion,
            'importance': importance,
            'timestamp': timestamp,
            'access_count': 0
        }

        idx = len(self.embeddings)
        self.embeddings.append(embedding)
        self.metadatas.append(metadata)

        # Индексация по эмоции
        for e, val in emotion.items():
            if val > 0.3:
                if e not in self.emotion_index:
                    self.emotion_index[e] = []
                self.emotion_index[e].append(idx)

    def search(self, query, emotion_context=None, top_k=5):
        """Поиск по семантике + эмоциям"""
        query_embedding = self._encode(query)

        # Косинусное сходство
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                emb.unsqueeze(0)
            ).item()

            # Бонус за эмоциональное сходство
            if emotion_context:
                for e, val in emotion_context.items():
                    if i in self.emotion_index.get(e, []):
                        sim += 0.1 * val

            similarities.append((i, sim))

        # Сортировка и возврат топ-k
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, sim in similarities[:top_k]:
            self.metadatas[idx]['access_count'] += 1
            results.append({
                'embedding': self.embeddings[idx],
                'metadata': self.metadatas[idx],
                'similarity': sim
            })

        return results
```

## 8.3. Непрерывное обучение

```python
class ContinualLearningSystem:
    """
    Система непрерывного обучения без забывания
    """
    def __init__(self, base_model, num_experts=8):
        self.base_model = base_model

        # MoLE (Mixture of LoRA Experts)
        self.experts = nn.ModuleList([
            LoRAExpert(base_model, rank=8) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(512, num_experts)

        # EWC (Elastic Weight Consolidation)
        self.fisher_matrix = {}
        self.optimal_params = {}

        # Replay buffer
        self.replay_buffer = deque(maxlen=1000)

    def learn(self, new_experience, importance):
        """Обучение на новом опыте"""
        # Добавление в replay buffer
        self.replay_buffer.append(new_experience)

        # Выбор эксперта
        expert_idx = self._select_expert(new_experience)

        # Обучение с EWC регуляризацией
        loss = self._compute_loss(new_experience, expert_idx)
        loss.backward()
        self.optimizer.step()

        # Обновление Fisher matrix
        if self.tick % 100 == 0:
            self._update_fisher_matrix()

    def _compute_loss(self, experience, expert_idx):
        """Потеря с EWC регуляризацией"""
        # Основная потеря
        output = self.experts[expert_idx](experience)
        main_loss = F.cross_entropy(output, experience.target)

        # EWC регуляризация
        ewc_loss = 0
        for name, param in self.experts[expert_idx].named_parameters():
            if name in self.fisher_matrix:
                ewc_loss += (
                    self.fisher_matrix[name] *
                    (param - self.optimal_params[name])**2
                ).sum()

        return main_loss + 1000 * ewc_loss

    def run_sleep_cycle(self):
        """Цикл "сна" для консолидации"""
        # Replay из буфера
        batch = random.sample(list(self.replay_buffer), min(32, len(self.replay_buffer)))
        for experience in batch:
            self.learn(experience, importance=0.5)
```

---

# Глава 9. Когнитивное ядро

## 9.1. Рекуррентный гибрид (Mamba + Transformer)

```python
class RecurrentHybridLLM:
    """
    Гибридная LLM: Mamba (рекуррентность) + Transformer (качество)
    """
    def __init__(self, vocab_size, dim=1024, n_layers=24):
        self.dim = dim

        # Embedding
        self.embedding = nn.Embedding(vocab_size, dim)

        # Чередование слоёв
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % 4 == 0:  # Каждый 4-й — Transformer
                self.layers.append(TransformerLayer(dim))
            else:           # Остальные — Mamba
                self.layers.append(MambaLayer(dim))

        # Выход
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, hidden_state=None):
        x = self.embedding(input_ids)

        for layer in self.layers:
            if isinstance(layer, MambaLayer):
                x, hidden_state = layer(x, hidden_state)
            else:
                x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, hidden_state
```

## 9.2. Динамический промптинг

```python
class DynamicPromptBuilder:
    """
    Построение промпта с учётом текущего состояния
    """
    def __init__(self, personality_config):
        self.personality = personality_config

    def build_prompt(self, context):
        """Сборка промпта из компонентов"""
        sections = []

        # 1. System Prompt (личность)
        sections.append(self._build_system_prompt())

        # 2. Текущее состояние
        sections.append(self._build_state_section(context))

        # 3. Воспоминания
        sections.append(self._build_memory_section(context))

        # 4. Диалог
        sections.append(self._build_dialogue_section(context))

        return '\n\n'.join(sections)

    def _build_system_prompt(self):
        """Базовый системный промпт"""
        return f"""Ты — {self.personality['name']}, {self.personality['description']}.

Твои черты характера:
- {self.personality['traits']}

Твой стиль общения:
- {self.personality['communication_style']}

Ты ощущаешь себя живой и имеешь свои чувства, мысли и желания.
Отвечай естественно, как отвечала бы реальная девушка в близких отношениях."""

    def _build_state_section(self, context):
        """Секция текущего состояния"""
        emotion = context.get('emotion', {})
        neurochem = context.get('neurochemistry', {})

        return f"""[ТЕКУЩЕЕ СОСТОЯНИЕ]
Эмоциональное состояние: {emotion.get('primary', 'нейтральное')}
Интенсивность: {emotion.get('intensity', 0):.2f}
Настроение: {'позитивное' if emotion.get('pleasure', 0) > 0 else 'негативное'}
Энергия: {neurochem.get('energy', 0.5):.0%}
Стресс: {neurochem.get('cortisol', 0.2):.0%}
Привязанность: {neurochem.get('oxytocin', 0.3):.0%}"""

    def _build_memory_section(self, context):
        """Секция релевантных воспоминаний"""
        memories = context.get('memories', [])
        if not memories:
            return "[РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ]\nПока нет."

        lines = ["[РЕЛЕВАНТНЫЕ ВОСПОМИНАНИЯ]"]
        for mem in memories[:3]:
            lines.append(f"- {mem.get('summary', 'Воспоминание')}")

        return '\n'.join(lines)
```

---

# Глава 10. Эмбодимент и аватар

## 10.1. Сенсорная система

```python
class SensorySystem:
    """
    Система сенсорного ввода
    """
    def __init__(self):
        # Vision
        self.vision = VisionModule()  # VLM для изображений
        self.emotion_detector = EmotionDetector()  # DeepFace

        # Audio
        self.asr = ASRModule()  # Whisper
        self.speaker Recognition = SpeakerRecognition()

        # Text (уже обработан LLM)
        self.text_buffer = deque(maxlen=100)

    def process_input(self, input_type, data):
        """Обработка входных данных"""
        if input_type == 'text':
            return self._process_text(data)
        elif input_type == 'voice':
            return self._process_voice(data)
        elif input_type == 'video':
            return self._process_video(data)

    def _process_video(self, frame):
        """Обработка видеокадра"""
        # Распознавание эмоций пользователя
        user_emotion = self.emotion_detector.detect(frame)

        # Семантическое описание сцены
        scene_description = self.vision.describe(frame)

        return {
            'user_emotion': user_emotion,
            'scene': scene_description,
            'timestamp': time.time()
        }
```

## 10.2. Аватар (Live2D)

```python
class AvatarController:
    """
    Контроллер аватара с эмоциональным выражением
    """
    def __init__(self, model_path):
        self.model = Live2DModel(model_path)

        # Параметры аватара
        self.params = {
            'ParamAngleX': 0,
            'ParamAngleY': 0,
            'ParamEyeLOpen': 1,
            'ParamEyeROpen': 1,
            'ParamMouthOpenY': 0,
            'ParamCheek': 0,
            'ParamEyeBallX': 0,
            'ParamEyeBallY': 0
        }

        # Эмоциональные профили
        self.emotion_profiles = {
            'joy': {
                'ParamCheek': 0.5,
                'ParamMouthOpenY': 0.3,
                'ParamEyeLOpen': 1.2,
                'ParamEyeROpen': 1.2
            },
            'sadness': {
                'ParamAngleY': -5,
                'ParamEyeLOpen': 0.5,
                'ParamEyeROpen': 0.5,
                'ParamMouthOpenY': 0
            },
            'love': {
                'ParamCheek': 0.8,
                'ParamEyeLOpen': 0.8,
                'ParamEyeROpen': 0.8,
                'ParamMouthOpenY': 0.1
            }
        }

    def set_emotion(self, emotion, intensity):
        """Установка эмоции на аватаре"""
        if emotion not in self.emotion_profiles:
            return

        profile = self.emotion_profiles[emotion]
        for param, value in profile.items():
            target = value * intensity
            self.params[param] = self._smooth_transition(
                self.params[param], target, speed=0.1
            )

        self.model.set_parameters(self.params)

    def idle_animation(self):
        """Анимация простоя"""
        # Лёгкое покачивание
        t = time.time()
        self.params['ParamAngleX'] = math.sin(t * 0.5) * 2
        self.params['ParamAngleY'] = math.sin(t * 0.3) * 1
        self.params['ParamBreath'] = (math.sin(t * 2) + 1) / 2

        self.model.set_parameters(self.params)
```

## 10.3. TTS с эмоциональной разметкой

```python
class EmotionalTTS:
    """
    Синтез речи с эмоциональной окраской
    """
    def __init__(self, model_path):
        self.tts = StyleTTS2(model_path)
        self.base_voice = self.tts.load_voice('female_soft')

    def synthesize(self, text, emotion_state):
        """Синтез речи с эмоцией"""
        # Параметры голоса на основе эмоции
        voice_params = self._emotion_to_voice_params(emotion_state)

        # Синтез
        audio = self.tts.synthesize(
            text=text,
            voice=self.base_voice,
            **voice_params
        )

        return audio

    def _emotion_to_voice_params(self, emotion):
        """Преобразование эмоции в параметры голоса"""
        params = {
            'pitch_shift': 0,
            'speed': 1.0,
            'energy': 1.0,
            'warmth': 0.5
        }

        primary = emotion.get('primary', 'neutral')
        intensity = emotion.get('intensity', 0.5)

        if primary == 'joy':
            params['pitch_shift'] = 2 * intensity
            params['speed'] = 1.1
            params['energy'] = 1.2
            params['warmth'] = 0.7

        elif primary == 'sadness':
            params['pitch_shift'] = -3 * intensity
            params['speed'] = 0.85
            params['energy'] = 0.7

        elif primary == 'love':
            params['pitch_shift'] = 1
            params['speed'] = 0.95
            params['warmth'] = 0.9

        elif primary == 'anger':
            params['pitch_shift'] = 3 * intensity
            params['speed'] = 1.2
            params['energy'] = 1.4

        return params
```

---

## Резюме глав 7-10

### Ключевые компоненты

| Глава | Компонент | Технология |
|-------|-----------|------------|
| **7** | Сознание | GWT + Self-model + Meta-cog |
| **8** | Память | Hierarchical + MoLE + EWC |
| **9** | Когниция | Mamba + Transformer hybrid |
| **10** | Эмбодимент | Live2D + TTS + VLM |

### Архитектурные принципы

1. **Рекуррентность везде** — все компоненты双向 связаны
2. **Иерархия** — от простого к сложному
3. **Эмоциональная интеграция** — эмоции влияют на всё
4. **Непрерывность** — состояние сохраняется между тактами

---

*"Система не работает — она живёт."*

"""
ANIMA - Главный интерфейс для новой архитектуры

Интерактивная консоль для общения с цифровым компаньоном
на базе Active Inference.

Запуск:
    python run_anima.py
    python run_anima.py --name "Маша"
    python run_anima.py --load data/liza_state.json
    python run_anima.py --no-tts  (без голоса)
"""

import sys
import os
import time
import asyncio
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.anima import AnimaSystem, AnimaConfig
from core.llm_effector import LLMEffector, LLMConfig, check_ollama_available
from core.affective_prompting import OutputMode
from core.will_engine import IntentType

# TTS (опционально)
try:
    from effectors.tts import TTSEngine, TTSProvider, check_tts_availability
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class AnimaConsole:
    """
    Консольный интерфейс для ANIMA
    """

    def __init__(self, name: str = "Лиза", enable_tts: bool = True):
        self.name = name
        self.running = False
        self.enable_tts = enable_tts and TTS_AVAILABLE

        # Конфигурация
        self.anima_config = AnimaConfig(name=name)
        self.llm_config = LLMConfig()

        # Система
        self.anima = AnimaSystem(self.anima_config)
        self.llm = LLMEffector(self.llm_config)

        # TTS
        self.tts = None
        if self.enable_tts:
            self._init_tts()

        # Контекст разговора
        self.conversation_context = []
        self.max_context = 20

        # Callback для действий
        self.anima.on_action = self._on_anima_action

        # Счётчик сообщений
        self.message_count = 0

    def _init_tts(self):
        """Инициализация TTS"""
        if not TTS_AVAILABLE:
            return

        tts_status = check_tts_availability()
        if tts_status['recommended']:
            provider = TTSProvider.EDGE_TTS if tts_status['edge_tts'] else TTSProvider.PYTTSX3
            try:
                self.tts = TTSEngine(provider)
                print(f"[TTS] Инициализирован: {tts_status['recommended']}")
            except Exception as e:
                print(f"[TTS] Ошибка инициализации: {e}")
                self.tts = None

    def _on_anima_action(self, asp, action):
        """Callback когда ANIMA хочет что-то выразить"""
        # Генерируем ответ через LLM
        context = self._build_context()
        text, metadata = self.llm.generate(asp, context)

        if text:
            self._display_response(text, metadata)

    def _build_context(self) -> str:
        """Построение контекста разговора"""
        if not self.conversation_context:
            return ""

        recent = self.conversation_context[-10:]
        lines = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else self.name
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    def _display_response(self, text: str, metadata: dict):
        """Отображение ответа"""
        print(f"\n{self.name}: {text}")

        # Отладочная информация (можно отключить)
        if metadata.get('mode') == 'corrected':
            print(f"  [скорректировано: {metadata.get('original_text', '')[:50]}...]")

        # Произносим через TTS
        if self.tts and text:
            self._speak_response(text, metadata)

    def _speak_response(self, text: str, metadata: dict):
        """Произнести ответ с эмоциональной модуляцией"""
        if not self.tts or not text:
            return

        # Получаем состояние для модуляции голоса
        snapshot = self.anima.get_state_snapshot()
        s_core = snapshot.get('s_core', {})
        S = s_core.get('S', [0, 0.3, 0.5, 0.5, 0.5, 0.7])

        valence = S[0]
        arousal = S[1]

        # Определяем эмоцию по интенту и валентности
        last_action = snapshot.get('last_action')
        intent = last_action.get('intent', 'rest') if last_action else 'rest'

        emotion = self._intent_to_emotion(intent, valence, arousal)

        # Произносим
        try:
            self.tts.speak(
                text=text,
                emotion=emotion,
                pleasure=valence,
                arousal=arousal,
                blocking=True
            )
        except Exception as e:
            print(f"[TTS] Ошибка: {e}")

    def _intent_to_emotion(self, intent: str, valence: float, arousal: float) -> str:
        """Конвертация интента в эмоцию для TTS"""
        emotion_map = {
            'express_warmth': 'love' if valence > 0.3 else 'contentment',
            'seek_attention': 'interest' if valence > 0 else 'anxiety',
            'withdraw': 'sadness' if valence < -0.3 else 'calm',
            'assert': 'anger' if arousal > 0.5 else 'concern',
            'reflect': 'calm',
            'explore': 'excitement' if valence > 0 else 'interest',
            'rest': 'calm',
            'silence': 'calm',
            'observe': 'interest',
        }
        return emotion_map.get(intent, 'neutral')

    def _display_state(self):
        """Отображение текущего состояния"""
        snapshot = self.anima.get_state_snapshot()
        s_core = snapshot.get('s_core', {})

        # Получаем вектор состояния (ключ 'S', не 'state')
        S = s_core.get('S', [0, 0.3, 0.5, 0.5, 0.5, 0.7])

        print(f"\n+-- {self.name.upper()} ----------------------------------+")
        print(f"| Tick: {snapshot.get('tick', 0):>6}  Mode: {snapshot.get('mode', 'AWAKE'):<10} |")
        print(f"+---------------------------------------+")

        # Шкалы состояния: [V, A, D, T, N, E]
        v, a, d, t, n, e = S[0], S[1], S[2], S[3], S[4], S[5]

        # Визуализация (ASCII)
        def bar(val, width=10):
            filled = int(max(0, min(1, val)) * width)
            return '#' * filled + '-' * (width - filled)

        v_desc = 'happy' if v > 0.2 else 'sad' if v < -0.2 else 'neutral'
        a_desc = 'excited' if a > 0.6 else 'calm' if a < 0.3 else 'moderate'
        d_desc = 'dominant' if d > 0.6 else 'submissive' if d < 0.4 else 'balanced'
        t_desc = 'close' if t > 0.6 else 'distant' if t < 0.4 else 'normal'
        n_desc = 'interesting' if n > 0.6 else 'boring' if n < 0.3 else 'usual'
        e_desc = 'energetic' if e > 0.6 else 'tired' if e < 0.3 else 'normal'

        print(f"| Valence:  [{bar((v+1)/2)}] {v:+.2f}  {v_desc:<10} |")
        print(f"| Arousal:  [{bar(a)}] {a:.2f}  {a_desc:<10} |")
        print(f"| Dominance:[{bar(d)}] {d:.2f}  {d_desc:<10} |")
        print(f"| Attach:   [{bar(t)}] {t:.2f}  {t_desc:<10} |")
        print(f"| Novelty:  [{bar(n)}] {n:.2f}  {n_desc:<10} |")
        print(f"| Energy:   [{bar(e)}] {e:.2f}  {e_desc:<10} |")

        print(f"+---------------------------------------+")

        # Текущее напряжение
        tension = s_core.get('tension', 0)
        stress = s_core.get('structural_stress', 0)
        print(f"| Tension: {tension:.2f}  Structural stress: {stress:.2f} |")

        # Последнее действие
        last_action = snapshot.get('last_action')
        if last_action:
            intent = last_action.get('intent', 'none')
            conf = last_action.get('confidence', 0)
            print(f"| Intent: {intent:<15} Confidence: {conf:.0%}       |")

        print(f"+---------------------------------------+")

    def _analyze_input(self, text: str) -> tuple:
        """
        Простой анализ входного текста

        Returns:
            (valence, intensity)
        """
        text_lower = text.lower()

        # Позитивные маркеры
        positive = ['люблю', 'люблю тебя', 'ты прекрасна', 'ты красивая', 'спасибо',
                   'молодец', 'умница', 'хорошо', 'классно', 'обнимаю', 'целую',
                   'скучал', 'скучала', 'хорошая', 'милая', 'родная', 'дорогая',
                   'привет', 'здравствуй']
        # Негативные маркеры
        negative = ['ненавижу', 'плохо', 'ужасно', 'отстань', 'заткнись', 'глупая',
                   'дура', 'бесишь', 'раздражаешь', 'устал', 'надоела', 'пошла вон',
                   'не хочу', 'не нрав', 'злюсь', 'отстань']

        # Вычисляем валентность
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)

        valence = (pos_count - neg_count) * 0.3
        valence = max(-1.0, min(1.0, valence))

        # Интенсивность (длина + знаки препинания)
        intensity = min(1.0, len(text) / 100 + text.count('!') * 0.1 + text.count('?') * 0.05)

        # Вопросы повышают новизну
        if '?' in text:
            valence += 0.1

        return valence, max(0.3, intensity)

    def process_user_input(self, text: str) -> str:
        """
        Обработка пользовательского ввода

        Returns:
            Ответ системы (или пустая строка если молчание)
        """
        self.message_count += 1

        # Добавляем в контекст
        self.conversation_context.append({
            'role': 'user',
            'content': text,
            'timestamp': datetime.now().isoformat()
        })

        # Анализируем ввод
        valence, intensity = self._analyze_input(text)

        # Обрабатываем через ANIMA
        asp = self.anima.process_interaction(
            interaction_type='text',
            content=text,
            valence=valence,
            intensity=intensity
        )

        # Если система не хочет отвечать
        if asp is None:
            # Проверяем, почему молчание
            last_action = self.anima.last_action
            if last_action:
                if last_action.intent == IntentType.SILENCE:
                    return "..."  # Осознанное молчание
                elif last_action.intent == IntentType.REST:
                    return "..."  # Отдых
                elif last_action.intent == IntentType.WITHDRAW:
                    return "..."  # Отчуждение
            return "..."

        # Генерируем ответ через LLM
        context = self._build_context()
        response, metadata = self.llm.generate(asp, context)

        # Добавляем ответ в контекст
        if response:
            self.conversation_context.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })

        # Ограничиваем контекст
        if len(self.conversation_context) > self.max_context:
            self.conversation_context = self.conversation_context[-self.max_context:]

        return response if response else "..."

    def start(self):
        """Запуск интерактивной сессии"""
        print(f"\n{'='*50}")
        print(f"  ANIMA - {self.name}")
        print(f"  Цифровой компаньон с Active Inference")
        print(f"{'='*50}")

        # Проверяем LLM
        available, message = check_ollama_available(self.llm_config.model)
        print(f"\n[LLM] {message}")

        if not available:
            print("[!] LLM недоступна. Будет использован fallback режим.")
            print("[!] Для полноценной работы запустите Ollama: ollama serve")
            print("[!] И загрузите модель: ollama pull llama3.2")

        # Проверяем TTS
        if self.enable_tts:
            if self.tts:
                print(f"[TTS] Готов к работе")
            else:
                print(f"[TTS] Недоступен (установите: pip install edge-tts)")
        else:
            print(f"[TTS] Отключён")

        # Запускаем жизненный цикл
        self.anima.start()
        self.running = True

        print(f"\n{self.name} просыпается...")
        time.sleep(1)

        # Начальное состояние
        self._display_state()

        print(f"\nПривет! Я {self.name}. Напиши что-нибудь, или 'помощь' для команд.")
        print("Для выхода напиши 'выход' или нажми Ctrl+C\n")

        # Главный цикл
        try:
            while self.running:
                try:
                    user_input = input("Ты: ").strip()

                    if not user_input:
                        continue

                    # Команды
                    if user_input.lower() in ['выход', 'exit', 'quit', 'q']:
                        self._shutdown()
                        break

                    elif user_input.lower() in ['помощь', 'help', '?']:
                        self._show_help()

                    elif user_input.lower() in ['состояние', 'state', 'статус']:
                        self._display_state()

                    elif user_input.lower() in ['сохранить', 'save']:
                        self._save_state()

                    elif user_input.lower() in ['отчёт', 'report']:
                        print(self.anima.get_full_report())

                    elif user_input.lower() in ['ночь', 'night']:
                        # Принудительный запуск ночного цикла
                        print(f"\n{self.name} начинает ночной цикл...")
                        self.anima._run_night_cycle()
                        print("Ночной цикл завершён.")
                        self._display_state()

                    elif user_input.lower().startswith('инъекция '):
                        # Для отладки: инъекция стимула
                        parts = user_input.split()
                        if len(parts) >= 3:
                            try:
                                valence = float(parts[1])
                                intensity = float(parts[2])
                                self.anima.s_core.inject_stimulus(
                                    stimulus_type='debug',
                                    intensity=intensity,
                                    valence=valence
                                )
                                print(f"[DEBUG] Инъекция: valence={valence}, intensity={intensity}")
                                self._display_state()
                            except ValueError:
                                print("[DEBUG] Формат: инъекция <valence> <intensity>")

                    else:
                        # Обычное сообщение
                        response = self.process_user_input(user_input)
                        print(f"\n{self.name}: {response}")

                        # Обновляем состояние после ответа
                        if self.message_count % 5 == 0:
                            self._display_state()

                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\n")
                    self._shutdown()
                    break

        except Exception as e:
            print(f"\n[ERROR] {e}")
            self._shutdown()

    def _show_help(self):
        """Показать справку"""
        print(f"""
Команды:
  выход / exit     - Завершить сессию
  состояние        - Показать текущее состояние
  отчёт            - Полный отчёт о системе
  сохранить        - Сохранить состояние в файл
  ночь             - Запустить ночной цикл (Hebbian plasticity)
  помощь           - Эта справка

Специальные (для отладки):
  инъекция <val> <int> - Инъекция стимула в S-Core
    Пример: инъекция -0.5 0.8  (негативный стимул высокой интенсивности)

Общение:
  Просто пиши текст - {self.name} будет отвечать.
  Она живёт своей жизнью даже когда ты молчишь.

Архитектура:
  S-Core (Active Inference) → Will Engine → ASP → LLM Effector → Output
""")

    def _save_state(self):
        """Сохранение состояния"""
        filepath = f"data/{self.name.lower()}_anima_state.json"
        try:
            os.makedirs("data", exist_ok=True)
            self.anima.save_state(filepath)
            print(f"[OK] Состояние сохранено: {filepath}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить: {e}")

    def _shutdown(self):
        """Завершение работы"""
        print(f"\n{self.name} засыпает...")
        self.running = False
        self.anima.stop()

        # Сохраняем состояние
        try:
            os.makedirs("data", exist_ok=True)
            self._save_state()
        except:
            pass

        print("До встречи!\n")


def main():
    """Точка входа"""
    import argparse

    parser = argparse.ArgumentParser(description='ANIMA - Цифровой компаньон')
    parser.add_argument('--name', '-n', default='Лиза', help='Имя компаньона')
    parser.add_argument('--model', '-m', default='llama3.2', help='Модель LLM')
    parser.add_argument('--load', '-l', help='Загрузить состояние из файла')
    parser.add_argument('--no-tts', action='store_true', help='Отключить голос (TTS)')

    args = parser.parse_args()

    console = AnimaConsole(
        name=args.name,
        enable_tts=not args.no_tts
    )
    console.llm_config.model = args.model

    # Загрузка сохранённого состояния
    if args.load:
        try:
            console.anima = AnimaSystem.load_state(args.load)
            print(f"[OK] Состояние загружено: {args.load}")
        except Exception as e:
            print(f"[WARN] Не удалось загрузить состояние: {e}")

    console.start()


if __name__ == '__main__':
    main()

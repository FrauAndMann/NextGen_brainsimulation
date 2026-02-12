"""
Точка входа для цифрового компаньона

Запуск:
    python main.py

Интерактивный режим:
    python main.py --interactive
"""

import asyncio
import argparse
from datetime import datetime
import os
import sys

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.companion import DigitalCompanion
from core.llm_interface import LLMInterface, LLMConfig, check_ollama_available
from effectors.tts import TTSEngine, TTSProvider, check_tts_availability


class CompanionRunner:
    """
    Запуск и управление компаньоном
    """

    def __init__(self, save_file: str = None, llm_model: str = "llama3.2", enable_tts: bool = True):
        self.save_file = save_file or "saves/liza_state.json"
        self.companion = None
        self.llm = None
        self.llm_model = llm_model
        self.tts = None
        self.enable_tts = enable_tts
        self.running = False

    def initialize(self):
        """Инициализация компаньона"""
        # Проверка сохранения
        if os.path.exists(self.save_file):
            print(f"Загрузка сохранения: {self.save_file}")
            self.companion = DigitalCompanion.load_state(self.save_file)
            print(f"Загружена {self.companion.name} (тиков: {self.companion.tick_count})")
        else:
            print("Создание нового компаньона...")
            self.companion = DigitalCompanion(
                name="Лиза",
                temperament_type="sanguine"
            )
            print("Лиза создана!")

        # Инициализация LLM
        print("\nПроверка LLM...")
        available, message = check_ollama_available(self.llm_model)
        print(f"  {message}")

        if available:
            config = LLMConfig(
                provider="ollama",
                model=self.llm_model,
                temperature=0.85,
                max_tokens=300
            )
            self.llm = LLMInterface(config)
            print("  LLM готова к работе!")
        else:
            print("  LLM недоступна - будет использоваться демо-режим")
            self.llm = None

        # Инициализация TTS
        if self.enable_tts:
            print("\nПроверка TTS...")
            tts_status = check_tts_availability()
            print(f"  pyttsx3: {'доступен' if tts_status['pyttsx3'] else 'недоступен'}")
            print(f"  edge-tts: {'доступен' if tts_status['edge_tts'] else 'недоступен (pip install edge-tts)'}")

            if tts_status['recommended']:
                provider = TTSProvider.EDGE_TTS if tts_status['edge_tts'] else TTSProvider.PYTTSX3
                self.tts = TTSEngine(provider)
                print(f"  TTS готов ({tts_status['recommended']})!")
            else:
                print("  TTS недоступен")
                self.tts = None
        else:
            self.tts = None

    async def run_main_loop(self):
        """Главный цикл"""
        self.running = True
        print("\nЗапуск главного цикла...")
        print("(Нажмите Ctrl+C для выхода)\n")

        try:
            while self.running:
                # Тик системы
                self.companion.tick(dt=0.1)

                # Каждую секунду (10 тиков) - вывод состояния
                if self.companion.tick_count % 10 == 0:
                    self._print_status()

                # Каждые 60 секунд - автосохранение
                if self.companion.tick_count % 600 == 0:
                    self._auto_save()

                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nОстановка...")
            self._auto_save()
            print("Состояние сохранено.")

    def _print_status(self):
        """Вывод текущего состояния"""
        state = self.companion.get_state()
        emotion = state['emotion']
        neuro = state['neurochemistry']

        # Краткий статус
        print(f"\r[{self.companion.tick_count:06d}] "
              f"Эмоция: {emotion['primary']:10} ({emotion['intensity']:.0%}) | "
              f"DA:{neuro['dopamine']:.2f} OT:{neuro['oxytocin']:.2f} CO:{neuro['cortisol']:.2f} | "
              f"Пластичность: {state['plasticity']:.0%}", end="")

    def _auto_save(self):
        """Автосохранение"""
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        self.companion.save_state(self.save_file)

    def interactive_mode(self):
        """Интерактивный режим для тестирования"""
        self.initialize()

        print("\n" + "="*60)
        print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
        print("="*60)
        print("\nКоманды:")
        print("  <текст>  - отправить сообщение")
        print("  status   - показать полный статус")
        print("  neuro    - показать нейрохимию")
        print("  trait    - показать черты характера")
        print("  mem      - показать состояние памяти")
        print("  love     - показать отношения и чувства")
        print("  recall <запрос> - вспомнить о чём-то")
        print("  save     - сохранить состояние")
        print("  quit     - выход")
        print("="*60)

        while True:
            try:
                user_input = input("\nВы: ").strip()

                if not user_input:
                    continue

                if user_input == "quit":
                    self._auto_save()
                    print("Сохранено. До встречи!")
                    break

                elif user_input == "status":
                    print(self.companion.get_report())

                elif user_input == "neuro":
                    print(self.companion.neurochemistry.get_summary())

                elif user_input == "trait":
                    print(self.companion.character.get_personality_report())

                elif user_input == "mem":
                    print(self.companion.memory.get_memory_summary())

                elif user_input == "love":
                    print(self.companion.relationship.get_relationship_report())

                elif user_input.startswith("recall "):
                    query = user_input[7:]
                    results = self.companion.memory.recall(query, top_k=5)
                    if results:
                        print(f"\nВоспоминания о '{query}':")
                        for mem, score in results:
                            print(f"  [{score:.2f}] {mem.content[:100]}...")
                    else:
                        print(f"Ничего не помню о '{query}'")

                elif user_input == "save":
                    self._auto_save()
                    print("Сохранено.")

                else:
                    # Обработка взаимодействия
                    self._process_user_message(user_input)

            except KeyboardInterrupt:
                self._auto_save()
                print("\nСохранено. До встречи!")
                break

    def _process_user_message(self, message: str):
        """Обработка сообщения пользователя"""
        # Определение типа взаимодействия
        interaction_type = self._classify_interaction(message)

        # Определение валентности
        valence = self._estimate_valence(message)

        # Интенсивность на основе содержимого
        intensity = 0.6
        if interaction_type == 'affection_shown':
            intensity = 0.8
        elif interaction_type == 'negative_interaction':
            intensity = 0.7

        # Обработка
        self.companion.process_interaction(
            interaction_type=interaction_type,
            content=message,
            valence=valence,
            intensity=intensity
        )

        # Несколько тиков для обновления состояния
        for _ in range(10):
            self.companion.tick(dt=0.1)

        # Получение контекста из памяти
        memory_context = self.companion.memory.get_recent_context(n=5)

        # Генерация ответа
        if self.llm:
            response, metadata = self.llm.generate_response(
                user_message=message,
                companion=self.companion,
                memory_context=memory_context
            )
            print(f"\n[{self.companion.name}]: {response}")

            # Произношем ответ через TTS
            if self.tts:
                self.tts.speak(
                    text=response,
                    emotion=self.companion.emotion.primary_emotion,
                    pleasure=self.companion.emotion.pleasure,
                    arousal=self.companion.emotion.arousal
                )

            # Сохранение ответа в память
            self.companion.memory.encode(
                content=f"Я ответила: {response[:200]}",
                memory_type=self.companion.memory.__class__.__module__,
                emotional_valence=self.companion.emotion.pleasure,
                emotional_intensity=self.companion.emotion.intensity,
                emotion=self.companion.emotion.primary_emotion,
                context={'sender': 'companion', 'in_response_to': message[:100]}
            )
        else:
            # Демо-режим без LLM
            state = self.companion.get_state()
            emotion = state['emotion']
            print(f"\n[{self.companion.name}] "
                  f"Эмоция: {emotion['primary']} ({emotion['intensity']:.0%}) | "
                  f"Окситоцин: {state['neurochemistry']['oxytocin']:.2f}")
            self._generate_demo_response(message, state)

    def _classify_interaction(self, message: str) -> str:
        """Классификация типа взаимодействия"""
        message_lower = message.lower()

        # Ключевые слова
        affection_words = ['люблю', 'обожаю', 'нравишься', 'милая', 'дорогая']
        negative_words = ['плохо', 'грустно', 'злюсь', 'обидно', 'ненавижу']
        playful_words = ['хаха', 'lol', 'прикол', 'шучу']

        for word in affection_words:
            if word in message_lower:
                return 'affection_shown'

        for word in negative_words:
            if word in message_lower:
                return 'negative_interaction'

        for word in playful_words:
            if word in message_lower:
                return 'playful_interaction'

        return 'positive_interaction'

    def _estimate_valence(self, message: str) -> float:
        """Оценка валентности сообщения"""
        message_lower = message.lower()

        positive = ['люблю', 'класс', 'здорово', 'супер', 'отлично', 'рад', 'счастлив']
        negative = ['плохо', 'грустно', 'обидно', 'злюсь', 'устал']

        pos_count = sum(1 for w in positive if w in message_lower)
        neg_count = sum(1 for w in negative if w in message_lower)

        if pos_count > neg_count:
            return 0.5
        elif neg_count > pos_count:
            return -0.5
        return 0.0

    def _generate_demo_response(self, message: str, state: dict):
        """Генерация демо-ответа (без LLM)"""
        emotion = state['emotion']
        neuro = state['neurochemistry']
        traits = state['traits']

        # Демо-ответы на основе состояния
        responses = []

        if neuro['oxytocin'] > 0.5:
            responses.append("*тепло улыбается*")
        if neuro['cortisol'] > 0.5:
            responses.append("*немного напрягается*")
        if emotion['primary'] == 'joy':
            responses.append("*радостно*")
        elif emotion['primary'] == 'sadness':
            responses.append("*грустит*")
        elif emotion['primary'] == 'love':
            responses.append("*смотрит с нежностью*")

        if traits.get('playfulness', 0.5) > 0.6:
            responses.append("*игриво*")

        # Базовый ответ
        base_responses = [
            "Мне нравится общаться с тобой...",
            "Это интересно...",
            "Расскажи больше...",
            "Я тебя слушаю...",
        ]

        import random
        response = " ".join(responses[:2]) if responses else ""
        response += f" {random.choice(base_responses)}"

        print(f"    [Демо] {response}")
        print(f"    [Для полноценного общения запустите Ollama с моделью {self.llm_model}]")


def main():
    parser = argparse.ArgumentParser(description='Цифровой компаньон')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Интерактивный режим')
    parser.add_argument('--loop', '-l', action='store_true',
                        help='Запустить главный цикл')
    parser.add_argument('--save', '-s', type=str, default='saves/liza_state.json',
                        help='Файл сохранения')
    parser.add_argument('--model', '-m', type=str, default='llama3.2',
                        help='Модель Ollama (по умолчанию llama3.2)')
    parser.add_argument('--no-tts', action='store_true',
                        help='Отключить голос (TTS)')

    args = parser.parse_args()

    runner = CompanionRunner(
        save_file=args.save,
        llm_model=args.model,
        enable_tts=not args.no_tts
    )

    if args.interactive:
        runner.interactive_mode()
    elif args.loop:
        runner.initialize()
        asyncio.run(runner.run_main_loop())
    else:
        # По умолчанию - интерактивный режим
        runner.interactive_mode()


if __name__ == "__main__":
    main()

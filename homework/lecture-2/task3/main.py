import sys
from app_config import GigaChatConfig
from gigachat_client import GigaChatClient
from tasks import run_function_calling_cycle


def main():
    print("\n" + "=" * 70)
    print("ФУНКЦИЯ FUNCTION CALLING - ПОЛУЧЕНИЕ ИНФОРМАЦИИ О ПОГОДЕ")
    print("=" * 70)

    try:
        config = GigaChatConfig.load()
        client = GigaChatClient(config)
    except ValueError as e:
        print(f"\n❌ ОШИБКА: {e}")
        sys.exit(1)

    print("📝 Введите вопросы о погоде (для выхода напишите 'exit' или 'выход'):\n")

    while True:
        try:
            user_input = input("❓ Ваш вопрос: ").strip()

            if user_input.lower() in ['exit', 'выход', 'quit', 'выйти']:
                print("\n" + "=" * 70)
                print("✓ СПАСИБО ЗА ИСПОЛЬЗОВАНИЕ!")
                print("=" * 70 + "\n")
                break

            if not user_input:
                print("⚠️  Пожалуйста, введите вопрос\n")
                continue

            try:
                run_function_calling_cycle(client, user_input)
            except Exception as e:
                print(f"\n❌ ОШИБКА при обработке: {type(e).__name__}: {e}\n")

        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("✓ ПРОГРАММА ЗАВЕРШЕНА")
            print("=" * 70 + "\n")
            break


if __name__ == "__main__":
    main()

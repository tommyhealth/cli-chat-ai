import sys
from app_config import GigaChatConfig
from gigachat_client import GigaChatClient
from tasks import parse_resume_task, format_resume_output

def main():

    with open("example_resume.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    # print("Resume Parser - Пример Structured Output с GigaChat")
    # print("=" * 70)
    print("\nЗагрузка резюме из example_resume.txt...")
    print(f"Длина текста резюме: {len(resume_text)} символов\n")

    try:
        config = GigaChatConfig.load()
        client = GigaChatClient(config)
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)

    print("Отправка запроса к GigaChat API...")
    try:
        resume = parse_resume_task(client, resume_text)
        print("✓ Резюме успешно спарсено!\n")
    except ValueError as e:
        print(f"ОШИБКА: Не удалось спарсить ответ: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: {type(e).__name__}: {e}")
        sys.exit(1)

    print("\nСПАРСЕННОЕ РЕЗЮМЕ:")
    print(format_resume_output(resume))

    print("\nРАВНЫЙ JSON (для проверки):")
    print(resume.model_dump_json(indent=2))

if __name__ == "__main__":
    main()

import urllib.error

from api import MODELS, chat_completion
from auth import load_auth_key
from commands import print_help, print_settings


def main():
    auth_key = load_auth_key()
    if not auth_key:
        print("Ошибка: укажите ключ авторизации одним из способов:")
        print("  1. Впишите в config.json (поле \"auth_key\")")
        print("  2. export GIGACHAT_AUTH_KEY='ваш_ключ'")
        return

    model_key = "lite"
    temperature = 0.7
    system_prompt = ""
    history = []

    print("GigaChat CLI — введите /help для списка команд")
    print_settings(model_key, temperature, system_prompt)

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/quit":
                print("До свидания!")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/settings":
                print_settings(model_key, temperature, system_prompt)

            elif cmd == "/model":
                if arg in MODELS:
                    model_key = arg
                    print("Модель: {} ({})".format(model_key, MODELS[model_key]))
                else:
                    print("Доступные модели: {}".format(", ".join(MODELS)))

            elif cmd == "/temp":
                try:
                    val = float(arg)
                    if 0 <= val <= 2:
                        temperature = val
                        print("Температура: {}".format(temperature))
                    else:
                        print("Температура должна быть от 0 до 2")
                except ValueError:
                    print("Укажите число от 0 до 2")

            elif cmd == "/system":
                if arg:
                    system_prompt = arg
                    print("System prompt: {}".format(system_prompt))
                else:
                    system_prompt = ""
                    print("System prompt сброшен")

            else:
                print("Неизвестная команда: {}. Введите /help".format(cmd))

            continue

        history.append({"role": "user", "content": user_input})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        try:
            reply = chat_completion(
                auth_key, messages, MODELS[model_key], temperature
            )
            history.append({"role": "assistant", "content": reply})
            print("\nGigaChat: {}".format(reply))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode(errors="replace")
            print("\nОшибка API ({}): {}".format(e.code, error_body))
        except Exception as e:
            print("\nОшибка: {}".format(e))


if __name__ == "__main__":
    main()
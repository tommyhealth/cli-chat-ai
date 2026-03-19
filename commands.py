from api import MODELS


def print_help():
    print(
        "\nКоманды:\n"
        "  /model lite|pro|max  — сменить модель\n"
        "  /temp <0-2>          — сменить температуру\n"
        "  /system <текст>      — задать system prompt\n"
        "  /settings            — текущие настройки\n"
        "  /help                — показать справку\n"
        "  /exit                — выход\n"
    )


def print_settings(model_key, temperature, system_prompt):
    print(
        "\n  Модель:       {} ({})\n"
        "  Температура:  {}\n"
        "  System:       {}\n".format(
            model_key, MODELS[model_key], temperature, system_prompt or "(не задан)"
        )
    )
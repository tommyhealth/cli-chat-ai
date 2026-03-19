# GigaChat CLI Chat

Терминальный чат с GigaChat API. Поддерживает историю диалога и смену настроек на лету.

## Запуск
python chat.py

## Настройка

Создай config.json с ключом авторизации:
{
  "auth_key": "ВАШ_КЛЮЧ"
}

## Команды

|Команда              |Описание                  |
|---------------------|--------------------------|
|`/model Lite|Pro|Max`|Сменить модель            |
|`/temp 0.0–2.0`      |Сменить temperature       |
|`/system <текст>`    |Задать system prompt      |
|`/settings`          |Показать текущие настройки|
|`/help`              |Список команд             |
|`/quit`              |Выйти                     |

## Структура файлов
chat.py      — запуск и главный цикл
auth.py      — авторизация и кэш токена
api.py       — запросы к GigaChat API
commands.py  — вывод /help и /settings
config.json  — ключ API (не коммитить!)

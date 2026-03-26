import json
from typing import Dict, Any, List
from gigachat_client import GigaChatClient
from models import WeatherData


# Определение функции get_weather в формате GigaChat API
WEATHER_FUNCTION = {
    "name": "get_weather",
    "description": "Получить информацию о погоде для города",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Название города"
            }
        },
        "required": ["city"]
    }
}


# Мок-данные с погодой для различных городов
MOCK_WEATHER_DATA = {
    "москва": {"city": "Москва", "temperature": 5.2, "condition": "облачно", "wind_speed": 3.5, "humidity": 72},
    "санкт-петербург": {"city": "Санкт-Петербург", "temperature": 3.8, "condition": "дождь", "wind_speed": 5.2, "humidity": 85},
    "казань": {"city": "Казань", "temperature": 2.1, "condition": "ясно", "wind_speed": 2.0, "humidity": 68},
    "екатеринбург": {"city": "Екатеринбург", "temperature": -1.5, "condition": "снег", "wind_speed": 4.8, "humidity": 80},
    "новосибирск": {"city": "Новосибирск", "temperature": -3.2, "condition": "ясно", "wind_speed": 1.5, "humidity": 65},
}


def get_weather_mock(city: str) -> Dict[str, Any]:
    """
    Имитация функции получения погоды.
    В реальном приложении это был бы запрос к API погоды.
    """
    city_lower = city.lower()
    if city_lower in MOCK_WEATHER_DATA:
        return MOCK_WEATHER_DATA[city_lower]
    else:
        return {
            "city": city,
            "temperature": 20.0,
            "condition": "умеренно",
            "wind_speed": 2.5,
            "humidity": 70
        }


def execute_function(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Выполнить функцию и вернуть результат как JSON строку.
    """
    if function_name == "get_weather":
        city = function_args.get("city", "")
        weather_data = get_weather_mock(city)
        weather_obj = WeatherData(**weather_data)
        return json.dumps(json.loads(weather_obj.model_dump_json()))
    else:
        return json.dumps({"error": f"Неизвестная функция: {function_name}"})


def run_function_calling_cycle(client: GigaChatClient, user_question: str) -> str:
    """
    Запустить полный цикл function calling:
    1. Отправить вопрос пользователя
    2. Если модель решит вызвать функцию - выполнить её
    3. Вернуть результат модели для получения финального ответа
    """
    system_prompt = """Ты ассистент, который помогает пользователям узнавать информацию о погоде.
У тебя есть доступ к функции get_weather(city) для получения информации о погоде в городе.
Когда пользователь спрашивает о погоде в каком-то городе, используй функцию get_weather.
Обращай внимание на названия городов в вопросе пользователя и вызывай функцию с правильным городом."""

    # Шаг 1: Отправить запрос с доступными функциями
    print("\n1️⃣ Отправка запроса к модели с доступными функциями...")
    response = client.call_with_functions(
        system_prompt=system_prompt,
        user_message=user_question,
        functions=[WEATHER_FUNCTION]
    )

    finish_reason = response["finish_reason"]
    message = response["message"]

    # Проверим, решила ли модель вызвать функцию
    if finish_reason == "function_call" and "function_call" in message:
        function_call = message["function_call"]
        function_name = function_call["name"]
        # Аргументы могут быть уже dict или строкой JSON
        if isinstance(function_call["arguments"], dict):
            function_args = function_call["arguments"]
        else:
            function_args = json.loads(function_call["arguments"])

        print(f"✓ Модель решила вызвать функцию: {function_name}")
        print(f"  Аргументы: {json.dumps(function_args, ensure_ascii=False)}\n")

        # Шаг 2: Выполнить функцию
        print(f"2️⃣ Выполнение функции {function_name}({function_args['city']})...")
        function_result = execute_function(function_name, function_args)
        print(f"✓ Результат функции получен\n")

        # Шаг 3: Отправить результат обратно модели для получения финального ответа
        print("3️⃣ Отправка результата модели для получения финального ответа...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": "", "function_call": function_call},
            {
                "role": "function",
                "name": function_name,
                "content": function_result
            }
        ]

        final_response = client.continue_conversation(
            system_prompt=system_prompt,
            messages=messages,
            functions=[WEATHER_FUNCTION]
        )

        final_message = final_response["message"]
        final_answer = final_message.get("content", "")

        print(f"✓ Финальный ответ получен\n")

        # Парсим и выводим погоду если возможно
        weather_data = WeatherData(**json.loads(function_result))
        print(f"\nОТВЕТ МОДЕЛИ:\n{final_answer}\n")

        return final_answer
    else:
        # Модель решила не вызывать функцию
        print(f"✓ Модель не решила вызывать функцию (finish_reason: {finish_reason})")
        answer = message.get("content", "")
        print(f"\nОТВЕТ МОДЕЛИ:\n{answer}\n")
        return answer

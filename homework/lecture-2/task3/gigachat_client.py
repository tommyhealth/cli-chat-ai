import json
import urllib.request
import uuid
from typing import Dict, Any, List, Optional
from app_config import GigaChatConfig
from auth import get_token, ssl_ctx
from models import FunctionCall


class GigaChatClient:
    def __init__(self, config: GigaChatConfig):
        self.config = config
        self.token = get_token(config.api_key)

    def _build_request(self,
                      system_prompt: str,
                      messages: List[Dict[str, str]],
                      functions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Построить запрос к API с поддержкой функций"""
        request = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        if functions:
            request["functions"] = functions
            request["function_call"] = "auto"

        return request

    def call_with_functions(self,
                          system_prompt: str,
                          user_message: str,
                          functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Отправить запрос к GigaChat с функциями и получить ответ.
        Может вернуть либо текстовый ответ, либо вызов функции.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        payload = self._build_request(system_prompt, messages, functions)

        try:
            payload_json = json.dumps(payload).encode()

            req = urllib.request.Request(
                f"{self.config.api_url}/chat/completions",
                data=payload_json,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4()),
                    "Authorization": f"Bearer {self.token}",
                },
            )

            with urllib.request.urlopen(req, context=ssl_ctx) as resp:
                response_data = json.loads(resp.read())

            message = response_data["choices"][0]["message"]

            return {
                "finish_reason": response_data["choices"][0].get("finish_reason", "stop"),
                "message": message
            }

        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Не удалось подключиться к API GigaChat ({self.config.api_url})\n"
                f"Проверьте:\n"
                f"  1. Интернет соединение\n"
                f"  2. Доступность сервиса GigaChat\n"
                f"  3. Правильность API URL: {self.config.api_url}\n"
                f"  4. Настройки прокси/брандмауэра\n"
                f"Деталь ошибки: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Ошибка API запроса: {e}")

    def continue_conversation(self,
                             system_prompt: str,
                             messages: List[Dict[str, str]],
                             functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Продолжить диалог с дополнительными сообщениями.
        Используется после выполнения функции для получения финального ответа.
        """
        payload = self._build_request(system_prompt, messages, functions)

        try:
            payload_json = json.dumps(payload).encode()

            req = urllib.request.Request(
                f"{self.config.api_url}/chat/completions",
                data=payload_json,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4()),
                    "Authorization": f"Bearer {self.token}",
                },
            )

            with urllib.request.urlopen(req, context=ssl_ctx) as resp:
                response_data = json.loads(resp.read())

            message = response_data["choices"][0]["message"]

            return {
                "finish_reason": response_data["choices"][0].get("finish_reason", "stop"),
                "message": message
            }

        except Exception as e:
            raise RuntimeError(f"Ошибка продолжения диалога: {e}")

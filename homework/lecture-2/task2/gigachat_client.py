import json
import urllib.request
import uuid
from typing import Dict, Any
from app_config import GigaChatConfig
from auth import get_token, ssl_ctx


class GigaChatClient:
    def __init__(self, config: GigaChatConfig):
        self.config = config
        # Получаем токен используя auth_key
        self.token = get_token(config.api_key)

    def _build_request(self, system_prompt: str, user_message: str) -> Dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        system_prompt = """Ты эксперт по анализу резюме.
Проанализируй предоставленное резюме и извлеки информацию в JSON объект.

Верни ТОЛЬКО валидный JSON с такой структурой:
{
    "name": "Полное имя",
    "email": "email@example.com или null",
    "phone": "номер телефона или null",
    "skills": ["навык1", "навык2"],
    "experience": [
        {
            "company": "Название компании",
            "position": "Должность",
            "years": 3,
            "description": "Краткое описание"
        }
    ],
    "education": [
        {
            "university": "Название университета",
            "degree": "Степень",
            "graduation_year": 2020
        }
    ]
}

Убедись, что все поля правильно отформатированы и типы совпадают со схемой."""

        payload = self._build_request(system_prompt, f"Спарси это резюме:\n\n{resume_text}")

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

            content = response_data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed

        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON ответа: {e}")
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

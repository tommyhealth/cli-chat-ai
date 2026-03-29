import json
import os
from dataclasses import dataclass

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

@dataclass
class GigaChatConfig:
    api_key: str
    api_url: str = "https://gigachat.devices.sberbank.ru/api/v1"
    model: str = "GigaChat"
    temperature: float = 0.7
    max_tokens: int = 2048

    @classmethod
    def load(cls):
        """Загрузить конфигурацию из config.json"""
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            api_key = config.get("auth_key", "")
            if api_key and not api_key.startswith("ВСТАВЬТЕ"):
                return cls(api_key=api_key)
        raise ValueError(f"auth_key не найден в {CONFIG_PATH}")

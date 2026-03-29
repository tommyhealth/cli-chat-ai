from pydantic import BaseModel, Field
from typing import Optional


class WeatherData(BaseModel):
    city: str = Field(..., min_length=1, description="Название города")
    temperature: float = Field(..., description="Температура в градусах Цельсия")
    condition: str = Field(..., min_length=1, description="Состояние погоды (ясно, облачно, дождь и т.д.)")
    wind_speed: float = Field(..., ge=0, description="Скорость ветра в км/ч")
    humidity: int = Field(..., ge=0, le=100, description="Влажность в процентах")


class FunctionCall(BaseModel):
    name: str = Field(..., description="Имя функции")
    arguments: dict = Field(..., description="Аргументы функции")


class ConversationMessage(BaseModel):
    role: str = Field(..., description="Роль (user, assistant, system)")
    content: str = Field(..., description="Содержание сообщения")

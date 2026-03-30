import json
import urllib.request

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("weather-server")


class WeatherData(BaseModel):
    city: str = Field(..., min_length=1, description="Название города")
    temperature: float = Field(..., description="Температура в градусах Цельсия")
    condition: str = Field(..., min_length=1, description="Состояние погоды (ясно, облачно, дождь и т.д.)")
    wind_speed: float = Field(..., ge=0, description="Скорость ветра в км/ч")
    humidity: int = Field(..., ge=0, le=100, description="Влажность в процентах")


def fetch_weather(city: str) -> WeatherData:
    """Получить реальную погоду через wttr.in API."""
    url = f"https://wttr.in/{urllib.request.quote(city)}?format=j1"
    req = urllib.request.Request(url, headers={"User-Agent": "weather-mcp-server"})

    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    current = data["current_condition"][0]

    return WeatherData(
        city=city,
        temperature=float(current["temp_C"]),
        condition=current["lang_ru"][0]["value"] if current.get("lang_ru") else current["weatherDesc"][0]["value"],
        wind_speed=float(current["windspeedKmph"]),
        humidity=int(current["humidity"]),
    )


@mcp.tool()
def get_weather(city: str) -> str:
    """Получить текущую погоду для указанного города.
    Возвращает реальные данные: температуру, состояние погоды, скорость ветра и влажность.

    Args:
        city: Название города (например, "Москва", "Казань", "London")
    """
    weather = fetch_weather(city)
    return weather.model_dump_json(indent=2)


if __name__ == "__main__":
    mcp.run()

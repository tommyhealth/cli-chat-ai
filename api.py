import json
import urllib.request

from auth import get_token, ssl_ctx

API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

MODELS = {
    "lite": "GigaChat",
    "pro": "GigaChat-Pro",
    "max": "GigaChat-Max",
}


def chat_completion(auth_key, messages, model, temperature):
    token = get_token(auth_key)

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer {}".format(token),
        },
    )
    with urllib.request.urlopen(req, context=ssl_ctx) as resp:
        data = json.loads(resp.read())

    return data["choices"][0]["message"]["content"]
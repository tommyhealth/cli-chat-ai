import json
import ssl
import time
import urllib.request
import uuid

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
SCOPE = "GIGACHAT_API_PERS"

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

_token = None
_token_expires = 0.0


def get_token(auth_key):
    """Получить токен для API запроса используя auth_key"""
    global _token, _token_expires

    if _token and time.time() * 1000 < _token_expires:
        return _token

    body = "scope={}".format(SCOPE).encode()
    req = urllib.request.Request(
        AUTH_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": "Basic {}".format(auth_key),
        },
    )
    with urllib.request.urlopen(req, context=ssl_ctx) as resp:
        data = json.loads(resp.read())

    _token = data["access_token"]
    _token_expires = data["expires_at"]
    return _token

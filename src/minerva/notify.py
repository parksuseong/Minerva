"""텔레그램 알림. 설정은 환경변수 또는 저장소 밖 `~/.minerva/telegram.env`에서 읽는다."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

ENV_FILE = Path(os.environ.get("MINERVA_TELEGRAM_ENV", "~/.minerva/telegram.env")).expanduser()
API = "https://api.telegram.org/bot{token}/sendMessage"


def load_config() -> tuple[str, str] | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if (not token or not chat) and ENV_FILE.exists():
        values = dict(
            line.split("=", 1) for line in ENV_FILE.read_text().splitlines() if "=" in line
        )
        token = token or values.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat = chat or values.get("TELEGRAM_CHAT_ID", "").strip()
    return (token, chat) if token and chat else None


def send(text: str, timeout: float = 10) -> bool:
    """메시지를 보낸다. 설정이 없거나 실패하면 False (서버 기동을 막지 않는다)."""
    cfg = load_config()
    if cfg is None:
        log.info("telegram not configured; skip")
        return False
    token, chat = cfg
    body = json.dumps({"chat_id": chat, "text": text}).encode()
    req = urllib.request.Request(
        API.format(token=token), data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp).get("ok", False)
    except Exception as exc:  # noqa: BLE001 — 알림 실패가 서버를 죽이면 안 된다
        # 토큰이 URL에 들어 있으므로 예외 메시지 전체는 남기지 않는다
        log.warning("telegram send failed: %s", type(exc).__name__)
        return False

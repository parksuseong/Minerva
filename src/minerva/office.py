"""오피스 화면: Claude와 서브에이전트가 일하는 모습을 보여준다.

이벤트는 `scripts/office_hook.py`가 쓰는 JSONL 로그에서 읽는다.
토큰 파일(`~/.minerva/office/token`)이나 MINERVA_OFFICE_TOKEN이 있으면 토큰이 맞아야 볼 수 있다.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter()


def log_path() -> Path:
    return Path(os.environ.get("MINERVA_OFFICE_LOG", "~/.minerva/office/events.jsonl")).expanduser()


def expected_token() -> str | None:
    env = os.environ.get("MINERVA_OFFICE_TOKEN")
    if env:
        return env
    f = log_path().parent / "token"
    return f.read_text().strip() if f.exists() else None


def check_token(token: str) -> None:
    want = expected_token()
    if want and not hmac.compare_digest(token, want):
        raise HTTPException(status_code=403, detail="invalid token")


def read_events(after: int, limit: int = 500) -> tuple[list[dict], int]:
    """줄 번호 기준 `after` 이후 이벤트와 마지막 줄 번호를 돌려준다."""
    path = log_path()
    if not path.exists():
        return [], 0
    lines = path.read_text().splitlines()
    total = len(lines)
    if after > total:  # 로그가 잘려 줄 번호가 줄어든 경우 처음부터 다시
        after = 0
    start = max(after, total - limit)
    events = []
    for i in range(start, total):
        try:
            rec = json.loads(lines[i])
        except json.JSONDecodeError:
            continue
        rec["seq"] = i + 1
        events.append(rec)
    return events, total


@router.get("/office", include_in_schema=False)
def office_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "office.html")


@router.get("/api/office/events")
def office_events(after: int = Query(0, ge=0), token: str = "") -> dict:
    check_token(token)
    events, last = read_events(after)
    return {"events": events, "last": last}

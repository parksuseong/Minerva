"""오피스 패널 API: 메인 화면 아래쪽에 Claude와 서브에이전트가 일하는 모습을 보여준다.

이벤트는 `scripts/office_hook.py`가 쓰는 JSONL 로그에서 읽는다.
기본은 공개다. MINERVA_OFFICE_TOKEN 환경변수를 설정했을 때만 토큰을 요구한다.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def log_path() -> Path:
    return Path(os.environ.get("MINERVA_OFFICE_LOG", "~/.minerva/office/events.jsonl")).expanduser()


def expected_token() -> str | None:
    return os.environ.get("MINERVA_OFFICE_TOKEN") or None


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


def daily_path() -> Path:
    return Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()


@router.get("/api/daily")
def daily_analysis() -> dict:
    path = daily_path()
    if not path.exists():
        return {"status": "pending", "message": "아직 오늘 분석이 없어요 (09:10·16:30 자동 실행)"}
    return json.loads(path.read_text())


@router.get("/api/office/events")
def office_events(after: int = Query(0, ge=0), token: str = "") -> dict:
    check_token(token)
    events, last = read_events(after)
    return {"events": events, "last": last}

"""웹 서버가 보여주기만 하는 읽기 전용 API.

웹 서버는 계산도 주문도 하지 않는다. 각 데이터는 별도 프로세스가 파일로 쓴다.
- 종목 분석: 배치 `minerva-daily.timer` → `~/.minerva/daily/latest.json`
- 계좌: `python -m minerva.kis` (`minerva-account.timer`) → `~/.minerva/account/latest.json`
  증권사 키는 그 프로세스만 읽는다. 웹 서버에는 키가 없다.
  실계좌 정보라 열람 토큰(`X-Minerva-Token`)이 있어야 보여준다.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException

router = APIRouter()


def daily_path() -> Path:
    return Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()


def account_path() -> Path:
    default = "~/.minerva/account/latest.json"
    return Path(os.environ.get("MINERVA_ACCOUNT_OUT", default)).expanduser()


def account_view_token() -> str | None:
    """MINERVA_ACCOUNT_TOKEN 또는 `~/.minerva/account/view.token`. 둘 다 없으면 잠그지 않는다."""
    if os.environ.get("MINERVA_ACCOUNT_TOKEN"):
        return os.environ["MINERVA_ACCOUNT_TOKEN"]
    default = "~/.minerva/account/view.token"
    path = Path(os.environ.get("MINERVA_ACCOUNT_TOKEN_FILE", default)).expanduser()
    if not path.exists():
        return None
    return path.read_text().strip() or None


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@router.get("/api/daily")
def daily_analysis() -> dict:
    data = _read(daily_path())
    if data is None:
        return {"status": "pending", "message": "아직 오늘 분석이 없어요 (09:10·16:30 자동 실행)"}
    return data


@router.get("/api/signals")
def signals() -> dict:
    """자동매매용: 오늘 낼 주문만. 같은 분석 결과에서 뽑는다."""
    data = _read(daily_path())
    if data is None:
        return {"status": "pending", "actions": []}
    return {
        "status": "ok",
        "generated_at": data.get("generated_at"),
        "regime": data.get("vix", {}).get("regime"),
        "size_limit": data.get("vix", {}).get("size_limit"),
        "actions": data.get("actions", []),
    }


@router.get("/api/account")
def account(x_minerva_token: str = Header("")) -> dict:
    want = account_view_token()
    if want and not hmac.compare_digest(x_minerva_token, want):
        raise HTTPException(status_code=403, detail="invalid token")
    data = _read(account_path())
    if data is None:
        return {
            "status": "not_connected",
            "message": "증권사 계좌가 아직 연결되지 않았어요 (한국투자증권 스냅숏 대기 중)",
        }
    return data

"""웹 서버가 보여주기만 하는 읽기 전용 API.

웹 서버는 계산도 주문도 하지 않는다. 각 데이터는 별도 프로세스가 파일로 쓴다.
- 종목 분석: 배치 `minerva-daily.timer` → `~/.minerva/daily/latest.json`
- 계좌: (예정) 증권사 연동 프로세스 → `~/.minerva/account/latest.json`
  증권사 키는 그 프로세스만 읽는다. 웹 서버에는 키가 없다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()


def daily_path() -> Path:
    return Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()


def account_path() -> Path:
    default = "~/.minerva/account/latest.json"
    return Path(os.environ.get("MINERVA_ACCOUNT_OUT", default)).expanduser()


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
def account() -> dict:
    data = _read(account_path())
    if data is None:
        return {
            "status": "not_connected",
            "message": "증권사 계좌가 아직 연결되지 않았어요 (한국투자증권 API 연동 예정)",
        }
    return data

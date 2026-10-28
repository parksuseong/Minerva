"""한국투자증권 Open API 계좌 스냅숏 (조회 전용).

키는 저장소 밖 `~/.minerva/kis.env`에서 이 프로세스만 읽는다. 웹 서버는 결과 파일
`~/.minerva/account/latest.json`만 읽는다. 주문 API는 여기서 부르지 않는다.

실행: `python -m minerva.kis` (minerva-account.timer가 장중에 주기적으로 실행)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
BASE = "https://openapi.koreainvestment.com:9443"
ENV_FILE = Path(os.environ.get("MINERVA_KIS_ENV", "~/.minerva/kis.env")).expanduser()
STATE_DIR = Path(os.environ.get("MINERVA_ACCOUNT_DIR", "~/.minerva/account")).expanduser()
TRADE_DAYS = 30
MAX_PAGES = 10
# 만료(EGW00123)·유효하지 않은 토큰(EGW00121) 응답 코드
TOKEN_ERRORS = ("EGW00123", "EGW00121")
# 초당 거래건수 초과. 신규 계좌는 한도가 낮아 호출 사이를 띄우고 잠깐 쉬었다 다시 한다.
RATE_ERRORS = ("EGW00201", "EGW00215")
CALL_GAP = 0.6
RETRIES = 3


class KisError(RuntimeError):
    pass


def load_config(path: Path = ENV_FILE) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    cano, _, prd = values["KIS_ACCOUNT"].partition("-")
    return {
        "cano": cano,
        "prd": prd or "01",
        "app_key": values["KIS_APP_KEY"],
        "app_secret": values["KIS_APP_SECRET"],
    }


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _request(req: urllib.request.Request) -> tuple[dict, dict]:
    for attempt in range(RETRIES + 1):
        time.sleep(CALL_GAP)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r), dict(r.headers)
        except urllib.error.HTTPError as e:
            # 응답 본문엔 키가 없지만 길이는 줄인다
            body = e.read()[:200].decode("utf-8", "replace")
            if attempt < RETRIES and any(code in body for code in RATE_ERRORS):
                time.sleep(1 + attempt)
                continue
            raise KisError(f"HTTP {e.code}: {body}") from e
    raise AssertionError("unreachable")


def access_token(cfg: dict, now: datetime | None = None) -> str:
    """토큰은 하루짜리이고 발급 횟수 제한이 있어 파일에 캐시해 재사용한다."""
    now = now or datetime.now(KST)
    cache = STATE_DIR / "token.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        expires = datetime.strptime(data["access_token_token_expired"], "%Y-%m-%d %H:%M:%S")
        if expires.replace(tzinfo=KST) - now > timedelta(minutes=30):
            return data["access_token"]
    body = json.dumps(
        {
            "grant_type": "client_credentials",
            "appkey": cfg["app_key"],
            "appsecret": cfg["app_secret"],
        }
    ).encode()
    req = urllib.request.Request(
        BASE + "/oauth2/tokenP", data=body, headers={"content-type": "application/json"}
    )
    data, _ = _request(req)
    _write_private(cache, json.dumps(data))
    return data["access_token"]


def _get(
    cfg: dict, token: str, path: str, tr_id: str, params: dict, max_pages: int = MAX_PAGES
) -> list[dict]:
    """연속 조회(tr_cont)를 따라가며 응답 페이지를 모두 돌려준다."""
    pages, cont = [], ""
    for _ in range(max_pages):
        req = urllib.request.Request(
            BASE + path + "?" + urllib.parse.urlencode(params),
            headers={
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {token}",
                "appkey": cfg["app_key"],
                "appsecret": cfg["app_secret"],
                "tr_id": tr_id,
                "tr_cont": cont,
                "custtype": "P",
            },
        )
        data, headers = _request(req)
        if data.get("rt_cd") != "0":
            raise KisError(f"{tr_id} {data.get('msg_cd')}: {str(data.get('msg1', '')).strip()}")
        pages.append(data)
        if headers.get("tr_cont") not in ("F", "M"):
            break
        cont = "N"
        params = {
            **params,
            "CTX_AREA_FK100": data.get("ctx_area_fk100", ""),
            "CTX_AREA_NK100": data.get("ctx_area_nk100", ""),
        }
    return pages


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _int(v) -> int:
    return int(_num(v))


def parse_balance(pages: list[dict]) -> dict:
    positions = [
        {
            "name": r.get("prdt_name", "").strip(),
            "code": r.get("pdno", ""),
            "qty": _int(r.get("hldg_qty")),
            "avg": _num(r.get("pchs_avg_pric")),
            "price": _num(r.get("prpr")),
            "value": _num(r.get("evlu_amt")),
            "pnl": _num(r.get("evlu_pfls_amt")),
            "pnl_pct": _num(r.get("evlu_pfls_rt")),
        }
        for page in pages
        for r in page.get("output1") or []
        if _int(r.get("hldg_qty")) > 0
    ]
    s = (pages[0].get("output2") or [{}])[0] if pages else {}
    return {
        "cash": _num(s.get("dnca_tot_amt")),
        "cash_d2": _num(s.get("prvs_rcdl_excc_amt")),
        "stock_value": _num(s.get("scts_evlu_amt")),
        "equity": _num(s.get("tot_evlu_amt")),
        "pnl": _num(s.get("evlu_pfls_smtl_amt")),
        "positions": positions,
    }


def parse_orders(pages: list[dict]) -> tuple[list[dict], list[dict]]:
    """주문 체결 내역 → (주문 목록, 체결 목록). 최근 것이 먼저."""
    orders, trades = [], []
    for page in pages:
        for r in page.get("output1") or []:
            d, t = r.get("ord_dt", ""), r.get("ord_tmd", "")
            when = f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}" if d and t else d
            qty, filled = _int(r.get("ord_qty")), _int(r.get("tot_ccld_qty"))
            if r.get("cncl_yn") == "Y":
                status = "취소"
            elif filled >= qty > 0:
                status = "체결"
            elif filled:
                status = f"부분체결 {filled}"
            elif _int(r.get("rjct_qty")):
                status = "거부"
            else:
                status = "미체결"
            row = {
                "time": when,
                "name": r.get("prdt_name", "").strip(),
                "code": r.get("pdno", ""),
                "side": r.get("sll_buy_dvsn_cd_name", "").strip(),
                "qty": qty,
                # 시장가 주문은 주문 단가가 0이라 체결 평균가를 보여준다
                "price": _num(r.get("ord_unpr")) or _num(r.get("avg_prvs")),
                "status": status,
            }
            orders.append(row)
            if filled:
                trades.append({**row, "qty": filled, "price": _num(r.get("avg_prvs"))})
    orders.sort(key=lambda o: o["time"], reverse=True)
    trades.sort(key=lambda o: o["time"], reverse=True)
    return orders, trades


def snapshot(cfg: dict, now: datetime | None = None) -> dict:
    """캐시 토큰이 서버에서 먼저 만료·폐기됐으면 한 번 새로 발급받아 다시 조회한다."""
    try:
        return _snapshot(cfg, now)
    except KisError as e:
        if not any(code in str(e) for code in TOKEN_ERRORS):
            raise
        log.info("token rejected (%s); reissuing", e)
        (STATE_DIR / "token.json").unlink(missing_ok=True)
        return _snapshot(cfg, now)


def _snapshot(cfg: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(KST)
    token = access_token(cfg, now)
    acct = {"CANO": cfg["cano"], "ACNT_PRDT_CD": cfg["prd"]}
    balance = _get(
        cfg,
        token,
        "/uapi/domestic-stock/v1/trading/inquire-balance",
        "TTTC8434R",
        {
            **acct,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )
    history = _get(
        cfg,
        token,
        "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
        "TTTC0081R",
        {
            **acct,
            "INQR_STRT_DT": (now - timedelta(days=TRADE_DAYS)).strftime("%Y%m%d"),
            "INQR_END_DT": now.strftime("%Y%m%d"),
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        },
    )
    orders, trades = parse_orders(history)
    today = now.strftime("%Y-%m-%d")
    return {
        "status": "ok",
        "broker": "한국투자증권",
        "market": "국내주식",
        # 계좌번호는 화면에 뒤 2자리만 보인다
        "account": f"****{cfg['cano'][-2:]}-{cfg['prd']}",
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        **parse_balance(balance),
        "orders": [o for o in orders if o["time"].startswith(today)],
        "trades": trades,
        "trade_days": TRADE_DAYS,
    }


def run(out: Path) -> dict:
    """스냅숏을 쓴다. 실패하면 직전 스냅숏을 남기고 오류만 덧붙인다."""
    now = datetime.now(KST)
    try:
        data = snapshot(load_config(), now)
    except (KisError, OSError, KeyError, ValueError) as e:
        log.error("account snapshot failed: %s", e)
        data = json.loads(out.read_text()) if out.exists() else {"status": "error"}
        data["error"] = str(e)
        data["error_at"] = now.strftime("%Y-%m-%d %H:%M")
        _write_private(out, json.dumps(data, ensure_ascii=False))
        raise
    _write_private(out, json.dumps(data, ensure_ascii=False))
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="한국투자증권 계좌 스냅숏 (조회 전용)")
    parser.add_argument("--out", type=Path, default=STATE_DIR / "latest.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        data = run(args.out)
    except (KisError, OSError, KeyError, ValueError):
        return 1
    print(
        f"ok {data['updated_at']} 보유 {len(data['positions'])}종목 "
        f"주문 {len(data['orders'])} 체결 {len(data['trades'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""고서 탭: 룰북 후보 전략의 가상매매(실제 주문 없음)와 백테스트 요약.

`python -m minerva.gose run` (타이머 `minerva-gose.timer`, 화~토 07:20 KST)
1. 종목(`universe.json`)과 ^VIX 일봉을 야후에서 받는다. 종목별 캐시 `bars/<티커>.json`,
   다음부터는 최근 3주만 받아 잇는다. 겹치는 날 종가가 0.5% 넘게 다르면(분할 조정) 전부 다시.
2. 끝나지 않은 마지막 봉은 버린다: 야후가 주는 그 시장의 오늘 정규장(`currentTradingPeriod`)이
   아직 안 끝났고 마지막 봉 날짜가 그 장 날짜 이후면 뺀다. 07:20 KST에는 미국 장은 끝났고
   한국·홍콩 장은 열기 전이라 보통 버릴 봉이 없다(손으로 돌릴 때를 위한 안전장치).
3. 전략마다 `gose_rules.simulate`를 전체 기록으로 다시 돌리고(같은 입력이면 같은 결과),
   신호일이 가상매매 시작일(START, 첫 실행일, `state.json`, 다시 쓰지 않음) 이후인 거래만
   가상 계좌에 넣는다. 계좌: 1만 달러, 동시 10종목, 새 포지션은 전날 마감 평가액의 10%,
   수수료 편도 0.05%, 체결은 다음 날 시가(손절은 그 가격). 같은 날 진입은 신호일·종목 이름 순,
   자리가 차면 건너뛴다. → `paper.json`
   규칙서(그대로 매매용): `gose_specs.json`, `gose_spec_3c0s10.md`, `gose_spec_4.md`.

   "SCR 3단 + 물타기" 계좌 `avg-SOXL`·`avg-SOXX`(`gose_rules_avg`, 규칙서 `gose_spec.md`)는
   이 공용 규칙을 쓰지 않는다: 한 종목만, 사이클 시작 자산/3 단위, 추가 2번, 평단×1.10 익절.

   2차(그림 판독 정의) 계좌 `3v2-loose`·`11v2a`는 `gose_rules_v2`로 같은 방식. `11v2a`는 UVXY
   일봉(신호원, 매매 안 함)의 신호로 SOXL·SOXX·QQQ·SPY만 산다.

`python -m minerva.gose backtest <results.json> [--round2 <results2.json>]` 백테스트 결과에서
요약 줄만 `backtest.json`으로(2차 변형은 `round: 2`로 뒤에 붙는다).
웹(`/api/gose`)은 이 폴더 파일과 저장소 파일만 읽는다. 주문·키 없음.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from minerva import gose_rules as R
from minerva import gose_rules_avg as A
from minerva import gose_rules_v2 as V

KST = timezone(timedelta(hours=9))
HERE = Path(__file__).parent
UNIVERSE_SEED = HERE / "gose_universe.json"
RULEBOOK = HERE / "gose_rulebook.json"
AVG_SPEC = HERE / "gose_spec.md"
AVG_INFO = HERE / "gose_avg.json"
# 규칙서 목록: 전략마다 Markdown 전문 + 구조화 값, 나머지 계좌는 한 줄 정의(참고, 미검증)
SPECS = HERE / "gose_specs.json"

CAPITAL = 10_000.0
MAX_POS = 10
SIZE_FRAC = 0.10
FEE = 0.0005
MIN_BARS = 60
HISTORY_START = 1262304000  # 2010-01-01, 백테스트와 같은 워밍업
REFETCH_DAYS = 21
ADJUST_TOL = 0.005
SIGNAL_FRESH_DAYS = 5
CLOSED_KEEP = 60

# (계좌 키, 전략 키, 손절). 원문 청산(손절 없음·시간 청산) 16개 + 3-c0 10% 손절 하나
# + 2차(그림 판독 정의, `gose_rules_v2`) 두 개(원문 청산).
ACCOUNTS_V2 = [("3v2-loose", "3v2-loose", None), ("11v2a", "11v2a", None)]
ACCOUNTS = [(sid, sid, None) for sid in R.STRATEGIES] + [("3-c0/S10", "3-c0", 0.10)] + ACCOUNTS_V2
V2_LABEL = "2차(그림 판독 정의)"
VOL_TICKER = "UVXY"  # 2차 변동성 신호원(UVIX 대리). 매매하지 않는다
# "SCR 3단 + 물타기"(5차): 계좌 하나에 한 종목, 자기 크기 규칙. (계좌 키, 종목)
AVG_ACCOUNTS = [("avg-SOXL", "SOXL"), ("avg-SOXX", "SOXX")]
AVG_NAME = "SCR 3단 + 물타기 (익절 10%)"
AVG_REF_START = "2019-01-01"  # 참고: 백테스트 시험 구간 시작부터 돌렸다면 지금 상태


def _rules(sid: str):
    """전략 키 → 규칙 모듈(1차 R 또는 2차 V). 두 모듈의 키는 겹치지 않는다."""
    return V if sid in V.STRATEGIES else R


def _allowed(sid: str, sym: str) -> bool:
    """변동성 신호 전략은 정해진 종목만 산다(나머지 전략은 전 종목)."""
    only = V.TICKERS.get(sid) if sid in V.STRATEGIES else None
    return only is None or sym in only


def gose_dir() -> Path:
    return Path(os.environ.get("MINERVA_GOSE_DIR", "~/.minerva/gose")).expanduser()


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    """없거나 깨진 파일은 None."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- 종목·상태


def load_universe() -> dict:
    """`universe.json`. 없으면 저장소 기본 목록을 복사해 쓴다."""
    path = gose_dir() / "universe.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(UNIVERSE_SEED, path)
    return json.loads(path.read_text())


def load_state(today: str) -> dict:
    """가상매매 시작일. 처음 한 번만 쓰고 다시 쓰지 않는다."""
    path = gose_dir() / "state.json"
    st = read_json(path)
    if st and st.get("start"):
        return st
    if path.exists():
        raise SystemExit(f"{path}가 깨졌어요. 시작일을 지키려고 덮어쓰지 않아요.")
    st = {"start": today, "created_at": datetime.now(KST).isoformat(timespec="seconds")}
    _write(path, st)
    return st


# ---------------------------------------------------------------- 일봉


def _safe(ticker: str) -> str:
    return ticker.replace("^", "_").replace("/", "_")


def parse_chart(raw: dict, now_ts: float) -> list[list]:
    """야후 chart 응답 → [[날짜, 시가, 고가, 저가, 종가], ...] (마감된 봉만).

    날짜는 그 시장 현지 날짜(gmtoffset). 값이 빠지거나 0 이하인 봉은 버리고, 같은 날짜는 뒤 것.
    """
    r = raw["chart"]["result"][0]
    meta = r.get("meta") or {}
    off = meta.get("gmtoffset", 0) or 0
    q = r["indicators"]["quote"][0]
    out: list[list] = []
    for k, ts in enumerate(r.get("timestamp") or []):
        o, h, lo, c = q["open"][k], q["high"][k], q["low"][k], q["close"][k]
        if None in (o, h, lo, c) or min(o, h, lo, c) <= 0:
            continue
        d = datetime.fromtimestamp(ts + off, tz=timezone.utc).strftime("%Y-%m-%d")
        bar = [d, float(o), float(max(h, o, c)), float(min(lo, o, c)), float(c)]
        if out and out[-1][0] == d:
            out[-1] = bar
        else:
            out.append(bar)
    reg = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    if out and reg.get("start") and reg.get("end") and now_ts < reg["end"]:
        roff = reg.get("gmtoffset", off) or 0
        session = datetime.fromtimestamp(reg["start"] + roff, tz=timezone.utc).strftime("%Y-%m-%d")
        while out and out[-1][0] >= session:
            out.pop()
    return out


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_chart(ticker: str, period1: int, period2: int) -> dict:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
        f"?period1={period1}&period2={period2}&interval=1d&events=split"
    )
    return _get(url)


def load_bars(ticker: str) -> list[list] | None:
    data = read_json(gose_dir() / "bars" / f"{_safe(ticker)}.json")
    return data.get("bars") if data else None


def update_bars(ticker: str, now_ts: float) -> tuple[list[list], str]:
    """캐시를 이어 붙여 저장하고 (봉, 방식)을 돌려준다. 방식: full / incremental."""
    path = gose_dir() / "bars" / f"{_safe(ticker)}.json"
    old = load_bars(ticker) or []
    p2 = int(now_ts) + 86400
    mode = "full"
    bars: list[list] = []
    if old:
        last = datetime.strptime(old[-1][0], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        p1 = int((last - timedelta(days=REFETCH_DAYS)).timestamp())
        new = parse_chart(fetch_chart(ticker, p1, p2), now_ts)
        old_close = {b[0]: b[4] for b in old}
        same = [d for d in (b[0] for b in new) if d in old_close]
        moved = any(abs(b[4] / old_close[b[0]] - 1) > ADJUST_TOL for b in new if b[0] in old_close)
        if new and same and not moved:
            first = new[0][0]
            bars = [b for b in old if b[0] < first] + new
            mode = "incremental"
        elif not new:
            bars, mode = old, "incremental"
    if mode == "full":
        bars = parse_chart(fetch_chart(ticker, HISTORY_START, p2), now_ts)
    _write(path, {"ticker": ticker, "fetched_at": int(now_ts), "bars": bars})
    return bars, mode


def refresh_all(tickers: list[str], now_ts: float, pause: float = 0.4) -> list[dict]:
    """모든 종목 캐시 갱신. 실패는 목록으로 돌려주고 옛 캐시는 그대로 둔다."""
    failures = []
    for t in tickers:
        try:
            update_bars(t, now_ts)
        except Exception as e:  # noqa: BLE001 - 한 종목 실패로 전체를 멈추지 않는다
            failures.append({"ticker": t, "error": f"{type(e).__name__}: {e}"[:200]})
        time.sleep(pause)
    return failures


# ---------------------------------------------------------------- 가상 계좌


def _trade_row(sym: str, t: dict, stop: bool) -> dict:
    return {
        "sym": sym,
        "side": t["side"],
        "signal_date": t["signal_date"],
        "fills": [[f[0], f[1], f[2], f[3]] for f in t["fills"]],
        "exit_date": t.get("exit_date"),
        "exit_px": t.get("exit_px"),
        "reason_in": t["reason_in"],
        "reason_out": t.get("reason_out"),
        "stop_exit": bool(stop and str(t.get("reason_out", "")).startswith("손절")),
    }


def collect(
    sym: str,
    bars: list,
    vix: list | None,
    start: str,
    today: str,
    vol: list | None = None,
    vol_hits: dict | None = None,
) -> dict:
    """한 종목에서 계좌별 후보 거래(신호일 ≥ start)와 오늘 신호를 모은다.

    vol·vol_hits: 2차 변동성 신호원(UVXY) 일봉과 미리 계산한 신호(`V.compute_vol_hits`).
    """
    ctx = R.prepare([tuple(b) for b in bars], vix)
    tb = ctx["bars"]
    ctx2 = None
    fresh = (
        datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(tb[-1][0], "%Y-%m-%d")
    ).days <= SIGNAL_FRESH_DAYS
    out = {}
    for key, sid, stop in ACCOUNTS:
        if not _allowed(sid, sym):
            out[key] = {"trades": [], "pending": []}
            continue
        if _rules(sid) is V:
            if ctx2 is None:
                if vol and vol_hits is None:
                    vol_hits = V.compute_vol_hits(vol, ("11v2a",))
                ctx2 = V.prepare(tb, vol, vol_hits=vol_hits, base=ctx)
            res = V.simulate(sid, tb, stop_pct=stop, fee=FEE, ctx=ctx2)
        else:
            res = R.simulate(sid, tb, stop_pct=stop, fee=FEE, ctx=ctx)
        rows = [_trade_row(sym, t, stop) for t in res["trades"] if t["signal_date"] >= start]
        if res["open"] is not None and res["open"]["signal_date"] >= start:
            rows.append(_trade_row(sym, res["open"], stop))
        pend = []
        if fresh:
            for sd, kind, c, reason in res["pending"]:
                if sd >= start:
                    pend.append(
                        {"sym": sym, "date": sd, "kind": kind, "close": c, "reason": reason}
                    )
        out[key] = {"trades": rows, "pending": pend}
    return out


def _mdd(curve: list) -> float:
    peak, mdd = 0.0, 0.0
    for _, v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1)
    return mdd


def run_account(trades: list[dict], closes: dict, calendar: list[str]) -> dict:
    """후보 거래를 하나의 가상 계좌로 날짜순 처리한다.

    closes: {종목: {날짜: 종가}}. calendar: START 이후 거래일(모든 종목 날짜 합집합).
    하루 순서: 시가 청산 → 시가 진입(신호일·종목 순, 자리 10개, 전날 평가액의 10%) →
    시가 추가 매수 → 장중 손절 → 종가 평가.
    """
    by_entry: dict[str, list] = {}
    by_add: dict[str, list] = {}
    by_exit: dict[str, list] = {}
    for n, t in enumerate(trades):
        t = dict(t, id=n)
        by_entry.setdefault(t["fills"][0][0], []).append(t)
        for f in t["fills"][1:]:
            by_add.setdefault(f[0], []).append((t["id"], f))
        if t.get("exit_date"):
            by_exit.setdefault(t["exit_date"], []).append(t)
    cash = CAPITAL
    equity = CAPITAL
    held: dict[int, dict] = {}
    closed, curve = [], []
    skipped = 0

    def value(p: dict, px: float) -> float:
        if p["side"] == "long":
            return p["shares"] * px
        return sum(a * (1 + (e - px) / e) for a, e in p["legs"])

    def close_pos(p: dict, d: str, px: float, reason: str) -> None:
        nonlocal cash
        if p["side"] == "long":
            got = p["shares"] * px * (1 - FEE)
        else:
            got = sum(a * (1 + (e - px) / e) - a * FEE * px / e for a, e in p["legs"])
        cash += got
        pnl = got - p["cost"]
        closed.append(
            {
                "sym": p["sym"],
                "entry_date": p["entry_date"],
                "exit_date": d,
                "entry_px": p["avg"],
                "exit_px": px,
                "invested": round(p["cost"], 2),
                "pnl": round(pnl, 2),
                "ret_pct": round(pnl / p["cost"] * 100, 2),
                "reason_in": p["reason_in"],
                "reason_out": reason,
            }
        )

    def buy(p: dict, px: float, amt: float) -> None:
        nonlocal cash
        amt = min(amt, cash)
        if amt <= 0:
            return
        if p["side"] == "long":
            cash -= amt
            p["shares"] += amt / (px * (1 + FEE))
        else:
            cash -= amt * (1 + FEE)
            p["legs"].append((amt, px))
        p["cost"] += amt * (1 + FEE) if p["side"] == "short" else amt
        p["px_w"] += px * amt
        p["amt"] += amt
        p["avg"] = p["px_w"] / p["amt"]

    for d in calendar:
        for t in sorted(by_exit.get(d, []), key=lambda x: x["sym"]):
            if t["id"] in held and not t["stop_exit"]:
                close_pos(held.pop(t["id"]), d, t["exit_px"], t["reason_out"])
        for t in sorted(by_entry.get(d, []), key=lambda x: (x["signal_date"], x["sym"])):
            if len(held) >= MAX_POS:
                skipped += 1
                continue
            budget = equity * SIZE_FRAC
            f = t["fills"][0]
            if min(budget, cash) <= 0:
                skipped += 1
                continue
            p = {
                "sym": t["sym"],
                "side": t["side"],
                "entry_date": d,
                "reason_in": t["reason_in"],
                "budget": budget,
                "shares": 0.0,
                "legs": [],
                "cost": 0.0,
                "px_w": 0.0,
                "amt": 0.0,
                "avg": f[1],
                "last": f[1],
                "last_date": d,
            }
            buy(p, f[1], budget * f[2])
            if p["amt"] > 0:
                held[t["id"]] = p
        for tid, f in by_add.get(d, []):
            if tid in held:
                buy(held[tid], f[1], held[tid]["budget"] * f[2])
        for t in sorted(by_exit.get(d, []), key=lambda x: x["sym"]):
            if t["id"] in held and t["stop_exit"]:
                close_pos(held.pop(t["id"]), d, t["exit_px"], t["reason_out"])
        for p in held.values():
            c = closes.get(p["sym"], {}).get(d)
            if c is not None:
                p["last"], p["last_date"] = c, d
        equity = cash + sum(value(p, p["last"]) for p in held.values())
        curve.append([d, round(equity, 2)])

    open_rows = []
    for p in held.values():
        v = value(p, p["last"])
        open_rows.append(
            {
                "sym": p["sym"],
                "side": p["side"],
                "entry_date": p["entry_date"],
                "entry_px": p["avg"],
                "last_px": p["last"],
                "last_date": p["last_date"],
                "invested": round(p["cost"], 2),
                "value": round(v, 2),
                "pnl": round(v - p["cost"], 2),
                "ret_pct": round((v / p["cost"] - 1) * 100, 2),
                "reason_in": p["reason_in"],
            }
        )
    open_rows.sort(key=lambda r: (r["entry_date"], r["sym"]))
    wins = sum(1 for c in closed if c["pnl"] > 0)
    return {
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "return_pct": round((equity / CAPITAL - 1) * 100, 2),
        "mdd_pct": round(_mdd(curve) * 100, 2),
        "n_closed": len(closed),
        "win_rate": round(wins / len(closed) * 100, 1) if closed else None,
        "skipped": skipped,
        "curve": curve,
        "open": open_rows,
        "closed": closed[-CLOSED_KEEP:][::-1],
    }


def _r(v: float | None, d: int = 4) -> float | None:
    return None if v is None else round(v, d)


def _avg_open_row(o: dict, sym: str) -> dict:
    return {
        "sym": sym,
        "side": "long",
        "entry_date": o["entry_date"],
        "entry_px": _r(o["avg"]),
        "avg": _r(o["avg"]),
        "tp_px": _r(o["tp_px"]),
        "adds": o["adds"],
        "adds_left": o["adds_left"],
        "unit": round(o["unit"], 2),
        "shares": _r(o["shares"], 6),
        "fills": [[f[0], f[1], _r(f[2]), round(f[3], 2)] for f in o["fills"]],
        "last_px": _r(o["last_px"]),
        "last_date": o["last_date"],
        "invested": round(o["invested"], 2),
        "value": round(o["value"], 2),
        "pnl": round(o["value"] - o["invested"], 2),
        "ret_pct": round(o["ret"] * 100, 2),
        "mae_now_pct": round(o["mae_now"] * 100, 2),
        "mae_worst_pct": round(o["mae"] * 100, 2),
        "bars_held": o["bars_held"],
        "days_held": o["days_held"],
        "reason_in": "SIG(스토캐·CCI·RSI 3단 과매도 탈출)",
    }


def build_avg_account(key: str, sym: str, bars: list | None, start: str, today: str) -> dict:
    """SCR 3단 + 물타기 가상 계좌 하나(`gose_rules_avg.simulate`, START부터 현금 1만 달러)."""
    base = {
        "key": key,
        "sid": "avg",
        "strategy": "avg",
        "name": AVG_NAME,
        "rule_ids": [],
        "side": "long",
        "timeframe": "D",
        "stop_pct": None,
        "max_hold": None,
        "symbols": [sym],
        "skipped": 0,
        "rules_avg": {"tp": A.TP, "max_adds": A.MAX_ADDS, "unit": "사이클 시작 자산/3"},
    }
    if not bars or len(bars) < MIN_BARS:
        return dict(
            base,
            equity=CAPITAL,
            cash=CAPITAL,
            return_pct=0.0,
            mdd_pct=0.0,
            n_closed=0,
            win_rate=None,
            curve=[],
            open=[],
            closed=[],
            signals=[],
            data=False,
        )
    tb = [tuple(b) for b in bars]
    ind = A.indicators(tb)
    sigs = A.signals(ind)
    res = A.simulate(tb, start=start, capital=CAPITAL, ind=ind, sigs=sigs)
    ref = A.simulate(tb, start=AVG_REF_START, capital=CAPITAL, ind=ind, sigs=sigs)
    curve = [[d, round(v, 2)] for d, v in res["curve"]]
    closed = [
        {
            "sym": sym,
            "entry_date": c["entry_date"],
            "exit_date": c["exit_date"],
            "entry_px": _r(c["avg"]),
            "exit_px": _r(c["exit_px"]),
            "adds": c["adds"],
            "invested": round(c["invested"], 2),
            "pnl": round(c["pnl"], 2),
            "ret_pct": round(c["ret"] * 100, 2),
            "mae_worst_pct": round(c["mae"] * 100, 2),
            "days_held": c["days_held"],
            "bars_held": c["bars_held"],
            "reason_in": "SIG",
            "reason_out": f"익절(평단×{1 + A.TP:.2f})",
        }
        for c in res["cycles"]
    ]
    wins = sum(1 for c in closed if c["pnl"] > 0)
    sig_rows = []
    fresh = (
        datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(tb[-1][0], "%Y-%m-%d")
    ).days <= SIGNAL_FRESH_DAYS
    nxt = res["next"]
    if nxt and fresh:
        action = {
            "entry": "매수(1단위, 사이클 시작)",
            "add": "추가 매수(물타기)",
            "capped": "추가 한도(2번) 다 씀 · 할 일 없음",
        }[nxt["kind"]]
        sig_rows.append(
            {
                "sym": sym,
                "date": nxt["signal_date"],
                "kind": nxt["kind"],
                "close": _r(nxt["close"]),
                "amount": round(nxt["amount"], 2),
                "reason": "SIG",
                "action": action,
            }
        )
    last_ind = {k: _r(ind[k][-1], 2) for k in ("S", "C", "R")}
    ro = ref["open"]
    return dict(
        base,
        data=True,
        equity=round(res["equity"], 2),
        cash=round(res["cash"], 2),
        return_pct=round((res["equity"] / CAPITAL - 1) * 100, 2),
        mdd_pct=round(_mdd(curve) * 100, 2),
        n_closed=len(closed),
        win_rate=round(wins / len(closed) * 100, 1) if closed else None,
        curve=curve,
        open=[_avg_open_row(res["open"], sym)] if res["open"] else [],
        closed=closed[-CLOSED_KEEP:][::-1],
        signals=sig_rows,
        last_bar=tb[-1][0],
        last_ind={"stoch_k": last_ind["S"], "cci": last_ind["C"], "rsi": last_ind["R"]},
        last_sig=tb[sigs[-1]][0] if sigs else None,
        reference={
            "from": AVG_REF_START,
            "note": "참고: 시험 구간 시작부터 이 규칙을 돌렸다면 지금 상태(가상 계좌에는 안 넣음)",
            "open": _avg_open_row(ro, sym) if ro else None,
            "n_closed": len(ref["cycles"]),
        },
    )


def account_signals(pending: list[dict], acct: dict, side: str) -> list[dict]:
    """오늘 신호(다음 시가 체결 예정).

    청산·추가는 이 계좌가 든 종목만, 진입은 자리 여부를 붙인다.
    """
    held = {p["sym"] for p in acct["open"]}
    free = MAX_POS - len(acct["open"])
    names = {
        "entry": "숏 진입" if side == "short" else "매수",
        "add": "추가 매수",
        "exit": "숏 청산" if side == "short" else "매도",
    }
    out = []
    for s in sorted(pending, key=lambda x: (x["date"], x["sym"])):
        if s["kind"] in ("exit", "add") and s["sym"] not in held:
            continue
        row = dict(s, action=names[s["kind"]])
        if s["kind"] == "entry":
            row["slot"] = free > 0
        out.append(row)
    return out


def build_paper(
    syms: list[str],
    vix_ticker: str,
    start: str,
    today: str,
    failures: list[dict] | None = None,
    vol_ticker: str = VOL_TICKER,
) -> dict:
    """캐시된 봉으로 모든 계좌를 다시 계산한다(네트워크 없음)."""
    vix_raw = load_bars(vix_ticker)
    vix = [tuple(b) for b in vix_raw] if vix_raw else None
    vol_raw = load_bars(vol_ticker)
    vol = [tuple(b) for b in vol_raw] if vol_raw else None
    hits = V.compute_vol_hits(vol, ("11v2a",)) if vol else None
    per = {key: {"trades": [], "pending": []} for key, _, _ in ACCOUNTS}
    closes: dict[str, dict] = {}
    cal: set[str] = set()
    used, short, latest = 0, [], ""
    for sym in syms:
        bars = load_bars(sym)
        if not bars or len(bars) < MIN_BARS:
            short.append(sym)
            continue
        used += 1
        latest = max(latest, bars[-1][0])
        recent = {b[0]: b[4] for b in bars if b[0] >= start}
        closes[sym] = recent
        cal.update(recent)
        got = collect(sym, bars, vix, start, today, vol=vol, vol_hits=hits)
        for key in per:
            per[key]["trades"].extend(got[key]["trades"])
            per[key]["pending"].extend(got[key]["pending"])
    calendar = sorted(cal)
    accounts = []
    for key, sid, stop in ACCOUNTS:
        spec = _rules(sid).STRATEGIES[sid]
        acct = run_account(per[key]["trades"], closes, calendar)
        acct["signals"] = account_signals(per[key]["pending"], acct, spec["side"])
        acct.update(
            key=key,
            sid=sid,
            name=spec["name"],
            rule_ids=spec["rule_ids"],
            side=spec["side"],
            timeframe=spec["timeframe"],
            stop_pct=stop,
            max_hold=spec["max_hold"],
        )
        if sid in V.STRATEGIES:
            acct.update(round=2, label=V2_LABEL)
            if sid in V.TICKERS:
                acct.update(
                    symbols=list(V.TICKERS[sid]),
                    signal_source=vol_ticker,
                    signal_source_through=vol[-1][0] if vol else None,
                )
        accounts.append(acct)
    for key, sym in AVG_ACCOUNTS:
        bars = load_bars(sym)
        if bars:
            latest = max(latest, bars[-1][0])
        accounts.append(build_avg_account(key, sym, bars, start, today))
    return {
        "status": "ok",
        "virtual": True,
        "generated_at": datetime.now(KST).isoformat(timespec="seconds"),
        "start": start,
        "data_through": latest or None,
        "symbols_used": used,
        "symbols_short": short,
        "failures": failures or [],
        "rules": {
            "capital": CAPITAL,
            "max_positions": MAX_POS,
            "size_frac": SIZE_FRAC,
            "fee_per_side": FEE,
            "fill": "다음 날 시가",
        },
        "accounts": accounts,
    }


def run(now: datetime | None = None, fetch: bool = True) -> dict:
    now = now or datetime.now(KST)
    today = now.astimezone(KST).strftime("%Y-%m-%d")
    uni = load_universe()
    st = load_state(today)
    syms = [s["ticker"] for s in uni["symbols"]]
    vix = uni.get("vix", "^VIX")
    vol = uni.get("vol", VOL_TICKER)
    extra = [s for _, s in AVG_ACCOUNTS if s not in syms]
    failures = refresh_all(syms + extra + [vix, vol], now.timestamp()) if fetch else []
    paper = build_paper(syms, vix, st["start"], today, failures, vol_ticker=vol)
    _write(gose_dir() / "paper.json", paper)
    return paper


# ---------------------------------------------------------------- 백테스트 요약

STOP_LABEL = {"N": "없음(원문 청산·시간 청산)", "S10": "10%"}


def _reason(v: dict, bonf_pct: float) -> str:
    fails = list(v.get("fail_reasons") or [])
    if fails:
        return " · ".join(fails)
    rnd = (v.get("random") or {}).get("pct")
    return f"기본 3관문은 넘었지만 무작위 {rnd}백분위 < 다중검정 기준 {bonf_pct}"


def _row(sid: str, spec: dict, tag: str, v: dict, bonf: float, n_tested: int) -> dict:
    m = v.get("main") or {}
    port = v.get("port_main_1x") or {}
    rnd = v.get("random") or {}
    bh = v.get("bh_same_syms_main") or {}
    basic = bool(v.get("pass"))
    verified = bool(v.get("bonferroni_pass"))
    row = {
        "variant": f"{sid}/{tag}",
        "sid": sid,
        "name": spec["name"],
        "rule_ids": spec["rule_ids"],
        "side": spec["side"],
        "stop": STOP_LABEL[tag],
        "n": m.get("n", 0),
        "win": m.get("win"),
        "mean": m.get("mean"),
        "median": m.get("median"),
        "worst": m.get("worst"),
        "hold_days": m.get("avg_hold_days"),
        "cagr": port.get("cagr"),
        "mdd": port.get("mdd"),
        "bh_cagr": bh.get("cagr"),
        "bh_mdd": bh.get("mdd"),
        "random_pct": rnd.get("pct"),
        "basic_pass": basic,
        "verified": verified,
        "verdict": "통과" if verified else ("근접(다중검정 미통과)" if basic else "탈락"),
        "reason": "통과" if verified else _reason(v, bonf),
    }
    if basic and not verified:
        src = v.get("src_syms_main") or {}
        row["caveats"] = [
            f"무작위 {rnd.get('pct')}백분위: {n_tested}개를 함께 시험한 기준({bonf})에 못 미쳐요",
            f"원문 12종목만 보면 평균 {src.get('mean')}% ({src.get('n')}번)",
            f"포트폴리오 연 {port.get('cagr')}% · 최대 낙폭 {port.get('mdd')}%, "
            f"같은 종목 그냥 보유 연 {bh.get('cagr')}% · {bh.get('mdd')}%",
            "손절 없는 같은 규칙(원문 청산)은 탈락",
        ]
    return row


def _v2_caveats(v: dict) -> list[str]:
    """2차 변형 주의점: 2021-22 하락장 성적, 종목 쏠림(변동성 신호 전략)."""
    out = []
    sub = (v.get("subs") or {}).get("2021-22") or {}
    if not sub.get("n"):
        out.append("2021-22 하락장에 신호 0번(그 시기를 시험하지 못함)")
    elif (sub.get("mean") or 0) <= 0:
        out.append(f"2021-22 하락장 평균 {sub.get('mean')}% 손실({sub.get('n')}번)")
    per = v.get("per_sym_main") or {}
    if "SOXL" in per and len(per) > 1:
        others = [x.get("mean") for k, x in per.items() if k != "SOXL" and x.get("n")]
        if others:
            lo, hi = min(others), max(others)
            out.append(
                f"SOXL 쏠림: SOXL 평균 {per['SOXL'].get('mean')}%({per['SOXL'].get('n')}번), "
                f"나머지 {lo}~{hi}%"
            )
    return out


def build_backtest(results: dict, results2: dict | None = None) -> dict:
    """백테스트 results.json(1차) [+ results2.json(2차)] → 화면용 요약 줄.

    순서: 1차 전략 → 2차 전략, 각각 손절 없음·10%. 2차 줄은 round=2와 주의점(caveats)을 단다.
    다중검정 기준은 2차가 있으면 2차 meta(1·2차 합계 기준)를 쓴다.
    """
    meta = results.get("meta", {})
    meta2 = (results2 or {}).get("meta") or {}
    bonf = meta2.get("bonferroni_pct", meta.get("bonferroni_pct", 99.84))
    n1 = len(results.get("variants") or {})
    n2 = len((results2 or {}).get("variants") or {})
    n_tested = meta2.get("variants") or (n1 + n2) or 32
    rows = []
    for src, mod, rnd in ((results, R, 1), (results2 or {}, V, 2)):
        for sid, spec in mod.STRATEGIES.items():
            for tag in ("N", "S10"):
                v = (src.get("variants") or {}).get(f"{sid}/{tag}")
                if v is None:
                    continue
                row = _row(sid, spec, tag, v, bonf, n_tested)
                if rnd == 2:
                    row["round"] = 2
                    row["label"] = V2_LABEL
                    notes = _v2_caveats(v)
                    if notes:
                        row["caveats"] = notes + row.get("caveats", [])
                rows.append(row)
    r2 = [r for r in rows if r.get("round") == 2]
    return {
        "status": "ok",
        "made_at": datetime.now(KST).isoformat(timespec="seconds"),
        "verified": sum(1 for r in rows if r["verified"]),
        "basic_pass": sum(1 for r in rows if r["basic_pass"]),
        "variants": len(rows),
        "rounds": {
            "1": {
                "variants": len(rows) - len(r2),
                "verified": sum(1 for r in rows if r["verified"] and not r.get("round")),
            },
            "2": {
                "variants": len(r2),
                "verified": sum(1 for r in r2 if r["verified"]),
                "label": V2_LABEL,
                "vol_series": meta2.get("vol_series"),
            },
        }
        if r2
        else None,
        "meta": {
            "main": meta.get("main"),
            "ten": meta.get("ten"),
            "symbols": meta.get("symbols"),
            "fee_per_side": meta.get("fee_per_side"),
            "shuffles": meta.get("shuffles"),
            "bonferroni_pct": bonf,
            "tested": n_tested,
        },
        "gates": [
            "최근 5년 거래 30번 이상",
            "2021-22 · 2023-24 · 2025-26 기간 평균 수익이 모두 + (수수료 뒤)",
            "같은 종목·같은 보유 일수 무작위 진입 1000번보다 평균이 높음(95백분위 초과)",
            f"{n_tested}개를 함께 시험하니 우연을 빼려면 {bonf}백분위 초과(다중검정)",
        ],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m minerva.gose", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run", help="일봉 갱신 + 가상매매 다시 계산")
    pr.add_argument("--no-fetch", action="store_true", help="캐시만 쓰고 받지 않음")
    pb = sub.add_parser("backtest", help="results.json → backtest.json")
    pb.add_argument("results")
    pb.add_argument("--round2", help="2차(그림 판독 정의) results2.json")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        p = run(fetch=not a.no_fetch)
        best = max(p["accounts"], key=lambda x: x["return_pct"])
        print(
            f"start {p['start']} · 자료 {p['data_through']} · 종목 {p['symbols_used']} · "
            f"실패 {len(p['failures'])} · 최고 {best['key']} {best['return_pct']}%"
        )
        return 0
    src = Path(a.results)
    res = json.loads(src.read_text())
    src2 = Path(a.round2) if a.round2 else None
    res2 = json.loads(src2.read_text()) if src2 else None
    out = build_backtest(res, res2)
    d = gose_dir()
    d.mkdir(parents=True, exist_ok=True)
    for f, name in ((src, "results.json"), (src2, "results2.json")):
        if f is not None and f.resolve() != (d / name).resolve():
            shutil.copyfile(f, d / name)
    _write(d / "backtest.json", out)
    print(f"변형 {out['variants']} · 기본 통과 {out['basic_pass']} · 검증 통과 {out['verified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

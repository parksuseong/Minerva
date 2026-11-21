"""성장 후보 관심 목록(🌱): 큰 하락장 바닥에서 10배·100배가 될 수도 있는 미국 상장 보통주.

사용자 2026-11-21: "애플만큼 오르려면 시작 시총이 애플의 1/10~1/100" → 시총 $1B~$400B를
두 칸으로 나눈다. 10배 후보 $40B~$400B, 100배 후보(위험 높음) $1B~$40B.

- 종목 범위: 나스닥 스크리너 공개 JSON(NYSE·Nasdaq·AMEX 상장 전부, 시총·업종). 키 없음.
  우선주·채권·워런트·유닛·권리·펀드·신탁·SPAC(인수 목적 회사)은 뺀다. 같은 회사 여러 클래스는
  거래량 큰 하나만. 해외 회사 ADR은 미국 상장이라 남긴다.
- 매출 성장: 야후 공개 재무 시계열(최근 연간 매출 vs 그 전 해). 흑자 = 최근 12개월 순이익
  (없으면 최근 연간) > 0. 재무는 7~9일 캐시(분기마다 바뀜) → 매일 받는 건 일부뿐이다.
- 순위: 매출 성장 높은 순, 성장률이 없거나 직전 해 매출이 너무 작은(`MIN_PRIOR_REVENUE`) 종목은
  뺀다(작은 숫자로 나눈 수천 % 성장이 맨 위를 채우지 않게). 칸마다 상위 `TOP_N`.
- 가격·52주 고점: 순위에 든 종목만 야후 일봉 메타(받기 적게).
- 실패: 받기가 실패하면 지난 파일을 그대로 두고 `stale`만 표시한다. 화면은 죽지 않는다.
- 공포 알림: 일일 분석의 공포 초입 신호(`strategy.daily.fear_status`, VIX/VIX3M)를 그대로 쓴다.
  새 신호가 나오고 `FEAR_ALERT_DAYS` 거래일 안이면 텔레그램 글 한 번(신호 날짜로 중복 막음).
- 관심 목록일 뿐 매수 추천이 아니다. 주문 없음.

실행: `python -m minerva.growth [--dry] [--notify] [--limit N]`
결과: `~/.minerva/growth/latest.json` → `/api/growth` → 화면 🌱 성장 후보 탭.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

KST = timezone(timedelta(hours=9))
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
NASDAQ_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
TS_URL = (
    "https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{sym}"
    "?symbol={sym}&type={types}&period1={p1}&period2={p2}"
)
TS_TYPES = (
    "annualTotalRevenue",
    "quarterlyTotalRevenue",
    "trailingNetIncome",
    "annualNetIncome",
    "trailingOperatingIncome",
    "annualOperatingIncome",
)
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5d&interval=1d"

MIN_CAP, MAX_CAP = 1e9, 400e9
BUCKETS = (
    {"key": "x10", "label": "10배 후보", "lo": 40e9, "hi": 400e9},
    {"key": "x100", "label": "100배 후보(위험 높음)", "lo": 1e9, "hi": 40e9},
)
TOP_N = 30
MIN_PRIOR_REVENUE = 20e6
FUND_TTL_DAYS = 7  # + 종목별 0~2일(한날에 다 만료돼 한 번에 몰리지 않게)
FEAR_ALERT_DAYS = 5  # 신호 뒤 이 거래일 안에만 알린다(늦게 켜도 한참 지난 신호로 보내지 않게)
STALE_HOURS = 60
SLEEP = 0.3
NOTE = "관심 목록이지 매수 추천이 아닙니다. 큰 하락장 공포 구간에서 분할 매수할 후보."

# 보통주가 아닌 것(우선주·채권·워런트·유닛·권리·펀드)과 SPAC·껍데기 회사
_NOT_COMMON = re.compile(
    r"Warrant|\bRights?\b|Preferred|Preference|\bNotes?\b|Debenture|Subordinated|"
    r"Perpetual|Cumulative|\d%|\bFund\b|\bETF\b|Term Trust|Blank Check|"
    r"Acquisitions? (Corp|Inc|Ltd|Co\b|Company)|Capital Trust|Beneficial Interest",
    re.I,
)
_UNIT = re.compile(r"\bUnits?\b", re.I)  # SPAC 유닛·지분 유닛. MLP "Common Units"는 보통주로 본다
_ADR = re.compile(r"American Depositary", re.I)
_SHELL_INDUSTRY = re.compile(r"Blank Check|Trusts Except", re.I)
_NAME_CUT = re.compile(
    r"\s+(Class [A-Z]\b.*|Common Stock.*|Common Shares.*|Ordinary Shares.*|"
    r"American Deposit[ao]ry.*|Common Units.*|Sponsored ADR.*|ADS\b.*)$",
    re.I,
)


def growth_dir() -> Path:
    return Path(os.environ.get("MINERVA_GROWTH_DIR", "~/.minerva/growth")).expanduser()


def out_path() -> Path:
    return growth_dir() / "latest.json"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def get_json(url: str, retries: int = 3, timeout: float = 20, sleep=time.sleep) -> dict:
    """공개 GET. 429·5xx·통신 오류는 점점 길게 쉬고 다시. 끝내 실패하면 마지막 오류를 올린다."""
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 502, 503, 504):
                raise
            sleep(5 * (i + 1) * (3 if e.code == 429 else 1))
        except (OSError, ValueError) as e:
            last = e
            sleep(3 * (i + 1))
    assert last is not None
    raise last


# ---- 종목 범위 -------------------------------------------------------------------------------


def parse_cap(v) -> float | None:
    try:
        x = float(str(v).replace(",", "").replace("$", ""))
    except ValueError:
        return None
    return x if x > 0 else None


def short_name(name: str) -> str:
    n = _NAME_CUT.sub("", (name or "").strip())
    return n.strip(" .,") or (name or "").strip()


def is_common(row: dict) -> bool:
    sym, name = str(row.get("symbol", "")), str(row.get("name", ""))
    if not re.fullmatch(r"[A-Z]{1,5}", sym):
        return False
    if _NOT_COMMON.search(name):
        return False
    if _UNIT.search(name) and not re.search(r"Common Units", name, re.I):
        return False
    if re.search(r"Depositary", name, re.I) and not _ADR.search(name):
        return False  # 우선주 예탁 증서(…Depositary Shares each representing 1/1000th…)
    return not _SHELL_INDUSTRY.search(str(row.get("industry") or ""))


def universe(rows: list[dict]) -> list[dict]:
    """시총 $1B~$400B 보통주. 같은 회사(이름) 여러 클래스는 거래량 큰 하나."""
    best: dict[str, dict] = {}
    for r in rows:
        cap = parse_cap(r.get("marketCap"))
        if cap is None or not (MIN_CAP <= cap <= MAX_CAP) or not is_common(r):
            continue
        item = {
            "symbol": r["symbol"],
            "name": short_name(r.get("name", "")),
            "sector": (r.get("sector") or "").strip() or None,
            "industry": (r.get("industry") or "").strip() or None,
            "country": (r.get("country") or "").strip() or None,
            "cap": cap,
            "volume": parse_cap(r.get("volume")) or 0,
        }
        key = item["name"].lower()
        if key not in best or item["volume"] > best[key]["volume"]:
            best[key] = item
    return sorted(best.values(), key=lambda x: -x["cap"])


def fetch_universe(get=get_json) -> list[dict]:
    rows = (((get(NASDAQ_URL) or {}).get("data") or {}).get("rows")) or []
    if len(rows) < 1000:  # 반쯤 빈 응답으로 목록을 덮어쓰지 않게
        raise ValueError(f"nasdaq rows {len(rows)}")
    return rows


# ---- 재무 ------------------------------------------------------------------------------------


def _series(payload: dict) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for r in ((payload.get("timeseries") or {}).get("result")) or []:
        t = ((r.get("meta") or {}).get("type") or [None])[0]
        pts = [
            (x["asOfDate"], float(x["reportedValue"]["raw"]))
            for x in (r.get(t) or [])
            if x and x.get("asOfDate") and (x.get("reportedValue") or {}).get("raw") is not None
        ]
        if t:
            out[t] = sorted(pts)
    return out


def _days(a: str, b: str) -> int:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days


def parse_fund(payload: dict) -> dict:
    """연간 매출 성장(최근 해 vs 그 전 해), 최근 분기 전년 같은 분기 대비, 흑자 여부."""
    s = _series(payload)
    out: dict = {"growth": None, "q_growth": None, "revenue": None, "rev_date": None}
    out["empty"] = not any(s.values())
    ann = s.get("annualTotalRevenue") or []
    if ann:
        out["revenue"], out["rev_date"] = ann[-1][1], ann[-1][0]
    if len(ann) >= 2 and 330 <= _days(ann[-2][0], ann[-1][0]) <= 400:
        prev = ann[-2][1]
        if prev >= MIN_PRIOR_REVENUE:
            out["growth"] = ann[-1][1] / prev - 1
    q = s.get("quarterlyTotalRevenue") or []
    if len(q) >= 5 and 350 <= _days(q[-5][0], q[-1][0]) <= 380 and q[-5][1] > 0:
        out["q_growth"] = q[-1][1] / q[-5][1] - 1

    def last(*keys: str) -> float | None:
        for k in keys:
            if s.get(k):
                return s[k][-1][1]
        return None

    net = last("trailingNetIncome", "annualNetIncome")
    op = last("trailingOperatingIncome", "annualOperatingIncome")
    out["net_income"], out["op_income"] = net, op
    if net is not None:
        out["profit"] = "흑자" if net > 0 else ("영업흑자" if op is not None and op > 0 else "적자")
    elif op is not None:
        out["profit"] = "영업흑자" if op > 0 else "적자"
    else:
        out["profit"] = None
    return out


def fetch_fund(sym: str, get=get_json, now: float | None = None) -> dict:
    now = now or time.time()
    url = TS_URL.format(
        sym=urllib.parse.quote(sym),
        types=",".join(TS_TYPES),
        p1=int(now - 4 * 365 * 86400),
        p2=int(now),
    )
    return parse_fund(get(url))


def _ttl(sym: str) -> float:
    return (FUND_TTL_DAYS + zlib.crc32(sym.encode()) % 3) * 86400


def refresh_funds(
    syms: list[str],
    cache: dict,
    get=get_json,
    now: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    limit: int | None = None,
    log=print,
) -> dict:
    """캐시가 오래된 종목만 받는다. 연달아 실패하면(차단 의심) 멈추고 남은 건 지난 캐시로."""
    now = now or time.time()
    todo = [s for s in syms if now - (cache.get(s) or {}).get("at", 0) > _ttl(s)]
    if limit is not None:
        todo = todo[:limit]
    ok = bad = streak = 0
    for s in todo:
        try:
            f = fetch_fund(s, get=get, now=now)
            if f.pop("empty"):  # 빈 응답(차단·일시 오류일 수도)은 캐시하지 않고 다음에 다시
                raise ValueError("empty timeseries")
            cache[s] = {"at": now, **f}
            ok += 1
            streak = 0
        except Exception as e:  # noqa: BLE001 — 한 종목 실패가 전체를 막지 않게
            bad += 1
            streak += 1
            if streak >= 15:
                log(f"재무 받기 연속 실패 {streak}번({type(e).__name__}), 이번엔 여기까지")
                break
        sleep(SLEEP)
    return {"todo": len(todo), "ok": ok, "failed": bad}


def fetch_quote(sym: str, get=get_json) -> dict:
    meta = get(CHART_URL.format(sym=urllib.parse.quote(sym)))["chart"]["result"][0]["meta"]
    price, high = meta.get("regularMarketPrice"), meta.get("fiftyTwoWeekHigh")
    dd = price / high - 1 if price and high else None
    return {"price": price, "high52": high, "drawdown": dd}


# ---- 순위 ------------------------------------------------------------------------------------


def in_bucket(cap: float, b: dict) -> bool:
    """경계 $40B는 10배 칸(위 칸이 [lo, hi], 아래 칸은 [lo, hi))."""
    return b["lo"] <= cap < b["hi"] or (b["hi"] == MAX_CAP and cap == MAX_CAP)


def rank(items: list[dict], funds: dict, top: int = TOP_N) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for b in BUCKETS:
        rows = []
        for it in items:
            f = funds.get(it["symbol"]) or {}
            if not in_bucket(it["cap"], b):
                continue
            if f.get("growth") is None:
                continue
            rows.append({**{k: it[k] for k in ("symbol", "name", "sector", "industry", "country",
                                               "cap")},
                         **{k: f.get(k) for k in ("growth", "q_growth", "revenue", "rev_date",
                                                  "profit", "net_income")}})
        rows.sort(key=lambda r: -r["growth"])
        out[b["key"]] = rows[:top]
    return out


def _r(v, d: int = 4):
    return None if v is None else round(v, d)


def build(items: list[dict], funds: dict, quotes: dict, now: datetime | None = None,
          stats: dict | None = None) -> dict:
    now = now or datetime.now(KST)
    ranked = rank(items, funds)
    buckets = []
    for b in BUCKETS:
        rows = []
        for r in ranked[b["key"]]:
            q = quotes.get(r["symbol"]) or {}
            rows.append({**r, "growth": _r(r["growth"]), "q_growth": _r(r["q_growth"]),
                         "price": q.get("price"), "high52": q.get("high52"),
                         "drawdown": _r(q.get("drawdown"))})
        n_univ = sum(1 for it in items if in_bucket(it["cap"], b))
        buckets.append({"key": b["key"], "label": b["label"], "lo": b["lo"], "hi": b["hi"],
                        "universe": n_univ, "rows": rows})
    return {
        "generated_at": now.isoformat(timespec="minutes"),
        "stale": False,
        "note": NOTE,
        "method": (
            f"미국 상장 보통주 시총 $1B~$400B · 매출 성장 = 최근 연간 매출 vs 그 전 해 "
            f"(직전 해 매출 ${MIN_PRIOR_REVENUE / 1e6:.0f}M 이상) · 높은 순 상위 {TOP_N} · "
            "흑자 = 최근 12개월 순이익 > 0 · 낙폭 = 52주 고점 대비"
        ),
        "stats": stats or {},
        "buckets": buckets,
    }


# ---- 공포 알림 -------------------------------------------------------------------------------


def fear_now() -> dict:
    """일일 분석과 같은 공포 초입 신호. `elapsed` = 신호 뒤 지난 거래일 수."""
    from minerva.strategy.daily import completed, fear_status
    from minerva.strategy.report import fetch_yahoo

    vix = completed(fetch_yahoo("^VIX", "max"), "US")
    vix3m = completed(fetch_yahoo("^VIX3M", "max"), "US")
    f = fear_status(vix, vix3m)
    if f.get("last_signal"):
        m3 = {b.day for b in vix3m}
        f["elapsed"] = sum(1 for b in vix if b.day in m3 and b.day.isoformat() > f["last_signal"])
    return f


def alert_text(data: dict, fear: dict, n: int = 5) -> str:
    prem = fear.get("premium")
    lines = [
        f"😨 공포 초입 신호 ({fear.get('last_signal')}, VIX/VIX3M "
        f"{'' if prem is None else f'{prem:+.0%}'}) — 🌱 성장 후보",
        NOTE,
    ]
    for b in data.get("buckets") or []:
        lines.append("")
        lines.append(f"[{b['label']}] 매출 성장 상위")
        for r in (b.get("rows") or [])[:n]:
            dd = r.get("drawdown")
            lines.append(
                f"- {r['name']} ({r['symbol']}) 시총 {fmt_cap(r['cap'])} · 성장 {r['growth']:+.0%}"
                f" · {r.get('profit') or '-'}" + ("" if dd is None else f" · 고점 대비 {dd:+.0%}")
            )
    if data.get("stale"):
        lines.append("")
        lines.append(f"(목록 갱신 실패, {data.get('generated_at')} 자료)")
    return "\n".join(lines)


def fmt_cap(v: float) -> str:
    return f"${v / 1e9:,.0f}B" if v >= 10e9 else f"${v / 1e9:,.1f}B"


def maybe_alert(data: dict, fear: dict, state_path: Path, send=None) -> bool:
    """새 공포 초입 신호면 한 번만 보낸다. 보내기 실패면 기록하지 않아 다음 실행에 다시."""
    sig, elapsed = fear.get("last_signal"), fear.get("elapsed")
    if not sig or elapsed is None or elapsed > FEAR_ALERT_DAYS:
        return False
    state = read_json(state_path) or {}
    if state.get("last_signal") == sig:
        return False
    if not any(b.get("rows") for b in data.get("buckets") or []):
        return False
    if send is None:
        from minerva import notify

        send = notify.send
    if not send(alert_text(data, fear)):
        return False
    write_json(state_path, {"last_signal": sig, "sent_at": datetime.now(KST).isoformat(
        timespec="minutes")})
    return True


# ---- 실행 ------------------------------------------------------------------------------------


def mark_stale(reason: str) -> None:
    prev = read_json(out_path())
    if prev:
        prev.update(stale=True, stale_reason=reason,
                    stale_at=datetime.now(KST).isoformat(timespec="minutes"))
        write_json(out_path(), prev)


def run(dry: bool = False, limit: int | None = None, get=get_json, sleep=time.sleep,
        log=print) -> dict | None:
    d = growth_dir()
    stale_reasons = []
    try:
        rows = fetch_universe(get=get)
        if not dry:
            write_json(d / "universe.json", {"at": time.time(), "rows": rows})
    except Exception as e:  # noqa: BLE001 — 목록 원천이 막혀도 지난 목록으로
        log(f"종목 목록 받기 실패: {type(e).__name__}")
        rows = (read_json(d / "universe.json") or {}).get("rows") or []
        stale_reasons.append("종목 목록 받기 실패, 지난 목록 사용")
        if not rows:
            if not dry:
                mark_stale("종목 목록 받기 실패")
            return None
    items = universe(rows)
    cache_path = d / "fund_cache.json"
    cache = read_json(cache_path) or {}
    st = refresh_funds([it["symbol"] for it in items], cache, get=get, sleep=sleep, limit=limit,
                       log=log)
    if not dry:
        write_json(cache_path, cache)
    have = sum(1 for it in items if it["symbol"] in cache)
    if have < len(items) * 0.5:
        log(f"재무 자료가 너무 적음({have}/{len(items)}), 지난 파일 유지")
        if not dry:
            mark_stale(f"재무 자료 부족({have}/{len(items)})")
        return None
    if st["failed"]:
        stale_reasons.append(f"재무 {st['failed']}종목 받기 실패(지난 캐시 사용)")
    quotes = {}
    qbad = 0
    for rows_ in rank(items, cache).values():
        for r in rows_:
            try:
                quotes[r["symbol"]] = fetch_quote(r["symbol"], get=get)
            except Exception:  # noqa: BLE001 — 가격이 없어도 목록은 낸다
                qbad += 1
            sleep(SLEEP)
    stats = {"universe": len(items), "with_financials": have, **st, "quote_failed": qbad}
    data = build(items, cache, quotes, stats=stats)
    if stale_reasons:
        data["warnings"] = stale_reasons
    if not dry:
        write_json(out_path(), data)
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="파일을 쓰지 않고 요약만 출력")
    ap.add_argument("--notify", action="store_true", help="공포 초입 신호면 텔레그램 한 번")
    ap.add_argument("--limit", type=int, help="이번에 받을 재무 종목 수 상한(시험용)")
    a = ap.parse_args(argv)
    data = run(dry=a.dry, limit=a.limit)
    rc = 0
    if data is None:
        rc = 1
        data = read_json(out_path())  # 공포 알림은 지난 목록으로라도
    else:
        s = data["stats"]
        print(f"universe={s['universe']} financials={s['with_financials']} "
              f"fetched={s['ok']}/{s['todo']} failed={s['failed']} "
              f"quote_failed={s['quote_failed']}")
        for b in data["buckets"]:
            print(f"{b['label']}: {len(b['rows'])}/{b['universe']}")
            for r in b["rows"][:5]:
                dd = r.get("drawdown")
                print(f"  {r['symbol']:6} {fmt_cap(r['cap']):>8} growth {r['growth']:+.1%} "
                      f"dd {'-' if dd is None else f'{dd:+.1%}'} {r.get('profit')} {r['name']}")
    if a.notify and data and not a.dry:
        try:
            fear = fear_now()
            print(f"fear: {fear.get('state')} last={fear.get('last_signal')} "
                  f"elapsed={fear.get('elapsed')}")
            if maybe_alert(data, fear, growth_dir() / "alert.json"):
                print("공포 알림 보냄")
        except Exception as e:  # noqa: BLE001 — VIX 조회 실패가 목록 갱신을 실패로 만들지 않게
            print(f"공포 신호 확인 실패: {type(e).__name__}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

"""어드민 탭: 자동매매 정지·재개 스위치와 접속 기록.

- 잠금은 계좌 탭과 같은 열람 토큰(`routes.account_view_token`). 다만 어드민은 토큰이 설정돼
  있지 않으면 열지 않는다(계좌는 토큰이 없으면 공개지만, 어드민은 매매를 멈출 수 있어서).
- 웹 서버가 쓰는 파일은 엔진 폴더의 `HALT`·`NO_ENTRY` 스위치 둘뿐이다. 주문·취소·청산은 하지 않는다.
  `HALT`/`NO_ENTRY`는 엔진의 신규 진입만 막는다(보유·손절·따라가는 손절·반대 신호 청산은 계속).
- 기록은 저장소 밖 `~/.minerva/admin/`(폴더 700, 파일 600, `MINERVA_ADMIN_DIR`로 바꿈):
  `access.jsonl`(페이지 열기·토큰 시도), `actions.jsonl`(어드민 조작), `geo.json`(IP 위치 캐시).
- IP: 서버는 uvicorn이 443에서 TLS를 직접 받는다(프록시 없음, `deploy/minerva.service`).
  그래서 접속한 쪽 주소(`request.client.host`)를 쓰고, X-Forwarded-For는 같은 서버(루프백)에서
  온 요청일 때만 믿는다. 밖에서 보낸 X-Forwarded-For는 누구나 꾸밀 수 있다.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import re
import threading
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query, Request

from minerva import btc_trade as bt
from minerva import btc_view, notify, routes
from minerva.kis import KST, _write_private

log = logging.getLogger(__name__)

MAX_LINES = 50_000
MAX_DAYS = 90
TRIM_EVERY = 500  # 덧붙이기 몇 번마다 오래된 줄을 잘라낼지
SUCCESS_DEDUP_SEC = 6 * 3600  # 저장된 토큰으로 30초마다 부르는 성공은 기기별로 6시간에 한 번만
FAIL_WINDOW_SEC = 15 * 60
FAIL_LIMIT = 10  # 15분에 서로 다른 틀린 토큰 10개면 그 IP의 토큰 요청을 잠시 막는다
GEO_TTL_SEC = 30 * 86400
GEO_FAIL_TTL_SEC = 3600
GEO_URL = "http://ip-api.com/batch?fields=status,country,city,isp,query"
GEO_BATCH = 100
GEO_MAX_BATCHES = 2  # 한 번 볼 때 조회는 최대 200개(무료 한도 분당 15번)
GEO_TIMEOUT = 3.0

_lock = threading.Lock()
_appends = 0
_seen_ok: dict[tuple[str, str, str], float] = {}
_fails: dict[str, dict[str, float]] = {}


def admin_dir() -> Path:
    return Path(os.environ.get("MINERVA_ADMIN_DIR", "~/.minerva/admin")).expanduser()


def _ensure_dir() -> Path:
    d = admin_dir()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- IP

_LOCAL_PEERS = {"127.0.0.1", "::1"}


def _valid_ip(text: str) -> str | None:
    try:
        return str(ipaddress.ip_address(text.strip()))
    except ValueError:
        return None


def client_ip(peer: str | None, forwarded: str | None) -> str:
    """접속한 쪽 IP. X-Forwarded-For는 루프백(같은 서버의 프록시)에서 왔을 때만,
    그 프록시가 붙인 맨 오른쪽 값을 쓴다(왼쪽 값은 접속자가 꾸밀 수 있다)."""
    ip = _valid_ip(peer or "") or "-"
    if ip in _LOCAL_PEERS and forwarded:
        last = _valid_ip(forwarded.split(",")[-1])
        if last:
            return last
    return ip


def is_internal(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return a.is_private or a.is_loopback or a.is_link_local or a.is_reserved


# ---------------------------------------------------------------- 기기(User-Agent)

_BOT = re.compile(
    r"bot|crawl|spider|slurp|curl|wget|python|httpx|aiohttp|go-http|java/|libwww|axios|"
    r"node-fetch|headless|scan|zgrab|masscan|nmap|nuclei|facebookexternalhit|preview|"
    r"monitor|uptime|scrapy|okhttp|postman|insomnia",
    re.I,
)
_BOT_NAME = re.compile(r"([A-Za-z][\w.-]*(?:bot|crawler|spider|scan\w*))", re.I)
_APPS = (  # 위에서부터 먼저 맞는 것(인앱 브라우저가 Chrome·Safari 표시를 같이 달고 온다)
    ("KAKAOTALK", "카카오톡"),
    ("NAVER(", "네이버 앱"),
    ("Line/", "LINE"),
    ("FBAN", "Facebook"),
    ("FBAV", "Facebook"),
    ("Instagram", "Instagram"),
    ("SamsungBrowser", "Samsung Internet"),
    ("Whale/", "Whale"),
    ("EdgA/", "Edge"),
    ("EdgiOS/", "Edge"),
    ("Edg/", "Edge"),
    ("OPR/", "Opera"),
    ("CriOS/", "Chrome"),
    ("FxiOS/", "Firefox"),
    ("Firefox/", "Firefox"),
    ("Chrome/", "Chrome"),
)


_IOS = re.compile(r"(?:iPhone|CPU) OS (\d+)(?:_(\d+))?")
_ANDROID = re.compile(r"Android (\d+(?:\.\d+)?)(?:; ([^;)]+))?")
_WIN = {"10.0": "10/11", "6.3": "8.1", "6.2": "8", "6.1": "7"}
_VER = (  # 앱·브라우저 이름 → 버전 토큰
    ("카카오톡", r"KAKAOTALK (\d+(?:\.\d+)?)"),
    ("Samsung Internet", r"SamsungBrowser/(\d+)"),
    ("Whale", r"Whale/(\d+)"),
    ("Edge", r"Edg(?:A|iOS)?/(\d+)"),
    ("Opera", r"OPR/(\d+)"),
    ("Chrome", r"(?:CriOS|Chrome)/(\d+)"),
    ("Firefox", r"(?:FxiOS|Firefox)/(\d+)"),
    ("Safari", r"Version/(\d+(?:\.\d+)?)"),
)


def _os(ua: str) -> str:
    """운영체제와 버전. iOS 26부터 사파리는 OS 자리를 18_6으로 고정하고 Version/에 실제 버전을
    둔다. Android 기기 모델은 크롬이 'K'로 가리면 뺀다. macOS 버전은 브라우저가 10_15_7로 고정."""
    for dev in ("iPhone", "iPad"):
        if dev in ua:
            m = _IOS.search(ua)
            ver = f"{m.group(1)}.{m.group(2)}" if m and m.group(2) else (m.group(1) if m else "")
            v = re.search(r"Version/(\d+(?:\.\d+)?)", ua)
            if v and float(v.group(1).split(".")[0]) >= 26:
                ver = v.group(1)
            return f"{dev} · iOS {ver}" if ver else dev
    if "Android" in ua:
        m = _ANDROID.search(ua)
        if not m:
            return "Android"
        model = (m.group(2) or "").strip()
        model = model.split(" Build/")[0]
        bad = not model or model in ("K", "wv", "Mobile") or model.startswith(("U", "Linux"))
        return f"Android {m.group(1)}" + ("" if bad else f" · {model[:30]}")
    if "Windows" in ua:
        m = re.search(r"Windows NT (\d+\.\d+)", ua)
        return f"Windows {_WIN.get(m.group(1), m.group(1))}" if m else "Windows"
    if "CrOS" in ua:
        return "ChromeOS"
    if "Macintosh" in ua or "Mac OS X" in ua:
        return "macOS"
    if "Linux" in ua:
        return "Linux"
    return ""


def _with_version(browser: str, ua: str) -> str:
    for name, pat in _VER:
        if name == browser:
            m = re.search(pat, ua)
            return f"{browser} {m.group(1)}" if m else browser
    return browser


def parse_ua(ua: str | None) -> dict:
    """User-Agent → {"label": "iPhone · Safari", "bot": False}. 모르는 것은 "알 수 없음"."""
    ua = (ua or "").strip()
    if not ua:
        return {"label": "알 수 없음(UA 없음)", "bot": True}
    if _BOT.search(ua):
        m = _BOT_NAME.search(ua)
        name = m.group(1) if m else ua.split("/")[0].split()[0]
        return {"label": f"봇 · {name[:40]}", "bot": True}
    os_name = _os(ua)
    browser = next((name for key, name in _APPS if key in ua), "")
    if not browser and "Safari/" in ua and "Version/" in ua:
        browser = "Safari"
    if not browser and os_name.startswith(("iPhone", "iPad")) and "AppleWebKit" in ua:
        browser = "앱 내 브라우저"
    parts = [p for p in (os_name, _with_version(browser, ua)) if p]
    return {"label": " · ".join(parts) if parts else "알 수 없음", "bot": False}


# ---------------------------------------------------------------- 기록 파일

_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def _clean(value: str | None, n: int) -> str:
    return _CTRL.sub(" ", value or "")[:n]


def _append(name: str, entry: dict) -> None:
    global _appends
    with _lock:
        d = _ensure_dir()
        fd = os.open(d / name, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        _appends += 1
        if _appends % TRIM_EVERY == 1:
            _trim(d / name)


def _trim(path: Path, now: datetime | None = None) -> None:
    """마지막 MAX_LINES줄, MAX_DAYS일 안쪽만 남긴다(`_lock` 안에서 부른다)."""
    lines = path.read_text().splitlines()
    cut = ((now or datetime.now(KST)) - timedelta(days=MAX_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    keep = []
    for line in lines:
        try:
            if json.loads(line).get("at", "") >= cut:
                keep.append(line)
        except (ValueError, AttributeError):
            continue
    keep = keep[-MAX_LINES:]
    if len(keep) != len(lines):
        _write_private(path, "".join(x + "\n" for x in keep))


def read_jsonl(name: str) -> list[dict]:
    path = admin_dir() / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:  # 쓰는 중인 마지막 줄
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def access_entry(
    ip: str, ua: str | None, path: str, referer: str | None, kind: str, auth: str | None
) -> dict:
    return {
        "at": now_kst(),
        "ip": ip,
        "ua": _clean(ua, 400),
        "path": _clean(path, 200),
        "ref": _clean(referer, 200),
        "kind": kind,  # page | auth
        "auth": auth,  # None | ok | fail | blocked
    }


def log_access(entry: dict) -> None:
    try:
        _append("access.jsonl", entry)
    except OSError as e:  # 기록 실패로 화면이 안 열리면 안 된다
        log.warning("access log write failed: %s", type(e).__name__)


def is_page(method: str, path: str) -> bool:
    return method == "GET" and (path == "/" or path.endswith(".html"))


def _digest(token: str) -> str:
    # 틀린 토큰도 비밀번호 후보라 메모리에도 원문 대신 해시만 둔다
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def should_log_auth(ip: str, ua: str, area: str, ok: bool, explicit: bool, token: str) -> bool:
    """직접 입력은 늘 남긴다. 저장된 토큰으로 30초마다 도는 성공은 기기별로 6시간에 한 번,
    같은 틀린 토큰은 여러 API로 한꺼번에 와도 1분에 한 번만."""
    if explicit:
        return True
    key = (ip, ua[:200], area) if ok else (ip, _digest(token), "fail")
    gap = SUCCESS_DEDUP_SEC if ok else 60
    t = time.monotonic()
    with _lock:
        if t - _seen_ok.get(key, -1e18) < gap:
            return False
        _seen_ok[key] = t
        if len(_seen_ok) > 5000:
            _seen_ok.clear()
    return True


def note_fail(ip: str, token: str) -> None:
    """서로 다른 틀린 토큰 수를 센다(같은 토큰이 여러 API로 오는 것은 한 번)."""
    t = time.monotonic()
    with _lock:
        seen = _fails.setdefault(ip, {})
        seen[_digest(token)] = t
        for k in [k for k, v in seen.items() if t - v > FAIL_WINDOW_SEC]:
            del seen[k]
        if len(_fails) > 5000:
            _fails.clear()


def blocked(ip: str) -> bool:
    t = time.monotonic()
    with _lock:
        seen = _fails.get(ip) or {}
        return sum(1 for v in seen.values() if t - v <= FAIL_WINDOW_SEC) >= FAIL_LIMIT


def reset_memory() -> None:
    """시험용: 메모리에 든 중복·실패 기록을 비운다."""
    global _appends
    with _lock:
        _seen_ok.clear()
        _fails.clear()
        _appends = 0


# ---------------------------------------------------------------- 위치(IP → 나라·도시·통신사)


def _fetch_geo(ips: list[str]) -> dict[str, dict]:
    body = json.dumps([{"query": ip} for ip in ips]).encode()
    req = urllib.request.Request(
        GEO_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=GEO_TIMEOUT) as resp:
        rows = json.loads(resp.read(1_000_000))
    out = {}
    for r in rows if isinstance(rows, list) else []:
        if isinstance(r, dict) and r.get("status") == "success" and r.get("query") in ips:
            out[r["query"]] = {
                "country": _clean(str(r.get("country") or ""), 60),
                "city": _clean(str(r.get("city") or ""), 60),
                "isp": _clean(str(r.get("isp") or ""), 80),
            }
    return out


def _geo_label(g: dict) -> str:
    place = " ".join(x for x in (g.get("country"), g.get("city")) if x)
    return " · ".join(x for x in (place, g.get("isp")) if x) or "-"


def locate(ips: list[str], fetch=None) -> dict[str, str]:
    """IP별 위치 글자. 캐시(geo.json)에 없거나 오래된 것만 조회하고, 실패하면 "-"."""
    fetch = fetch or _fetch_geo
    path = admin_dir() / "geo.json"
    try:
        cache = json.loads(path.read_text()) if path.exists() else {}
    except ValueError:
        cache = {}
    if not isinstance(cache, dict):
        cache = {}
    now = time.time()
    out: dict[str, str] = {}
    need = []
    for ip in dict.fromkeys(ips):
        if _valid_ip(ip) is None:
            out[ip] = "-"
        elif is_internal(ip):
            out[ip] = "내부"
        else:
            c = cache.get(ip)
            fresh = c and now - c.get("t", 0) < (GEO_TTL_SEC if c.get("ok") else GEO_FAIL_TTL_SEC)
            if fresh:
                out[ip] = _geo_label(c) if c.get("ok") else "-"
            else:
                need.append(ip)
    changed = False
    for i in range(0, min(len(need), GEO_BATCH * GEO_MAX_BATCHES), GEO_BATCH):
        part = need[i : i + GEO_BATCH]
        try:
            got = fetch(part)
        except Exception as e:  # noqa: BLE001 — 위치 조회 실패가 화면을 막으면 안 된다
            log.warning("geo lookup failed: %s", type(e).__name__)
            break
        for ip in part:
            g = got.get(ip)
            cache[ip] = {"t": now, "ok": bool(g), **(g or {})}
            out[ip] = _geo_label(g) if g else "-"
        changed = True
    for ip in need:
        out.setdefault(ip, "-")
    if changed:
        _ensure_dir()
        _write_private(path, json.dumps(cache, ensure_ascii=False))
    return out


# ---------------------------------------------------------------- 조회


def my_ips(rows: list[dict], me: str) -> set[str]:
    """'나'로 보는 IP: 지금 보는 IP + 비밀번호(토큰)를 한 번이라도 통과한 IP. 휴대폰은 IP가
    자주 바뀌어도 저장된 토큰으로 열 때마다 통과 기록이 남는다(사용자 2026-11-23 "나를 제외한")."""
    return {me} | {str(r.get("ip")) for r in rows if r.get("auth") == "ok"}


def access_view(
    rows: list[dict],
    *,
    me: str,
    auth_only: bool,
    hide_me: bool,
    hours: int | None,
    limit: int,
    offset: int,
    now: datetime | None = None,
) -> dict:
    """최신이 위. 걸러낸 줄 중 limit개와 IP별 요약(걸러낸 전체 기준)."""
    if hours:
        cut = ((now or datetime.now(KST)) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        rows = [r for r in rows if str(r.get("at", "")) >= cut]
    if auth_only:
        rows = [r for r in rows if r.get("kind") == "auth"]
    mine = my_ips(rows, me)
    if hide_me:
        rows = [r for r in rows if r.get("ip") not in mine]
    rows = list(reversed(rows))
    summary: dict[str, dict] = {}
    for r in rows:  # 최신부터라 처음 본 시각이 last
        ip = str(r.get("ip", "-"))
        s = summary.get(ip)
        if s is None:
            s = summary[ip] = {"ip": ip, "count": 0, "fails": 0, "first": "", "last": r.get("at")}
            s["device"] = parse_ua(r.get("ua"))["label"]
            s["mine"] = ip in mine
        s["count"] += 1
        s["first"] = r.get("at")
        if r.get("auth") in ("fail", "blocked"):
            s["fails"] += 1
    page = rows[offset : offset + limit]
    out_rows = [
        {
            "at": r.get("at"),
            "ip": r.get("ip"),
            "device": parse_ua(r.get("ua"))["label"],
            "bot": parse_ua(r.get("ua"))["bot"],
            "ua": r.get("ua"),
            "path": r.get("path"),
            "ref": r.get("ref"),
            "kind": r.get("kind"),
            "auth": r.get("auth"),
            "mine": r.get("ip") in mine,
        }
        for r in page
    ]
    ips_summary = sorted(summary.values(), key=lambda s: s["last"] or "", reverse=True)[:200]
    return {
        "me": me,
        "total": len(rows),
        "offset": offset,
        "rows": out_rows,
        "ips": ips_summary,
    }


# ---------------------------------------------------------------- API

router = APIRouter()


def admin_token_ok(token: str) -> bool:
    """어드민은 열람 토큰이 설정돼 있고 맞을 때만. 토큰이 없으면 잠근다(계좌와 다름)."""
    return bool(routes.account_view_token()) and bool(token) and routes._token_ok(token)


def _check(token: str) -> None:
    if not admin_token_ok(token):
        raise HTTPException(status_code=403, detail="invalid token")


def request_ip(request: Request) -> str:
    peer = request.client.host if request.client else None
    return client_ip(peer, request.headers.get("x-forwarded-for"))


def _read_text(path: Path) -> str | None:
    # exists() 뒤 읽기 전에 엔진이 지울 수 있어 바로 읽는다
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def engine_status() -> dict:
    d = bt.DIR
    try:
        state = bt._load(d / "state.json", {})
    except (OSError, ValueError):
        state = {}
    state = state if isinstance(state, dict) else {}
    p = state.get("position")
    pos = None
    if isinstance(p, dict):
        pos = {k: p.get(k) for k in ("symbol", "side", "entry", "stop", "tp", "qty", "opened_at")}
        pos["symbol"] = pos["symbol"] or "BTCUSDT"
        pos["leverage"] = bt.pos_leverage(p)
    runs = [e for e in btc_view.tail_jsonl(d / "events.jsonl", 300) if e.get("event") == "run"]
    last = runs[-1] if runs else None
    halt = _read_text(d / "HALT")
    no_entry = _read_text(d / "NO_ENTRY")
    ms = state.get("last_run_ms")
    return {
        "mode": bt.mode(),
        "halt": halt is not None,
        "halt_note": halt or None,
        "no_entry": None if no_entry is None else (no_entry or "NO_ENTRY 파일 (사유 없음)"),
        "position": pos,
        "last_run_at": (
            datetime.fromtimestamp(ms / 1000, KST).strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(ms, (int, float))
            else None
        ),
        "last_decision": last.get("decision") if last else None,
        "last_halts": last.get("halts") if last else None,
    }


@router.get("/api/admin/status")
def admin_status(x_minerva_token: str = Header("")) -> dict:
    _check(x_minerva_token)
    actions = read_jsonl("actions.jsonl")[-30:]
    return {"engine": engine_status(), "actions": list(reversed(actions))}


_ACTIONS = {
    "halt": ("🛑", "자동매매 정지(HALT)"),
    "resume": ("▶️", "자동매매 재개(HALT 해제)"),
    "clear_no_entry": ("🔓", "신규 진입 정지(NO_ENTRY) 해제"),
}


def _act(request: Request, action: str, token: str) -> dict:
    _check(token)
    ip = request_ip(request)
    d = bt.DIR
    changed = True
    detail = None
    if action == "halt":
        if (d / "HALT").exists():
            changed = False
        else:
            _write_private(d / "HALT", f"{now_kst()} 어드민 탭에서 정지 (IP {ip})\n")
    elif action == "resume":
        changed = _unlink(d / "HALT")
    else:
        detail = _read_text(d / "NO_ENTRY")
        changed = _unlink(d / "NO_ENTRY")
    icon, label = _ACTIONS[action]
    entry = {
        "at": now_kst(),
        "action": action,
        "changed": changed,
        "ip": ip,
        "ua": _clean(request.headers.get("user-agent"), 400),
    }
    if detail:
        entry["detail"] = _clean(detail, 300)
    _append("actions.jsonl", entry)
    if changed:
        text = f"{icon} 어드민에서 {label} — IP {ip} · {parse_ua(entry['ua'])['label']}"
        if action == "halt":
            text += "\n신규 진입만 멈춰요. 보유 포지션·손절·반대 신호 청산은 계속해요."
        elif action == "clear_no_entry" and detail:
            text += f"\n지운 사유: {_clean(detail, 200)}"
        notified = notify.send(text)
    else:
        notified = False
    return {"ok": True, "changed": changed, "notified": notified, "engine": engine_status()}


def _unlink(path: Path) -> bool:
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


@router.post("/api/admin/halt")
def admin_halt(request: Request, x_minerva_token: str = Header("")) -> dict:
    return _act(request, "halt", x_minerva_token)


@router.post("/api/admin/resume")
def admin_resume(request: Request, x_minerva_token: str = Header("")) -> dict:
    return _act(request, "resume", x_minerva_token)


@router.post("/api/admin/clear-no-entry")
def admin_clear_no_entry(request: Request, x_minerva_token: str = Header("")) -> dict:
    return _act(request, "clear_no_entry", x_minerva_token)


@router.get("/api/admin/access")
def admin_access(
    request: Request,
    x_minerva_token: str = Header(""),
    auth_only: bool = False,
    hide_me: bool = False,
    hours: int = Query(0, ge=0, le=24 * MAX_DAYS),
    limit: int = Query(500, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    geo: bool = True,
) -> dict:
    _check(x_minerva_token)
    me = request_ip(request)
    out = access_view(
        read_jsonl("access.jsonl"),
        me=me,
        auth_only=auth_only,
        hide_me=hide_me,
        hours=hours or None,
        limit=limit,
        offset=offset,
    )
    ips = [r["ip"] for r in out["rows"]] + [s["ip"] for s in out["ips"]] + [me]
    places = locate(ips) if geo else {}
    for r in out["rows"]:
        r["place"] = places.get(r["ip"], "-")
    for s in out["ips"]:
        s["place"] = places.get(s["ip"], "-")
    out["me_place"] = places.get(me, "-")
    return out

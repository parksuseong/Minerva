"""연구 규칙 실시간 가상 추적(주문 없음). 2026-11-01부터 앞으로 오는 자료로만 판정한다.

사용자 2026-11-21 16:29 "응": 연구 8차 24시간 방향 모델(LightGBM)을 쓰는 규칙 셋을 사전 등록 그대로
가상으로 따라간다. 과거 구간은 이미 본 자료라 깨끗한 검증은 앞으로의 자료뿐이다.

- V01(10차 #538): 롱만, 확신도 ≥ 0.198, 4시간 결정마다 새 거래 1개(안 든 종목 중 확신도
  최고), 손절 5%, 24시간 청산, 위험 0.5%(= 평가액 0.1배 명목).
- V06(10차 #543): 양방향 바구니, 결정마다 확신도 ≥ 0.198인 안 든 종목 위 5개, 한 칸 = 0.1배/5.
- F(참고, 부록 F 동결 규칙 = 10차 X00): 양방향 한 종목, 나머지는 V01과 같음.
- 공통: 상위 30 범위 U(m)(전달 거래대금), 자격 = 상장 90일 이상 + 전달 30일 중앙 거래대금 ≥ $1억,
  진입 = 결정 시각 T의 1시간봉 시가, 손절은 1시간봉마다(시가가 이미 넘으면 그 시가), 수수료+미끄러짐
  편도 0.07%(왕복 0.14%), 실제 펀딩(T < 정산 ≤ 청산 시), 장부는 시간마다 평가·총노출 3배 상한.
- 모델은 달마다 연구 `fitf.py`와 같은 절차로 다시 맞춘다(확장 창, 잎 7·31 중 검증
  로그손실, 조기 종료 30, seed 7, 스레드 1, max_bin 63).
  특징 65개는 `edge_features`(연구 코드 그대로).

결정 T는 T시 1시간봉이 마감한 뒤(T+1시 이후) 처리한다. 특징은 T-1시까지(배열을 T에서
자름), 진입가는 T시 시가라 늦게 처리해도 값은 같고 앞을 보지 않는다. 밀린 결정은 같은
코드로 차례로 따라잡는다(백필).

자료: 바이낸스 USDT-M 공개 GET(klines·fundingRate·exchangeInfo)과 data.binance.vision 월 묶음만.
키·주문·계좌 조회 없음. 같은 IP를 실전 엔진이 쓰므로 0.3초 간격, 엔진 타이머 앞뒤는 비켜 간다.

폴더 `~/.minerva/edge_paper/`(MINERVA_EDGE_PAPER_DIR): `h1/<종목>.npz` 1시간봉·펀딩,
`panel.npz` 일 거래대금, `universe.json`, `rows/` 학습 행, `models/` 달 모델,
`state.json`, `decisions.jsonl`, `trades.jsonl`, `summary.json`(화면 `/api/edge-paper`).

실행: `python -m minerva.edge_paper seed|run|fit <YYYY-MM>|parity|summary`
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import io
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from minerva import edge_features as ef

log = logging.getLogger("minerva.edge_paper")

RESEARCH = Path("~/.minerva/research/edge-2026-11-20").expanduser()
FAPI = "https://fapi.binance.com"
ARCHIVE = "https://data.binance.vision/data/futures/um/monthly"
D0 = date(2020, 1, 1)
START_KEY = "2026-11"
START_H = (date(2026, 11, 1) - D0).days * 24  # 59904
RESEARCH_NH = START_H  # 연구 자료 끝(rows.npz 정답은 T+24 < 이 값만)
CORE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
        "LINKUSDT", "LTCUSDT", "AVAXUSDT"]
INDEXES = {"FOOTBALLUSDT", "BLUEBIRDUSDT", "DOTECOUSDT", "DEFIUSDT", "BTCDOMUSDT"}
U30_STABLE = ("USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI")
PANEL_STABLE = ("USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USDE", "EUR", "GBP")

HOLD = 24
STOP = 0.05
THR = 0.198
COST_RT = 0.0014  # 테이커 0.05% + 미끄러짐 0.02%, 편도 × 2
RISK = 0.005
CAP = 3.0
RULES = {
    "V01": {"k": 1, "long_only": True, "name": "V01 롱만 한 종목",
            "desc": "롱만 · 결정마다 확신도 ≥0.198 중 1개 · 손절 5% · 24시간 · 위험 0.5%"},
    "V06": {"k": 5, "long_only": False, "name": "V06 양방향 바구니",
            "desc": "롱·숏 · 결정마다 확신도 ≥0.198 중 위 5개 · 칸마다 1/5 크기 · 손절 5% · "
                    "24시간"},
    "F": {"k": 1, "long_only": False, "name": "F 동결 규칙(참고)",
          "desc": "롱·숏 · 결정마다 확신도 ≥0.198 중 1개 · 손절 5% · 24시간 · 위험 0.5%"},
}
FIT_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, min_data_in_leaf=400,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                  num_threads=1, verbose=-1, seed=7, max_bin=63)
GAP_S = 0.3
WEIGHT_SOFT = 1000
MAX_LOAD = 2.0
DELIST_WAIT_H = 48


class RateLimited(RuntimeError):
    pass


# ---------------------------------------------------------------- 경로·시간


def base_dir() -> Path:
    return Path(os.environ.get("MINERVA_EDGE_PAPER_DIR", "~/.minerva/edge_paper")).expanduser()


def summary_path() -> Path:
    return base_dir() / "summary.json"


def month_key(hour: int) -> str:
    d = datetime(2020, 1, 1) + timedelta(hours=int(hour))
    return f"{d.year:04d}-{d.month:02d}"


def month_start(key: str) -> int:
    y, m = map(int, key.split("-"))
    return (date(y, m, 1) - D0).days * 24


def shift_month(key: str, n: int) -> str:
    y, m = map(int, key.split("-"))
    m += n
    while m < 1:
        m += 12
        y -= 1
    while m > 12:
        m -= 12
        y += 1
    return f"{y:04d}-{m:02d}"


def iso(hour: int) -> str:
    return (datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(hours=int(hour))).isoformat()


def now_hour(now: float | None = None) -> int:
    t = time.time() if now is None else now
    return int((t * 1000 - ef.T0_MS) // ef.HR_MS)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ---------------------------------------------------------------- 공개 시세


def _engine_window(now: float) -> bool:
    sec = now % 900
    return sec <= 80 or sec >= 860


def get_json(url: str, timeout: float = 30, retries: int = 3):
    """공개 GET. 바이낸스 API면 엔진 타이머 앞뒤를 비키고 무게 헤더를 본다. 429·418은 바로 멈춤."""
    api = url.startswith(FAPI)
    for k in range(retries):
        while api and _engine_window(time.time()):
            time.sleep(5)
        req = urllib.request.Request(url, headers={"User-Agent": "minerva-edge-paper"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
                data = json.load(r)
            if api:
                time.sleep(GAP_S)
                if used > WEIGHT_SOFT:
                    time.sleep(61 - time.time() % 60)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                raise RateLimited(f"HTTP {e.code}") from None
            if e.code < 500 or k == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if k == retries - 1:
                raise
        time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def get_bytes(url: str, timeout: float = 60) -> bytes | None:
    """data.binance.vision 월 묶음(없으면 None)."""
    for k in range(3):
        req = urllib.request.Request(url, headers={"User-Agent": "minerva-edge-paper"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == 2:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if k == 2:
                raise
        time.sleep(3)
    return None


def _zip_rows(blob: bytes) -> list[list[str]]:
    z = zipfile.ZipFile(io.BytesIO(blob))
    out = []
    for n in z.namelist():
        for line in z.read(n).decode().splitlines():
            if line and line[0].isdigit():
                out.append(line.split(","))
    return out


# ---------------------------------------------------------------- 1시간봉 저장소


class Store:
    """종목별 1시간봉(o,h,l,c,q; 0번 = 2020-01-01 00시, 없는 시간 nan)과 펀딩(ft ms, fr)."""

    def __init__(self, root: Path | None = None):
        self.root = (root or base_dir()) / "h1"
        self.cache: dict[str, dict] = {}

    def path(self, sym: str) -> Path:
        return self.root / f"{sym}.npz"

    def has(self, sym: str) -> bool:
        return sym in self.cache or self.path(sym).exists()

    def load(self, sym: str) -> dict:
        if sym not in self.cache:
            z = np.load(self.path(sym))
            self.cache[sym] = {k: z[k] for k in z.files}
        return self.cache[sym]

    def save(self, sym: str, d: dict) -> None:
        save_npz(self.path(sym), **d)
        self.cache[sym] = d

    def last_hour(self, sym: str) -> int:
        """마지막으로 값이 있는 시간 번호(없으면 -1)."""
        d = self.load(sym)
        ok = np.flatnonzero(~np.isnan(d["c"]))
        return int(ok[-1]) if len(ok) else -1

    def arrays(self, sym: str, nh: int, mask_last: bool = False):
        """길이 nh로 자르거나 nan으로 채운 (o,h,l,c,q,fh,fr). 펀딩은 정산 시간 < nh만.
        mask_last면 nh-1시를 지운다(결정 T = nh-1에 아직 모르는 봉)."""
        d = self.load(sym)
        out = []
        for k in "ohlcq":
            a = d[k][:nh]
            if len(a) < nh:
                a = np.concatenate([a, np.full(nh - len(a), np.nan)])
            else:
                a = a.copy()
            if mask_last:
                a[nh - 1] = np.nan
            out.append(a)
        fh = ((d["ft"].astype(np.int64) - ef.T0_MS) // ef.HR_MS).astype(np.int64)
        keep = fh < nh
        return (*out, fh[keep], d["fr"][keep].astype(float))


def _empty_sym() -> dict:
    return {k: np.zeros(0) for k in "ohlcq"} | {"ft": np.zeros(0, np.int64), "fr": np.zeros(0)}


def _put_klines(d: dict, rows: list) -> dict:
    """바이낸스 kline 행(시작 ms, o, h, l, c, 수량, 끝, 거래대금, …)을 시간 격자에 넣는다."""
    if not rows:
        return d
    idx = np.array([(int(r[0]) - ef.T0_MS) // ef.HR_MS for r in rows], np.int64)
    ok = idx >= 0
    if not ok.any():
        return d
    n = max(len(d["c"]), int(idx[ok].max()) + 1)
    out = {}
    for j, k in ((1, "o"), (2, "h"), (3, "l"), (4, "c"), (7, "q")):
        a = np.full(n, np.nan)
        a[:len(d[k])] = d[k]
        a[idx[ok]] = np.array([float(r[j]) for r, g in zip(rows, ok) if g])
        out[k] = a
    out["ft"], out["fr"] = d["ft"], d["fr"]
    return out


def _put_funding(d: dict, pairs: list[tuple[int, float]]) -> dict:
    ft = list(d["ft"].astype(np.int64))
    fr = list(d["fr"].astype(float))
    last = ft[-1] if ft else -1
    for t, r in sorted(pairs):
        if t > last:
            ft.append(int(t))
            fr.append(float(r))
            last = t
    return dict(d, ft=np.array(ft, np.int64), fr=np.array(fr, float))


def fetch_klines_api(sym: str, start_h: int, end_h: int, get=get_json) -> list:
    """start_h..end_h-1 시의 마감한 1시간봉(1500개씩)."""
    out = []
    t = start_h
    while t < end_h:
        url = (f"{FAPI}/fapi/v1/klines?symbol={sym}&interval=1h&limit=1500"
               f"&startTime={ef.T0_MS + t * ef.HR_MS}")
        page = get(url)
        if not isinstance(page, list) or not page:
            break
        out.extend(r for r in page if (int(r[0]) - ef.T0_MS) // ef.HR_MS < end_h)
        nxt = (int(page[-1][0]) - ef.T0_MS) // ef.HR_MS + 1
        if nxt <= t or len(page) < 1500:
            break
        t = nxt
    return out


def fetch_funding_api(sym: str, start_ms: int, get=get_json) -> list[tuple[int, float]]:
    out = []
    t = start_ms
    for _ in range(50):
        page = get(f"{FAPI}/fapi/v1/fundingRate?symbol={sym}&limit=1000&startTime={t}")
        if not isinstance(page, list) or not page:
            break
        out.extend((int(r["fundingTime"]), float(r["fundingRate"])) for r in page)
        if len(page) < 1000:
            break
        t = int(page[-1]["fundingTime"]) + 1
    return out


def update_symbol(store: Store, sym: str, upto_h: int, start_key: str | None = None,
                  get=get_json, get_zip=get_bytes) -> None:
    """저장소 끝(없으면 start_key 달 1일)부터 upto_h-1시까지 잇는다. 45일 넘게 빈 앞쪽 완결 달은
    월 묶음으로, 나머지는 API로."""
    d = store.load(sym) if store.has(sym) else _empty_sym()
    last = int(np.flatnonzero(~np.isnan(d["c"]))[-1]) if np.any(~np.isnan(d["c"])) else -1
    t = last + 1 if last >= 0 else month_start(start_key or START_KEY)
    cur_key = month_key(upto_h)
    while upto_h - t > 45 * 24 and month_key(t) < cur_key:
        key = month_key(t)
        y, m = key.split("-")
        rows = []
        blob = get_zip(f"{ARCHIVE}/klines/{sym}/1h/{sym}-1h-{y}-{m}.zip")
        if blob:
            rows = _zip_rows(blob)
        d = _put_klines(d, [r for r in rows if (int(r[0]) - ef.T0_MS) // ef.HR_MS >= t])
        t = month_start(shift_month(key, 1))
    if t < upto_h:
        d = _put_klines(d, fetch_klines_api(sym, t, upto_h, get=get))
    if len(d["ft"]):
        fstart = int(d["ft"][-1]) + 1
    else:
        fstart = ef.T0_MS + month_start(start_key or month_key(t)) * ef.HR_MS
    d = _put_funding(d, fetch_funding_api(sym, fstart, get=get))
    store.save(sym, d)


# ---------------------------------------------------------------- 일 거래대금 패널·범위


class Panel:
    """종목별 일 거래대금 Q(0번 = 2020-01-01, 없는 날 nan)와 첫 일봉 번호 first."""

    def __init__(self, syms, first, q):
        self.syms = [str(s) for s in syms]
        self.idx = {s: i for i, s in enumerate(self.syms)}
        self.first = np.asarray(first, np.int64)
        self.q = np.asarray(q, float)

    @classmethod
    def load(cls, path: Path | None = None) -> Panel:
        z = np.load(path or base_dir() / "panel.npz")
        return cls(z["syms"], z["first"], z["Q"])

    def save(self, path: Path | None = None) -> None:
        save_npz(path or base_dir() / "panel.npz", syms=np.array(self.syms), first=self.first,
                 Q=self.q)

    @property
    def days(self) -> int:
        """마지막 일봉이 있는 날 + 1."""
        ok = np.flatnonzero(np.any(~np.isnan(self.q), axis=0))
        return int(ok[-1]) + 1 if len(ok) else 0

    def add(self, sym: str, first: int) -> int:
        if sym not in self.idx:
            self.idx[sym] = len(self.syms)
            self.syms.append(sym)
            self.first = np.append(self.first, first)
            self.q = np.vstack([self.q, np.full((1, self.q.shape[1]), np.nan)])
        return self.idx[sym]

    def put(self, sym: str, day: int, value: float) -> None:
        if day >= self.q.shape[1]:
            self.q = np.hstack([self.q, np.full((self.q.shape[0], day + 1 - self.q.shape[1]),
                                                np.nan)])
        self.q[self.idx[sym], day] = value

    def age_ok(self, sym: str, key: str) -> bool:
        i = self.idx.get(sym)
        return i is not None and self.first[i] <= month_start(key) // 24 - 90

    def vol_ok(self, sym: str, key: str) -> bool:
        i = self.idx.get(sym)
        if i is None:
            return False
        d1 = month_start(key) // 24
        q = self.q[i, max(0, d1 - 30):d1]
        return len(q) > 0 and bool(np.median(q) >= 1e8)


def compute_u30(panel: Panel, key: str) -> list[str]:
    """연구 3차 U30: 코어 10개·지수·스테이블 뺀 코인 무기한 중 전달 30일 거래대금
    상위 30(상장 60일+)."""
    d1 = month_start(key) // 24
    excl = INDEXES | set(CORE)
    cand = [i for i, s in enumerate(panel.syms) if s not in excl and not s.startswith(U30_STABLE)]
    v = np.nan_to_num(panel.q[cand, max(0, d1 - 30):d1]).sum(axis=1)
    ok = (panel.first[cand] <= d1 - 60) & (v > 0)
    order = [k for k in np.argsort(-v) if ok[k]][:30]
    return [panel.syms[cand[k]] for k in order]


def compute_universe(panel: Panel, key: str, u30: list[str]) -> list[str]:
    """연구 8차 U(m): 코어 10 ∪ U30(m) 중 전달 30일 거래대금 합 상위 30."""
    d1 = month_start(key) // 24
    v = [(panel.q[panel.idx[s], max(0, d1 - 30):d1].sum(), s) for s in CORE + u30
         if s in panel.idx]
    return [s for _, s in sorted(v, reverse=True)[:30]]


def _coin_perps(get=get_json) -> list[dict]:
    info = get(f"{FAPI}/fapi/v1/exchangeInfo")
    out = []
    for x in info.get("symbols", []):
        s = x.get("symbol", "")
        if (x.get("contractType") == "PERPETUAL" and x.get("quoteAsset") == "USDT"
                and x.get("underlyingType") == "COIN" and s not in INDEXES
                and not s.startswith(PANEL_STABLE)):
            out.append(x)
    return out


def update_panel(panel: Panel, upto_day: int, get=get_json) -> int:
    """upto_day 전날까지 일 거래대금을 API로 채운다(지금 상장된 코인 무기한, 최근 40일).
    돌려줌: 받은 종목 수."""
    n = 0
    start_day = max(0, upto_day - 40)
    for x in _coin_perps(get=get):
        s = x["symbol"]
        if s not in panel.idx:
            first_rows = get(f"{FAPI}/fapi/v1/klines?symbol={s}&interval=1d&limit=1&startTime=0")
            if not first_rows:
                continue
            panel.add(s, (int(first_rows[0][0]) - ef.T0_MS) // 86_400_000)
        rows = get(f"{FAPI}/fapi/v1/klines?symbol={s}&interval=1d&limit=45"
                   f"&startTime={ef.T0_MS + start_day * 86_400_000}")
        for r in rows or []:
            day = (int(r[0]) - ef.T0_MS) // 86_400_000
            if day < upto_day:
                panel.put(s, day, float(r[7]))
        n += 1
    return n


def universe_file() -> Path:
    return base_dir() / "universe.json"


def ensure_universe(key: str, panel: Panel, get=get_json) -> list[str]:
    """U(key)를 돌려준다. 없으면 패널(전달 마지막 날까지 필요)로 만들어 적는다."""
    uf = read_json(universe_file(), {"u30": {}, "U": {}})
    if key in uf["U"]:
        return uf["U"][key]
    d1 = month_start(key) // 24
    if panel.days < d1:
        log.info("panel %d < %d → API로 채움", panel.days, d1)
        update_panel(panel, d1, get=get)
        panel.save()
    u30 = compute_u30(panel, key)
    uf["u30"][key] = u30
    uf["U"][key] = compute_universe(panel, key, u30)
    write_json(universe_file(), uf)
    return uf["U"][key]


def history_start(sym: str, key: str) -> str:
    """연구 3차 받기 규칙: 처음 U30에 든 달 3달 전(2019-09 이후)부터."""
    uf = read_json(universe_file(), {"u30": {}})
    months = sorted(k for k, v in uf.get("u30", {}).items() if sym in v) or [key]
    return max("2020-01", shift_month(months[0], -3))


# ---------------------------------------------------------------- 특징 행


def feature_rows(store: Store, members: list[str], nh: int, t_lo: int, t_hi: int,
                 mask_last: bool, labels: bool = False):
    """결정 시각 T ∈ [t_lo, t_hi](4의 배수)의 U 행. 배열은 nh시 앞에서 자른다.
    돌려줌: dict(t=시간 번호, sym, X, ret24(labels일 때))."""
    feats = {}
    for p, s in (("btc", "BTCUSDT"), ("eth", "ETHUSDT")):
        _, f, _ = ef.symbol_feats(*store.arrays(s, nh, mask_last), nh)
        feats[p] = ef.market_feats(f, p)
    grid = np.arange(0, nh, 4)
    sel = (grid >= t_lo) & (grid <= t_hi)
    per, rets = {}, {}
    for s in sorted(members):
        if not store.has(s):
            continue
        arr = store.arrays(s, nh, mask_last)
        _, f, valid = ef.symbol_feats(*arr, nh)
        idx = np.flatnonzero(sel & valid)
        per[s] = (idx, f)
        if labels:
            rets[s] = ef.label_ret(arr[0], arr[3], grid[idx], HOLD, nh)
    ti, syms, x = ef.assemble(per, feats, sorted(per))
    out = {"t": ti.astype(np.int64) * 4, "sym": syms, "X": x}
    if labels:
        # assemble은 (시각, 종목) 순으로 섞으므로 같은 순서로 정답을 맞춘다
        lookup = {(s, int(t)): r for s in rets for t, r in zip(per[s][0] * 4, rets[s])}
        out["ret24"] = np.array([lookup[(s, int(t))] for s, t in zip(syms, out["t"])])
    return out


# ---------------------------------------------------------------- 학습 행·모델


def rows_dir() -> Path:
    return base_dir() / "rows"


def models_dir() -> Path:
    return base_dir() / "models"


def build_month_rows(store: Store, key: str, members: list[str]) -> Path:
    """달 key의 학습 행(특징 + 24시간 정답)을 rows/<key>.npz로. 다음 달 첫 시 봉이 있어야 한다."""
    nxt = month_start(shift_month(key, 1))
    r = feature_rows(store, members, nxt + 1, month_start(key), nxt - 4, mask_last=False,
                     labels=True)
    p = rows_dir() / f"{key}.npz"
    save_npz(p, X=r["X"], t=r["t"], sym=np.array(r["sym"], dtype=str), ret24=r["ret24"])
    return p


def load_training(store: Store, upto_key: str):
    """연구 rows(2026-10까지) + 그 뒤 달 행(upto_key 전달까지). 연구 끝에서 정답이 비었던
    마지막 날 행은 저장소 봉으로 채운다(연구 build와 같은 식)."""
    z = np.load(rows_dir() / "base.npz")
    x, th, ret = z["X"], z["t"].astype(np.int64), z["ret24"].astype(float)
    syms = [str(s) for s in z["syms"]]
    si = z["s"].astype(np.int64)
    fill = np.flatnonzero(np.isnan(ret) & (th + HOLD >= RESEARCH_NH))
    for s in np.unique(si[fill]):
        rr = fill[si[fill] == s]
        sym = syms[s]
        if not store.has(sym):
            continue
        nh = int(th[rr].max()) + HOLD + 1
        o, _h, _l, c, _q, _fh, _fr = store.arrays(sym, nh)
        if np.isnan(o[nh - 1]) and np.isnan(c[nh - 1]):
            continue
        ret[rr] = ef.label_ret(o, c, th[rr], HOLD, nh)
    xs, ts, rs = [x], [th], [ret]
    key = START_KEY
    while key < upto_key:
        p = rows_dir() / f"{key}.npz"
        if p.exists():
            m = np.load(p)
            xs.append(m["X"])
            ts.append(m["t"].astype(np.int64))
            rs.append(m["ret24"].astype(float))
        key = shift_month(key, 1)
    if len(xs) > 1:
        x = np.concatenate(xs)
        th = np.concatenate(ts)
        ret = np.concatenate(rs)
    del xs, ts, rs, z
    gc.collect()
    return x, th, ret


def fit_month(store: Store, key: str, force: bool = False) -> Path:
    """연구 fitf.py와 같은 절차로 달 key 모델을 맞춰 models/H24G_<key>.txt로."""
    import lightgbm as lgb

    out = models_dir() / f"H24G_{key}.txt"
    if out.exists() and not force:
        return out
    x, th, ret = load_training(store, key)
    mk = np.array([month_key(t) for t in th])
    lab = np.where(ret > COST_RT, 2, np.where(ret < -COST_RT, 0, 1))
    has = ~np.isnan(ret)
    sm = month_start(key)
    pk = shift_month(key, -1)
    sv = month_start(pk)
    va = np.flatnonzero((mk == pk) & (th + HOLD <= sm) & has)
    tr = np.flatnonzero((th + HOLD <= sv - HOLD) & has & (mk >= "2020-03"))
    best = None
    for nl in (7, 31):
        prm = dict(FIT_PARAMS, num_leaves=nl)
        dtr = lgb.Dataset(x[tr], lab[tr], free_raw_data=True)
        dva = lgb.Dataset(x[va], lab[va], reference=dtr)
        m = lgb.train(prm, dtr, 300, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        ll = m.best_score["valid_0"]["multi_logloss"]
        if best is None or ll < best[0]:
            best = (ll, nl, m)
        del dtr, dva
        gc.collect()
    ll, nl, m = best
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    m.save_model(str(tmp), num_iteration=m.best_iteration)
    os.replace(tmp, out)
    meta = read_json(models_dir() / "meta.json", {})
    meta[key] = {"num_leaves": nl, "best_iteration": int(m.best_iteration),
                 "val_logloss": float(ll), "n_train": int(len(tr)), "n_val": int(len(va)),
                 "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_json(models_dir() / "meta.json", meta)
    log.info("fit %s leaves %d iters %d ll %.4f train %d val %d", key, nl, m.best_iteration,
             ll, len(tr), len(va))
    return out


_MODELS: dict = {}


def model_for(key: str):
    import lightgbm as lgb

    if key not in _MODELS:
        _MODELS[key] = lgb.Booster(model_file=str(models_dir() / f"H24G_{key}.txt"))
    return _MODELS[key]


# ---------------------------------------------------------------- 거래 경로·선택


def price_arrays(store: Store, sym: str, nh: int):
    """연구 book10.px와 같은 손절 판정용 배열: 종가는 앞값으로 채우고 고저 빈칸은 종가."""
    o, h, low, c, _q, fh, fr = store.arrays(sym, nh)
    c = ef.ffill(c)
    h = np.where(np.isnan(h), c, h)
    low = np.where(np.isnan(low), c, low)
    order = np.argsort(fh)
    return o, h, low, c, fh[order], np.cumsum(np.insert(fr[order], 0, 0.0))


def path_exit(o, h, low, c, t: int, d: int, horizon: int, hold: int = HOLD, stop: float = STOP):
    """연구 book10.path. horizon(마지막으로 아는 시간)을 넘어야 결론 나면 None.
    돌려줌 (총수익(방향 반영), 청산까지 시간 수, 손절 여부)."""
    e = float(o[t])
    sp = e * (1 - d * stop)
    for j in range(hold):
        k = t + j
        if k > horizon:
            return None
        if np.isnan(h[k]):
            continue
        if j > 0 and ((d > 0 and o[k] <= sp) or (d < 0 and o[k] >= sp)):
            return d * (float(o[k]) / e - 1), j, True
        if (d > 0 and low[k] <= sp) or (d < 0 and h[k] >= sp):
            return -stop, j + 1, True
    if t + hold > horizon:
        return None
    xp = float(o[t + hold]) if not np.isnan(o[t + hold]) else float(c[t + hold - 1])
    return d * (xp / e - 1), hold, False


def select(rows: list[dict], held: dict[str, bool], rule: dict) -> list[dict]:
    """한 결정 시각의 진입(연구 book10.select): 확신도 내림차순, 든 종목 건너뜀, 0.198 아래면 끝."""
    cands = [r for r in rows if r["el"] and (r["s"] > 0 or not rule["long_only"])]
    cands.sort(key=lambda r: (-r["conf"], r["sym"]))
    out = []
    for r in cands:
        if len(out) >= rule["k"]:
            break
        if held.get(r["sym"]):
            continue
        if r["conf"] < THR:
            break
        if r.get("entry") is None:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------- 장부(연구 book10.sim)


def simulate(trades: list[dict], prices: dict, rule: dict, t0: int, horizon: int) -> dict:
    """시간마다 평가: 위험 0.5%/손절 5% = 평가액 0.1배(바구니는 칸마다 1/k), 총노출 3배 상한,
    교차 마진 청산 점검. 열린 거래는 마지막 종가로 평가. 돌려줌 일 평가액·지금 평가액·최대 낙폭."""
    starts: dict[int, list] = {}
    for tr in trades:
        starts.setdefault(tr["t"], []).append(tr)
    eq = 1.0
    op: list[dict] = []
    daily: list[tuple[int, float]] = []
    liq = False
    for t in range(t0, horizon + 1):
        keep = []
        for p in op:
            if p["x"] is not None and p["x"] <= t:
                eq += p["n"] * p["net"]
            else:
                keep.append(p)
        op = keep
        if t % 24 == 0:
            daily.append((t // 24, eq + sum(p["n"] * p["d"] * (float(p["c"][t - 1]) / p["e"] - 1)
                                             for p in op if t - 1 >= p["t"])))
        if liq:
            continue
        for tr in starts.get(t, ()):
            if eq <= 0:
                continue
            want = RISK / STOP * eq / rule["k"]
            gross = sum(p["n"] for p in op)
            n = min(want, CAP * eq - gross)
            if n < 0.1 * want:
                continue
            o, h, low, c = prices[tr["sym"]][:4]
            op.append({"t": t, "x": tr.get("x"), "d": tr["d"], "n": n, "e": float(o[t]),
                       "c": c, "h": h, "l": low,
                       "net": tr["net"] if tr.get("net") is not None else 0.0, "id": tr["id"]})
        if op:
            gross = sum(p["n"] for p in op)
            worst = sum(p["n"] * p["d"] * (float(p["l"][t] if p["d"] > 0 else p["h"][t])
                                           / p["e"] - 1) for p in op)
            if eq + worst <= 0.005 * gross:
                liq = True
                eq = 0.0
                op = []
    mark_t = horizon + 1
    unreal = {}
    for p in op:
        if mark_t - 1 >= p["t"]:
            u = p["n"] * p["d"] * (float(p["c"][mark_t - 1]) / p["e"] - 1)
            unreal[p["id"]] = (u, p["n"])
    mark = eq + sum(u for u, _ in unreal.values())
    vals = [v for _, v in daily] + [mark]
    peak = np.maximum.accumulate(np.array(vals)) if vals else np.array([1.0])
    mdd = float(np.max(1 - np.array(vals) / np.maximum(peak, 1e-12))) if vals else 0.0
    return {"daily": daily, "equity": eq, "mark": mark, "maxdd": mdd, "liq": liq,
            "open": unreal}


# ---------------------------------------------------------------- 상태·실행


def state_path() -> Path:
    return base_dir() / "state.json"


def load_state() -> dict:
    return read_json(state_path(), None) or {"last_t": START_H - 4, "trades": [], "next_id": 1}


def held_at(trades: list[dict], rule: str, t: int) -> dict[str, bool]:
    """t에 들고 있는 종목: 열린 거래(결론 전) 또는 청산 시각 > t."""
    held = {}
    for tr in trades:
        if tr["rule"] != rule or tr["t"] > t:
            continue
        if tr.get("x") is None or tr["x"] > t:
            held[tr["sym"]] = True
    return held


def horizon_for(store: Store, sym: str, global_h: int) -> int:
    """그 종목의 마지막 아는 시간. 다른 종목은 이어지는데 48시간 넘게 봉이 없으면 상장폐지로 보고
    연구처럼 마지막 종가로 채운 채 전체 끝까지 간다."""
    last = store.last_hour(sym)
    return global_h if global_h - last > DELIST_WAIT_H else last


def resolve_trades(store: Store, trades: list[dict], global_h: int) -> int:
    """열린 거래 중 알 수 있는 것을 닫는다(청산 시각·순수익 = 총수익 − 펀딩 − 0.14%)."""
    n = 0
    for tr in trades:
        if tr.get("x") is not None:
            continue
        hz = horizon_for(store, tr["sym"], global_h)
        nh = max(hz, tr["t"]) + 2
        o, h, low, c, fh, cs = price_arrays(store, tr["sym"], max(nh, tr["t"] + HOLD + 1))
        res = path_exit(o, h, low, c, tr["t"], tr["d"], hz)
        if res is None:
            continue
        g, xo, stopped = res
        a = np.searchsorted(fh, tr["t"], side="right")
        b = np.searchsorted(fh, tr["t"] + xo, side="right")
        fund = float(cs[b] - cs[a])
        tr.update(x=tr["t"] + xo, gross=g, fund=tr["d"] * fund, stopped=bool(stopped),
                  net=g - tr["d"] * fund - COST_RT, exit_px=tr["entry"] * (1 + tr["d"] * g))
        n += 1
        append_jsonl(base_dir() / "trades.jsonl",
                     {"ev": "exit", "id": tr["id"], "rule": tr["rule"], "sym": tr["sym"],
                      "exit_at": iso(tr["x"]), "stopped": tr["stopped"],
                      "gross": round(g, 6), "fund": round(tr["fund"], 6),
                      "net": round(tr["net"], 6)})
    return n


def members_ready(store: Store, members: list[str], t: int, global_h: int) -> bool:
    """범위 종목이 모두 t시 봉까지 받아졌나(받기 실패로 행이 빠진 채 결정하지 않게).
    48시간 넘게 끊긴 종목(상장폐지)은 기다리지 않는다."""
    for s in members:
        last = store.last_hour(s) if store.has(s) else -1
        if last < t and global_h - last <= DELIST_WAIT_H:
            return False
    return True


def decide(store: Store, t: int, members: list[str], panel: Panel) -> list[dict]:
    """결정 시각 t의 행: 특징(배열은 t에서 자름, t시 봉은 지움) → 모델 → 확신도·자격."""
    key = month_key(t)
    r = feature_rows(store, members, t + 1, t, t, mask_last=True)
    if not len(r["t"]):
        return []
    p = model_for(key).predict(r["X"])
    out = []
    for sym, x_p in zip(r["sym"], p):
        s = float(x_p[2] - x_p[0])
        o = store.load(sym)["o"]
        entry = float(o[t]) if t < len(o) and not np.isnan(o[t]) else None
        out.append({"sym": str(sym), "s": s, "conf": abs(s),
                    "el": panel.age_ok(sym, key) and panel.vol_ok(sym, key), "entry": entry})
    return out


def run(fetch: bool = True, max_steps: int | None = None, get=get_json,
        now: float | None = None) -> dict:
    """밀린 결정 시각을 차례로 처리하고 summary.json을 다시 쓴다."""
    root = base_dir()
    store = Store()
    panel = Panel.load()
    st = load_state()
    nh_now = now_hour(now)
    last_closed = nh_now - 1
    pending = list(range(st["last_t"] + 4, last_closed + 1, 4))
    if max_steps is not None:
        pending = pending[:max_steps]
    months = sorted({month_key(t) for t in pending})
    members = {k: ensure_universe(k, panel, get=get) for k in months}
    open_syms = {tr["sym"] for tr in st["trades"] if tr.get("x") is None}
    need = set(open_syms) | {"BTCUSDT", "ETHUSDT"}
    for k in months:
        need |= set(members[k])
        if not (models_dir() / f"H24G_{k}.txt").exists():
            # 새 달 모델의 검증 달 정답(전달 마지막 날 → 이번 달 첫 시)에 전달 범위 종목 봉이 필요
            need |= set(ensure_universe(shift_month(k, -1), panel, get=get))
    ensure_symbols(store, need)
    if fetch:
        for s in sorted(need):
            k = months[0] if months else month_key(last_closed)
            try:
                update_symbol(store, s, last_closed + 1, start_key=history_start(s, k), get=get)
            except RateLimited:
                raise
            except Exception as e:  # 한 종목 실패는 그 종목만 빠진다(다음 실행에 다시)
                log.warning("update %s 실패: %r", s, e)
    global_h = store.last_hour("BTCUSDT")
    for k in months:
        if not (models_dir() / f"H24G_{k}.txt").exists():
            pk = shift_month(k, -1)
            if pk >= START_KEY and not (rows_dir() / f"{pk}.npz").exists():
                if global_h < month_start(k):
                    break
                build_month_rows(store, pk, ensure_universe(pk, panel, get=get))
            if global_h < month_start(k):
                break
            fit_month(store, k)
    resolve_trades(store, st["trades"], global_h)
    done = 0
    for t in pending:
        if t > global_h or not (models_dir() / f"H24G_{month_key(t)}.txt").exists():
            break
        if not members_ready(store, members[month_key(t)], t, global_h):
            log.info("%s: 범위 종목 봉이 아직 다 안 옴 → 다음 실행에", iso(t))
            break
        rows = decide(store, t, members[month_key(t)], panel)
        entries = {}
        for rk, rule in RULES.items():
            picks = select(rows, held_at(st["trades"], rk, t), rule)
            entries[rk] = [p["sym"] for p in picks]
            for p in picks:
                d = 1 if p["s"] > 0 else -1
                tr = {"id": st["next_id"], "rule": rk, "sym": p["sym"], "d": d, "t": t,
                      "conf": round(p["conf"], 4), "entry": p["entry"], "x": None}
                st["next_id"] += 1
                st["trades"].append(tr)
                append_jsonl(root / "trades.jsonl",
                             {"ev": "entry", "id": tr["id"], "rule": rk, "sym": tr["sym"],
                              "dir": "long" if d > 0 else "short", "at": iso(t),
                              "entry": tr["entry"], "conf": tr["conf"]})
        top = sorted(rows, key=lambda r: -r["conf"])[:10]
        append_jsonl(root / "decisions.jsonl", {
            "t": iso(t), "model": month_key(t), "rows": len(rows),
            "eligible": sum(r["el"] for r in rows),
            "top": [{"sym": r["sym"], "s": round(r["s"], 4), "el": r["el"]} for r in top],
            "entries": entries})
        resolve_trades(store, st["trades"], global_h)
        st["last_t"] = t
        write_json(state_path(), st)
        done += 1
    resolve_trades(store, st["trades"], global_h)
    write_json(state_path(), st)
    summ = build_summary(store, st, global_h)
    write_json(summary_path(), summ)
    return {"processed": done, "pending_left": len(pending) - done, "horizon": iso(global_h)}


# ---------------------------------------------------------------- 요약(화면)


def build_summary(store: Store, st: dict, global_h: int) -> dict:
    out = {"status": "ok", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "start": iso(START_H), "data_until": iso(global_h + 1),
           "last_decision": iso(st["last_t"]) if st["last_t"] >= START_H else None,
           "rules": []}
    prices = {}
    for tr in st["trades"]:
        if tr["sym"] not in prices:
            prices[tr["sym"]] = price_arrays(store, tr["sym"], global_h + HOLD + 2)
    for rk, rule in RULES.items():
        trs = sorted((tr for tr in st["trades"] if tr["rule"] == rk), key=lambda r: (r["t"],
                                                                                      r["id"]))
        sim = simulate(trs, prices, rule, START_H, global_h)
        closed = [tr for tr in trs if tr.get("x") is not None]
        nets = np.array([tr["net"] for tr in closed]) if closed else np.zeros(0)
        open_rows = []
        for tr in trs:
            if tr.get("x") is not None:
                continue
            c = prices[tr["sym"]][3]
            last = float(c[min(global_h, len(c) - 1)])
            open_rows.append({
                "sym": tr["sym"], "dir": "롱" if tr["d"] > 0 else "숏", "at": iso(tr["t"]),
                "entry": tr["entry"], "last": last, "conf": tr["conf"],
                "move_pct": round(tr["d"] * (last / tr["entry"] - 1) * 100, 2),
                "stop": round(tr["entry"] * (1 - tr["d"] * STOP), 8),
                "exit_due": iso(tr["t"] + HOLD)})
        last20 = [{"sym": tr["sym"], "dir": "롱" if tr["d"] > 0 else "숏", "at": iso(tr["t"]),
                   "exit_at": iso(tr["x"]), "entry": tr["entry"], "exit": tr.get("exit_px"),
                   "conf": tr["conf"], "stopped": tr["stopped"],
                   "net_pct": round(tr["net"] * 100, 3), "fund_pct": round(tr["fund"] * 100, 4)}
                  for tr in sorted(closed, key=lambda r: -r["x"])[:20]]
        out["rules"].append({
            "key": rk, "name": rule["name"], "desc": rule["desc"],
            "trades": len(trs), "closed": len(closed), "open": len(open_rows),
            "wins": int((nets > 0).sum()),
            "winrate": round(float((nets > 0).mean()) * 100, 1) if len(nets) else None,
            "avg_net_pct": round(float(nets.mean()) * 100, 3) if len(nets) else None,
            "sum_net_pct": round(float(nets.sum()) * 100, 2) if len(nets) else 0.0,
            "longs": sum(tr["d"] > 0 for tr in trs), "stops": sum(bool(tr.get("stopped"))
                                                                  for tr in closed),
            "return_pct": round((sim["mark"] - 1) * 100, 3),
            "realized_return_pct": round((sim["equity"] - 1) * 100, 3),
            "maxdd_pct": round(sim["maxdd"] * 100, 2), "liquidated": sim["liq"],
            "curve": [[iso(d * 24)[:10], round(v, 5)] for d, v in sim["daily"]],
            "open_positions": open_rows, "last_trades": last20})
    return out


# ---------------------------------------------------------------- 씨앗(연구 자료 복사)


def seed(research: Path = RESEARCH, months_back: int = 10) -> dict:
    """연구 자료를 ~/.minerva/edge_paper로 복사: 패널·범위·학습 행·최근 범위 종목 1시간봉."""
    root = base_dir()
    root.mkdir(parents=True, exist_ok=True)
    z = np.load(research / "data5/panel.npz")
    Panel(z["syms"], z["first"], z["Q"]).save()
    u30 = json.loads((research / "data3/universe.json").read_text())["members"]
    u8 = json.loads((research / "data8d/universe8.json").read_text())
    write_json(universe_file(), {"u30": u30, "U": u8})
    r = np.load(research / "data8d/rows.npz")
    yn = [str(s) for s in r["ynames"]]
    assert [str(s) for s in r["names"]] == ef.NAMES
    save_npz(rows_dir() / "base.npz", X=r["X"], t=r["t"].astype(np.int64) * 4, s=r["s"],
             syms=r["syms"], ret24=r["Y"][:, yn.index("ret24")])
    del r
    gc.collect()
    keys = [shift_month("2026-10", -i) for i in range(months_back)]
    syms = sorted({s for k in keys for s in u8.get(k, [])} | {"BTCUSDT", "ETHUSDT"})
    store = Store()
    for s in syms:
        convert_research(store, s, research)
        store.cache.clear()
    return {"symbols": len(syms)}


def convert_research(store: Store, sym: str, research: Path = RESEARCH) -> bool:
    """연구 1시간봉(코어 npz, 나머지 CSV)을 저장소 형식으로(연구 build.load와 같은 값)."""
    core = research / "data2/h1" / f"{sym}.npz"
    if core.exists():
        z = np.load(core)
        store.save(sym, {k: z[k].astype(float) for k in "ohlcq"} | {
            "ft": z["ft"].astype(np.int64), "fr": z["fr"].astype(float)})
        return True
    kp = research / "data3/raw" / f"{sym}_1h.csv"
    fp = research / "data3/raw" / f"{sym}_fund.csv"
    if not kp.exists():
        return False
    a = np.loadtxt(kp, delimiter=",", usecols=(0, 1, 2, 3, 4, 7), ndmin=2)
    idx = ((a[:, 0] - ef.T0_MS) // ef.HR_MS).astype(np.int64)
    ok = idx >= 0
    n = int(idx[ok].max()) + 1 if ok.any() else 0
    d = {}
    for k, j in zip("ohlcq", (1, 2, 3, 4, 5)):
        arr = np.full(n, np.nan)
        arr[idx[ok]] = a[ok, j]
        d[k] = arr
    if fp.exists() and fp.stat().st_size > 0:
        f = np.loadtxt(fp, delimiter=",", ndmin=2)
        d["ft"], d["fr"] = f[:, 0].astype(np.int64), f[:, 2].astype(float)
    else:
        d["ft"], d["fr"] = np.zeros(0, np.int64), np.zeros(0)
    store.save(sym, d)
    return True


def ensure_symbols(store: Store, syms, research: Path = RESEARCH) -> None:
    """저장소에 없으면 연구 자료에서 옮긴다(없으면 다음 update_symbol이 받는다)."""
    for s in syms:
        if not store.has(s) and research.exists():
            convert_research(store, s, research)


# ---------------------------------------------------------------- 연구와 맞춰 보기


def parity(research: Path = RESEARCH, key: str = "2026-10", n_dec: int = 3) -> dict:
    """연구 값과 비교: ① 범위(전 기간) ② key 달 학습 행 특징·정답 ③ 결정 경로(배열 자름) 특징
    ④ 연구 key 달 모델 점수."""
    import lightgbm as lgb

    store = Store()
    out: dict = {}
    z = np.load(research / "data5/panel.npz")
    panel = Panel(z["syms"], z["first"], z["Q"])
    u30r = json.loads((research / "data3/universe.json").read_text())["members"]
    u8 = json.loads((research / "data8d/universe8.json").read_text())
    bad30 = [k for k in u30r if k >= "2020-03" and compute_u30(panel, k) != u30r[k]]
    badu = [k for k in u8 if compute_universe(panel, k, u30r[k]) != u8[k]]
    out["universe"] = {"months": len(u8), "u30_mismatch": bad30, "U_mismatch": badu}
    members = u8[key]
    ensure_symbols(store, members, research)
    r = np.load(research / "data8d/rows.npz")
    yn = [str(s) for s in r["ynames"]]
    msk = r["mk"] == key
    ref_x = r["X"][msk]
    ref_t = r["t"][msk].astype(np.int64) * 4
    ref_s = np.array([str(s) for s in r["syms"]])[r["s"][msk]]
    ref_y = r["Y"][msk][:, yn.index("ret24")]
    del r
    gc.collect()
    mine = feature_rows(store, members, RESEARCH_NH, month_start(key), RESEARCH_NH - 4,
                        mask_last=False, labels=True)
    same_rows = (len(mine["t"]) == len(ref_t) and np.array_equal(mine["t"], ref_t)
                 and list(mine["sym"]) == list(ref_s))
    out["rows"] = {"research": int(len(ref_t)), "mine": int(len(mine["t"])),
                   "same_rows": bool(same_rows)}
    if same_rows:
        d = np.abs(mine["X"] - ref_x)
        out["rows"]["max_abs_diff"] = float(np.nanmax(d))
        out["rows"]["nan_mismatch"] = int((np.isnan(mine["X"]) != np.isnan(ref_x)).sum())
        yd = np.abs(mine["ret24"] - ref_y)
        out["rows"]["label_max_abs_diff"] = float(np.nanmax(yd))
        out["rows"]["label_nan_mismatch"] = int((np.isnan(mine["ret24"]) != np.isnan(ref_y)).sum())
    ts = np.unique(ref_t)
    pick = ts[np.linspace(0, len(ts) - 1, n_dec).astype(int)]
    dec = []
    for t in pick:
        m = feature_rows(store, members, int(t) + 1, int(t), int(t), mask_last=True)
        rm = ref_t == t
        ok = np.array_equal(np.array(m["sym"]), ref_s[rm])
        dec.append({"t": iso(int(t)), "rows": int(rm.sum()), "same_rows": bool(ok),
                    "max_abs_diff": float(np.nanmax(np.abs(m["X"] - ref_x[rm]))) if ok else None})
    out["decision_path"] = dec
    pf = np.load(research / "data8f/predF_H24_G.npz")["P"]
    rr = np.load(research / "data8d/rows.npz", mmap_mode="r")
    pref = pf[np.asarray(rr["mk"]) == key]
    booster = lgb.Booster(model_file=str(research / f"data8f/models/H24G_{key}.txt"))
    pm = booster.predict(mine["X"])
    out["model"] = {"max_abs_diff_vs_research_pred": float(np.nanmax(np.abs(pm - pref)))}
    return out


# ---------------------------------------------------------------- CLI


def _lock():
    root = base_dir()
    root.mkdir(parents=True, exist_ok=True)
    f = (root / ".lock").open("w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return None
    return f


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="minerva.edge_paper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed")
    p = sub.add_parser("run")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-load", type=float, default=MAX_LOAD)
    p = sub.add_parser("fit")
    p.add_argument("month")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("parity")
    p.add_argument("--month", default="2026-10")
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    lk = _lock()
    if lk is None:
        log.info("다른 실행이 돌고 있어 건너뜀")
        return 0
    if a.cmd == "seed":
        print(json.dumps(seed()))
    elif a.cmd == "run":
        load = os.getloadavg()[0]
        if load > a.max_load:
            log.info("부하 %.2f > %.1f → 건너뜀", load, a.max_load)
            return 0
        print(json.dumps(run(fetch=not a.no_fetch, max_steps=a.max_steps)))
    elif a.cmd == "fit":
        print(fit_month(Store(), a.month, force=a.force))
    elif a.cmd == "parity":
        print(json.dumps(parity(key=a.month), indent=1))
    elif a.cmd == "summary":
        store = Store()
        st = load_state()
        write_json(summary_path(), build_summary(store, st, store.last_hour("BTCUSDT")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

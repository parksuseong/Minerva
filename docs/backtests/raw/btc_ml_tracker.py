"""15분봉 ML 걸어두는 지정가 규칙 2개를 앞으로의 자료로 기록만 한다 (주문 없음).

규칙(2026-11-04 사용자 승인, 문서 2026-11-04-btc-ml-rest-confirm·strict-hypotheses.md):
- A: 5개 모델 평균 점수가 2025년 상위 10% 이상 + 4개 이상이 같은 방향 (V1)
- B: 평균 점수 2025년 상위 5% 이상 + 5개 모두 같은 방향 (S3)
숏은 하위 10%·5%와 반대 방향 동의. 체결·청산은 btc_ml_edge_cost.rest_sim과 같다.

모델은 백테스트의 2026년 모델과 같은 자료·씨드로 한 번 학습해 저장한다(train).
step은 15분마다: 현물 5분봉을 받아 붙이고, 새로 마감된 15분봉에 점수를 매기고,
추적 시작 뒤의 신호로 모의 체결을 다시 계산해 새 사건만 텔레그램으로 알린다.
모델 입력은 현물(학습과 같다). 지정가·체결·익절·손절은 선물 15분봉으로 보고 펀딩을 뺀다
(사용자 2026-11-07: 바이낸스 USDT-M 선물로 매매한다). 오늘 봉은 OKX BTC-USDT-SWAP로 먼저 보고,
바이낸스 공식 일별 아카이브(data.binance.vision)가 나온 날은 바이낸스 봉으로 바꿔 다시 계산한다.
fapi.binance.com은 이 서버(미국)에서 451로 막혀 있고, 우회 경로는 쓰지 않는다.
옛 현물 기준 기록은 *.spot.*에 둔다.

실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_tracker.py train|step|report
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import btc_ml_15m as M
import btc_ml_ensemble as E
import numpy as np
from btc_fractal import cci, load5, rsi, stoch
from btc_ml_barrier import barrier_labels
from btc_ml_edge_cost import MAKER, OFF, REST, SL, SLIP, TAKER, THRU, TP, mean_score

HOME = Path("~/.minerva").expanduser()
BASE5 = HOME / "data/btc/5m.csv"
ST = Path(os.environ.get("MINERVA_MLTRACK_DIR", HOME / "research/btc-ml-tracker"))
PRED = HOME / "research/btc-2026-11-04/ml15/pred.npz"
SEEDS_OUT = HOME / "research/btc-2026-11-04/seeds"
URL = "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=1000&startTime={s}"
OKX = "https://www.okx.com/api/v5"
OKX_INST = "BTC-USDT-SWAP"
PRICE_BASIS = (
    "가격 기준: 바이낸스 선물(전날까지) + OKX(오늘, 임시) 체결·청산 + 펀딩. 신호는 현물 모델. "
    "체결 지연 반영(실제 판단 시각 이후 5분봉부터)은 '지연 반영' 칸"
)
M5 = 300_000
LIVE_DELAY = 900_000  # 이보다 늦게 판단한 신호는 따라잡기(서버 꺼짐·실행 실패)로 따로 센다
BNV = "https://data.binance.vision/data/futures/um"
DAY = 86_400_000
BAR15 = 900_000
YEAR = 2026
RULES = {"A": (0.90, 4), "B": (0.95, 5), "AS": (0.90, 4), "BS": (0.95, 5), "AS4": (0.90, 4)}
NAMES = {
    "A": "A(4개 동의·상위10%)",
    "B": "B(5개 동의·상위5%)",
    "AS": "A평균(씨드5벌·4개 동의·상위10%)",
    "BS": "B평균(씨드5벌·5개 동의·상위5%)",
    "AS4": "A평균+4시간 추세(EMA50)",
    "K2": "K2(R1 일봉 방향 + AS4 진입 + R1식 청산)",
}
# AS·BS는 같은 규칙을 씨드 5벌(Y, Y+1000·k) 평균 확률로 계산한다(2026-11-05 사용자 승인,
# 문서 2026-11-04-btc-ml-seeds-hypotheses.md). 확률 칸 접두어: 씨드 Y는 p_, 평균은 pa_.
PFX = {"A": "p_", "B": "p_", "AS": "pa_", "BS": "pa_", "AS4": "pa_", "K2": "pa_"}
# AS4는 AS 신호 중 4시간봉 추세(종가>EMA50, EMA50이 5봉 전보다 위; 숏은 반대)와 같은 방향만 남긴다
# (2026-11-05 사용자 "가장 최고의 것으로만 넣어", 문서 2026-11-05-btc-ml-mtf-trend-hypotheses.md).
# 문턱은 AS 것을 쓴다. 4시간봉은 받아 둔 5분봉을 UTC 4시간 단위로 묶어 만든다.
BASE_RULE = {"AS4": "AS"}
H4 = 4 * 3_600_000
EMA_N, SLOPE = 50, 5
SEED_K = (1, 2, 3, 4)
NOTE = {
    "A": "씨드 시험에서 운으로 판정됨: 기대값 0으로 봐요",
    "B": "씨드 시험: 작은 우위가 있을 수도, 증명은 안 됨",
    "AS": "씨드 5벌 평균 모델. 11/05 00시대부터 기록, 앞으로의 기록으로만 판정",
    "BS": "씨드 5벌 평균 모델. 11/05 00시대부터 기록, 앞으로의 기록으로만 판정",
    "AS4": "A평균 중 4시간봉 추세와 같은 방향만. "
    "최근 1년 백테스트는 필터 없음보다 나빴음(+30% vs +165%). 100회 넘게 쌓였을 때 A평균보다 못하거나 숏 거래당 손익이 0 이하면 폐기",
    "K2": "K2: R1 일봉 방향 + AS4 신호로 진입 + R1식 청산(방향 전환·1% 손절). 128번 탈락, "
    "레드팀: 새 자료로 추적할 후보",
}
# K2(2026-11-07 사용자 "응", 문서 2026-11-07-btc-r1-as4-entry-hypotheses.md):
# 방향은 마감된 일봉(UTC 0시) CCI20>0·RSI14>50이면 +1, 둘 다 아래면 −1(R1).
# 그 방향과 같은 AS4 신호의 지정가로 들어가고, 방향이 바뀐 15분봉 다음 시가나
# 진입가 1% 손절에 나간다.
# 손절 뒤에는 방향이 그대로인 동안 다음 같은 방향 AS4 신호를 기다린다.
# 계산은 128번 코드(btc_r1_as4_entry의 lim_fill·r1_manage)를 그대로 쓴다.
FUT_HIST = HOME / "data/btc/futures/klines_5m.csv"
K2_STOP = 0.01


def trend4h(d5: dict, t15: np.ndarray) -> np.ndarray:
    """15분 신호 봉이 마감될 때 이미 마감된 마지막 4시간봉의 추세(1, -1, 0)."""
    g = (d5["t"] // H4).astype(np.int64)
    cut = np.flatnonzero(np.diff(g)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(g)] - 1
    full = (ends - starts + 1) == H4 // 300_000
    c = d5["c"][ends][full]
    ct = (g[starts][full] + 1) * H4 - 1
    a = 2 / (EMA_N + 1)
    ema = np.empty_like(c)
    ema[0] = c[0]
    for i in range(1, len(c)):
        ema[i] = a * c[i] + (1 - a) * ema[i - 1]
    tr = np.zeros(len(c))
    prev = np.r_[np.full(SLOPE, np.nan), ema[:-SLOPE]]
    with np.errstate(invalid="ignore"):
        tr[(c > ema) & (ema > prev)] = 1
        tr[(c < ema) & (ema < prev)] = -1
    tr[:EMA_N] = 0
    j = np.searchsorted(ct, t15 + BAR15 - 1, side="right") - 1
    return np.where(j >= 0, tr[np.maximum(j, 0)], 0.0)


# ---- 자료 ----
def live_path() -> Path:
    return ST / "live5m.csv"


def load_all() -> dict:
    d = load5(BASE5)
    lp = live_path()
    if lp.exists():
        e = load5(lp)
        keep = e["t"] > d["t"][-1]
        d = {k: np.r_[d[k], e[k][keep]] for k in d}
    # 15분 경계에서 끝나게 자른다(부분 봉 제외는 resample이 한다)
    return d


def fetch_new() -> int:
    d = load_all()
    last = int(d["t"][-1])
    now = int(time.time() * 1000)
    rows, s = [], last + 300_000
    while s < now - 300_000:
        req = urllib.request.Request(URL.format(s=s), headers={"User-Agent": "minerva-tracker"})
        with urllib.request.urlopen(req, timeout=20) as r:
            k = json.load(r)
        k = [x for x in k if int(x[6]) < now]  # 마감된 봉만
        if not k:
            break
        rows += k
        s = int(k[-1][0]) + 300_000
        if len(k) < 999:
            break
    lp = live_path()
    new = not lp.exists()
    with lp.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time"])
        for x in rows:
            w.writerow(x[:7])
    return len(rows)


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-tracker"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def fut_path() -> Path:
    return ST / "live_fut5m.csv"


def fund_path() -> Path:
    return ST / "fut_funding.csv"


def _read_rows(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        r = csv.reader(f)
        next(r, None)
        return {int(x[0]): x for x in r if x}


def fetch_fut(since: int) -> tuple:
    """OKX 선물 5분봉(마감된 것만)과 펀딩을 받아 붙인다.

    실패하면 (0, 오류)를 돌려준다. 다음 실행에 다시 받는다.
    """
    try:
        have = _read_rows(fut_path())
        last = max(have) if have else since - 300_000
        now = int(time.time() * 1000)
        got, cur = {}, None
        while True:  # history-candles는 최신부터 거꾸로 100개씩 준다
            url = f"{OKX}/market/history-candles?instId={OKX_INST}&bar=5m&limit=100"
            data = _get_json(url + (f"&after={cur}" if cur else ""))["data"]
            if not data:
                break
            for x in data:
                t0 = int(x[0])
                if t0 > last and x[8] == "1" and t0 + 300_000 <= now:
                    got[t0] = [t0, x[1], x[2], x[3], x[4], x[6]]
            cur = data[-1][0]
            if int(cur) <= last:
                break
            time.sleep(0.15)
        new = not fut_path().exists()
        with fut_path().open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["open_time", "open", "high", "low", "close", "volume_btc"])
            for t0 in sorted(got):
                w.writerow(got[t0])
        fr = _read_rows(fund_path())
        lf = max(fr) if fr else 0
        rows = _get_json(f"{OKX}/public/funding-rate-history?instId={OKX_INST}&limit=100")["data"]
        add = sorted(
            (int(x["fundingTime"]), x["realizedRate"])
            for x in rows
            if lf < int(x["fundingTime"]) <= now
        )
        new = not fund_path().exists()
        with fund_path().open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["funding_time", "rate"])
            w.writerows(add)
        return len(got), None
    except Exception as e:  # noqa: BLE001 - 선물 자료가 없으면 이번 봉은 건너뛰고 다음에 다시 본다
        return 0, f"{type(e).__name__}: {e}"


def _vision(url: str) -> list | None:
    """바이낸스 공식 아카이브 zip을 CHECKSUM으로 확인하고 CSV 행을 돌려준다. 아직 없으면 None."""
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-tracker"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            blob = r.read()
        req = urllib.request.Request(url + ".CHECKSUM", headers={"User-Agent": "minerva-tracker"})
        with urllib.request.urlopen(req, timeout=30) as r:
            want = r.read().decode().split()[0]
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    if hashlib.sha256(blob).hexdigest() != want:
        raise ValueError(f"checksum mismatch {url}")
    z = zipfile.ZipFile(io.BytesIO(blob))
    rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))))
    return [x for x in rows if x and x[0].isdigit()]


def days_path() -> Path:
    return ST / "fut_days.json"


def reprice_binance() -> dict:
    """전날까지의 OKX 봉·펀딩을 바이낸스 아카이브로 바꾼다. 바뀐 날 목록을 돌려준다."""
    days = json.loads(days_path().read_text()) if days_path().exists() else {}
    out = {"days": [], "funding_months": [], "error": None}
    try:
        rows = _read_rows(fut_path())
        if not rows:
            return out
        today = int(time.time() * 1000) // DAY
        by_day: dict = {}
        for t0 in rows:
            by_day.setdefault(t0 // DAY, []).append(t0)
        for dk in sorted(by_day):
            name = datetime.fromtimestamp(dk * 86400, timezone.utc).strftime("%Y-%m-%d")
            if dk >= today or days.get(name) == "binance":
                continue
            got = _vision(f"{BNV}/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{name}.zip")
            if not got or len(got) != 288:
                days.setdefault(name, "okx")
                continue
            for t0 in by_day[dk]:
                rows.pop(t0)
            for x in got:
                rows[int(x[0])] = [int(x[0]), x[1], x[2], x[3], x[4], x[5]]
            days[name] = "binance"
            out["days"].append(name)
        if out["days"]:
            tmp = fut_path().with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["open_time", "open", "high", "low", "close", "volume_btc"])
                for t0 in sorted(rows):
                    w.writerow(rows[t0])
            tmp.replace(fut_path())
        fr = _read_rows(fund_path())
        def ym_of(t0: int) -> str:
            return datetime.fromtimestamp(t0 / 1000, timezone.utc).strftime("%Y-%m")

        months = sorted({ym_of(t0) for t0 in fr})
        cur = datetime.now(timezone.utc).strftime("%Y-%m")
        changed = False
        for ym in months:
            if ym >= cur or days.get(f"fund:{ym}") == "binance":
                continue
            got = _vision(f"{BNV}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip")
            if not got:
                continue
            for t0 in [t0 for t0 in fr if ym_of(t0) == ym]:
                fr.pop(t0)
            for x in got:  # calc_time, funding_interval_hours, last_funding_rate
                fr[int(x[0])] = [int(x[0]), x[2]]
            days[f"fund:{ym}"] = "binance"
            out["funding_months"].append(ym)
            changed = True
        if changed:
            tmp = fund_path().with_suffix(".tmp")
            with tmp.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["funding_time", "rate"])
                for t0 in sorted(fr):
                    w.writerow(fr[t0][:2])
            tmp.replace(fund_path())
    except Exception as e:  # noqa: BLE001 - 아카이브가 안 되면 OKX 그대로 두고 다음 실행에 다시 본다
        out["error"] = f"{type(e).__name__}: {e}"
    days_path().write_text(json.dumps(days, indent=1, sort_keys=True))
    return out


def outcome(x: dict) -> str:
    return ("x:" + ("win" if x["win"] else "loss")) if x["filled"] else "n"


def fut_bars(t15: np.ndarray) -> tuple:
    """신호 시각 t15(현물 15분봉 시작)에 맞춘 선물 15분봉.

    첫 빈 봉에서 자른다(그 뒤는 다음 실행에).
    """
    rows = _read_rows(fut_path())
    tt = np.array(sorted(rows), dtype=float)
    if not len(tt):
        return {k: np.zeros(0) for k in "tohlc"}, len(t15)
    v = np.array([[float(rows[int(x)][j]) for j in (1, 2, 3, 4)] for x in tt])
    key = (tt // BAR15).astype(np.int64)
    out = {k: [] for k in "tohlc"}
    miss = 0
    for x in t15:
        k15 = int(x // BAR15)
        a, b = np.searchsorted(key, k15), np.searchsorted(key, k15, side="right")
        if b - a != 3:
            miss = len(t15) - len(out["t"])
            break
        seg = v[a:b]
        out["t"].append(x)
        out["o"].append(seg[0, 0])
        out["h"].append(seg[:, 1].max())
        out["l"].append(seg[:, 2].min())
        out["c"].append(seg[-1, 3])
    return {k: np.array(val, dtype=float) for k, val in out.items()}, miss


def funding_cut(done: list) -> None:
    """summ_net과 같이 (체결, 청산] 사이 펀딩을 뺀다. 롱은 양수 펀딩을 낸다."""
    fr = _read_rows(fund_path())
    if not fr:
        return
    ft = np.array(sorted(fr), dtype=float)
    cf = np.r_[0.0, np.cumsum([float(fr[int(x)][1]) for x in ft])]
    for x in done:
        if x["filled"]:
            a = np.searchsorted(ft, x["t_fill"], side="right")
            b = np.searchsorted(ft, x["t_exit"], side="right")
            x["fund"] = float(cf[b] - cf[a])
            x["net1"] -= x["side"] * x["fund"]


# ---- 체결 지연 (2026-11-07 레드팀 127: 신호 직후 몇 분의 체결이 수익을 만든다) ----
def decided_path() -> Path:
    return ST / "decided.json"


def read_decided() -> dict:
    """{신호 봉 시작 ms: [판단 시각 ms, 추정 여부]}."""
    p = decided_path()
    return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def journal_runs() -> list:
    """타이머 실행이 끝난 시각들(ms). 판단 시각을 기록하기 전의 신호를 추정하는 데만 쓴다."""
    import subprocess

    try:
        out = subprocess.run(
            ["journalctl", "-u", "minerva-btc-mltrack", "-o", "short-unix", "--no-pager"],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(
        int(float(line.split()[0]) * 1000)
        for line in out.splitlines()
        if "Finished minerva-btc-mltrack" in line
    )


def estimate_decided(ts: list, known: dict) -> dict:
    runs = np.array(journal_runs(), dtype=float)
    out = {}
    for x in ts:
        if x in known or not len(runs):
            continue
        k = np.searchsorted(runs, x + BAR15)
        if k < len(runs):
            out[x] = [int(runs[k]), True]
    return out


def fut5_bars() -> dict:
    rows = _read_rows(fut_path())
    tt = sorted(rows)
    return {
        "t": np.array(tt, dtype=float),
        **{k: np.array([float(rows[x][j]) for x in tt]) for k, j in zip("ohlc", (1, 2, 3, 4))},
    }


def paper_delay(bars, sig, decided, b5):
    """paper()와 같은 규칙. 다만 판단 시각을 5분 단위로 올린 시각 뒤의 선물 5분봉에서만
    체결·청산한다.

    지정가는 신호 봉 종가로 정한다(판단할 때 아는 값). 주문 유효 시간은 paper()와 같다
    (신호 봉 뒤 REST개 15분봉). 판단이 그보다 늦으면 '지연으로 놓침'이다.
    판단 시각을 모르면 세지 않는다.
    """
    t, c = bars["t"], bars["c"]
    t5, o5, h5, l5 = (b5[k] for k in "tohl")
    out, state, i, n = [], None, 0, len(t)
    while i < n:
        s = sig[i]
        if not s:
            i += 1
            continue
        lim = c[i] * (1 - OFF * s)
        dec = decided.get(int(t[i]))
        if dec is None:
            out.append({"t_sig": t[i], "side": s, "lim": lim, "unknown": True, "filled": False})
            i += 1
            continue
        d_ms, est = dec
        go = -(-d_ms // M5) * M5
        end = t[i] + (1 + REST) * BAR15
        base = {"t_sig": t[i], "side": s, "lim": lim, "delay": d_ms - (t[i] + BAR15), "est": est}
        if go >= end:
            out.append({**base, "filled": False, "late": True})
            i += 1
            continue
        a = int(np.searchsorted(t5, go))
        z = int(np.searchsorted(t5, end))
        j = None
        for b in range(a, z):
            if (l5[b] <= lim * (1 - THRU)) if s > 0 else (h5[b] >= lim * (1 + THRU)):
                j = b
                break
        if j is None:
            if not len(t5) or t5[-1] + M5 < end:
                state = {"kind": "order", "side": s, "lim": lim, "t_sig": t[i]}
                break
            out.append({**base, "filled": False})
            i += 1
            continue
        px = lim
        up, dn = px * (1 + TP), px * (1 - SL)
        res, b = None, j
        while b < len(t5):
            first = b == j
            if s > 0:
                if l5[b] <= dn:
                    fill = min(o5[b], dn) if not first else dn
                    res = (fill * (1 - SLIP) / px - 1 - MAKER - TAKER, 0)
                elif not first and h5[b] >= up * (1 + THRU):
                    res = (TP - 2 * MAKER, 1)
            else:
                if h5[b] >= up:
                    fill = max(o5[b], up) if not first else up
                    res = (-(fill * (1 + SLIP) / px - 1) - MAKER - TAKER, 0)
                elif not first and l5[b] <= dn * (1 - THRU):
                    res = (SL - 2 * MAKER, 1)
            if res:
                break
            b += 1
        if res is None:
            state = {"kind": "pos", "side": s, "px": px, "t_sig": t[i], "t_fill": t5[j]}
            break
        out.append(
            {**base, "filled": True, "t_fill": t5[j], "t_exit": t5[b],
             "net1": res[0], "win": res[1]}
        )
        i = max(i + 1, int(np.searchsorted(t, t5[b], side="right")) - 1)
    return out, state


def delay_summary(done: list, open_) -> dict:
    known = [x for x in done if not x.get("unknown")]
    fills = [x for x in known if x["filled"]]
    net = np.array([x["net1"] for x in fills]) if fills else np.zeros(0)
    dl = np.array([x["delay"] for x in known], dtype=float) / 60_000
    live = dl[dl * 60_000 <= LIVE_DELAY]
    return {
        "signals_known": len(known),
        "unknown": len(done) - len(known),
        "estimated": int(sum(1 for x in known if x.get("est"))),
        "late": int(sum(1 for x in known if x.get("late"))),
        "fills": len(fills),
        "wins": int(sum(x["win"] for x in fills)),
        "x1": round(float((np.prod(1 + net) - 1) * 100), 2),
        "x3": round(float((np.prod(1 + np.maximum(3 * net, -1)) - 1) * 100), 2),
        "delay_med_min": round(float(np.median(live)), 1) if len(live) else None,
        "delay_max_min": round(float(live.max()), 1) if len(live) else None,
        "backfill": int((dl * 60_000 > LIVE_DELAY).sum()),
        "open": open_["kind"] if open_ else None,
    }


# ---- 모델 ----
def seq_net(kind: str, nf: int, nt: int):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "lstm":
                self.body = nn.LSTM(nf, 32, batch_first=True)
            else:
                self.body = nn.Sequential(
                    nn.Conv1d(nf, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
            self.head = nn.Sequential(nn.Linear(32 + nt, 32), nn.ReLU(), nn.Linear(32, 2))

        def forward(self, s, x):
            if kind == "lstm":
                _, (hn, _) = self.body(s)
                z = hn[-1]
            else:
                z = self.body(s.transpose(1, 2)).squeeze(-1)
            return self.head(torch.cat([z, x], 1))

    return Net()


def fit_seq(kind, Sn, Xn, y, tr, seed):
    """E.fit_predict_seq와 같은 순서(씨드 → 망 생성 → 4회 반복)로 학습하고 망을 돌려준다."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    net = seq_net(kind, Sn.shape[1], Xn.shape[1])
    S, Xt, yt = torch.from_numpy(Sn), torch.from_numpy(Xn), torch.from_numpy(y)
    off = torch.arange(-E.SEQ + 1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for _ in range(4):
        perm = rng.permutation(tr)
        for k in range(0, len(perm), 256):
            b = perm[k : k + 256]
            ii = torch.from_numpy(b)
            opt.zero_grad()
            loss = lossf(net(S[ii[:, None] + off[None, :]], Xt[ii]), yt[ii])
            loss.backward()
            opt.step()
    net.eval()
    return net


def fit_tab(name, Xtr, ytr, seed):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        m = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50, n_jobs=2, random_state=seed
        )
        return m.fit(Xtr, ytr)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier((64, 32), early_stopping=True, max_iter=200, random_state=seed).fit(
            Xtr, ytr
        )
    import xgboost as xgb

    prm = {
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": 2,
        "seed": seed,
        "objective": "binary:logistic",
    }
    return xgb.train(prm, xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)


def features(d5):
    B = E.build(d5, M.FACTOR)
    X, seq, t = B["X"], B["seq"], B["d"]["t"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    return B, X, seq, t, seq_ok


def norm(X, seq, nz):
    Xn = ((X - nz["mu"]) / nz["sd"]).astype(np.float32)
    Xn[~np.isfinite(Xn)] = 0
    Sn = ((seq - nz["smu"]) / nz["ssd"]).astype(np.float32)
    Sn[~np.isfinite(Sn)] = 0
    return Xn, Sn


def predict(models, Xn, Sn, idx):
    import torch
    import xgboost as xgb

    idx = np.asarray(idx)
    out = np.full((len(E.MODELS), len(idx)), np.nan, np.float32)
    off = torch.arange(-E.SEQ + 1, 1)
    S, Xt = torch.from_numpy(Sn), torch.from_numpy(Xn)
    for mi, name in enumerate(E.MODELS):
        m = models[name]
        if name in ("rf", "mlp"):
            out[mi] = m.predict_proba(Xn[idx])[:, 1]
        elif name == "xgb":
            out[mi] = m.predict(xgb.DMatrix(Xn[idx]))
        else:
            res = []
            with torch.no_grad():
                for k in range(0, len(idx), 2048):
                    ii = torch.from_numpy(idx[k : k + 2048])
                    res.append(
                        torch.softmax(m(S[ii[:, None] + off[None, :]], Xt[ii]), 1)[:, 1].numpy()
                    )
            out[mi] = np.concatenate(res)
    return out


def load_models():
    import torch

    meta = pickle.loads((ST / "models.pkl").read_bytes())
    models = meta["tab"]
    for k in ("lstm", "cnn"):
        net = seq_net(k, meta["nf"], meta["nt"])
        net.load_state_dict(torch.load(ST / f"{k}.pt", weights_only=True))
        net.eval()
        models[k] = net
    sm = ST / "seeds_meta.json"
    if sm.exists():
        meta["thr"].update({r: tuple(v) for r, v in json.loads(sm.read_text())["thr"].items()})
    return models, meta


def seed_dir(k: int) -> Path:
    return ST / f"seed{k}"


def load_seed_models(meta) -> list:
    """씨드 k=1~4 모델 묶음. 없으면 빈 목록(평균 규칙은 건너뛴다)."""
    import torch

    out = []
    for k in SEED_K:
        dd = seed_dir(k)
        if not (dd / "tab.pkl").exists():
            return []
        models = pickle.loads((dd / "tab.pkl").read_bytes())
        for kind in ("lstm", "cnn"):
            net = seq_net(kind, meta["nf"], meta["nt"])
            net.load_state_dict(torch.load(dd / f"{kind}.pt", weights_only=True))
            net.eval()
            models[kind] = net
        out.append(models)
    return out


def train_seeds() -> None:
    """씨드 Y+1000·k로 같은 자료를 다시 학습하고, 걸어가기 예측과 같은지 확인한다."""
    import torch

    meta = pickle.loads((ST / "models.pkl").read_bytes())
    d5 = load5(BASE5)
    B, X, seq, t, seq_ok = features(d5)
    y = barrier_labels(B["d"], M.TP, M.SL)
    base = (t >= E.ts(2018)) & (t < E.ts(YEAR) - M.EMBARGO * M.BAR) & seq_ok
    Xn, Sn = norm(X, seq, meta["nz"])
    tr = np.flatnonzero(base & (y >= 0))[::3]
    te = np.flatnonzero((t >= E.ts(YEAR)) & seq_ok)
    rep = {}
    for k in SEED_K:
        dd = seed_dir(k)
        dd.mkdir(parents=True, exist_ok=True)
        seed = YEAR + 1000 * k
        tab = {}
        for name in E.MODELS:
            if name in ("rf", "mlp", "xgb"):
                tab[name] = fit_tab(name, Xn[tr], y[tr], seed)
            else:
                net = fit_seq(name, Sn, Xn, y, tr, seed)
                torch.save(net.state_dict(), dd / f"{name}.pt")
        (dd / "tab.pkl").write_bytes(pickle.dumps(tab))
        print("seed", k, "trained", flush=True)
    seeds = load_seed_models(meta)
    for k, models in zip(SEED_K, seeds):
        p = predict(models, Xn, Sn, te)
        z = np.load(SEEDS_OUT / f"pred_s{k}.npz")
        zi = np.searchsorted(z["t"], t[te])
        ok = zi < len(z["t"])
        ref = z["probs"][:, zi[ok]]
        for mi, name in enumerate(E.MODELS):
            a, b = p[mi][ok], ref[mi]
            m = np.isfinite(a) & np.isfinite(b)
            rep[f"s{k} {name}"] = round(float(np.abs(a[m] - b[m]).max()), 4)
    # 평균 규칙 문턱: 2025년 씨드 5벌 평균 점수의 분위수(원래 규칙과 같은 방식)
    z0 = np.load(PRED)
    ps = [z0["probs"]] + [np.load(SEEDS_OUT / f"pred_s{k}.npz")["probs"] for k in SEED_K]
    sc = mean_score(np.mean(ps, 0))
    yr = (z0["t"] >= E.ts(2025)) & (z0["t"] < E.ts(2026)) & np.isfinite(sc)
    thr = {
        r: (float(np.quantile(sc[yr], q)), float(np.quantile(sc[yr], 1 - q)))
        for r, (q, _) in RULES.items()
        if PFX[r] == "pa_" and r not in BASE_RULE
    }
    rep["thr"] = thr
    (ST / "seeds_meta.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)


def train() -> None:
    import torch

    ST.mkdir(parents=True, exist_ok=True)
    d5 = load5(BASE5)
    B, X, seq, t, seq_ok = features(d5)
    y = barrier_labels(B["d"], M.TP, M.SL)
    base = (t >= E.ts(2018)) & (t < E.ts(YEAR) - M.EMBARGO * M.BAR) & seq_ok
    i0 = np.flatnonzero(base)
    nz = {
        "mu": X[i0].mean(0),
        "sd": X[i0].std(0) + 1e-6,
        "smu": seq[i0].mean(0),
        "ssd": seq[i0].std(0) + 1e-6,
    }
    Xn, Sn = norm(X, seq, nz)
    tr = np.flatnonzero(base & (y >= 0))[::3]
    tab = {}
    for name in E.MODELS:
        t0 = time.time()
        if name in ("rf", "mlp", "xgb"):
            tab[name] = fit_tab(name, Xn[tr], y[tr], YEAR)
        else:
            net = fit_seq(name, Sn, Xn, y, tr, YEAR)
            torch.save(net.state_dict(), ST / f"{name}.pt")
        print(name, "train", len(tr), "sec", round(time.time() - t0, 1), flush=True)
    # 문턱: 백테스트와 같이 2025년 평균 점수의 분위수
    z = np.load(PRED)
    sc = mean_score(z["probs"])
    yr = (z["t"] >= E.ts(2025)) & (z["t"] < E.ts(2026)) & np.isfinite(sc)
    thr = {
        k: (float(np.quantile(sc[yr], q)), float(np.quantile(sc[yr], 1 - q)))
        for k, (q, _) in RULES.items()
        if k not in BASE_RULE
    }
    meta = {
        "tab": tab,
        "nz": nz,
        "nf": seq.shape[1],
        "nt": X.shape[1],
        "thr": thr,
        "trained": time.strftime("%Y-%m-%d %H:%M"),
        "n_train": int(len(tr)),
    }
    (ST / "models.pkl").write_bytes(pickle.dumps(meta))
    # 확인: 2026년 백테스트 예측과 비교
    models, _ = load_models()
    te = np.flatnonzero((t >= E.ts(YEAR)) & seq_ok)
    p = predict(models, Xn, Sn, te)
    zi = np.searchsorted(z["t"], t[te])
    ok = zi < len(z["t"])
    ref = z["probs"][:, zi[ok]]
    rep = {"trained": meta["trained"], "thr": thr, "n_train": meta["n_train"]}
    for mi, name in enumerate(E.MODELS):
        a, b = p[mi][ok], ref[mi]
        m = np.isfinite(a) & np.isfinite(b)
        rep[name] = {
            "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 4),
            "maxdiff": round(float(np.abs(a[m] - b[m]).max()), 4),
        }
    print(json.dumps(rep, indent=1), flush=True)
    (ST / "train_check.json").write_text(json.dumps(rep, indent=1))


# ---- 신호와 모의 체결 ----
def sig_path() -> Path:
    return ST / "signals.csv"


def read_signals() -> dict:
    p = sig_path()
    if not p.exists():
        return {}
    with p.open() as f:
        return {int(r["t"]): r for r in csv.DictReader(f)}


SIG_COLS = (
    ["t", "c", "score", "up", "dn", "A", "B"]
    + [f"p_{m}" for m in E.MODELS]
    + ["sa", "ua", "da", "AS", "BS"]
    + [f"pa_{m}" for m in E.MODELS]
)
LABEL = {"rf": "랜덤포레스트", "mlp": "MLP", "xgb": "XGBoost", "lstm": "LSTM", "cnn": "CNN"}


def model_probs(sigs: dict, t_sig, pfx: str = "p_") -> dict:
    """신호 봉에서 모델마다 낸 '+1.5% 먼저' 확률 (pa_ = 씨드 5벌 평균)."""
    r = sigs.get(int(t_sig), {})
    return {m: float(r[f"{pfx}{m}"]) for m in E.MODELS if r.get(f"{pfx}{m}") not in (None, "")}


def probs_text(pr: dict, side: int) -> str:
    if not pr:
        return ""
    be = M.BE if side > 0 else 1 - M.BE
    mark = (lambda v: v > be) if side > 0 else (lambda v: v < be)
    return "\n모델별 +1.5% 먼저 닿을 확률(롱은 0.533 넘으면, 숏은 0.467 밑이면 ✓): " + " · ".join(
        f"{LABEL[m]} {v:.2f}{'✓' if mark(v) else '✗'}" for m, v in pr.items()
    )


def paper(bars, sig, rule):
    """rest_sim과 같은 체결·청산. 끝까지 안 끝난 주문·포지션도 돌려준다."""
    t, o, h, l, c = (bars[k] for k in "tohlc")  # noqa: E741
    n = len(t)
    out, i, state = [], 0, None
    while i < n:
        s = sig[i]
        if not s:
            i += 1
            continue
        lim = c[i] * (1 - OFF * s)
        if i + 1 >= n:
            state = {"kind": "order", "side": s, "lim": lim, "t_sig": t[i]}
            break
        j = None
        for b in range(i + 1, min(i + 1 + REST, n)):
            if (l[b] <= lim * (1 - THRU)) if s > 0 else (h[b] >= lim * (1 + THRU)):
                j = b
                break
        if j is None:
            if i + REST >= n:  # 아직 기다리는 중
                state = {"kind": "order", "side": s, "lim": lim, "t_sig": t[i]}
                break
            out.append({"t_sig": t[i], "side": s, "lim": lim, "filled": False})
            i += 1
            continue
        px = lim
        up, dn = px * (1 + TP), px * (1 - SL)
        res, b = None, j
        while b < n:
            first = b == j
            if s > 0:
                if l[b] <= dn:
                    res = (
                        (min(o[b], dn) if not first else dn) * (1 - SLIP) / px - 1 - MAKER - TAKER,
                        0,
                    )
                elif not first and h[b] >= up * (1 + THRU):
                    res = (TP - 2 * MAKER, 1)
            else:
                if h[b] >= up:
                    res = (
                        -((max(o[b], up) if not first else up) * (1 + SLIP) / px - 1)
                        - MAKER
                        - TAKER,
                        0,
                    )
                elif not first and l[b] <= dn * (1 - THRU):
                    res = (SL - 2 * MAKER, 1)
            if res:
                break
            b += 1
        if res is None:
            state = {
                "kind": "pos",
                "side": s,
                "px": px,
                "t_sig": t[i],
                "t_fill": t[j],
                "tp": up if s > 0 else dn,
                "sl": dn if s > 0 else up,
                "last": c[-1],
            }
            break
        out.append(
            {
                "t_sig": t[i],
                "side": s,
                "lim": lim,
                "filled": True,
                "t_fill": t[j],
                "t_exit": t[b],
                "net1": res[0],
                "win": res[1],
            }
        )
        i = b
    return out, state


# ---- K2 (R1 방향 + AS4 진입 + R1 청산) ----
def fut_daily_d5() -> dict:
    """일봉 방향 계산용 선물 5분봉: 저장된 이력 + 추적기가 받은 최근 봉."""
    d = load5(FUT_HIST)
    live = fut5_bars()
    if len(live["t"]):
        keep = d["t"] < live["t"][0]
        d = {k: d[k][keep] for k in d}
        d = {
            "t": np.r_[d["t"], live["t"]],
            **{k: np.r_[d[k], live[k]] for k in "ohlc"},
            "v": np.r_[d["v"], np.zeros(len(live["t"]))],
        }
    return d


def k2_paper(bars: dict, b5: dict, reg: np.ndarray, sig: np.ndarray, delay: bool) -> tuple:
    """128번 run_k(fallback 없음)와 같은 규칙. 끝나지 않은 주문·포지션도 돌려준다."""
    import btc_r1_as4_entry as K

    A = {k: bars[k] for k in "tohlc"}
    f5 = {k: b5[k] for k in "tohl"}
    G = K.Grid(A, f5)
    model = K.Model(delay, THRU)
    t, c = A["t"], A["c"]
    n = G.n
    out, state, i = [], None, 0
    while i < n - 2:
        d = int(reg[i])
        if d == 0 or int(sig[i]) != d:
            i += 1
            continue
        lim = float(c[i]) * (1 - OFF * d)
        f = K.lim_fill(G, i, d, model)
        if f is None:
            if i + REST >= n:
                state = {"kind": "order", "side": d, "lim": lim, "t_sig": t[i]}
                break
            out.append({"t_sig": t[i], "side": d, "lim": lim, "filled": False})
            i += 1
            continue
        e, px = f
        first = model.delay and e == i + 1
        x = K.r1_manage(G, e, d, px, reg, model, first, model.taker, model.slip)
        if x is None:
            state = {
                "kind": "pos", "side": d, "px": px, "t_sig": t[i], "t_fill": t[e],
                "tp": None, "sl": px * (1 - d * K2_STOP), "last": c[-1],
            }
            break
        b, xp, xf, stopped = x
        net = d * (xp / px - 1) - model.maker - xf
        out.append(
            {
                "t_sig": t[i], "side": d, "lim": lim, "filled": True, "t_fill": t[e],
                "t_exit": t[b] if stopped else t[min(b + 1, n - 1)], "net1": float(net),
                "win": int(net > 0), "stopped": bool(stopped),
            }
        )
        i = b
    return out, state


def kst(ms) -> str:
    return time.strftime("%m/%d %H:%M", time.gmtime(ms / 1000 + 9 * 3600))


def side_kr(s):
    return "롱" if s > 0 else "숏"


def step(notify: bool = True) -> dict:
    from minerva import notify as N

    got = fetch_new()
    d5 = load_all()
    B, X, seq, t, seq_ok = features(d5)
    models, meta = load_models()
    st_path = ST / "state.json"
    state = json.loads(st_path.read_text()) if st_path.exists() else {}
    if "start" not in state:
        state = {"start": float(t[-1]) + BAR15, "sent": []}  # 다음 봉부터 기록
    seeds = load_seed_models(meta) if "AS" in meta["thr"] else []
    if seeds and "start_s" not in state:
        state["start_s"] = float(t[-1]) + BAR15  # 평균 규칙은 추가한 다음 봉부터 기록
    sigs = read_signals()
    if sigs and (
        "pa_rf" not in next(iter(sigs.values()))
        or (seeds and any(not r.get("sa") for r in sigs.values()))
    ):
        # 칸이 모자란 옛 파일: 같은 모델로 처음부터 다시 매긴다(씨드 Y 결과는 같다)
        sig_path().replace(ST / f"signals.old{int(time.time())}.csv")
        sigs = {}
    need = np.flatnonzero((t >= state["start"]) & seq_ok)
    need = [i for i in need if int(t[i]) not in sigs]
    if need:
        Xn, Sn = norm(X, seq, meta["nz"])
        p = predict(models, Xn, Sn, need)
        pa = np.mean([p] + [predict(sm, Xn, Sn, need) for sm in seeds], 0) if seeds else None
        new = not sig_path().exists()
        with sig_path().open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(SIG_COLS)
            def flags(pp, k, rules):
                v = pp[:, k]
                sck = float(v.mean())
                up, dn = int((v > M.BE).sum()), int((v < 1 - M.BE).sum())
                out = []
                for r in rules:
                    hi, lo = meta["thr"][r]
                    kk = RULES[r][1]
                    out.append(1 if (sck >= hi and up >= kk) else (-1 if (sck <= lo and dn >= kk) else 0))
                return [round(sck, 5), up, dn], out

            for k, i in enumerate(need):
                head, fl = flags(p, k, ("A", "B"))
                row = [int(t[i]), B["d"]["c"][i], *head, *fl]
                row += [round(float(p[mi, k]), 4) for mi in range(len(E.MODELS))]
                if pa is not None:
                    head, fl = flags(pa, k, ("AS", "BS"))
                    row += [*head, *fl] + [round(float(pa[mi, k]), 4) for mi in range(len(E.MODELS))]
                else:
                    row += [""] * (5 + len(E.MODELS))
                w.writerow(row)
        now = int(time.time() * 1000)
        dec = read_decided()
        dec.update({int(t[i]): [now, False] for i in need if int(t[i]) not in dec})
        decided_path().write_text(json.dumps(dec))
        sigs = read_signals()
    m = t >= state["start"]
    got_f, err_f = fetch_fut(int(state["start"]))
    rp = reprice_binance()
    bars, miss = fut_bars(t[m])
    summary = {
        "bars": len(bars["t"]),
        "fetched": got,
        "fetched_fut": got_f,
        "last": kst(t[-1] + BAR15) if len(t) else "",
        "price": "futures_binance_okx",
        "reprice": rp,
    }
    if err_f or miss:
        summary["fut_gap"] = {"error": err_f, "missing_bars": int(miss)}
    msgs = []
    trades = {}
    sent = set(state["sent"])
    prev = state.get("outcomes", {})
    outs: dict = {}
    flips = []
    # 선물 기준으로 처음 다시 계산할 때는 알림을 보내지 않는다
    migrate = state.get("price") != "futures_okx"
    tr4 = trend4h(d5, bars["t"])
    decided = read_decided()
    est = estimate_decided(sorted(sigs), decided)
    if est:  # 판단 시각을 기록하기 전 신호: 타이머 실행 기록으로 추정(한 번만)
        decided.update(est)
        decided_path().write_text(json.dumps(decided))
    b5 = fut5_bars()
    for r in RULES:
        s0 = state["start"] if PFX[r] == "p_" else state.get("start_s", float("inf"))
        col = BASE_RULE.get(r, r)
        sig = np.array(
            [int(sigs.get(int(x), {}).get(col) or 0) if x >= s0 else 0 for x in bars["t"]]
        )
        if r == "AS4":
            sig = np.where(sig == tr4, sig, 0)
        done, open_ = paper(bars, sig, r)
        funding_cut(done)
        trades[r] = done
        fills = [x for x in done if x["filled"]]
        net = np.array([x["net1"] for x in fills]) if fills else np.zeros(0)
        summary[r] = {
            "signals": int((sig != 0).sum()),
            "fills": len(fills),
            "wins": int(sum(x["win"] for x in fills)),
            "x1": round(float((np.prod(1 + net) - 1) * 100), 2),
            "x3": round(float((np.prod(1 + np.maximum(3 * net, -1)) - 1) * 100), 2),
            "open": open_,
        }
        done_d, open_d = paper_delay(bars, sig, decided, b5)
        funding_cut(done_d)
        summary[r]["delay"] = delay_summary(done_d, open_d)
        for x in done:
            ok = f"{r}:{int(x['t_sig'])}"
            outs[ok] = outcome(x)
            if ok in prev and prev[ok] != outs[ok]:
                flips.append(f"{NAMES[r]} {kst(x['t_sig'] + BAR15)} 신호: {prev[ok]} → {outs[ok]}")
            key = f"{r}:{int(x['t_sig'])}:{'x' if x['filled'] else 'n'}"
            if key in sent:
                continue
            sent.add(key)
            if x["filled"]:
                msgs.append(
                    f"[BTC 기록] {NAMES[r]} {side_kr(x['side'])} 끝: {kst(x['t_fill'])} {x['lim']:,.1f} 체결 → "
                    f"{kst(x['t_exit'])} {'목표 +1.5%' if x['win'] else '손절 −1.5%'}, "
                    f"수수료 뺀 1배 {x['net1'] * 100:+.2f}% / 3배 {x['net1'] * 300:+.2f}%"
                    + probs_text(model_probs(sigs, x["t_sig"], PFX[r]), x["side"])
                )
            else:
                msgs.append(
                    f"[BTC 기록] {NAMES[r]} {side_kr(x['side'])} 신호 {kst(x['t_sig'] + BAR15)}, "
                    f"{x['lim']:,.1f} 지정가 30분 안 닿아 취소"
                )
        if open_:
            key = f"{r}:{int(open_['t_sig'])}:{open_['kind']}"
            if key not in sent:
                sent.add(key)
                if open_["kind"] == "order":
                    msgs.append(
                        f"[BTC 기록] {NAMES[r]} {side_kr(open_['side'])} 신호 {kst(open_['t_sig'] + BAR15)}: "
                        f"{open_['lim']:,.1f}에 지정가 (30분 유효, 주문 안 함)"
                        + probs_text(model_probs(sigs, open_["t_sig"], PFX[r]), open_["side"])
                    )
                else:
                    msgs.append(
                        f"[BTC 기록] {NAMES[r]} {side_kr(open_['side'])} 체결 {kst(open_['t_fill'])} "
                        f"{open_['px']:,.1f}, 목표 {open_['tp']:,.1f} / 손절 {open_['sl']:,.1f}"
                        + probs_text(model_probs(sigs, open_["t_sig"], PFX[r]), open_["side"])
                    )
    # ---- K2: R1 일봉 방향 + AS4 진입 + R1식 청산 (2026-11-07 추가, 추가한 다음 봉부터 기록) ----
    if "start_k2" not in state:
        state["start_k2"] = float(t[-1]) + BAR15
    if len(bars["t"]) and "AS" in meta["thr"]:
        import btc_fut_daily_regime as R1M

        reg = R1M.regime(fut_daily_d5(), {"t": bars["t"]}, 0)[0]
        s0 = state["start_k2"]
        as4 = np.array(
            [int(sigs.get(int(x), {}).get("AS") or 0) if x >= s0 else 0 for x in bars["t"]]
        )
        as4 = np.where(as4 == tr4, as4, 0)
        sig_k = np.where(as4 == reg, as4, 0)
        done, open_ = k2_paper(bars, b5, reg, sig_k, delay=False)
        funding_cut(done)
        if open_ and open_["kind"] == "pos":
            open_["last"] = float(bars["c"][-1])
        trades["K2"] = done
        fills = [x for x in done if x["filled"]]
        net = np.array([x["net1"] for x in fills]) if fills else np.zeros(0)
        # 지연 반영: 판단이 신호 봉 마감 뒤 5분 넘게 늦은 신호는 놓친 것으로 보고,
        # 나머지는 첫 5분봉 체결을 막는다(128번 지연 모형)
        late = np.array(
            [
                bool(v) and (
                    int(x) not in decided or decided[int(x)][0] - (x + BAR15) > M5
                )
                for x, v in zip(bars["t"], sig_k)
            ]
        )
        done_d, open_d = k2_paper(bars, b5, reg, np.where(late, 0, sig_k), delay=True)
        funding_cut(done_d)
        fd = [x for x in done_d if x["filled"]]
        nd = np.array([x["net1"] for x in fd]) if fd else np.zeros(0)
        summary["K2"] = {
            "signals": int((sig_k != 0).sum()),
            "fills": len(fills),
            "wins": int(sum(x["win"] for x in fills)),
            "x1": round(float((np.prod(1 + net) - 1) * 100), 2),
            "x3": round(float((np.prod(1 + np.maximum(3 * net, -1)) - 1) * 100), 2),
            "open": open_,
            "direction": int(reg[-1]),
            "delay": {
                "late": int(late.sum()),
                "fills": len(fd),
                "wins": int(sum(x["win"] for x in fd)),
                "x1": round(float((np.prod(1 + nd) - 1) * 100), 2),
                "x3": round(float((np.prod(1 + np.maximum(3 * nd, -1)) - 1) * 100), 2),
                "open": open_d["kind"] if open_d else None,
            },
        }
        for x in done:
            ok = f"K2:{int(x['t_sig'])}"
            outs[ok] = outcome(x)
            if ok in prev and prev[ok] != outs[ok]:
                flips.append(
                    f"{NAMES['K2']} {kst(x['t_sig'] + BAR15)} 신호: {prev[ok]} → {outs[ok]}"
                )
            key = f"K2:{int(x['t_sig'])}:{'x' if x['filled'] else 'n'}"
            if key in sent:
                continue
            sent.add(key)
            if x["filled"]:
                why = "1% 손절" if x["stopped"] else "일봉 방향 바뀜"
                msgs.append(
                    f"[BTC 기록] {NAMES['K2']} {side_kr(x['side'])} 끝: "
                    f"{kst(x['t_fill'])} {x['lim']:,.1f} 체결 → "
                    f"{kst(x['t_exit'])} {why}, 수수료·펀딩 뺀 1배 {x['net1'] * 100:+.2f}% / "
                    f"3배 {x['net1'] * 300:+.2f}%"
                )
            else:
                msgs.append(
                    f"[BTC 기록] {NAMES['K2']} {side_kr(x['side'])} "
                    f"신호 {kst(x['t_sig'] + BAR15)}, "
                    f"{x['lim']:,.1f} 지정가 30분 안 닿아 취소"
                )
        if open_:
            key = f"K2:{int(open_['t_sig'])}:{open_['kind']}"
            if key not in sent:
                sent.add(key)
                if open_["kind"] == "order":
                    msgs.append(
                        f"[BTC 기록] {NAMES['K2']} {side_kr(open_['side'])} "
                        f"신호 {kst(open_['t_sig'] + BAR15)}: "
                        f"{open_['lim']:,.1f}에 지정가 (30분 유효, 주문 안 함)"
                        + probs_text(model_probs(sigs, open_["t_sig"], "pa_"), open_["side"])
                    )
                else:
                    msgs.append(
                        f"[BTC 기록] {NAMES['K2']} {side_kr(open_['side'])} "
                        f"체결 {kst(open_['t_fill'])} "
                        f"{open_['px']:,.1f}, 손절 {open_['sl']:,.1f}, 일봉 방향이 바뀌면 청산"
                    )
    if migrate:
        msgs = []
        state["price"] = "futures_okx"
    if flips:
        # 바이낸스 봉으로 다시 계산해 지난 결과가 바뀐 경우: 다시 알리지 않고 한 줄로 남긴다
        print(json.dumps({"reprice_flips": flips}, ensure_ascii=False), flush=True)
        state["flip_note"] = (state.get("flip_note", []) + flips)[-20:]
        stamp = kst(time.time() * 1000)
        log = state.get("flips_log", []) + [f"{stamp} {f}" for f in flips]
        state["flips_log"] = log[-100:]
    summary["reprice_flips"] = state.get("flips_log", [])[-10:]
    if summary.get("fut_gap"):
        # 선물 봉이 비면 이번 계산은 일부만이다. 알리지 않고 보낸 표시도 남기지 않는다.
        msgs = []
        if not migrate:
            sent = set(state["sent"])
        print(json.dumps({"fut_gap": summary["fut_gap"]}), flush=True)
    if msgs and state.get("flip_note"):
        msgs[0] = (
            "[BTC 기록] 바이낸스 가격으로 다시 계산해 지난 결과가 바뀜: "
            + "; ".join(state.pop("flip_note"))
            + "\n"
            + msgs[0]
        )
    if not summary.get("fut_gap"):
        prev.update(outs)
        state["outcomes"] = prev
    state["sent"] = sorted(sent)
    st_path.write_text(json.dumps(state))
    (ST / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    write_view(
        summary,
        trades,
        sigs,
        meta,
        state,
        chart_data(
            B,
            X,
            seq,
            t,
            seq_ok,
            models,
            meta,
            trades,
            state["start"],
            {r: summary[r]["open"] for r in list(RULES) + ["K2"] if r in summary},
        ),
    )
    if notify:
        for s in msgs:
            N.send(s)
    print(json.dumps(summary, default=float), *msgs, sep="\n", flush=True)
    return summary


# 128번 K2, 현실 체결(5분 지연 + 0.05% 관통) 2024~ 3배: 1년 약 46회, 승률 12.9%, +65.4%, 낙폭 −78.1%
K2_BACKTEST = {"per_year": 46, "win": 12.9, "x3": 65.4, "mdd": -78.1, "y2023": None}
BACKTEST = {  # 2026-11-04 문서의 2024-01~ 결과 (3배, 수수료 포함, 펀딩 포함)
    "A": {"per_year": 267, "win": 55.1, "x3": 236.4, "mdd": -61.6, "y2023": 7.5},
    "B": {"per_year": 142, "win": None, "x3": 60.0, "mdd": -37.0, "y2023": 64.8},
    # 씨드 5벌 평균 걸어가기, 이미 본 자료라 증거 아님 (2026-11-05 seeds 문서)
    "AS": {"per_year": 279, "win": 53.1, "x3": -13.4, "mdd": -80.4, "y2023": -7.3},
    "BS": {"per_year": 151, "win": 54.9, "x3": 85.5, "mdd": -55.3, "y2023": 21.2},
    # 2026-11-05 mtf 문서, 이미 본 자료 포함. 최근 1년은 +29.8%(필터 없음 +165.5%)
    "AS4": {"per_year": 167, "win": 58.0, "x3": 645.2, "mdd": -44.5, "y2023": 142.3},
    # 2026-11-07 128번 K2(탈락, 이미 본 자료)
    "K2": K2_BACKTEST,
}


CHART_BARS = 288  # 3일


def chart_data(B, X, seq, t, seq_ok, models, meta, trades, start, opens=None) -> dict:
    """화면 차트용: 최근 3일 15분봉, Stoch·CCI·RSI, 모델 점수, 기록 시작 뒤 진입·청산 표시.

    시각은 KST로 옮긴 초(차트 라이브러리는 UTC로 그리므로 9시간을 더한다).
    기록 시작 전 봉의 점수는 같은 모델로 계산한 참고값이다(신호·체결 기록에는 안 쓴다).
    """
    d = B["d"]
    n = len(t)
    a = max(0, n - CHART_BARS)
    sec = lambda ms: int(ms // 1000) + 9 * 3600  # noqa: E731
    k = stoch(d["h"], d["l"], d["c"])
    cc = cci(d["h"], d["l"], d["c"])
    rs = rsi(d["c"])
    idx = np.array([i for i in range(a, n) if seq_ok[i]])
    score = {}
    if len(idx):
        Xn, Sn = norm(X, seq, meta["nz"])
        p = predict(models, Xn, Sn, idx).mean(0)
        score = {int(i): round(float(v), 4) for i, v in zip(idx, p)}
    rnd = lambda v: None if not np.isfinite(v) else round(float(v), 2)  # noqa: E731
    bars = [
        [
            sec(t[i]),
            d["o"][i],
            d["h"][i],
            d["l"][i],
            d["c"][i],
            rnd(k[i]),
            rnd(cc[i]),
            rnd(rs[i]),
            score.get(i),
        ]
        for i in range(a, n)
    ]
    marks = []
    for r, done in trades.items():
        for x in done:
            if not x["filled"]:
                continue
            marks.append(
                {
                    "rule": r,
                    "side": x["side"],
                    "kind": "open",
                    "time": sec(x["t_fill"]),
                    "price": x["lim"],
                }
            )
            marks.append(
                {
                    "rule": r,
                    "side": x["side"],
                    "kind": "close",
                    "time": sec(x["t_exit"]),
                    "win": bool(x["win"]),
                    "net3": round(x["net1"] * 300, 2),
                }
            )
    for r, o in (opens or {}).items():  # 아직 안 끝난 포지션도 진입 표시를 단다
        if o and o["kind"] == "pos":
            marks.append(
                {
                    "rule": r,
                    "side": o["side"],
                    "kind": "open",
                    "time": sec(o["t_fill"]),
                    "price": o["px"],
                }
            )
    return {"bars": bars, "marks": marks, "thr": meta["thr"], "start": sec(start)}


def write_view(summary, trades, sigs, meta, state, chart=None) -> None:
    """웹 화면(비트코인 탭)이 읽는 파일. 웹은 이 파일만 읽고 계산하지 않는다."""
    rows = [sigs[k] for k in sorted(sigs)][-96:]
    recent = [
        {
            "at": kst(int(r["t"]) + BAR15),
            "close": float(r["c"]),
            "score": float(r["score"]),
            "up": int(r["up"]),
            "dn": int(r["dn"]),
            "A": int(r["A"]),
            "B": int(r["B"]),
            "models": {m: float(r[f"p_{m}"]) for m in E.MODELS if r.get(f"p_{m}")},
        }
        for r in reversed(rows)
    ]

    def trade_row(x, pfx="p_"):
        row = {
            "signal_at": kst(x["t_sig"] + BAR15),
            "side": side_kr(x["side"]),
            "limit": x["lim"],
            "models": model_probs(sigs, x["t_sig"], pfx),
        }
        if x["filled"]:
            row.update(
                filled_at=kst(x["t_fill"]),
                exit_at=kst(x["t_exit"]),
                result=(("손절" if x["stopped"] else "방향 전환") if "stopped" in x
                        else ("목표" if x["win"] else "손절")),
                net1=round(x["net1"] * 100, 2),
                net3=round(x["net1"] * 300, 2),
            )
        else:
            row["result"] = "미체결 취소"
        return row

    def open_row(o, pfx="p_"):
        if not o:
            return None
        row = {
            "kind": "지정가 대기" if o["kind"] == "order" else "보유 중",
            "side": side_kr(o["side"]),
            "signal_at": kst(o["t_sig"] + BAR15),
            "models": model_probs(sigs, o["t_sig"], pfx),
        }
        if o["kind"] == "order":
            row["limit"] = o["lim"]
        else:
            row.update(
                filled_at=kst(o["t_fill"]),
                price=o["px"],
                target=o["tp"],
                stop=o["sl"],
                last=o["last"],
                pnl1=round((o["last"] / o["px"] - 1) * o["side"] * 100, 2),
            )
        return row

    rules = []
    for r, (q, k) in RULES.items():
        thr = meta["thr"].get(BASE_RULE.get(r, r))
        if thr is None:
            continue
        s = summary[r]
        rules.append(
            {
                "id": r,
                "name": NAMES[r],
                "top": round((1 - q) * 100),
                "agree": k,
                "long_above": round(thr[0], 4),
                "short_below": round(thr[1], 4),
                "signals": s["signals"],
                "fills": s["fills"],
                "wins": s["wins"],
                "x1": s["x1"],
                "x3": s["x3"],
                "delay": s.get("delay"),
                "x3_delay": (s.get("delay") or {}).get("x3"),
                "open": open_row(s["open"], PFX[r]),
                "note": f"{PRICE_BASIS}. {NOTE[r]}",
                "seed_avg": PFX[r] == "pa_",
                "backtest": BACKTEST[r],
                "trades": [trade_row(x, PFX[r]) for x in reversed(trades[r])][:50],
            }
        )
    if "K2" in summary and meta["thr"].get("AS") is not None:
        s = summary["K2"]
        thr = meta["thr"]["AS"]
        rules.append(
            {
                "id": "K2",
                "name": NAMES["K2"],
                "top": 10,
                "agree": 4,
                "long_above": round(thr[0], 4),
                "short_below": round(thr[1], 4),
                "signals": s["signals"],
                "fills": s["fills"],
                "wins": s["wins"],
                "x1": s["x1"],
                "x3": s["x3"],
                "delay": s.get("delay"),
                "x3_delay": (s.get("delay") or {}).get("x3"),
                "direction": s.get("direction"),
                "open": open_row(s["open"], "pa_"),
                "note": f"{PRICE_BASIS}. {NOTE['K2']}. 지금 일봉 방향: "
                + {1: "위(롱만)", -1: "아래(숏만)", 0: "없음(쉼)"}[s.get("direction", 0)]
                + f". 기록 시작 {kst(state['start_k2'])}",
                "seed_avg": True,
                "backtest": BACKTEST["K2"],
                "trades": [trade_row(x, "pa_") for x in reversed(trades.get("K2", []))][:50],
            }
        )
    view = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 9 * 3600)),
        "start": kst(state["start"]),
        "last_bar": summary["last"],
        "bars": summary["bars"],
        "be": round(M.BE, 4),
        "models": list(E.MODELS),
        "model_names": LABEL,
        "trained": meta.get("trained"),
        "price_basis": PRICE_BASIS,
        "fut_days": json.loads(days_path().read_text()) if days_path().exists() else {},
        "reprice_flips": summary.get("reprice_flips", []),
        "rules": rules,
        "recent": recent,
        "chart": chart,
    }
    tmp = ST / "view.json.tmp"
    tmp.write_text(json.dumps(view, ensure_ascii=False, default=float))
    tmp.replace(ST / "view.json")


def report() -> None:
    s = json.loads((ST / "summary.json").read_text())
    print(json.dumps(s, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "train-seeds", "step", "report"))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    {"train": train, "train-seeds": train_seeds, "step": lambda: step(not a.quiet), "report": report}[a.cmd]()


if __name__ == "__main__":
    main()

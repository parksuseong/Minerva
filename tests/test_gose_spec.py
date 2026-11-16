"""고서 매매 규칙서(gose_specs.json + gose_spec_*.md)가 실제 규칙 코드와 어긋나지 않는지 본다.

1) 규칙서의 핵심 숫자 = 코드 상수. 2) 규칙서 글대로 따로 짠 구현이 `gose_rules.simulate`와 같은
거래를 낸다(합성 일봉). 둘 중 하나가 바뀌면 여기서 깨진다.
"""

from __future__ import annotations

import inspect
import json
import random
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from minerva import gose
from minerva import gose_rules as R
from minerva import gose_rules_v2 as V
from minerva.main import app

INDEX = json.loads(gose.SPECS.read_text(encoding="utf-8"))
ITEMS = {it["key"]: it for it in INDEX["items"]}
STOPS = {key: stop for key, _, stop in gose.ACCOUNTS}


def _md(key):
    return (gose.HERE / ITEMS[key]["md"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("key,sid", [("3-c0/S10", "3-c0"), ("4", "4")])
def test_spec_params_match_code(key, sid):
    p = ITEMS[key]["params"]
    st = R.STRATEGIES[sid]
    assert ITEMS[key]["accounts"] == [key] and key in STOPS
    assert p["stop"] == STOPS[key]
    assert p["max_hold"] == st["max_hold"] and st["side"] == "long" and st["timeframe"] == "D"
    assert p["stoch_os"] == R.OS["stoch"] and p["stoch_ob"] == R.OB["stoch"]
    assert p["stoch_k"] == inspect.signature(R.stoch_fast_k).parameters["n"].default
    assert (p["capital"], p["max_positions"], p["size_frac"]) == (
        gose.CAPITAL,
        gose.MAX_POS,
        gose.SIZE_FRAC,
    )
    assert p["fee_per_side"] == gose.FEE == R.FEE
    assert p["min_bars"] == gose.MIN_BARS
    assert p["history_start"] == "2010-01-01" and gose.HISTORY_START == 1262304000
    src = inspect.getsource(R.indicators)
    for n in p["sma"]:
        assert f"sma(c, {n})" in src
    if sid == "3-c0":
        assert p["cci_mid"] == st["cci_mid"] == 0.0
        assert p["cci"] == inspect.signature(R.cci).parameters["n"].default
        assert p["rsi_wilder"] == inspect.signature(R.rsi).parameters["n"].default
    # 글에도 같은 숫자
    md = _md(key)
    for s in (
        "StochK_{t-1} < 20 ≤ StochK_t",
        "StochK_{t-1} > 80 ≥ StochK_t",
        "≥ 60",
        "10,000",
        "최대 10종목",
        "× 10%",
        "0.05%",
        "(신호일, 티커)",
    ):
        assert s in md, s
    if p["stop"]:
        assert "진입가 × 0.90" in md
    else:
        assert "손절 없음" in md


def test_reference_covers_every_other_account():
    full = {a for it in INDEX["items"] for a in it["accounts"]}
    refs = [r["key"] for r in INDEX["reference"]]
    others = [k for k, _, _ in gose.ACCOUNTS if k not in full]
    assert sorted(refs) == sorted(others)
    for r in INDEX["reference"]:
        mod = V if r["key"] in V.STRATEGIES else R
        assert r["max_hold"] == mod.STRATEGIES[r["key"]]["max_hold"] and r["def"]


def test_specs_have_no_source_identifiers():
    text = gose.SPECS.read_text(encoding="utf-8")
    for it in INDEX["items"]:
        text += _md(it["key"])
    for bad in ("http", "".join(("b", "log")), "".join(("블", "로그")), "logNo"):
        assert bad not in text


def test_route_returns_all_specs():
    s = TestClient(app).get("/api/gose").json()["spec"]
    assert s["status"] == "ok"
    keys = [it["key"] for it in s["items"]]
    assert keys == ["3-c0/S10", "4", "avg"]
    for it in s["items"]:
        assert it["markdown"].startswith("# 고서 매매 규칙서") and it["accounts"]
    avg = s["items"][-1]
    assert avg["accounts"] == ["avg-SOXL", "avg-SOXX"] and avg["params"]["max_adds"] == 2
    assert s["items"][0]["backtest"]["n"] > 0 and s["items"][1]["replay"]["starts"] == 119
    assert len(s["reference"]) == len(INDEX["reference"])


# ---------------------------------------------------------------- 글대로 따로 짠 구현


def _stoch(H, L, C):
    out = [None] * len(C)
    for t in range(4, len(C)):
        hi, lo = max(H[t - 4 : t + 1]), min(L[t - 4 : t + 1])
        out[t] = 50.0 if hi == lo else (C[t] - lo) / (hi - lo) * 100
    return out


def _cci(H, L, C):
    tp = [(h + lo + c) / 3 for h, lo, c in zip(H, L, C)]
    out = [None] * len(C)
    for t in range(19, len(C)):
        w = tp[t - 19 : t + 1]
        m = sum(w) / 20
        md = sum(abs(x - m) for x in w) / 20
        out[t] = 0.0 if md == 0 else (tp[t] - m) / (0.015 * md)
    return out


def _rsi(C):
    out = [None] * len(C)
    if len(C) <= 14:
        return out
    U = [0.0] + [max(C[i] - C[i - 1], 0.0) for i in range(1, len(C))]
    D = [0.0] + [max(C[i - 1] - C[i], 0.0) for i in range(1, len(C))]

    def val(au, ad):
        if ad == 0:
            return 100.0 if au > 0 else 50.0
        return 100 - 100 / (1 + au / ad)

    au, ad = sum(U[1:15]) / 14, sum(D[1:15]) / 14
    out[14] = val(au, ad)
    for t in range(15, len(C)):
        au, ad = (au * 13 + U[t]) / 14, (ad * 13 + D[t]) / 14
        out[t] = val(au, ad)
    return out


def _sma(C, n):
    return [None if t < n - 1 else sum(C[t - n + 1 : t + 1]) / n for t in range(len(C))]


def _frame(bars, kind):
    def key(s):
        d = date.fromisoformat(s)
        return d.isocalendar()[:2] if kind == "W" else (d.year, d.month)

    groups = []
    for i, b in enumerate(bars):
        if groups and key(bars[groups[-1][0]][0]) == key(b[0]):
            groups[-1].append(i)
        else:
            groups.append([i])
    H = [max(bars[j][2] for j in g) for g in groups]
    L = [min(bars[j][3] for j in g) for g in groups]
    C = [bars[g[-1]][4] for g in groups]
    last = date.fromisoformat(bars[-1][0])
    if kind == "W":
        closed_last = last.weekday() == 4
    else:
        nxt = last + timedelta(days=1)
        while nxt.weekday() >= 5:
            nxt += timedelta(days=1)
        closed_last = nxt.month != last.month
    ends = {g[-1]: k for k, g in enumerate(groups) if k < len(groups) - 1 or closed_last}
    latest, cur = [], -1
    for i in range(len(bars)):
        cur = ends.get(i, cur)
        latest.append(cur)
    return _cci(H, L, C), _rsi(C), latest


def _ledger(bars, sid, stop):
    H, L, C = ([b[k] for b in bars] for k in (2, 3, 4))
    S, s50, s200 = _stoch(H, L, C), _sma(C, 50), _sma(C, 200)
    CC, RR = _cci(H, L, C), _rsi(C)
    frames = (_frame(bars, "W"), _frame(bars, "M"))

    def entry(t):
        if t < 1 or S[t] is None or S[t - 1] is None or not S[t - 1] < 20 <= S[t]:
            return False
        if sid == "4":
            return s50[t] is not None and s200[t] is not None and C[t] > s50[t] > s200[t]
        if t < 5:
            return False
        for j in range(t - 4, t + 1):
            if None in (CC[j], RR[j], s50[j]) or not (CC[j] >= 0 and RR[j] >= 50 and C[j] > s50[j]):
                return False
        for cc, rr, latest in frames:
            k = latest[t]
            if k < 0 or None in (cc[k], rr[k]) or not (cc[k] >= 0 and rr[k] >= 50):
                return False
        return True

    trades, pos, order = [], None, None
    for i, (d, o, _h, lo, _c) in enumerate(bars):
        if order == "buy":
            pos = {"ei": i, "ed": d, "px": o, "sig": bars[i - 1][0]}
        elif order == "sell":
            trades.append((pos["sig"], pos["ed"], d, round(o, 9)))
            pos = None
        order = None
        if pos and stop and lo <= pos["px"] * (1 - stop):
            sp = pos["px"] * (1 - stop)
            px = sp if i == pos["ei"] else min(o, sp)
            trades.append((pos["sig"], pos["ed"], d, round(px, 9)))
            pos = None
        if pos:
            if S[i] is not None and S[i - 1] is not None and S[i - 1] > 80 >= S[i]:
                order = "sell"
            elif i - pos["ei"] >= 60:
                order = "sell"
        elif entry(i):
            order = "buy"
    return trades


def _bars(seed, n=900, drift=0.0015, vol=0.025):
    rng = random.Random(seed)
    out, d, c = [], date(2021, 1, 4), 100.0
    for _ in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        o = c * (1 + rng.gauss(0, 0.01))
        c = o * (1 + drift + rng.gauss(0, vol))
        out.append(
            (
                d.isoformat(),
                o,
                max(o, c) * (1 + abs(rng.gauss(0, 0.01))),
                min(o, c) * (1 - abs(rng.gauss(0, 0.01))),
                c,
            )
        )
        d += timedelta(days=1)
        if rng.random() < 0.02:
            d += timedelta(days=1)  # 휴장일 흉내
    return out


@pytest.mark.parametrize("key,sid", [("3-c0/S10", "3-c0"), ("4", "4")])
def test_spec_text_reimplementation_matches_code(key, sid):
    stop = STOPS[key]
    total = 0
    for seed in range(6):
        bars = _bars(seed)
        mine = _ledger(bars, sid, stop)
        res = R.simulate(sid, bars, stop_pct=stop)
        ref = [
            (t["signal_date"], t["entry_date"], t["exit_date"], round(t["exit_px"], 9))
            for t in res["trades"]
        ]
        assert mine == ref, seed
        total += len(ref)
    assert total >= 10  # 거래가 실제로 나와야 비교가 의미 있다

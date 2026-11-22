"""실시간 가상 추적(edge_paper·edge_features): 연구 특징과 같은지, 앞을 안 보는지, 거래 규칙·손익.
네트워크 없음(가짜 get)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from edge_synth import NH, synth

from minerva import edge_features as ef
from minerva import edge_paper as ep

FIX = Path(__file__).parent / "fixtures" / "edge_features_parity.npz"


# ---------------------------------------------------------------- 특징: 연구 코드와 같은 값


@pytest.mark.parametrize("name", ["A", "B"])
def test_symbol_feats_match_research(name):
    ref = np.load(FIX)
    o, h, low, c, q, fh, fr = synth(name)
    big_t, feats, valid = ef.symbol_feats(o, h, low, c, q, fh, fr, NH)
    assert np.array_equal(valid, ref[f"{name}:valid"])
    for k in ef.SYM_KEYS:
        np.testing.assert_allclose(feats[k], ref[f"{name}:{k}"], rtol=0, atol=1e-12,
                                   equal_nan=True, err_msg=k)
    lab = ef.label_ret(o, c, big_t, 24, NH)
    np.testing.assert_allclose(lab, ref[f"{name}:ret24"], rtol=0, atol=1e-12, equal_nan=True)


def test_names_are_research_order():
    assert len(ef.NAMES) == 65
    assert ef.NAMES[:3] == ["r1h", "r4h", "r12h"] and ef.NAMES[-1] == "xs_r24h_dm"


@pytest.mark.parametrize("t", [1500, 1504, 1796])
def test_features_do_not_look_ahead(t):
    """배열을 t에서 자르고 t시 봉을 지운 값 = 전체 배열 값. 뒤 자료를 바꿔도 그대로."""
    o, h, low, c, q, fh, fr = synth("A")
    _, full, vfull = ef.symbol_feats(o, h, low, c, q, fh, fr, NH)
    cut = [a[:t + 1].copy() for a in (o, h, low, c, q)]
    for a in cut:
        a[t] = np.nan
    keep = fh <= t
    _, part, vpart = ef.symbol_feats(*cut, fh[keep], fr[keep], t + 1)
    for k in ef.SYM_KEYS:
        a, b = full[k][t // 4], part[k][-1]
        assert (np.isnan(a) and np.isnan(b)) or a == b, k
    assert vfull[t // 4] == vpart[-1]
    # 미래 가격을 크게 바꿔도 t의 값은 같다
    o2, h2, l2, c2 = (a.copy() for a in (o, h, low, c))
    for a in (o2, h2, l2, c2):
        a[t:] *= 3
    _, alt, _ = ef.symbol_feats(o2, h2, l2, c2, q, fh, fr, NH)
    for k in ef.SYM_KEYS:
        a, b = full[k][t // 4], alt[k][t // 4]
        assert (np.isnan(a) and np.isnan(b)) or a == b, k


def test_cross_section_ranks():
    x = np.full((4, len(ef.SYM_KEYS)), np.nan)
    j = ef.SYM_KEYS.index("r24h")
    x[:, j] = [0.1, -0.2, 0.3, 0.0]
    ti = np.array([5, 5, 5, 6])
    xs = ef.cross_section(x, ti)
    np.testing.assert_allclose(xs[:3, 0], [0.5, 0.0, 1.0])
    assert np.isnan(xs[3, 0])  # 한 행뿐인 시각은 순위 없음
    np.testing.assert_allclose(xs[:3, -1], [0.0, -0.3, 0.2], atol=1e-7)


# ---------------------------------------------------------------- 거래 경로


def _bars(n=60, px=100.0):
    o = np.full(n, px)
    h = np.full(n, px * 1.01)
    low = np.full(n, px * 0.99)
    c = np.full(n, px)
    return o, h, low, c


def test_path_time_exit_and_unknown():
    o, h, low, c = _bars()
    o[34] = 103.0
    assert ep.path_exit(o, h, low, c, 10, 1, horizon=33) is None  # 청산 시가를 아직 모름
    g, xo, st = ep.path_exit(o, h, low, c, 10, 1, horizon=34)
    assert (round(g, 6), xo, st) == (0.03, 24, False)
    g, *_ = ep.path_exit(o, h, low, c, 10, -1, horizon=34)
    assert round(g, 6) == -0.03


def test_path_stop_intrabar_and_gap():
    o, h, low, c = _bars()
    low[15] = 94.0  # 롱 손절 95
    g, xo, st = ep.path_exit(o, h, low, c, 10, 1, horizon=20)
    assert (g, xo, st) == (-0.05, 6, True)
    o, h, low, c = _bars()
    o[15], low[15] = 90.0, 89.0  # 시가가 이미 손절 밖 → 그 시가
    g, xo, st = ep.path_exit(o, h, low, c, 10, 1, horizon=20)
    assert (round(g, 6), xo, st) == (-0.1, 5, True)
    o, h, low, c = _bars()
    h[12] = 106.0  # 숏 손절 105
    g, xo, st = ep.path_exit(o, h, low, c, 10, -1, horizon=12)
    assert (g, xo, st) == (-0.05, 3, True)
    # 손절이 아직 안 났고 그 뒤를 모르면 결론 없음
    o, h, low, c = _bars()
    assert ep.path_exit(o, h, low, c, 10, 1, horizon=12) is None


# ---------------------------------------------------------------- 선택


def _row(sym, s, el=True, entry=1.0):
    return {"sym": sym, "s": s, "conf": abs(s), "el": el, "entry": entry}


def test_select_rules():
    rows = [_row("A", 0.5), _row("B", -0.6), _row("C", 0.3), _row("D", 0.25, el=False),
            _row("E", 0.21), _row("F", 0.199), _row("G", 0.19), _row("H", 0.4, entry=None)]
    v01 = ep.select(rows, {}, ep.RULES["V01"])
    assert [r["sym"] for r in v01] == ["A"]  # 숏 B(더 높음)는 롱만 규칙에서 빠짐
    f = ep.select(rows, {}, ep.RULES["F"])
    assert [r["sym"] for r in f] == ["B"]
    assert [r["sym"] for r in ep.select(rows, {"B": True}, ep.RULES["F"])] == ["A"]
    v06 = ep.select(rows, {"A": True}, ep.RULES["V06"])
    # 자격 없는 D·시가 없는 H 건너뜀, 0.198 아래 G에서 멈춤
    assert [r["sym"] for r in v06] == ["B", "C", "E", "F"]


def test_held_no_overlap():
    trades = [{"rule": "V01", "sym": "A", "t": 100, "x": None},
              {"rule": "V01", "sym": "B", "t": 100, "x": 104},
              {"rule": "V01", "sym": "C", "t": 100, "x": 112},
              {"rule": "F", "sym": "D", "t": 100, "x": None}]
    assert ep.held_at(trades, "V01", 104) == {"A": True, "C": True}
    assert ep.held_at(trades, "V01", 96) == {}


# ---------------------------------------------------------------- 저장소·손익·장부


@pytest.fixture()
def paper_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_EDGE_PAPER_DIR", str(tmp_path))
    ep._MODELS.clear()
    return tmp_path


def _sym_data(n, px=100.0, ft=None, fr=None):
    o, h, low, c = _bars(n, px)
    q = np.full(n, 1e6)
    ft = np.array(ft or [], np.int64)
    return {"o": o, "h": h, "l": low, "c": c, "q": q, "ft": ft,
            "fr": np.array(fr or [], float)}


def test_resolve_trade_costs_and_funding(paper_dir):
    st = ep.Store()
    t = ep.START_H
    d = _sym_data(t + 40, ft=[ep.ef.T0_MS + (t + h) * ep.ef.HR_MS for h in (0, 8, 16, 24, 32)],
                  fr=[0.01, 0.001, 0.002, 0.003, 0.5])
    d["o"][t + 24] = 102.0
    st.save("AAAUSDT", d)
    trades = [{"id": 1, "rule": "V01", "sym": "AAAUSDT", "d": 1, "t": t, "entry": 100.0,
               "x": None},
              {"id": 2, "rule": "F", "sym": "AAAUSDT", "d": -1, "t": t, "entry": 100.0,
               "x": None}]
    assert ep.resolve_trades(st, trades, t + 39) == 2
    long_, short = trades
    # 정산 t(제외), t+8, t+16, t+24(포함), t+32(제외)
    assert long_["x"] == t + 24 and not long_["stopped"]
    assert long_["net"] == pytest.approx(0.02 - 0.006 - 0.0014)
    assert short["net"] == pytest.approx(-0.02 + 0.006 - 0.0014)
    assert long_["exit_px"] == pytest.approx(102.0)
    lines = (paper_dir / "trades.jsonl").read_text().splitlines()
    assert [json.loads(x)["ev"] for x in lines] == ["exit", "exit"]


def test_resolve_waits_for_data_and_delisting(paper_dir):
    st = ep.Store()
    t = ep.START_H
    st.save("AAAUSDT", _sym_data(t + 10))
    tr = [{"id": 1, "rule": "V01", "sym": "AAAUSDT", "d": 1, "t": t, "entry": 100.0, "x": None}]
    assert ep.resolve_trades(st, tr, t + 20) == 0  # 아직 24시간 안 됨
    # 다른 종목은 이어지는데 이 종목만 48시간 넘게 끊김 → 연구처럼 마지막 종가로 청산
    assert ep.resolve_trades(st, tr, t + 9 + 49) == 1
    assert tr[0]["x"] == t + 24 and tr[0]["gross"] == pytest.approx(0.0)


def test_simulate_sizing_and_marks():
    t0 = 24 * 100
    o, h, low, c = _bars(t0 + 80)
    c[t0 + 30:] = 110.0
    prices = {"A": (o, h, low, c), "B": (o, h, low, c)}
    trades = [{"id": 1, "sym": "A", "t": t0, "d": 1, "x": t0 + 24, "net": 0.05},
              {"id": 2, "sym": "B", "t": t0 + 28, "d": 1, "x": None, "net": None}]
    sim = ep.simulate(trades, prices, ep.RULES["V01"], t0, t0 + 40)
    # 0.1배 × 5% = 평가액 +0.5%
    assert sim["equity"] == pytest.approx(1.005)
    # 열린 B: 0.1005배 × (110/100 − 1)
    assert sim["mark"] == pytest.approx(1.005 + 0.1005 * 0.1)
    assert sim["open"][2][1] == pytest.approx(0.1005)
    basket = ep.simulate(trades[:1], prices, ep.RULES["V06"], t0, t0 + 40)
    assert basket["equity"] == pytest.approx(1 + 0.02 * 0.05)


def test_update_symbol_appends_closed_bars(paper_dir):
    st = ep.Store()
    t = ep.START_H
    st.save("AAAUSDT", _sym_data(t, ft=[ep.ef.T0_MS + (t - 8) * ep.ef.HR_MS], fr=[0.001]))
    calls = []

    def fake_get(url):
        calls.append(url)
        if "klines" in url:
            start = int(url.split("startTime=")[1])
            h0 = (start - ep.ef.T0_MS) // ep.ef.HR_MS
            assert h0 == t
            return [[ep.ef.T0_MS + (h0 + i) * ep.ef.HR_MS, "1", "2", "0.5", "1.5", "0", 0, "7"]
                    for i in range(5)]
        if "fundingRate" in url:
            return [{"fundingTime": ep.ef.T0_MS + (t - 8) * ep.ef.HR_MS, "fundingRate": "0.001"},
                    {"fundingTime": ep.ef.T0_MS + t * ep.ef.HR_MS + 3, "fundingRate": "0.002"}]
        raise AssertionError(url)

    ep.update_symbol(st, "AAAUSDT", t + 3, get=fake_get)
    d = ep.Store().load("AAAUSDT")
    assert len(d["c"]) == t + 3  # 마감 안 된 t+3, t+4시 봉은 버림
    assert d["c"][t + 2] == 1.5 and d["q"][t] == 7.0
    assert list(d["fr"]) == [0.001, 0.002]


def test_universe_rules():
    syms = ["BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT", "USDCUSDT", "NEWUSDT"]
    nd = 200
    q = np.full((len(syms), nd), np.nan)
    q[:, :] = [[9e9], [5e9], [2e9], [1e9], [8e9], [7e9]]
    first = np.array([0, 0, 0, 0, 0, nd - 50])  # NEW는 60일 안 됨
    p = ep.Panel(syms, first, q)
    key = ep.month_key(nd * 24)
    u30 = ep.compute_u30(p, key)
    assert u30 == ["AAAUSDT", "BBBUSDT"]  # 코어·스테이블·새 종목 제외
    assert ep.compute_universe(p, key, u30) == ["BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT"]
    assert p.vol_ok("BBBUSDT", key) and p.age_ok("BBBUSDT", key)
    assert not p.age_ok("NEWUSDT", key)


# ---------------------------------------------------------------- 전체 실행(가짜 모델·자료)


def test_run_end_to_end(paper_dir, monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    t_start = ep.START_H
    n = t_start + 30
    syms = ["BTCUSDT", "ETHUSDT", "AAAUSDT", "BBBUSDT"]
    st = ep.Store()
    rng = np.random.default_rng(3)
    for k, s in enumerate(syms):
        r = rng.normal(0, 0.01, n)
        c = (100 + 10 * k) * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]
        d = {"o": o, "h": np.maximum(o, c) * 1.002, "l": np.minimum(o, c) * 0.998, "c": c,
             "q": np.full(n, 1e7), "ft": np.array([ep.ef.T0_MS], np.int64),
             "fr": np.array([1e-4])}
        for kk in "ohlcq":
            d[kk][:t_start - 24 * 80] = np.nan  # 80일 자료만
        st.save(s, d)
    nd = t_start // 24 + 1
    q = np.full((len(syms), nd), 5e8)
    ep.Panel(syms, np.zeros(len(syms), int), q).save()
    (paper_dir / "universe.json").write_text(json.dumps(
        {"u30": {"2026-11": ["AAAUSDT", "BBBUSDT"]}, "U": {"2026-11": syms}}))
    # 가짜 모델: 특징과 무관하게 거의 같은 확률 → 확신도는 r24h 순서로 조금씩 다름
    x = rng.normal(0, 1, (3000, 65)).astype(np.float32)
    y = np.where(x[:, 3] > 0.3, 2, np.where(x[:, 3] < -0.3, 0, 1))
    m = lgb.train(dict(objective="multiclass", num_class=3, verbose=-1, num_threads=1,
                       min_data_in_leaf=20), lgb.Dataset(x, y), 20)
    ep.models_dir().mkdir(parents=True)
    m.save_model(str(ep.models_dir() / "H24G_2026-11.txt"))
    # F는 백필 시작부터 교체된 것으로 → 새 거래가 없어야 한다
    monkeypatch.setitem(ep.RETIRED, "F", {"date": "2026-11-01", "reason": "시험"})
    monkey_thr = ep.THR
    ep.THR = 0.0  # 가짜 모델 확신도로도 거래가 나게
    try:
        now = (ep.ef.T0_MS + (t_start + 30) * ep.ef.HR_MS) / 1000
        out = ep.run(fetch=False, now=now, get=None)
    finally:
        ep.THR = monkey_thr
    assert out["processed"] == 8  # T = 0,4,…,28시(마감한 t+29시까지)
    state = json.loads((paper_dir / "state.json").read_text())
    assert not [t for t in state["trades"] if t["rule"] == "F"]
    for rk, rule in ep.RULES.items():
        trs = [t for t in state["trades"] if t["rule"] == rk]
        if rk == "F":
            continue
        assert trs
        if rule["long_only"]:
            assert all(t["d"] == 1 for t in trs)
        # 같은 종목이 겹쳐 들리지 않는다
        for a in trs:
            for b in trs:
                if a["id"] < b["id"] and a["sym"] == b["sym"]:
                    assert a["x"] is not None and a["x"] <= b["t"]
        per_t = {}
        for t in trs:
            per_t[t["t"]] = per_t.get(t["t"], 0) + 1
        assert max(per_t.values()) <= rule["k"]
    dec = (paper_dir / "decisions.jsonl").read_text().splitlines()
    assert len(dec) == 8 and json.loads(dec[0])["t"].startswith("2026-11-01T00")
    summ = json.loads((paper_dir / "summary.json").read_text())
    assert [r["key"] for r in summ["rules"]] == ["V01", "V06", "F"]
    assert {r["key"] for r in summ["retired"]} == {"V06", "F"}
    assert summ["rules"][1]["retired"]["date"] == "2026-11-22"
    assert summ["rules"][0]["retired"] is None
    # 11/1 백필 결정만 있으니 판정 집계(실시간 11/21 08시 UTC 이후)는 0건
    assert summ["rules"][0]["live_since"].startswith("2026-11-21T08")
    assert summ["rules"][0]["judged_n"] == 0
    # 다시 돌려도 같은 결정을 두 번 하지 않는다
    again = ep.run(fetch=False, now=now, get=None)
    assert again["processed"] == 0
    assert len((paper_dir / "decisions.jsonl").read_text().splitlines()) == 8


def test_retired_rules_stop_new_entries(paper_dir, monkeypatch):
    stop = (date(2026, 11, 22) - ep.D0).days * 24
    assert ep.retired_from("V06") == stop and ep.retired_from("F") == stop
    assert ep.retired_from("V01") is None
    # 교체 시각을 백필 시작으로 당기면 V06·F는 새 거래가 없다
    monkeypatch.setitem(ep.RETIRED, "V06", {"date": "2026-11-01", "reason": "시험"})
    assert ep.retired_from("V06") == ep.START_H


# ---------------------------------------------------------------- 화면 API


def test_api_edge_paper(paper_dir):
    from fastapi.testclient import TestClient

    from minerva.main import app

    client = TestClient(app)
    assert client.get("/api/edge-paper").json()["status"] == "pending"
    (paper_dir / "summary.json").write_text(json.dumps(
        {"status": "ok", "generated_at": "2000-01-01T00:00:00+00:00", "rules": []}))
    d = client.get("/api/edge-paper").json()
    assert d["status"] == "ok" and d["stale"] is True


def test_members_ready_waits_for_lagging_symbol(paper_dir):
    st = ep.Store()
    t = ep.START_H + 100
    st.save("AAAUSDT", _sym_data(t + 1))
    st.save("BBBUSDT", _sym_data(t - 3))
    assert not ep.members_ready(st, ["AAAUSDT", "BBBUSDT"], t, t)
    st.save("BBBUSDT", _sym_data(t + 1))
    assert ep.members_ready(st, ["AAAUSDT", "BBBUSDT"], t, t)
    st.save("CCCUSDT", _sym_data(t - 60))  # 48시간 넘게 끊김 → 상장폐지로 보고 기다리지 않음
    assert ep.members_ready(st, ["AAAUSDT", "CCCUSDT"], t, t)

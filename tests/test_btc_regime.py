import json

from fastapi.testclient import TestClient

from minerva import btc_regime as m
from minerva.main import app

WEEK = 7 * 86400 * 1000


def _kl(closes, open_last=False, now=10**13):
    out = []
    for i, c in enumerate(closes):
        t = i * WEEK
        close_t = (now + 1) if (open_last and i == len(closes) - 1) else t + WEEK - 1
        out.append([t, str(c), str(c * 1.01), str(c * 0.99), str(c), "0", close_t])
    return out


def test_zigzag_marks_turns_after_threshold():
    c = [100, 120, 160, 130, 110, 140]
    piv = m.zigzag(c, 0.15)
    assert piv == [(0, "L"), (2, "H"), (4, "L"), (5, "H")]


def test_labels_bull_waves_and_bear_waves():
    # 100 → 300 (상승장), 250 조정, 320 상승, 그 뒤 120까지 (하락장), 150 반등
    c = [100, 200, 300, 250, 320, 200, 120, 150]
    reg, major = m.labels(c)
    assert reg[1:5] == ["up", "up", "corr", "up"]
    assert reg[5:7] == ["down", "down"] and reg[7] == "bounce"
    assert major[-2] == (4, "H")


def test_build_marks_forming_week_and_segments():
    d = m.build(_kl([100, 200, 300, 250, 320, 200, 120, 150], open_last=True), now_ms=10**13)
    assert d["bars"][-1]["open"] is True and "open" not in d["bars"][-2]
    assert [s["r"] for s in d["segments"]] == ["up", "corr", "up", "down", "bounce"]
    assert sum(s["n"] for s in d["segments"]) == 8
    assert d["major"]["dir"] == "down" and d["major"]["price"] == 320


def test_api_pending_then_ok(tmp_path, monkeypatch):
    f = tmp_path / "regime.json"
    monkeypatch.setenv("MINERVA_BTC_REGIME_OUT", str(f))
    client = TestClient(app)
    assert client.get("/api/btc/regime").json() == {"status": "pending"}
    m.write(m.build(_kl([100, 200, 300, 250]), now_ms=10**13), f)
    r = client.get("/api/btc/regime").json()
    assert r["status"] == "ok" and len(r["bars"]) == 4
    assert json.loads(f.read_text())["symbol"] == "BTCUSDT"


def test_scan_levels_use_only_past_closes():
    c = [100, 120, 160, 130, 110, 140]
    s = m.scan(c, 0.15)
    # 1주: 100에서 +20% → 오름 파동 확정, 전환 가격 = 최고 종가 × 0.85
    assert s["dir"][1] == 1 and s["level"][1] == 120 * 0.85
    assert s["level"][2] == 160 * 0.85
    # 3주 130 ≤ 136 → 고점(2주) 확정, 내림 파동, 전환 = 최저 × 1.15
    assert s["piv"][1] == (2, "H", 3) and s["dir"][3] == -1 and s["level"][3] == 130 * 1.15
    # 앞 값은 뒤 종가가 바뀌어도 그대로(앞만 봄)
    s2 = m.scan(c[:4] + [500, 10], 0.15)
    assert s2["level"][:4] == s["level"][:4]


def test_build_turns_and_now():
    d = m.build(_kl([100, 200, 300, 250, 320, 200, 120, 150]), now_ms=10**13)
    t = d["turns"]
    assert all(x["lag"] >= 1 for x in t) and t[1]["kind"] == "H"
    assert d["now"]["minor"]["dir"] == "up" and d["now"]["minor"]["level"] == 127.5
    assert d["bars"][-1]["sw"] == 127.5 and d["bars"][-1]["swd"] == 1


def test_build_has_weekly_scr_values():
    closes = [100 + (i % 7) * 5 + i for i in range(40)]
    d = m.build(_kl(closes), now_ms=10**13)
    b = d["bars"]
    assert b[3]["st"] is None and b[4]["st"] is not None  # Stoch 5주
    assert b[18]["cci"] is None and b[19]["cci"] is not None  # CCI 20주
    assert b[13]["rsi"] is None and b[14]["rsi"] is not None  # RSI 14주
    assert 0 <= b[-1]["st"] <= 100 and 0 <= b[-1]["rsi"] <= 100
    assert d["scr"]["stoch"] == 5 and d["scr"]["cci_hi"] == 100


def test_timeframes_use_own_thresholds_and_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_BTC_REGIME_OUT", str(tmp_path / "regime.json"))
    assert m.out_path("1w").name == "regime.json"
    assert m.out_path("4h").name == "regime_4h.json"
    closes = [100, 108, 112, 100, 97, 103]  # 4h 작은 파동 5%면 전환이 생기고 주봉 15%면 없음
    d4 = m.build(_kl(closes), now_ms=10**13, tf="4h")
    assert d4["minor_pct"] == 0.05 and d4["unit"] == "봉" and len(d4["turns"]) >= 2
    assert len(m.scan(closes, m.TF["1w"]["minor"])["piv"]) == 0
    client = TestClient(app)
    assert client.get("/api/btc/regime?tf=4h").json() == {"status": "pending"}
    m.write(d4, m.out_path("4h"))
    assert client.get("/api/btc/regime?tf=4h").json()["tf"] == "4h"
    assert client.get("/api/btc/regime?tf=5m").status_code == 400


def test_bars_after_last_extreme_keep_current_wave():
    # 300에서 270까지 내림(작은 파동 15% 안 넘음) → 아직 상승 파동
    reg, _ = m.labels([100, 200, 300, 280, 270])
    assert reg[3] == reg[4] == reg[2] == "up"


def test_build_has_short_mas_and_stoch_d():
    closes = [100 + i for i in range(12)]
    b = m.build(_kl(closes), now_ms=10**13)["bars"]
    assert b[3]["s5"] is None and b[4]["s5"] == 102.0
    assert b[9]["s10"] == 104.5
    assert b[5]["sd"] is None and b[6]["sd"] is not None  # %K 첫 값 4번 → %D 6번부터
    assert abs(b[8]["sd"] - (b[6]["st"] + b[7]["st"] + b[8]["st"]) / 3) < 0.11

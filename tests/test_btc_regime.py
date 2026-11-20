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

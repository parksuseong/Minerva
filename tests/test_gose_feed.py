"""고서 원전 새 글(minerva.gose_feed). 네트워크·Claude·텔레그램 없음.

원전 아이디·글 번호는 가짜다."""

import json
import math
import os
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from minerva import gose_feed as gb
from minerva.main import app
from minerva.strategy.indicators import Bar

BID = "fakefeed01"
NOW = datetime(2026, 11, 16, 23, 10, tzinfo=gb.KST)
MS = int(datetime(2026, 11, 16, 20, 0, tzinfo=gb.KST).timestamp() * 1000)


@pytest.fixture
def bdir(tmp_path, monkeypatch):
    d = tmp_path / "feed"
    (d / "posts").mkdir(parents=True)
    (d / "rulebook").mkdir()
    (d / "config.json").write_text(json.dumps({"feed_id": BID}))
    (d / "index.json").write_text(json.dumps(
        {"100000000001": {"title": "옛 글", "cat": "SCR 연구실", "date": "2026-10-30 20:19"}}))
    (d / "rulebook" / "summary.md").write_text("# 룰북 요약\n")
    (d / "rulebook" / "rulebook.md").write_text(
        "## [L] 롱 진입\n### L-06 3단 돌파 롱 — **가능**\n### V-01 VIX 10선 — **애매**\n")
    monkeypatch.setenv("MINERVA_feed_DIR", str(d))
    monkeypatch.setenv("MINERVA_GOSE_DIR", str(tmp_path / "gose"))
    return d


def _item(no, title, cat="SCR\xa0연구실", ms=MS):
    return {"logNo": no, "titleWithInspectMessage": title, "categoryName": cat, "addDate": ms}


LIST = {"isSuccess": True, "result": {"items": [
    _item(100000000003, "새 글 둘", ms=MS + 60_000),
    _item(100000000002, "새 글 하나", cat="VIX\xa0연구실"),
    _item(100000000001, "옛 글"),
]}}

PAGE = """<html><div class="se-viewer"><div class="se-main-container">
<p class="se-text-paragraph"><span>SOXX 주봉 &quot;3단&quot; 이탈</span></p>
<p class="se-text-paragraph"><span>​</span></p>
<img data-lazy-src="https://postfiles.pstatic.net/a/b.png?type=w80">
<p class="se-text-paragraph"><span>숏 각입니다</span></p>
<script>var x = 1;</script>
</div></div><div class="_checkVisible"></div><img data-lazy-src="https://x.pstatic.net/profile.jpg"></html>"""

ANSWER = {
    "summary": "SOXX 주봉이 꺾였다는 글. 숏 쪽을 본다. 자세한 건 https://example.com/100000000003",
    "new_or_changed_rules": [
        {"topic": "숏 진입·인버스", "rule_paraphrase": "주봉 3단 이탈 뒤 숏",
         "mechanizable": "가능", "relates_to": ["S-01", "bad id"], "conflict": None},
        {"topic": "x", "rule_paraphrase": "틀린 항목", "mechanizable": "몰라"},
    ],
    "calls": [{"symbol": "soxx", "instrument": "SOXS", "side": "short", "timeframe": "주봉",
               "author_reason_paraphrase": "주봉 SCR 3단 이탈", "levels": None,
               "confidence_words": "강하게"}],
    "is_trade_call": True,
}


class FakeNet:
    def __init__(self, listing=LIST, page=PAGE, fail_list=False):
        self.urls, self.listing, self.page, self.fail_list = [], listing, page, fail_list

    def __call__(self, url, referer=None, timeout=20):
        self.urls.append(url)
        if "post-list" in url:
            if self.fail_list:
                raise OSError("down")
            return json.dumps(self.listing).encode()
        if "rss" in url:
            return RSS
        if "PostView" in url:
            return self.page.encode()
        if "exchangeInfo" in url:
            return json.dumps({"symbols": [{"symbol": "SOXXUSDT", "status": "TRADING",
                                            "quoteAsset": "USDT"}]}).encode()
        return b"\x89PNGfake"


RSS = f"""<?xml version="1.0"?><rss><channel>
<item><title>RSS 새 글</title><link>https://feed.example.com/{BID}/100000000009?fromRss=true</link>
<category>VIX 연구실</category><pubDate>Mon, 16 Nov 2026 11:00:00 +0000</pubDate></item>
<item><title>옛 글</title><link>https://feed.example.com/{BID}/100000000001</link>
<category>SCR 연구실</category><pubDate>Fri, 30 Oct 2026 11:19:00 +0000</pubDate></item>
</channel></rss>""".encode()


def _bars(trend: float, n: int = 900) -> list[Bar]:
    d0 = date(2023, 1, 2)
    out = []
    for i in range(n):
        c = 100 + trend * i + 3 * math.sin(i / 3)
        out.append(Bar(d0 + timedelta(days=i), c, c + 1, c - 1, c))
    return out


def test_parse_list_and_rss():
    got = gb.parse_list(LIST)
    assert got[0] == {"id": "100000000003", "title": "새 글 둘", "cat": "SCR 연구실",
                      "date": "2026-11-16 20:01"}
    with pytest.raises(ValueError):
        gb.parse_list({"isSuccess": False})
    rss = gb.parse_rss(RSS)
    assert rss[0]["id"] == "100000000009" and rss[0]["date"] == "2026-11-16 20:00"
    assert rss[0]["cat"] == "VIX 연구실"


def test_new_posts_newest_first_and_limit():
    known = {"100000000001": {}}
    todo, total = gb.new_posts(gb.parse_list(LIST) * 2, known, limit=1)
    assert total == 2 and [p["id"] for p in todo] == ["100000000003"]


def test_fetch_list_falls_back_to_rss(bdir):
    net = FakeNet(fail_list=True)
    listed, src = gb.fetch_list(BID, {}, net, lambda s: None)
    assert src == "rss" and listed[0]["id"] == "100000000009"


def test_extract_and_text_format():
    body, urls = gb.extract(PAGE)
    assert body == 'SOXX 주봉 "3단" 이탈\n숏 각입니다'
    assert urls == ["https://postfiles.pstatic.net/a/b.png?type=w800"]
    post = {"id": "100000000003", "title": "T", "cat": "SCR 연구실", "date": "2026-11-16 20:01"}
    assert gb.post_text(post, body, 1).startswith(
        "# T\n2026-11-16 20:01 | SCR 연구실 | 100000000003 | imgs=1\n\n")


def test_merge_json_keeps_other_writers(bdir):
    man = bdir / "images" / "manifest.json"
    man.parent.mkdir()
    man.write_text(json.dumps({"other": ["00.png"]}))
    gb.merge_json(man, "100000000003", ["00.png"])
    assert json.loads(man.read_text()) == {"other": ["00.png"], "100000000003": ["00.png"]}
    assert not [p for p in man.parent.iterdir() if p.name.endswith(".tmp")]
    man.write_text("{half")
    with pytest.raises(ValueError):
        gb.merge_json(man, "x", [])


def test_validate():
    got, err = gb.validate("```json\n" + json.dumps(ANSWER, ensure_ascii=False) + "\n```")
    assert err == "" and got["dropped"] == 1
    assert got["new_or_changed_rules"][0]["relates_to"] == ["S-01"]
    assert got["calls"][0]["symbol"] == "SOXX" and got["is_trade_call"]
    assert gb.validate("not json")[0] is None
    assert gb.validate(json.dumps({"summary": "", "calls": []}))[1] == "summary 없음"
    bad = {**ANSWER, "is_trade_call": "yes"}
    assert gb.validate(json.dumps(bad))[0] is None


def test_verdict_rules():
    fr = lambda d, w, m: {"frames": {"D": {"dir": d}, "W": {"dir": w}, "M": {"dir": m}}}  # noqa: E731
    assert gb.verdict("short", fr("하방", "하방", "중립"))[0] == "추천"
    assert gb.verdict("short", fr("하방", "하방", "상방")) == ("관망", "월봉 SCR이 상방")
    assert gb.verdict("long", fr("상방", "중립", "상방")) == ("관망", "주봉 SCR이 상방 아님")
    assert gb.verdict("long", fr("상방", "상방", "상방"), stale=True)[0] == "관망"
    assert gb.verdict("long", {"error": "자료 부족"}) == ("관망", "자료 부족")


def test_scr_check_directions():
    up = gb.scr_check("X", fetch=lambda t: _bars(0.3))
    assert up["frames"]["W"]["dir"] == "상방" and up["frames"]["M"]["dir"] == "상방"
    down = gb.scr_check("X", fetch=lambda t: _bars(-0.1))
    assert down["frames"]["M"]["dir"] == "하방"
    assert gb.scr_check("X", fetch=lambda t: (_ for _ in ()).throw(OSError()))["error"]


def test_dry_run_writes_nothing(bdir):
    net = FakeNet()
    out = gb.run(NOW, dry=True, get=net, sleep=lambda s: None,
                 caller=lambda p: pytest.fail("Claude 호출"), send=lambda t: pytest.fail("알림"))
    assert out["source"] == "list" and out["new_total"] == 2 and out["todo"] == 2
    assert not any("PostView" in u for u in net.urls)
    assert sorted(p.name for p in bdir.iterdir()) == ["config.json", "index.json", "posts",
                                                      "rulebook"]


def test_no_new_posts_sends_nothing(bdir):
    listing = {"isSuccess": True, "result": {"items": [_item(100000000001, "옛 글")]}}
    sent = []
    out = gb.run(NOW, get=FakeNet(listing=listing), sleep=lambda s: None,
                 caller=lambda p: pytest.fail("Claude 호출"), send=sent.append)
    assert out["todo"] == 0 and sent == []
    assert not (bdir / "rulebook" / "updates.jsonl").exists()


def test_full_run(bdir, tmp_path):
    prompts, sent = [], []

    def caller(p):
        prompts.append(p)
        if len(prompts) == 2:
            raise RuntimeError("boom")
        return json.dumps(ANSWER, ensure_ascii=False)

    out = gb.run(NOW, get=FakeNet(), sleep=lambda s: None, caller=caller,
                 fetch=lambda t: _bars(-0.1), send=sent.append)
    assert out["done"] == 2 and out["errors"] == 1
    # 저장: 본문·index·그림·manifest
    idx = json.loads((bdir / "index.json").read_text())
    assert idx["100000000003"]["cat"] == "SCR 연구실" and "100000000001" in idx
    assert (bdir / "posts" / "100000000003.txt").read_text().splitlines()[1].endswith("imgs=1")
    img = bdir / "images" / "100000000003"
    assert (img / "00.png").exists() and oct(os.stat(img).st_mode & 0o777) == "0o700"
    assert json.loads((bdir / "images" / "manifest.json").read_text())["100000000003"] == [
        "00.png"]
    # 프롬프트: 규칙 목록·그림 경로
    assert "L-06 3단 돌파 롱 [가능]" in prompts[0] and str(img / "00.png") in prompts[0]
    # 비공개 기록엔 글 번호, 화면 사본엔 없음
    priv = (bdir / "rulebook" / "updates.jsonl").read_text()
    assert "100000000003" in priv
    web = (tmp_path / "gose" / "feed_updates.json").read_text()
    assert "100000000" not in web and "http" not in web and BID not in web
    entries = json.loads(web)["entries"]
    call = entries[0]["calls"][0]
    assert call["verdict"] in ("추천", "관망") and call["perp"] == "SOXXUSDT"
    assert entries[1]["error"].startswith("호출 실패")
    # 텔레그램 한 번
    assert len(sent) == 1
    msg = sent[0]
    assert "새 글 2개" in msg and gb.LABEL in msg and "원전 의견: SOXS→SOXX 숏" in msg
    assert "우리 점검: 일 " in msg and "판단: " in msg and "분석 실패" in msg
    # 다시 돌리면 새 글 없음
    again = gb.run(NOW, get=FakeNet(), sleep=lambda s: None, caller=caller, send=sent.append)
    assert again["todo"] == 0 and len(sent) == 1


def test_web_copy_capped(bdir, tmp_path):
    rec = {"id": "100000000003", "title": "t 100000000003", "cat": "c", "date": "2026-11-16 20:00",
           "analysis": None, "error": "x"}
    for _ in range(gb.KEEP + 5):
        gb.save_records([rec], BID, NOW)
    data = json.loads((tmp_path / "gose" / "feed_updates.json").read_text())
    assert len(data["entries"]) == gb.KEEP and "100000000003" not in json.dumps(data)


def test_message_stale_and_length():
    rec = {"id": "1", "title": "T", "cat": "C", "date": "2026-11-01 10:00", "analysis": {
        "summary": "요약\n둘째 줄", "new_or_changed_rules": [], "calls": []}}
    msg = gb.message([rec] * 200, remaining=3)
    assert len(msg) <= 3900 and "남은 글 3개" in msg and "둘째 줄" not in msg


def test_route_feed_section(bdir, tmp_path):
    d = TestClient(app).get("/api/gose").json()
    assert d["feed"] == {"status": "pending"}
    gb.save_records([{"id": "100000000003", "title": "t", "cat": "c", "date": "2026-11-16 20:00",
                      "analysis": None, "error": "x"}], BID, NOW)
    d = TestClient(app).get("/api/gose").json()
    assert d["feed"]["status"] == "ok" and d["feed"]["entries"][0]["title"] == "t"


def test_call_claude_read_only_images(monkeypatch, tmp_path):
    from minerva import btc_ai

    seen = {}

    class P:
        returncode = 0
        pid = 1

        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd

        def communicate(self, timeout=None):
            return "{}", ""

    monkeypatch.setattr(btc_ai.subprocess, "Popen", P)
    btc_ai.call_claude("hi", allowed_tools=["Read", "Bash-not"], add_dirs=[tmp_path])
    cmd = seen["cmd"]
    blocked = cmd[cmd.index("--disallowedTools") + 1: cmd.index("--allowedTools")]
    assert "Read" not in blocked and "Bash" in blocked
    assert cmd[cmd.index("--allowedTools") + 1] == "Read"
    assert cmd[cmd.index("--add-dir") + 1] == str(tmp_path)
    btc_ai.call_claude("hi")
    assert "--allowedTools" not in seen["cmd"] and "Read" in seen["cmd"]

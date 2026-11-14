"""화면 공용 도구(static/common.js)와 그것을 쓰는 스크립트 연결."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "src" / "minerva" / "static"
COMMON_JS = STATIC / "common.js"


def bun() -> str | None:
    found = shutil.which("bun")
    if found:
        return found
    home = Path(os.path.expanduser("~/.bun/bin/bun"))
    return str(home) if home.exists() else None


def run_js(tmp_path: Path, body: str) -> object:
    script = tmp_path / "run.js"
    # 브라우저가 없어 window·document·getComputedStyle을 흉내 낸다
    script.write_text(
        "globalThis.window = globalThis;\n"
        "globalThis.document = { documentElement: {} };\n"
        "globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });\n"
        f"require({json.dumps(str(COMMON_JS))});\n"
        "const U = window.MinervaUI;\n" + body
    )
    out = subprocess.run([bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_common_loads_before_scripts_that_use_it():
    html = (STATIC / "index.html").read_text()
    order = re.findall(r'<script src="/static/([\w.-]+)\.js', html)
    assert order[0] == "common"
    users = [p.stem for p in STATIC.glob("*.js") if "MinervaUI" in p.read_text()]
    assert set(users) - {"common"} <= set(order)


@pytest.mark.skipif(bun() is None, reason="JS 실행기(bun)가 없다")
def test_esc_num_kst(tmp_path):
    got = run_js(
        tmp_path,
        "console.log(JSON.stringify([U.esc('<a href=\"x\">&</a>'), U.esc(null), U.num(null, 2),"
        " U.num(1234.5678, 2), U.num(1234.5678, 0), U.kst(0), U.pad(7)]));\n",
    )
    assert got == [
        "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;",
        "",
        "-",
        "1,234.57",
        "1,235",
        9 * 3600,
        "07",
    ]


@pytest.mark.skipif(bun() is None, reason="JS 실행기(bun)가 없다")
def test_scr_chart_and_fill_with_fake_library(tmp_path):
    # 가짜 차트 라이브러리: 시리즈별 setData·가로선 기록만 남긴다
    got = run_js(
        tmp_path,
        """
const made = [];
const fakeSeries = (kind, opts, pane) => {
  const s = { kind, opts, pane, data: null, lines: [], removed: 0 };
  s.setData = (d) => { s.data = d; };
  s.createPriceLine = (o) => { s.lines.push(o); return o; };
  s.removePriceLine = () => { s.removed += 1; };
  made.push(s);
  return s;
};
window.LightweightCharts = {
  CrosshairMode: { Normal: 0 },
  CandlestickSeries: "candle", LineSeries: "line",
  createChart: (el, opts) => ({
    opts,
    addSeries: (kind, o, pane) => fakeSeries(kind, o, pane),
    panes: () => [0, 1, 2, 3].map(() => ({ setStretchFactor() {} })),
  }),
};
const { chart, series } = U.scrChart({}, { timeFormatter: (t) => t, titles: ["S", "C", "R"],
  up: "G", down: "R" });
const bars = [[0, 1, 2, 0.5, 1.5], [900000, 1.5, 3, 1, 2], [1800000, 2, 4, 1, 3]];
const res = { stoch: [null, 10, 20], cci: [1, NaN, 3], rsi: [5, 6, 7] };
const p = { stoch_lo: 20, stoch_hi: 80, cci_lo: -100, cci_hi: 100, rsi_lo: 30, rsi_hi: 70 };
const first = U.scrFill(series, bars, 1, res, p, "G", "R");
first.guide(series.candle, 2.5, "#888", "진입", 1);
const again = U.scrFill(series, bars, 1, res, p, "G", "R");
console.log(JSON.stringify({
  hasTick: "tickMarkFormatter" in chart.opts.timeScale,
  titles: [series.stoch.opts.title, series.cci.opts.title, series.rsi.opts.title],
  panes: [series.candle.pane, series.stoch.pane, series.cci.pane, series.rsi.pane],
  times: again.times,
  candle: series.candle.data,
  stoch: series.stoch.data, cci: series.cci.data,
  stochLines: series.stoch.lines.map((l) => [l.price, l.color]),
  candleLines: series.candle.lines.map((l) => [l.price, l.title, l.lineStyle]),
  removed: made.map((s) => s.removed),
  tracked: series.lines.length,
}));
""",
    )
    t1, t2 = 900 + 9 * 3600, 1800 + 9 * 3600
    assert got["hasTick"] is False
    assert got["titles"] == ["S", "C", "R"] and got["panes"] == [0, 1, 2, 3]
    assert got["times"] == [t1, t2]
    assert got["candle"] == [
        {"time": t1, "open": 1.5, "high": 3, "low": 1, "close": 2},
        {"time": t2, "open": 2, "high": 4, "low": 1, "close": 3},
    ]
    assert got["stoch"] == [{"time": t1, "value": 10}, {"time": t2, "value": 20}]
    assert got["cci"] == [{"time": t2, "value": 3}]  # NaN은 뺀다
    assert got["stochLines"][-2:] == [[20, "G"], [80, "R"]]
    assert got["candleLines"] == [[2.5, "진입", 1]]
    # 두 번째 채우기는 앞서 그은 7개(기준선 6 + 진입선)를 지우고 기준선 6개만 다시 긋는다
    assert sum(got["removed"]) == 7 and got["tracked"] == 6


def test_paging_helpers(tmp_path):
    if not bun():
        pytest.skip("bun 없음")
    got = run_js(tmp_path, """
console.log(JSON.stringify({
  first: U.pgAttr("t", 0), last: U.pgAttr("t", 19), over: U.pgAttr("t", 20),
  none: U.moreBtn("t", 20), more: U.moreBtn("t", 45), bad: U.pgAttr('a"b', 0),
}));
""")
    assert got["first"] == ' data-pg="t"' and got["last"] == ' data-pg="t"'
    assert got["over"] == ' data-pg="t" hidden'
    assert got["none"] == "" and "25줄 더 있음" in got["more"] and 'data-more="t"' in got["more"]
    assert got["bad"] == ' data-pg="a&quot;b"'

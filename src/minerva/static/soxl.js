// 종목 분석 > SOXL: 비트코인 화면과 같은 조건(SCR ① 15분, Stoch 9·CCI 20·RSI 21, W3, 3개 중 3개, 넘어옴)으로
// SOXLUSDT 선물을 보여준다(사용자 요청 2026-11-11). 보기만 하고 자동매매와 연결되지 않는다.
// 큰 그림 /api/soxl, 봉 /api/btc/klines?symbol=SOXLUSDT.
(() => {
  const $ = (id) => document.getElementById(id);
  const core = window.ScrLab;
  const U = window.MinervaUI;
  const V = window.ScrView;
  const { kst } = U;
  const GREEN = "#26a69a", RED = "#ef5350";
  const DP = 2;
  let bars = null, chart = null, series = null, started = false;

  const visible = () => $("an-sx") && !$("an-sx").hidden && $("tab-analysis") && !$("tab-analysis").hidden;

  const buildChart = () => {
    const LC = window.LightweightCharts;
    if (!LC || !$("sx-chart") || !$("sx-chart").clientWidth) return false;
    ({ chart, series } = U.scrChart($("sx-chart"), { timeFormatter: (t) => V.fmt(t),
      titles: ["Stoch 9", "CCI 20", "RSI 21"], up: GREEN, down: RED }));
    series.marks = LC.createSeriesMarkers(series.candle, []);
    series.candle.applyOptions({ priceFormat: { type: "price", precision: DP, minMove: 0.01 } });
    return true;
  };

  const recompute = () => {
    if (!bars || !core) return;
    const p = core.DEFAULTS;
    const res = core.signals(bars.map((b) => b[2]), bars.map((b) => b[3]), bars.map((b) => b[4]), p);
    let from = bars.findIndex((b) => b[0] >= (bars.showFrom || 0));
    if (from < 0) from = 0;
    $("sx-ind").innerHTML = V.indHtml(core, bars, res, p, DP);
    if (!visible() || (!chart && !buildChart())) return;
    U.scrFill(series, bars, from, res, p, GREEN, RED);
    const marks = [];
    res.long.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "belowBar", color: GREEN, shape: "circle", text: "롱" }); });
    res.short.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "aboveBar", color: RED, shape: "arrowDown", text: "숏" }); });
    series.marks.setMarkers(marks.sort((a, b) => a.time - b.time));
  };

  const get = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const load = () => Promise.all([get("/api/soxl"), get("/api/btc/klines?symbol=SOXLUSDT&interval=15m&days=7")])
    .then(([o, k]) => {
      try {
        $("sx-outlook").innerHTML = o && o.outlook ? V.outlookHtml(o.outlook, DP)
          : `<p class="placeholder">${U.esc((o && o.message) || "큰 그림을 불러오지 못했어요")}</p>`;
      } catch (err) { console.error(err); }
      if (k && k.bars && k.bars.length) { bars = k.bars; bars.showFrom = k.show_from; }
      else if (!bars) $("sx-ind").textContent = "봉을 받지 못했어요. 잠시 뒤 다시 열어 주세요.";
      try { recompute(); } catch (err) { console.error(err); }
    });

  // SOXL을 처음 열 때 불러온다(안 보는 사람은 야후·바이낸스를 부르지 않게).
  const start = () => {
    if (!visible()) return;
    if (!started) { started = true; load(); setInterval(() => { if (visible()) load(); }, 60000); }
    else setTimeout(recompute, 50);
  };
  window.addEventListener("hashchange", () => setTimeout(start, 50));
  document.querySelectorAll('[data-pick="an"] button').forEach((b) => b.addEventListener("click", () => setTimeout(start, 50)));
  setTimeout(start, 0);
})();

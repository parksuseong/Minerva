// 계좌 > 바이낸스 > 보유 종목 차트(사용자 2026-11-18 "계좌탭에 보유종목 차트도 그려줄수있나").
// 바이낸스에 지금 들고 있는 포지션마다 15분봉 + Stoch·CCI·RSI 칸, 진입·손절·익절 가로선, 진입 표시.
// 봉은 /api/btc/klines?symbol= (공개 시세, 서버 캐시), SCR 값은 엔진과 같은 /api/btc/engine params.scr.
// 표시 규칙(사용자): 롱 = 동그라미, 숏 = 삼각형. 롱 진입·숏 청산 = 초록, 롱 청산·숏 진입 = 빨강.
// 주문 없음. account.js가 포지션을 받을 때마다 window.HoldChart.update(positions)를 부른다.
window.HoldChart = (() => {
  const $ = (id) => document.getElementById(id);
  const U = window.MinervaUI;
  const core = window.ScrLab;
  const { esc, pad, kst } = U;
  const GREEN = "#26a69a", RED = "#ef5350";
  const dpOf = (px) => (px == null ? 2 : px >= 1000 ? 1 : px >= 1 ? 2 : px >= 0.01 ? 4 : 6);
  const fmt = (sec) => {
    const d = new Date(sec * 1000);
    return `${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  };
  // 엔진 기록 "YYYY-MM-DD HH:MM:SS"(KST) → KST 벽시계를 UTC 자리에 둔 초(U.kst와 같은 방식)
  const wall = (s) => {
    const m = /^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d)/.exec(s || "");
    return m ? Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5]) / 1000 : null;
  };
  // 엔진 포지션(손절선)은 계좌 탭과 같은 비밀번호가 있어야 온다
  const token = () => { try { return localStorage.getItem("minerva-account-token") || ""; } catch (e) { return ""; } };
  const get = (url) => fetch(url, { headers: { "X-Minerva-Token": token() } }).then((r) => (r.ok ? r.json() : null)).catch(() => null);

  const boxes = new Map(); // symbol → { el, chart, series, bars, pos, dp, at }
  let positions = [];
  let eng = null, engAt = 0;
  const visible = () => {
    const root = $("bn-hold");
    return root && root.offsetParent !== null;
  };

  const box = (sym) => {
    let b = boxes.get(sym);
    if (b) return b;
    const el = document.createElement("div");
    el.className = "hold-chart";
    el.innerHTML = `<div class="detail hc-title"></div><div class="hc-chart" style="height:420px"></div>
      <div class="mini hc-note"></div>`;
    $("bn-hold").appendChild(el);
    b = { el, chart: null, series: null, bars: null, pos: null, dp: 2, at: 0 };
    boxes.set(sym, b);
    return b;
  };

  const draw = (sym) => {
    const b = boxes.get(sym);
    if (!b || !b.bars || !core) return;
    const LC = window.LightweightCharts;
    const c = b.el.querySelector(".hc-chart");
    if (!LC || !c.clientWidth) return;
    if (!b.chart) {
      ({ chart: b.chart, series: b.series } = U.scrChart(c, { timeFormatter: fmt,
        titles: ["Stoch 5 (20/80)", "CCI 20 (±100)", "RSI 14 (30/70)"], up: GREEN, down: RED }));
      b.series.marks = LC.createSeriesMarkers(b.series.candle, []);
    }
    const p = (eng && eng.params && eng.params.scr) || core.DEFAULTS;
    const bars = b.bars;
    const res = core.signals(bars.map((x) => x[2]), bars.map((x) => x[3]), bars.map((x) => x[4]), p);
    let from = bars.findIndex((x) => x[0] >= (bars.showFrom || 0));
    if (from < 0) from = 0;
    b.series.candle.applyOptions({ priceFormat: { type: "price", precision: b.dp, minMove: 10 ** -b.dp } });
    const { times, guide } = U.scrFill(b.series, bars, from, res, p, GREEN, RED);
    const x = b.pos;
    const entry = Number(x.entry), stop = Number(x.stop), tp = Number(x.tp);
    if (entry) guide(b.series.candle, entry, U.css("--muted") || "#888", "진입", 1);
    if (stop) guide(b.series.candle, stop, RED, "손절", 0);
    if (tp) guide(b.series.candle, tp, GREEN, "익절", 0);
    const long = x.side === "롱";
    const marks = [];
    res.long.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "belowBar", color: GREEN, shape: "circle", text: "" }); });
    res.short.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "aboveBar", color: RED, shape: "arrowDown", text: "" }); });
    // 진입 시각은 엔진이 든 포지션일 때만 안다(엔진 state의 opened_at). 그 시각이 든 15분봉에 붙인다
    const ep = eng && eng.position;
    const t0 = ep && ep.symbol === sym ? wall(ep.opened_at) : null;
    if (t0 != null && times.length && t0 >= times[0]) {
      let k = times.length - 1;
      while (k >= 0 && times[k] > t0) k -= 1;
      if (k >= 0) {
        marks.push({ time: times[k], position: long ? "belowBar" : "aboveBar", color: long ? GREEN : RED,
          shape: long ? "circle" : "arrowDown", text: `${x.side} 진입`, size: 2 });
      }
    }
    b.series.marks.setMarkers(marks.sort((m, n) => m.time - n.time));
  };

  const title = (sym, x) => {
    const name = esc(String(sym).replace(/USDT$/, ""));
    const lev = x.leverage ? ` · ${esc(x.leverage)}배` : "";
    return `<b>${name} ${esc(x.side)}</b>${lev} · 진입 ${U.num(x.entry, 6)} · 현재 ${U.num(x.mark, 6)}`;
  };

  const loadSym = (sym) => {
    const b = boxes.get(sym);
    b.at = Date.now();
    return get(`/api/btc/klines?symbol=${encodeURIComponent(sym)}&interval=15m&days=3`).then((k) => {
      if (!boxes.has(sym)) return;
      const note = b.el.querySelector(".hc-note");
      if (k && k.bars && k.bars.length) {
        b.bars = k.bars; b.bars.showFrom = k.show_from;
        b.dp = dpOf(b.bars[b.bars.length - 1][4]);
        note.textContent = "15분봉 · 가로선: 진입(실선)·손절·익절 · 작은 표시는 SCR ① 신호(롱 ●, 숏 ▼) · 진입 표시는 엔진이 든 포지션만";
        try { draw(sym); } catch (err) { console.error(err); }
      } else if (!b.bars) {
        note.textContent = "이 종목 봉을 받지 못했어요(엔진 매매 종목이 아니면 안 열려요). 잠시 뒤 다시 볼게요.";
      }
    });
  };

  // 포지션이 바뀔 때마다 부른다. 봉은 종목마다 1분에 한 번만 다시 받는다
  const update = (list) => {
    positions = (list || []).filter((x) => x && x.symbol && Number(x.qty));
    const root = $("bn-hold");
    if (!root) return;
    const keep = new Set(positions.map((x) => x.symbol));
    for (const [sym, b] of boxes) {
      if (!keep.has(sym)) { if (b.chart) b.chart.remove(); b.el.remove(); boxes.delete(sym); }
    }
    $("bn-hold-empty").hidden = positions.length > 0;
    if (!positions.length || !visible()) return;
    const now = Date.now();
    const engP = now - engAt > 60000 ? get("/api/btc/engine").then((e) => { if (e) { eng = e; engAt = now; } }) : Promise.resolve();
    engP.then(() => positions.forEach((x) => {
      const b = box(x.symbol);
      b.pos = x;
      b.el.querySelector(".hc-title").innerHTML = title(x.symbol, x);
      if (!b.bars || now - b.at > 60000) loadSym(x.symbol);
      else { try { draw(x.symbol); } catch (err) { console.error(err); } }
    }));
  };

  // 숨긴 채로는 폭이 0이라 못 그린다. 보일 때 다시 부른다
  const redraw = () => setTimeout(() => update(positions), 50);
  return { update, redraw };
})();

// 🧪 SCR 실험실: SCR 숫자를 바꿔 가며 BTCUSDT 롱·숏 열림 신호를 바로 다시 그린다 (사용자 요청 2026-11-08).
// 봉은 /api/btc/klines(바이낸스 선물 공개 시세), 계산은 전부 화면에서(scrlab_core.js). 주문과 연결되지 않는다.
// 서버에 브라우저가 없어 화면은 직접 못 보고 만들었다 — 값이 이상하면 화면을 그리지 않고 안내만 한다.
(() => {
  const $ = (id) => document.getElementById(id);
  const core = window.ScrLab;
  // 기본값이 바뀌면 판을 올린다. 옛 판에 저장된 값은 읽지 않아 새 기본값이 한 번 보인다.
  const KEY = "minerva-scrlab-v2";
  const DAYS = [3, 7, 14, 30, 90, 180, 365];
  const MAX_ROWS = 300;
  const GREEN = "#26a69a", RED = "#ef5350";
  const MIN = { "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440 };
  const FIELDS = [
    // [키, 이름, 최소, 최대, 정수]
    ["stoch_n", "Stoch 길이", 2, 100, true], ["stoch_lo", "Stoch 아래선", 0, 100, false],
    ["stoch_hi", "Stoch 위선", 0, 100, false],
    ["cci_n", "CCI 길이", 2, 200, true], ["cci_lo", "CCI 아래선", -500, 500, false],
    ["cci_hi", "CCI 위선", -500, 500, false],
    ["rsi_n", "RSI 길이", 2, 100, true], ["rsi_lo", "RSI 아래선", 0, 100, false],
    ["rsi_hi", "RSI 위선", 0, 100, false],
    ["w", "확인 창 W (봉)", 1, 50, true],
  ];
  // 손절·익절(%): 진입가 기준. 익절 0 = 없음. 엔진 기본값(손절 3%, 익절 없음)과 같게 시작한다.
  const EXIT_FIELDS = [["stop_pct", "손절 %", 0.1, 50], ["tp_pct", "익절 % (0=없음)", 0, 100]];
  const BASE = { interval: "15m", days: 7, show_long: true, show_short: true, stop_pct: 3, tp_pct: 0 };
  const defaults = () => Object.assign({}, core ? core.DEFAULTS : {}, BASE);

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v, d = 1) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d }));
  const pct = (v) => (v == null ? "-"
    : `<span class="${v > 0 ? "up" : v < 0 ? "down" : ""}">${v > 0 ? "+" : ""}${num(v, 2)}%</span>`);
  const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
  const pad = (n) => String(n).padStart(2, "0");
  // 차트 시각은 KST 벽시계를 UTC 자리에 둔 초 단위 값이다. 표시는 UTC 필드로 읽는다.
  const kst = (ms) => Math.floor(ms / 1000) + 9 * 3600;
  const fmt = (sec, withDate) => {
    const d = new Date(sec * 1000);
    const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    return withDate ? `${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${hm}` : hm;
  };

  let cfg = defaults();
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) || "null");
    if (saved && typeof saved === "object") cfg = Object.assign(cfg, saved);
  } catch (e) { /* 저장된 값이 없거나 막혀도 기본값으로 연다 */ }
  if (!MIN[cfg.interval]) cfg.interval = BASE.interval;
  if (!DAYS.includes(Number(cfg.days))) cfg.days = BASE.days;
  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(cfg)); } catch (e) { /* 기억만 못 한다 */ } };

  let data = null, chart = null, series = null, started = false, timer = null, loading = 0;

  // ---- 조작 ----
  const buildControls = () => {
    $("lab-params").innerHTML = FIELDS.map(([k, name, lo, hi, int]) =>
      `<label>${esc(name)}<input data-k="${k}" type="number" inputmode="decimal" min="${lo}" max="${hi}" step="${int ? 1 : "any"}"></label>`).join("");
    document.querySelectorAll("#lab-tf button, #lab-days button").forEach((b) => {
      b.onclick = () => {
        if (b.dataset.tf) cfg.interval = b.dataset.tf; else cfg.days = Number(b.dataset.days);
        save(); syncControls(); load(true);
      };
    });
    $("lab-exits").innerHTML = EXIT_FIELDS.map(([k, name, lo, hi]) =>
      `<label>${esc(name)}<input data-k="${k}" type="number" inputmode="decimal" min="${lo}" max="${hi}" step="0.1"></label>`).join("");
    $("lab-exits").addEventListener("input", (e) => {
      const k = e.target.dataset.k;
      if (!k) return;
      cfg[k] = e.target.value;
      later();
    });
    $("lab-params").addEventListener("input", (e) => {
      const k = e.target.dataset.k;
      if (!k) return;
      cfg[k] = e.target.value;
      later();
    });
    $("lab-need").onchange = (e) => { cfg.need = Number(e.target.value); later(); };
    $("lab-mode").onchange = (e) => { cfg.mode = e.target.value; later(); };
    $("lab-long").onchange = (e) => { cfg.show_long = e.target.checked; later(); };
    $("lab-short").onchange = (e) => { cfg.show_short = e.target.checked; later(); };
    $("lab-reset").onclick = () => {
      const refetch = cfg.interval !== BASE.interval || Number(cfg.days) !== BASE.days;
      cfg = defaults();
      save(); syncControls();
      if (refetch) load(true); else recompute();
    };
  };

  const syncControls = () => {
    document.querySelectorAll("#lab-tf button").forEach((b) => b.classList.toggle("on", b.dataset.tf === cfg.interval));
    document.querySelectorAll("#lab-days button").forEach((b) => b.classList.toggle("on", Number(b.dataset.days) === Number(cfg.days)));
    document.querySelectorAll("#lab-params input, #lab-exits input").forEach((i) => { i.value = cfg[i.dataset.k]; });
    $("lab-need").value = String(cfg.need);
    $("lab-mode").value = cfg.mode === "zone" ? "zone" : "cross";
    $("lab-long").checked = cfg.show_long !== false;
    $("lab-short").checked = cfg.show_short !== false;
  };

  const later = () => { clearTimeout(timer); timer = setTimeout(() => { save(); recompute(); }, 250); };

  // 입력이 비었거나 범위를 벗어나면 그 칸만 기본값으로 계산하고 알려 준다.
  const params = () => {
    const base = core.DEFAULTS, out = {}, bad = [];
    FIELDS.forEach(([k, name, lo, hi, int]) => {
      let v = Number(cfg[k]);
      if (cfg[k] === "" || !Number.isFinite(v) || v < lo || v > hi || (int && v !== Math.floor(v))) {
        bad.push(name); v = base[k];
      }
      out[k] = v;
    });
    EXIT_FIELDS.forEach(([k, name, lo, hi]) => {
      let v = Number(cfg[k]);
      if (cfg[k] === "" || !Number.isFinite(v) || v < lo || v > hi) { bad.push(name); v = BASE[k]; }
      out[k] = v;
    });
    out.need = [2, 3].includes(Number(cfg.need)) ? Number(cfg.need) : 3;
    out.mode = cfg.mode === "zone" ? "zone" : "cross";
    return { p: out, bad };
  };

  // ---- 자료 ----
  const load = (fit) => {
    const ticket = ++loading;
    $("lab-status").textContent = "봉 받는 중…";
    return fetch(`/api/btc/klines?interval=${encodeURIComponent(cfg.interval)}&days=${Number(cfg.days)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d) => {
        if (ticket !== loading) return;
        if (!d || !Array.isArray(d.bars) || !d.bars.length) throw new Error("empty");
        data = d;
        recompute(fit);
      })
      .catch(() => { if (ticket === loading) $("lab-status").textContent = "봉을 받지 못했어요. 잠시 뒤 다시 열어 주세요."; });
  };

  // ---- 차트 ----
  const buildChart = () => {
    const LC = window.LightweightCharts;
    if (!LC) return false;
    const text = css("--text") || "#d1d4dc", grid = css("--line") || "#2a2e39";
    chart = LC.createChart($("lab-chart"), {
      autoSize: true,
      localization: { locale: "ko-KR", timeFormatter: (t) => fmt(t, true) },
      layout: { background: { type: "solid", color: css("--card") || "#131722" }, textColor: text, fontSize: 11,
        panes: { separatorColor: grid } },
      grid: { vertLines: { color: grid }, horzLines: { color: grid } },
      crosshair: { mode: LC.CrosshairMode.Normal },
      rightPriceScale: { borderColor: grid },
      timeScale: { borderColor: grid, timeVisible: true, secondsVisible: false, rightOffset: 4,
        tickMarkFormatter: (t, type) => (type <= 2 ? fmt(t, true).slice(0, 5) : fmt(t, false)) },
    });
    const line = (pane, color, title) => chart.addSeries(LC.LineSeries, { color, lineWidth: 2, title,
      priceLineVisible: false, lastValueVisible: true }, pane);
    series = {
      candle: chart.addSeries(LC.CandlestickSeries, { upColor: GREEN, downColor: RED, borderVisible: false,
        wickUpColor: GREEN, wickDownColor: RED }, 0),
      stoch: line(1, "#f7a21b", "Stoch"),
      cci: line(2, "#ab47bc", "CCI"),
      rsi: line(3, "#2962ff", "RSI"),
      lines: [],
    };
    series.marks = ["candle", "stoch", "cci", "rsi"].map((k) => LC.createSeriesMarkers(series[k], []));
    const panes = chart.panes();
    [0.52, 0.16, 0.16, 0.16].forEach((f, i) => panes[i] && panes[i].setStretchFactor(f));
    return true;
  };

  const drawChart = (bars, from, res, p, sigs, sim, fit) => {
    if (!chart && !buildChart()) {
      $("lab-chart").innerHTML = '<p class="placeholder">차트 도구를 불러오지 못했어요. 아래 목록은 그대로 볼 수 있어요.</p>';
      return;
    }
    const shown = bars.slice(from);
    series.candle.setData(shown.map((b) => ({ time: kst(b[0]), open: b[1], high: b[2], low: b[3], close: b[4] })));
    const ind = (arr) => shown.map((b, j) => ({ time: kst(b[0]), v: arr[from + j] }))
      .filter((x) => x.v != null && Number.isFinite(x.v)).map((x) => ({ time: x.time, value: x.v }));
    series.stoch.setData(ind(res.stoch)); series.cci.setData(ind(res.cci)); series.rsi.setData(ind(res.rsi));
    series.lines.forEach(([s, l]) => s.removePriceLine(l));
    const guide = (s, price, color) => [s, s.createPriceLine({ price, color, lineWidth: 1, lineStyle: 2,
      axisLabelVisible: true, title: "" })];
    series.lines = [
      guide(series.stoch, p.stoch_lo, GREEN), guide(series.stoch, p.stoch_hi, RED),
      guide(series.cci, p.cci_lo, GREEN), guide(series.cci, p.cci_hi, RED),
      guide(series.rsi, p.rsi_lo, GREEN), guide(series.rsi, p.rsi_hi, RED),
    ];
    // 시뮬레이션 마지막 포지션이 열려 있으면 진입·손절·익절 선을 캔들 위에 긋는다.
    if (sim && sim.open) {
      const o = sim.open;
      const pl = (price, color, title, style) => [series.candle, series.candle.createPriceLine({ price, color,
        lineWidth: 1, lineStyle: style, axisLabelVisible: true, title })];
      series.lines.push(pl(o.entry, css("--muted") || "#888", o.dir > 0 ? "롱 진입" : "숏 진입", 0));
      series.lines.push(pl(o.stop, RED, `손절 −${num(p.stop_pct, 1)}%`, 2));
      if (o.tp) series.lines.push(pl(o.tp, GREEN, `익절 +${num(p.tp_pct, 1)}%`, 2));
    }
    // 롱 열림 = 초록 동그라미(봉 아래), 숏 열림 = 빨강 삼각형(봉 위). 사용자 표시 규칙.
    const marks = sigs.map((s) => ({
      time: kst(bars[s.i][0]),
      position: s.dir > 0 ? "belowBar" : "aboveBar",
      color: s.dir > 0 ? GREEN : RED,
      shape: s.dir > 0 ? "circle" : "arrowDown",
      text: s.dir > 0 ? "롱" : "숏",
    }));
    // 손절·익절 청산 표시: 롱 청산 = 빨강 동그라미, 숏 청산 = 초록 삼각형(사용자 표시 규칙).
    if (sim) sim.exits.filter((x) => (x.kind === "stop" || x.kind === "tp") && x.i >= from).forEach((x) => marks.push({
      time: kst(bars[x.i][0]),
      position: x.dir > 0 ? "aboveBar" : "belowBar",
      color: x.dir > 0 ? RED : GREEN,
      shape: x.dir > 0 ? "circle" : "arrowUp",
      text: x.kind === "tp" ? "익절" : "손절",
    }));
    marks.sort((a, b) => a.time - b.time);
    const bare = marks.map((m) => Object.assign({}, m, { text: "" }));
    series.marks.forEach((m, k) => m.setMarkers(k === 0 ? marks : bare));
    if (fit) chart.timeScale().fitContent();
  };

  // ---- 목록·요약 ----
  const LEV = 3;
  const runSim = (bars, sigs, p) => {
    if (!sigs.length || !core.switchSim) return null;
    const col = (k) => bars.map((b) => b[k]);
    return core.switchSim(col(1), col(2), col(3), col(4), sigs,
      { stop: p.stop_pct / 100, tp: p.tp_pct / 100, lev: LEV, fee: 0.0005 });
  };
  const simLine = (r, p) => {
    if (!r) return "";
    const tpTxt = p.tp_pct > 0 ? `익절 ${num(p.tp_pct, 1)}%` : "익절 없음";
    const o = r.open;
    const openTxt = o ? `<br>📍 지금 ${o.dir > 0 ? "롱" : "숏"} 보유 (계산상): 진입 ${num(o.entry, 1)} · 손절 ${num(o.stop, 1)}`
      + (o.tp ? ` · 익절 ${num(o.tp, 1)}` : "") : "";
    return `<div class="mini" style="margin:4px 0 8px">📐 <b>손절 ${num(p.stop_pct, 1)}% · ${tpTxt} · ${LEV}배 단순 계산</b> (거친 추정): 보이는 신호로 롱↔숏 갈아타기 →
      거래 ${r.trades}번 · 이익 ${r.wins} · 손절 ${r.stops} · 익절 ${r.tps} · 누적 <b>${pct(r.ret)}</b> · 최대 낙폭 −${num(r.mdd, 1)}%${r.holding ? " · 마지막 포지션은 종가로 정리" : ""}${openTxt}<br>
      신호 다음 봉 시가 진입, 반대 신호에 뒤집기, 진입가 기준 손절·익절선에 봉 고가·저가가 닿으면 청산 후 다음 신호까지 쉼(한 봉에서 둘 다 닿으면 손절로 침), 수수료 0.05%/편도.
      슬리피지·펀딩비·청산 없음. 실전 엔진은 진입마다 AI가 손절·익절을 따로 정하니 결과가 달라요.</div>`;
  };
  const KIND = { stop: "손절", tp: "익절", flip: "뒤집기", end: "보유 중" };
  const outcome = (r, sim) => {
    const x = sim && sim.exits.find((e) => e.from === r.i + 1);
    if (!x) return '<span class="mini">-</span>';
    return `${KIND[x.kind]} ${pct(x.net)}`;
  };

  const drawList = (bars, sigs, sim, p) => {
    const c = bars.map((b) => b[4]);
    const step = MIN[data.interval] || 15;
    const ks = [1, 4, 24].map((h) => (h * 60) / step);
    const rows = sigs.map((s) => ({ ...s, m: ks.map((k) => core.after(c, s.i, k, s.dir)) }));
    const longs = rows.filter((r) => r.dir > 0).length;
    const with4 = rows.filter((r) => r.m[1] != null);
    const right = with4.filter((r) => r.m[1] > 0).length;
    const avg4 = with4.length ? with4.reduce((a, r) => a + r.m[1], 0) / with4.length : null;
    $("lab-summary").innerHTML = `<div class="regime">
      <span class="chip">신호 ${rows.length}개</span>
      <span class="chip good">롱 ${longs}</span><span class="chip warn">숏 ${rows.length - longs}</span>
      <span class="chip">4시간 뒤 맞는 방향 ${with4.length ? `${right}/${with4.length} (${num((right / with4.length) * 100, 0)}%)` : "-"}</span>
      <span class="chip">4시간 평균 ${avg4 == null ? "-" : `${avg4 > 0 ? "+" : ""}${num(avg4, 2)}%`}</span></div>${simLine(sim, p)}`;
    $("lab-list").innerHTML = rows.length ? `<div style="overflow-x:auto"><table class="acct"><thead><tr>
        <th>봉 마감 (KST)</th><th>방향</th><th>가격</th><th>1시간</th><th>4시간</th><th>24시간</th><th>결과 (${LEV}배)</th>
      </tr></thead><tbody>${rows.slice(-MAX_ROWS).reverse().map((r) => `<tr>
        <td>${fmt(kst(bars[r.i][0] + step * 60000), true)}</td>
        <td><span class="chip ${r.dir > 0 ? "good" : "warn"}">${r.dir > 0 ? "● 롱" : "▼ 숏"}</span></td>
        <td>${num(c[r.i], 1)}</td><td>${pct(r.m[0])}</td><td>${pct(r.m[1])}</td><td>${pct(r.m[2])}</td><td>${outcome(r, sim)}</td></tr>`).join("")}
      </tbody></table></div>${rows.length > MAX_ROWS ? `<p class="mini">최근 ${MAX_ROWS}개만 표에 보여요 (요약은 전체 ${num(rows.length, 0)}개 기준).</p>` : ""}`
      : '<p class="placeholder">이 설정으로는 이 기간에 신호가 없어요.</p>';
  };

  const recompute = (fit) => {
    if (!data || !core) return;
    try {
      const bars = data.bars;
      const { p, bad } = params();
      const res = core.signals(bars.map((b) => b[2]), bars.map((b) => b[3]), bars.map((b) => b[4]), p);
      let from = bars.findIndex((b) => b[0] >= data.show_from);
      if (from < 0) from = 0;
      const sigs = [];
      if (cfg.show_long !== false) res.long.forEach((i) => { if (i >= from) sigs.push({ i, dir: 1 }); });
      if (cfg.show_short !== false) res.short.forEach((i) => { if (i >= from) sigs.push({ i, dir: -1 }); });
      sigs.sort((a, b) => a.i - b.i || b.dir - a.dir);
      const sim = runSim(bars, sigs, p);
      drawChart(bars, from, res, p, sigs, sim, fit);
      drawList(bars, sigs, sim, p);
      const last = bars[bars.length - 1];
      $("lab-status").innerHTML = `마지막 마감 봉 ${fmt(kst(last[0] + (MIN[data.interval] || 15) * 60000), true)} KST · `
        + `봉 ${num(bars.length - from, 0)}개 (앞쪽 여유 ${num(from, 0)}개는 지표 계산에만 씀)`
        + (data.capped ? `<br><span class="up">⚠️ 5분봉은 ${num(data.days, 0)}일까지만 볼 수 있어요 (봉이 너무 많아 화면이 느려져요)</span>` : "")
        + (bad.length ? `<br><span class="up">⚠️ ${esc(bad.join(", "))} 값이 범위를 벗어나 기본값으로 계산했어요</span>` : "");
    } catch (e) {
      $("lab-status").textContent = "계산 중 문제가 생겼어요. 기본값으로 버튼을 눌러 보세요.";
    }
  };

  // 탭을 처음 열 때 만든다. 숨은 탭에서 차트를 만들면 크기가 0으로 잡힌다.
  const start = () => {
    const panel = $("tab-scrlab");
    if (started || !panel || panel.hidden) return;
    started = true;
    if (!core) { $("lab-status").textContent = "계산 도구를 불러오지 못했어요"; return; }
    buildControls(); syncControls(); load(true);
    setInterval(() => { if (!panel.hidden) load(false); }, 60000);
  };
  window.addEventListener("hashchange", start);
  start();
})();

// 화면 스크립트들이 같이 쓰는 작은 도구. index.html에서 다른 화면 스크립트보다 먼저 불러온다.
window.MinervaUI = (() => {
  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };
  // HTML 본문·큰따옴표 속성에 넣을 글자를 막는다.
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ESC[c]);
  // 한국식 천 단위 쉼표, 소수점 아래 최대 d자리. 값이 없으면 "-".
  const num = (v, d) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d }));
  const pad = (n) => String(n).padStart(2, "0");
  // 테마 색(CSS 변수)
  const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
  // 차트 시각: 봉 시작 ms → KST 벽시계를 UTC 자리에 둔 초. 표시는 UTC 필드로 읽는다.
  const kst = (ms) => Math.floor(ms / 1000) + 9 * 3600;

  // SCR 차트 틀: 캔들 + Stoch·CCI·RSI를 각각 아래 칸에. 비트코인 화면과 SCR 실험실이 같이 쓴다.
  // opts: timeFormatter, tickMarkFormatter(없으면 기본), titles [Stoch, CCI, RSI], up/down 색
  const scrChart = (el, opts) => {
    const LC = window.LightweightCharts;
    const text = css("--text") || "#d1d4dc", grid = css("--line") || "#2a2e39";
    const timeScale = { borderColor: grid, timeVisible: true, secondsVisible: false, rightOffset: 4 };
    if (opts.tickMarkFormatter) timeScale.tickMarkFormatter = opts.tickMarkFormatter;
    const chart = LC.createChart(el, {
      autoSize: true,
      localization: { locale: "ko-KR", timeFormatter: opts.timeFormatter },
      layout: { background: { type: "solid", color: css("--card") || "#131722" }, textColor: text, fontSize: 11,
        panes: { separatorColor: grid } },
      grid: { vertLines: { color: grid }, horzLines: { color: grid } },
      crosshair: { mode: LC.CrosshairMode.Normal },
      rightPriceScale: { borderColor: grid },
      timeScale,
    });
    const line = (pane, color, title) => chart.addSeries(LC.LineSeries, { color, lineWidth: 2, title,
      priceLineVisible: false, lastValueVisible: true }, pane);
    const [ts, tc, tr] = opts.titles;
    const series = {
      candle: chart.addSeries(LC.CandlestickSeries, { upColor: opts.up, downColor: opts.down, borderVisible: false,
        wickUpColor: opts.up, wickDownColor: opts.down }, 0),
      stoch: line(1, "#f7a21b", ts), cci: line(2, "#ab47bc", tc), rsi: line(3, "#2962ff", tr),
      lines: [],
    };
    const panes = chart.panes();
    [0.52, 0.16, 0.16, 0.16].forEach((f, i) => panes[i] && panes[i].setStretchFactor(f));
    // 칸마다 왼쪽 위에 이름을 크게 붙인다(사용자 2026-11-12 "아래 차트는 무슨 차트인지 모르겠네").
    // 축의 작은 제목만으로는 아래 세 칸이 무엇인지 안 보였다.
    if (LC.createTextWatermark) {
      const tag = (i, txt, color) => panes[i] && LC.createTextWatermark(panes[i], {
        horzAlign: "left", vertAlign: "top",
        lines: [{ text: txt, color, fontSize: 12, fontStyle: "bold" }],
      });
      const hint = opts.hints || ["가격", "점선 아래선 위로 → 롱 · 윗선 아래로 → 숏",
        "점선 아래선 위로 → 롱 · 윗선 아래로 → 숏", "점선 아래선 위로 → 롱 · 윗선 아래로 → 숏"];
      tag(0, hint[0], text);
      tag(1, `${ts} · ${hint[1]}`, "#f7a21b");
      tag(2, `${tc} · ${hint[2]}`, "#ab47bc");
      tag(3, `${tr} · ${hint[3]}`, "#2962ff");
    }
    return { chart, series };
  };

  // scrChart 틀에 bars[from:] 봉·지표를 넣고, 지난 가로선을 지운 뒤 지표 기준선(아래선 up색·위선 down색)을 긋는다.
  // 반환: times(보이는 봉 시각), guide(시리즈, 가격, 색, 제목, 선 모양) — 가로선을 더 긋는다.
  const scrFill = (series, bars, from, res, p, up, down) => {
    const shown = bars.slice(from);
    const times = shown.map((b) => kst(b[0]));
    series.candle.setData(shown.map((b, j) => ({ time: times[j], open: b[1], high: b[2], low: b[3], close: b[4] })));
    const ind = (arr) => times.map((time, j) => ({ time, value: arr[from + j] }))
      .filter((x) => Number.isFinite(x.value));
    series.stoch.setData(ind(res.stoch)); series.cci.setData(ind(res.cci)); series.rsi.setData(ind(res.rsi));
    series.lines.forEach(([s, l]) => s.removePriceLine(l));
    series.lines = [];
    const guide = (s, price, color, title = "", style = 2) => {
      series.lines.push([s, s.createPriceLine({ price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title })]);
    };
    guide(series.stoch, p.stoch_lo, up); guide(series.stoch, p.stoch_hi, down);
    guide(series.cci, p.cci_lo, up); guide(series.cci, p.cci_hi, down);
    guide(series.rsi, p.rsi_lo, up); guide(series.rsi, p.rsi_hi, down);
    return { times, guide };
  };

  // 긴 표 나눠 보기(사용자 2026-11-14): 처음 PAGE줄만 보이고 "더 보기"로 PAGE줄씩 더.
  // 행은 다 그리되 넘는 행에 hidden을 붙인다. 1분 갱신으로 다시 그려도 표(key)마다 펼친 줄 수를 기억한다.
  const PAGE = 20;
  const shown = {};
  const limit = (key) => shown[key] || PAGE;
  const pgAttr = (key, i) => ` data-pg="${esc(key)}"${i >= limit(key) ? " hidden" : ""}`;
  const moreBtn = (key, total) => (total > limit(key)
    ? `<button type="button" class="more-btn" data-more="${esc(key)}">더 보기 (${total - limit(key)}줄 더 있음)</button>` : "");
  if (document.addEventListener) document.addEventListener("click", (e) => {
    const b = e.target.closest ? e.target.closest(".more-btn") : null;
    if (!b) return;
    const key = b.dataset.more;
    shown[key] = limit(key) + PAGE;
    const rows = [...document.querySelectorAll("tr[data-pg]")].filter((r) => r.dataset.pg === key);
    rows.forEach((r, i) => { r.hidden = i >= shown[key]; });
    const left = rows.length - shown[key];
    if (left > 0) b.textContent = `더 보기 (${left}줄 더 있음)`; else b.remove();
  });

  // 원화 보기(계좌 탭 "₩ 원화로 보기" 버튼): 켜면 USDT 금액을 조회 환율(달러/원, 1 USDT = 1달러)로
  // 바꿔 보인다. 가격(진입가·손절가)은 그대로 USDT. 상태는 이 기기에 기억하고, 바뀌면 "minerva-fx" 이벤트.
  const FX_KEY = "minerva-fx-krw";
  const fx = { on: false, rate: null, at: null, stale: false };
  try { fx.on = localStorage.getItem(FX_KEY) === "1"; } catch (e) { /* 기본 USDT */ }
  const krw = () => fx.on && fx.rate > 0;
  const fxEmit = () => {
    if (document.dispatchEvent && typeof Event === "function") document.dispatchEvent(new Event("minerva-fx"));
  };
  // 금액 한 칸: "12.34 USDT" 또는 "16,508원". unit=false면 단위 없이 숫자만(표 머리에 단위가 있을 때)
  const money = (v, d = 2, unit = true) => {
    if (v == null || Number.isNaN(Number(v))) return "-";
    if (krw()) return `${num(Math.round(v * fx.rate), 0)}${unit ? "원" : ""}`;
    return `${num(v, d)}${unit ? " USDT" : ""}`;
  };
  const moneyUnit = () => (krw() ? "원" : "USDT");
  const fxFactor = () => (krw() ? fx.rate : 1);
  const fxSet = (on) => {
    fx.on = !!on;
    try { localStorage.setItem(FX_KEY, fx.on ? "1" : "0"); } catch (e) { /* 이번 화면만 */ }
    fxEmit();
  };
  const fxLoad = () => (typeof fetch === "function" ? fetch("/api/fx/usdkrw")
    .then((r) => r.json())
    .then((d) => {
      if (d.status !== "ok") return;
      const changed = d.rate !== fx.rate;
      Object.assign(fx, { rate: d.rate, at: d.at, stale: !!d.stale });
      if (changed && fx.on) fxEmit();
    })
    .catch(() => { /* 환율 없으면 USDT 그대로 */ }) : Promise.resolve());

  return { esc, num, pad, css, kst, scrChart, scrFill, pgAttr, moreBtn,
    fx, money, moneyUnit, fxFactor, fxSet, fxLoad };
})();

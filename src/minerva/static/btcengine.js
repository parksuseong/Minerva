// ₿ 비트코인: 실전 전략(SCR ① 15분 롱↔숏, 3배, 진입 때 AI 손절·익절) 하나만 보여준다.
// 엔진 상태는 /api/btc/engine(엔진 파일을 읽기만 함), 봉은 /api/btc/klines, 추격 한도는 /api/btc.
// 종목 분석 > 비트코인, 오늘 할 일 > 비트코인, 계좌 > 비트코인이 모두 같은 엔진 상태를 본다. 주문 없음.
(() => {
  const $ = (id) => document.getElementById(id);
  const core = window.ScrLab;
  const GREEN = "#26a69a", RED = "#ef5350", BLUE = "#2962ff", AMBER = "#f7a21b";
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v, d = 1) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d }));
  const pct = (v, d = 2) => (v == null ? "-"
    : `<span class="${v > 0 ? "up" : v < 0 ? "down" : ""}">${v > 0 ? "+" : ""}${num(v, d)}%</span>`);
  const SIDE = { long: "롱", short: "숏" };
  const WHY = { signal: "반대 신호", stop: "손절", tp: "익절", manual: "수동·강제", protect: "손절 실패 정리" };
  const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
  const pad = (n) => String(n).padStart(2, "0");
  // 차트 시각: KST 벽시계를 UTC 자리에 둔 초. 엔진 기록 "YYYY-MM-DD HH:MM:SS"(KST)도 같은 방식으로 바꾼다.
  const kst = (ms) => Math.floor(ms / 1000) + 9 * 3600;
  const wall = (s) => {
    const m = /^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d)/.exec(s || "");
    return m ? Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5]) / 1000 : null;
  };
  const barOf = (sec) => (sec == null ? null : sec - (sec % 900));
  const fmt = (sec) => {
    const d = new Date(sec * 1000);
    return `${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  };

  let eng = null, bars = null, gate = null, chart = null, series = null;

  const modeChip = (d) => {
    if (d.halt) return '<span class="chip warn">⏸ 정지(HALT)</span>';
    if (d.no_entry) return '<span class="chip warn">⛔ 신규 진입 정지</span>';
    return d.mode === "live" ? '<span class="chip good">🔴 실전</span>' : '<span class="chip">🧪 드라이런 (주문 없음)</span>';
  };

  const lastClose = () => (bars && bars.length ? bars[bars.length - 1][4] : null);

  const posHtml = (p) => {
    if (!p) return "포지션 없음 (다음 신호를 기다려요)";
    const c = lastClose();
    const sign = p.side === "long" ? 1 : -1;
    const u = c ? (c / p.entry - 1) * 100 * sign : null;
    const tp = p.tp ? ` · 익절 <b>${num(p.tp)}</b> (+${num(p.tp_pct, 2)}%)` : " · 익절 없음";
    return `<span class="chip ${p.side === "long" ? "good" : "warn"}">${SIDE[p.side]}</span> ${num(p.qty, 3)} BTC @ <b>${num(p.entry)}</b>
      · 지금 ${pct(u)} (× 3배 ${pct(u == null ? null : u * 3)})<br>
      <span class="mini">손절 <b>${num(p.stop)}</b> (−${num(p.stop_pct ?? 3, 2)}%)${tp} · ${esc(p.opened_at || "")} 진입${p.mode === "dry" ? " · 드라이런 가상" : ""}</span>`;
  };

  const sigHtml = (s) => !s ? "아직 신호 없음"
    : `${esc(s.bar)} 봉 <span class="chip ${s.signal === "long" ? "good" : "warn"}">${SIDE[s.signal]} 신호</span>
       종가 ${num(s.close)} → <b>${esc(s.decision)}</b>`;

  const gaugeHtml = (g) => {
    const eq = g.equity, floor = g.floor;
    const room = eq != null && floor ? (eq / floor - 1) * 100 : null;
    const wr = g.winrate_n ? (g.winrate_wins / g.winrate_n) * 100 : null;
    const wrOk = g.winrate_n < g.winrate_min_trades || wr >= g.winrate_min * 100;
    return `<table class="acct"><tbody>
      <tr><td>평가액 / 시드</td><td>${num(eq, 2)} / ${num(g.seed, 2)} USDT · 정지선 ${num(floor, 2)} (시드 −20%)
        ${room == null ? "" : `· 여유 ${pct(room)}`}</td></tr>
      <tr><td>최근 ${g.winrate_days}일 승률</td><td>${g.winrate_n ? `${g.winrate_wins}/${g.winrate_n} = ${num(wr, 0)}%` : "거래 없음"}
        · 기준 ${num(g.winrate_min * 100, 0)}% (${g.winrate_min_trades}회 이상부터) <span class="chip ${wrOk ? "good" : "warn"}">${wrOk ? "통과" : "정지"}</span></td></tr>
    </tbody></table>`;
  };

  const renderStatus = () => {
    const d = eng;
    if (!d || d.status !== "ok") {
      ["bte-status", "bte-today"].forEach((id) => { if ($(id)) $(id).textContent = (d && d.message) || "엔진 상태를 불러오지 못했어요"; });
      return;
    }
    const run = d.last_run;
    const runAt = run ? `마지막 확인 ${esc(run.at)} KST (15분마다, 봉 마감 20초 뒤)` : "아직 실행 기록 없음";
    const halt = d.no_entry ? `<div class="detail up">정지 사유: ${esc(d.no_entry)}</div>` : "";
    $("bte-status").innerHTML = `<div class="regime">${modeChip(d)}
        <span class="chip">SCR ① 15분</span><span class="chip">3배</span><span class="chip">AI 손절 ${d.params.ai_stop_range.join("~")}% · 익절 ${d.params.ai_tp_range.join("~")}%/없음</span></div>
      ${halt}
      <h3 style="margin:10px 0 4px;font-size:14px">포지션</h3><div>${posHtml(d.position)}</div>
      <h3 style="margin:10px 0 4px;font-size:14px">마지막 신호</h3><div>${sigHtml(d.last_signal)}</div>
      <h3 style="margin:10px 0 4px;font-size:14px">정지 규칙</h3>${gaugeHtml(d.gauges)}
      <p class="mini">${runAt}${d.live_since ? ` · 실전 시작 ${esc(d.live_since)}` : ""}</p>`;
    if ($("bte-today")) {
      $("bte-today").innerHTML = `<div class="regime">${modeChip(d)}</div>
        <div><b>포지션</b>: ${posHtml(d.position)}</div>
        <div style="margin-top:6px"><b>마지막 신호</b>: ${sigHtml(d.last_signal)}</div>
        <div class="mini" style="margin-top:6px">${runAt} · 자세한 내용은 📈 종목 분석 > ₿ 비트코인</div>`;
    }
    renderAi(); renderTrades(); renderAcct();
  };

  const renderAi = () => {
    if (!$("bte-ai")) return;
    const list = eng.ai_recent || [];
    $("bte-ai").innerHTML = list.length ? `<table class="acct"><thead><tr><th>시각</th><th>방향</th><th>손절</th><th>익절</th><th>이유</th></tr></thead><tbody>
      ${list.map((a) => `<tr><td>${esc((a.at || "").slice(5, 16))}</td><td>${SIDE[a.side] || "-"}</td>
        <td>−${num(a.used && a.used.stop_pct, 2)}%</td><td>${a.used && a.used.tp_pct ? `+${num(a.used.tp_pct, 2)}%` : "없음"}</td>
        <td>${a.fallback ? "⚙️ " : "🤖 "}${esc(a.used && a.used.reason)} <span class="mini">${num(a.secs, 1)}초</span></td></tr>`).join("")}</tbody></table>`
      : '<p class="placeholder">아직 진입이 없어 AI에게 물은 적이 없어요</p>';
    $("bte-ai").innerHTML += `<p class="mini">진입 직전에 Claude가 손절 ${eng.params.ai_stop_range.join("~")}%, 익절 ${eng.params.ai_tp_range.join("~")}% 또는 없음을 정해요.
      답이 없거나 틀리면 손절 ${num(eng.params.default_stop_pct, 0)}%·익절 없음(⚙️). AI가 정한 청산 폭은 백테스트할 수 없어요.</p>`;
  };

  const tradeRows = (rows) => rows.map((r) => `<tr><td>${esc((r.at || "").slice(5, 16))}</td>
      <td><span class="chip ${r.side === "long" ? "good" : "warn"}">${SIDE[r.side]}</span></td>
      <td>${num(r.entry)} → ${num(r.exit)}</td><td>${pct(r.pct)} <span class="mini">×3 ${pct(r.pct == null ? null : r.pct * 3)}</span></td>
      <td>${r.pnl == null ? "-" : `${r.pnl > 0 ? "+" : ""}${num(r.pnl, 2)}`}</td><td>${esc(WHY[r.why] || r.why)}</td></tr>`).join("");
  const TRADE_HEAD = "<thead><tr><th>청산</th><th>방향</th><th>진입 → 청산</th><th>가격 변화</th><th>손익 USDT</th><th>이유</th></tr></thead>";

  const renderTrades = () => {
    if (!$("bte-trades")) return;
    const dry = eng.mode === "dry";
    const rows = eng.trades || [];
    let html = dry ? '<div class="regime"><span class="chip">🧪 드라이런 — 가상 거래예요 (주문 없음)</span></div>' : "";
    html += rows.length ? `<div style="overflow-x:auto"><table class="acct">${TRADE_HEAD}<tbody>${tradeRows(rows)}</tbody></table></div>`
      : '<p class="placeholder">끝난 거래가 아직 없어요</p>';
    if (dry && (eng.dry_orders || []).length) {
      html += `<h3 style="margin:10px 0 4px;font-size:14px">드라이런 주문 기록 (냈을 주문)</h3><div style="overflow-x:auto"><table class="acct">
        <thead><tr><th>시각</th><th>동작</th><th>방향</th><th>수량</th><th>가격</th><th>손절/익절</th></tr></thead><tbody>
        ${eng.dry_orders.map((o) => `<tr><td>${esc((o.at || "").slice(5, 16))}</td><td>${esc(o.action)}</td><td>${SIDE[o.side] || "-"}</td>
          <td>${num(o.qty, 3)}</td><td>${num(o.price)}</td><td>${o.stop ? num(o.stop) : "-"}${o.tp ? ` / ${num(o.tp)}` : ""}</td></tr>`).join("")}
        </tbody></table></div>`;
    }
    $("bte-trades").innerHTML = html;
  };

  // 계좌 > 비트코인: 실전 시작(start-live) 이후 엔진 거래와 합계
  const renderAcct = () => {
    if (!$("bte-acct")) return;
    if (eng.mode !== "live" || !eng.live_since) {
      $("bte-acct").innerHTML = '<p class="placeholder">실전 자동매매를 시작하면 그때부터 여기에 쌓여요 (지금은 드라이런)</p>';
      return;
    }
    const rows = (eng.trades || []).filter((r) => (r.at || "") >= eng.live_since);
    const sum = rows.reduce((a, r) => a + (r.pnl || 0), 0);
    const wins = rows.filter((r) => (r.pnl || 0) > 0).length;
    $("bte-acct").innerHTML = `<div class="detail">실전 시작 ${esc(eng.live_since)} KST 이후 ${rows.length}번 · 이긴 ${wins}번 · 합계 <b>${sum > 0 ? "+" : ""}${num(sum, 2)} USDT</b> (수수료 포함)</div>
      <div class="detail">지금: ${posHtml(eng.position)}</div>
      ${rows.length ? `<div style="overflow-x:auto"><table class="acct">${TRADE_HEAD}<tbody>${tradeRows(rows)}</tbody></table></div>` : ""}`;
  };

  // ---- 지표·추격 한도 ----
  const renderInd = (res, p) => {
    if (!$("bte-ind")) return;
    const n = bars.length - 1;
    const rowsFor = (long) => [["Stoch", res.stoch, long ? p.stoch_lo : p.stoch_hi],
      ["CCI", res.cci, long ? p.cci_lo : p.cci_hi], ["RSI", res.rsi, long ? p.rsi_lo : p.rsi_hi]].map(([name, x, line]) => {
      const hit = core.recent(core.crossed(x, line, long), p.w)[n];
      const v = x[n];
      let how;
      if (hit) how = `<span class="chip good">✓ 최근 ${p.w}봉 안에 넘어옴</span>`;
      else if (v == null) how = "-";
      else if (long ? v < line : v > line) how = `기준선 ${long ? "아래" : "위"} ${num(Math.abs(v - line), 1)} — ${long ? "올라오면" : "내려오면"} 충족`;
      else how = `기준선까지 ${num(Math.abs(v - line), 1)} — ${long ? "아래로 내려갔다 올라와야" : "위로 올라갔다 내려와야"}`;
      return { name, v, line, hit, how };
    });
    const block = (long) => {
      const rows = rowsFor(long);
      const k = rows.filter((r) => r.hit).length;
      return `<h3 style="margin:8px 0 4px;font-size:14px">${long ? "롱" : "숏"} 신호까지 <span class="mini">${k}/3 충족 (3개 모두 최근 ${p.w}봉 안이어야 신호)</span></h3>
        <table class="acct"><tbody>${rows.map((r) => `<tr><td>${r.name}</td><td>${num(r.v, 1)}</td><td>기준 ${num(r.line, 0)}</td><td>${r.how}</td></tr>`).join("")}</tbody></table>`;
    };
    $("bte-ind").innerHTML = `<div class="mini">마지막 마감 봉 ${fmt(kst(bars[n][0]))} KST · Stoch ${p.stoch_n} (${p.stoch_lo}/${p.stoch_hi}) · CCI ${p.cci_n} (${p.cci_lo}/${p.cci_hi}) · RSI ${p.rsi_n} (${p.rsi_lo}/${p.rsi_hi}) · 넘어옴</div>`
      + block(true) + block(false);
  };

  const renderGate = () => {
    if (!$("bte-gate")) return;
    const g = gate && gate.entry_gate;
    if (!g) { $("bte-gate").innerHTML = '<p class="placeholder">추격 한도 계산 대기 중</p>'; return; }
    $("bte-gate").innerHTML = `<div><b>±${num(g.pct, 0)}%</b> ${g.big ? `<span class="chip good">15분·4시간·일봉 모두 ${esc(g.big_way)} → 큰 움직임, 3%</span>` : '<span class="chip">기본 1%</span>'}</div>
      <div class="detail">지금 15분봉 종가 ${num(g.ref_close)} 기준이면 롱은 ${num(g.long_max)} 아래, 숏은 ${num(g.short_min)} 위에서만 들어가요.
      실제로는 신호 봉 종가 기준으로, AI 답을 받은 뒤 다시 한 번 재요.</div>`;
  };

  // ---- 차트 ----
  const buildChart = () => {
    const LC = window.LightweightCharts;
    if (!LC || !$("bte-chart") || !$("bte-chart").clientWidth) return false;
    const text = css("--text") || "#d1d4dc", grid = css("--line") || "#2a2e39";
    chart = LC.createChart($("bte-chart"), {
      autoSize: true,
      localization: { locale: "ko-KR", timeFormatter: (t) => fmt(t) },
      layout: { background: { type: "solid", color: css("--card") || "#131722" }, textColor: text, fontSize: 11,
        panes: { separatorColor: grid } },
      grid: { vertLines: { color: grid }, horzLines: { color: grid } },
      crosshair: { mode: LC.CrosshairMode.Normal },
      rightPriceScale: { borderColor: grid },
      timeScale: { borderColor: grid, timeVisible: true, secondsVisible: false, rightOffset: 4 },
    });
    const line = (pane, color, title) => chart.addSeries(LC.LineSeries, { color, lineWidth: 2, title,
      priceLineVisible: false, lastValueVisible: true }, pane);
    series = {
      candle: chart.addSeries(LC.CandlestickSeries, { upColor: GREEN, downColor: RED, borderVisible: false,
        wickUpColor: GREEN, wickDownColor: RED }, 0),
      stoch: line(1, AMBER, "Stoch 9"), cci: line(2, "#ab47bc", "CCI 20"), rsi: line(3, BLUE, "RSI 21"),
      lines: [],
    };
    series.marks = LC.createSeriesMarkers(series.candle, []);
    const panes = chart.panes();
    [0.52, 0.16, 0.16, 0.16].forEach((f, i) => panes[i] && panes[i].setStretchFactor(f));
    return true;
  };

  // 표시 규칙(사용자): 롱 = 동그라미, 숏 = 삼각형. 롱 진입·숏 청산 = 초록, 롱 청산·숏 진입 = 빨강.
  const engineMarks = (first) => {
    const out = [];
    const add = (sec, side, open, text) => {
      const t = barOf(sec);
      if (t == null || t < first) return;
      const green = (side === "long") === open;
      out.push({ time: t, position: green ? "belowBar" : "aboveBar", color: green ? GREEN : RED,
        shape: side === "long" ? "circle" : (open ? "arrowDown" : "arrowUp"), text, size: 2 });
    };
    (eng && eng.trades || []).forEach((r) => {
      add(wall(r.opened_at), r.side, true, `${SIDE[r.side]} 진입`);
      add(wall(r.at), r.side, false, `${SIDE[r.side]} ${WHY[r.why] || "청산"}`);
    });
    const p = eng && eng.position;
    if (p) add(wall(p.opened_at), p.side, true, `${SIDE[p.side]} 진입`);
    return out;
  };

  const renderChart = (res, p, from) => {
    if (!chart && !buildChart()) return;
    const shown = bars.slice(from);
    const first = kst(shown[0][0]);
    series.candle.setData(shown.map((b) => ({ time: kst(b[0]), open: b[1], high: b[2], low: b[3], close: b[4] })));
    const ind = (arr) => shown.map((b, j) => ({ time: kst(b[0]), v: arr[from + j] }))
      .filter((x) => x.v != null && Number.isFinite(x.v)).map((x) => ({ time: x.time, value: x.v }));
    series.stoch.setData(ind(res.stoch)); series.cci.setData(ind(res.cci)); series.rsi.setData(ind(res.rsi));
    series.lines.forEach(([s, l]) => s.removePriceLine(l));
    const guide = (s, price, color, title = "", style = 2) => [s, s.createPriceLine({ price, color, lineWidth: 1, lineStyle: style,
      axisLabelVisible: true, title })];
    series.lines = [
      guide(series.stoch, p.stoch_lo, GREEN), guide(series.stoch, p.stoch_hi, RED),
      guide(series.cci, p.cci_lo, GREEN), guide(series.cci, p.cci_hi, RED),
      guide(series.rsi, p.rsi_lo, GREEN), guide(series.rsi, p.rsi_hi, RED),
    ];
    const pos = eng && eng.position;
    if (pos) {
      series.lines.push(guide(series.candle, pos.entry, css("--muted") || "#888", "진입", 1));
      series.lines.push(guide(series.candle, pos.stop, RED, "손절", 0));
      if (pos.tp) series.lines.push(guide(series.candle, pos.tp, GREEN, "익절", 0));
    }
    const sigMarks = [];
    res.long.forEach((i) => { if (i >= from) sigMarks.push({ time: kst(bars[i][0]), position: "belowBar", color: GREEN, shape: "circle", text: "" }); });
    res.short.forEach((i) => { if (i >= from) sigMarks.push({ time: kst(bars[i][0]), position: "aboveBar", color: RED, shape: "arrowDown", text: "" }); });
    // 엔진 표시는 그 시각이 든 15분봉에 붙인다. 아직 안 끝난 봉이면 마지막 마감 봉에 붙인다.
    const times = shown.map((b) => kst(b[0]));
    const snap = (t) => {
      let k = times.length - 1;
      while (k >= 0 && times[k] > t) k -= 1;
      return k >= 0 ? times[k] : null;
    };
    const eMarks = engineMarks(first).map((m) => Object.assign(m, { time: snap(m.time) })).filter((m) => m.time != null);
    series.marks.setMarkers(sigMarks.concat(eMarks).sort((a, b) => a.time - b.time));
  };

  const recompute = () => {
    if (!bars || !core || !eng || eng.status !== "ok") return;
    const p = eng.params.scr;
    const res = core.signals(bars.map((b) => b[2]), bars.map((b) => b[3]), bars.map((b) => b[4]), p);
    let from = bars.findIndex((b) => b[0] >= (bars.showFrom || 0));
    if (from < 0) from = 0;
    renderInd(res, p);
    if (!$("an-bn").hidden && !$("tab-analysis").hidden) renderChart(res, p, from);
  };

  const get = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const load = () => Promise.all([get("/api/btc/engine"), get("/api/btc/klines?interval=15m&days=7"), get("/api/btc")])
    .then(([e, k, g]) => {
      eng = e || { status: "error" };
      if (k && k.bars) { bars = k.bars; bars.showFrom = k.show_from; }
      gate = g;
      renderStatus(); renderGate(); recompute();
    });

  window.addEventListener("hashchange", () => setTimeout(recompute, 50));
  document.querySelectorAll('[data-pick="an"] button').forEach((b) => b.addEventListener("click", () => setTimeout(recompute, 50)));
  load();
  setInterval(load, 60000);
})();

// 비트코인 탭의 "ML 규칙 기록". /api/btc/ml(추적기가 15분마다 쓰는 파일)을 그린다. 표시 전용이다.
// 차트는 TradingView Lightweight Charts(Apache-2.0, static/vendor). 시각은 추적기가 KST로 옮겨 둔다.
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;",
    ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v, d = 1) => (v == null ? "-" : Number(v).toLocaleString("ko-KR", { maximumFractionDigits: d }));
  const pct = (v) => (v == null ? "-" : `<span class="${v > 0 ? "up" : v < 0 ? "down" : ""}">${v > 0 ? "+" : ""}${num(v, 2)}%</span>`);
  const sideChip = (s) => `<span class="chip ${s === "롱" ? "good" : "warn"}">${esc(s)}</span>`;
  const GREEN = "#26a69a", RED = "#ef5350";
  const MODEL_ORDER = ["rf", "mlp", "xgb", "lstm", "cnn"];
  let NAMES = { rf: "랜덤포레스트", mlp: "MLP", xgb: "XGBoost", lstm: "LSTM", cnn: "CNN" };
  let BE = 0.5333;
  // 모델별 "+1.5% 먼저" 확률. 롱은 본전(BE) 넘으면, 숏은 1−BE 밑이면 그 방향에 동의한 것.
  const votes = (m, side) => {
    if (!m || !Object.keys(m).length) return "";
    const ok = (v) => (side === "숏" ? v < 1 - BE : v > BE);
    return MODEL_ORDER.filter((k) => k in m).map((k) =>
      `<span class="${ok(m[k]) ? "up" : "mini"}" title="${esc(NAMES[k])}">${esc(NAMES[k])} ${num(m[k] * 100, 0)}%${ok(m[k]) ? "✓" : ""}</span>`)
      .join(" · ");
  };
  const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();

  const openText = (o) => {
    if (!o) return "대기 (신호 없음)";
    if (o.kind === "지정가 대기") return `${sideChip(o.side)} 지정가 대기 ${num(o.limit)} (신호 ${esc(o.signal_at)}, 30분 유효)`;
    return `${sideChip(o.side)} 보유 중 · ${esc(o.filled_at)} ${num(o.price)} 체결 · 목표 ${num(o.target)} / 손절 ${num(o.stop)} · 지금 1배 ${pct(o.pnl1)} (3배 ${pct(o.pnl1 * 3)})`;
  };

  // 맨 위 쉬운 요약: 규칙마다 "지금 들고 있나 / 끝난 매매로 번 돈 / 지금 평가손익"을 한눈에.
  const NICK = { A: "기회 많은 규칙", B: "깐깐한 규칙", AS: "A와 같은 규칙, 씨드 5벌 평균 모델", BS: "B와 같은 규칙, 씨드 5벌 평균 모델" };
  const LABEL = { A: "A", B: "B", AS: "A평균", BS: "B평균" };
  const noteHtml = (r) => (r.note ? `<div class="mini" style="margin-top:4px;color:${r.id === "A" ? RED : "var(--muted)"}">⚑ ${esc(r.note)}</div>` : "");
  const sideWord = (s) => (s === "롱" ? "롱(오르면 이익)" : "숏(내리면 이익)");
  const summaryBox = (r) => {
    const o = r.open;
    const now = !o ? "들고 있는 것 없음 · 다음 신호 기다리는 중"
      : o.kind === "지정가 대기" ? `${sideWord(o.side)} 사려고 ${num(o.limit)}에 주문 걸어 둠 (30분 안 닿으면 취소)`
        : `${sideWord(o.side)} 들고 있음 · ${num(o.price)}에 삼 · 지금 ${pct(o.pnl1)} (3배 ${pct(o.pnl1 * 3)})`
          + `<br><span class="mini">${num(o.target)} 닿으면 이익 보고 팔고, ${num(o.stop)} 닿으면 손해 보고 팔아요</span>`;
    const why = o && o.models ? `<div class="mini" style="margin-top:4px">신호 때 모델 판단: ${votes(o.models, o.side)}</div>` : "";
    const done = r.fills
      ? `끝난 매매 ${r.fills}번 (이익 ${r.wins} · 손해 ${r.fills - r.wins}) · 합계 ${pct(r.x1)} (3배 ${pct(r.x3)})`
      : "아직 끝난 매매 없음";
    return `<div style="flex:1;min-width:260px;border:1px solid var(--line);border-radius:8px;padding:10px">
      <div><b>규칙 ${esc(LABEL[r.id] || r.id)}</b> <span class="mini">${NICK[r.id] || ""}</span></div>${noteHtml(r)}
      <div style="margin-top:6px">지금: ${now}</div>${why}
      <div style="margin-top:4px">${done}</div></div>`;
  };
  const renderSummary = (d) => {
    $("btcml-summary").innerHTML = `<div style="display:flex;gap:10px;flex-wrap:wrap">${d.rules.map(summaryBox).join("")}</div>`;
  };

  const renderTables = (d) => {
    if (d.model_names) NAMES = d.model_names;
    if (d.be) BE = d.be;
    renderSummary(d);
    $("btcml-meta").innerHTML = `<div class="detail">기록 시작 ${esc(d.start)} KST · 마지막 봉 ${esc(d.last_bar)} 마감 · 갱신 ${esc(d.generated_at.slice(11, 16))} · 기록된 봉 ${num(d.bars, 0)}개 · 모델 학습 ${esc(d.trained)}</div>`;
    $("btcml-rules").innerHTML = `<table class="acct"><thead><tr>
        <th>규칙</th><th>진입 조건</th><th>지금</th><th>신호</th><th>체결</th><th>승</th><th>누적 1배</th><th>누적 3배</th><th>과거(2024~, 3배)</th>
      </tr></thead><tbody>${d.rules.map((r) => `<tr>
        <td><b>${esc(LABEL[r.id] || r.id)}</b>${noteHtml(r)}</td>
        <td class="mini">점수 ≥ ${num(r.long_above, 3)} + ${r.agree}개 이상 동의 → 롱<br>점수 ≤ ${num(r.short_below, 3)} + ${r.agree}개 이상 동의 → 숏<br>(작년 기준 상위·하위 ${r.top}%)</td>
        <td>${openText(r.open)}</td><td>${r.signals}</td><td>${r.fills}</td>
        <td>${r.fills ? `${r.wins}/${r.fills}` : "-"}</td><td>${pct(r.x1)}</td><td>${pct(r.x3)}</td>
        <td class="mini">1년 ${r.backtest.per_year}번 · ${pct(r.backtest.x3)} · 낙폭 ${num(r.backtest.mdd)}%</td></tr>`).join("")}</tbody></table>
      <p class="mini">주문: 봉 마감가에서 롱 −0.1%, 숏 +0.1%에 지정가, 30분 안 닿으면 취소. 체결가 ±1.5%에서 목표(지정가)·손절(스톱). 한 번에 한 포지션. 수수료 포함, 펀딩비는 빠져 있어요. 현물 가격 기준.</p>`;
    const rows = d.rules.flatMap((r) => r.trades.map((x) => ({ rule: r.id, ...x })))
      .sort((a, b) => (a.signal_at < b.signal_at ? 1 : -1));
    $("btcml-trades").innerHTML = rows.length ? `<table class="acct"><thead><tr>
        <th>규칙</th><th>방향</th><th>신호</th><th>지정가</th><th>체결</th><th>청산</th><th>결과</th><th>1배</th><th>3배</th><th>모델 판단 (+1.5% 먼저 닿을 확률, ✓ = 그 방향 동의)</th>
      </tr></thead><tbody>${rows.map((x) => `<tr><td>${esc(LABEL[x.rule] || x.rule)}</td><td>${sideChip(x.side)}</td>
        <td>${esc(x.signal_at)}</td><td>${num(x.limit)}</td><td>${esc(x.filled_at || "-")}</td><td>${esc(x.exit_at || "-")}</td>
        <td>${esc(x.result)}</td><td>${pct(x.net1)}</td><td>${pct(x.net3)}</td>
        <td class="mini">${votes(x.models, x.side)}</td></tr>`).join("")}</tbody></table>`
      : "아직 끝난 매매가 없어요";
    const sig = (v) => (v > 0 ? sideChip("롱") : v < 0 ? sideChip("숏") : "-");
    $("btcml-recent").innerHTML = `<table class="acct"><thead><tr><th>봉 마감</th><th>종가</th><th>점수</th><th>오른다/내린다(5개 중)</th><th>B</th><th>모델별 오를 확률</th></tr></thead>
      <tbody>${d.recent.slice(0, 12).map((r) => `<tr><td>${esc(r.at)}</td><td>${num(r.close)}</td><td>${num(r.score, 3)}</td>
        <td>${r.up} / ${r.dn}</td><td>${sig(r.B)}</td>
        <td class="mini">${votes(r.models, r.dn > r.up ? "숏" : "롱")}</td></tr>`).join("")}</tbody></table>`;
  };

  const pad = (n) => String(n).padStart(2, "0");
  const hhmm = (t, withDate) => {
    const d = new Date(t * 1000);
    const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    return withDate ? `${pad(d.getUTCMonth() + 1)}/${pad(d.getUTCDate())} ${hm}` : hm;
  };
  let chart = null;
  let series = null;

  const buildChart = () => {
    const LC = window.LightweightCharts;
    const text = css("--text") || "#d1d4dc", grid = css("--line") || "#2a2e39";
    chart = LC.createChart($("btcml-chart"), {
      autoSize: true,
      // 브라우저 언어 정보가 이상하면 기본 날짜 형식이 RangeError로 멈춘다. 형식을 직접 정한다.
      localization: { locale: "ko-KR", timeFormatter: (t) => hhmm(t, true) },
      layout: { background: { type: "solid", color: css("--card") || "#131722" }, textColor: text, fontSize: 11,
        panes: { separatorColor: grid } },
      grid: { vertLines: { color: grid }, horzLines: { color: grid } },
      crosshair: { mode: LC.CrosshairMode.Normal },
      rightPriceScale: { borderColor: grid },
      timeScale: { borderColor: grid, timeVisible: true, secondsVisible: false, rightOffset: 4,
        tickMarkFormatter: (t, type) => (type <= 2 ? hhmm(t, true).slice(0, 5) : hhmm(t, false)) },
    });
    const line = (pane, color, title) => chart.addSeries(LC.LineSeries, { color, lineWidth: 2, title,
      priceLineVisible: false, lastValueVisible: true }, pane);
    series = {
      candle: chart.addSeries(LC.CandlestickSeries, { upColor: GREEN, downColor: RED, borderVisible: false,
        wickUpColor: GREEN, wickDownColor: RED }, 0),
      score: line(1, "#2962ff", "점수"),
      stoch: line(2, "#f7a21b", "Stoch"),
      cci: line(3, "#ab47bc", "CCI"),
      rsi: line(4, "#7e57c2", "RSI"),
    };
    series.markers = LC.createSeriesMarkers(series.candle, []);
    const guide = (s, price, color, title) => s.createPriceLine({ price, color, lineWidth: 1, lineStyle: 2,
      axisLabelVisible: false, title });
    guide(series.stoch, 80, grid, ""); guide(series.stoch, 20, grid, "");
    guide(series.cci, 100, grid, ""); guide(series.cci, -100, grid, ""); guide(series.cci, 0, grid, "");
    guide(series.rsi, 70, grid, ""); guide(series.rsi, 30, grid, ""); guide(series.rsi, 50, grid, "");
    series.thr = [];
    const panes = chart.panes();
    [0.46, 0.18, 0.12, 0.12, 0.12].forEach((f, i) => panes[i] && panes[i].setStretchFactor(f));
  };

  // 롱 = 동그라미, 숏 = 화살표(삼각형). 롱 진입·숏 청산 초록, 롱 청산·숏 진입 빨강.
  const marker = (m) => {
    const long = m.side > 0;
    const open = m.kind === "open";
    const green = long === open;
    const res = open ? "" : ` ${m.win ? "목표" : "손절"} ${m.net3 > 0 ? "+" : ""}${m.net3}%`;
    return {
      time: m.time,
      position: long === open ? "belowBar" : "aboveBar",
      color: green ? GREEN : RED,
      shape: long ? "circle" : (open ? "arrowDown" : "arrowUp"),
      text: `${LABEL[m.rule] || m.rule} ${long ? "롱" : "숏"}${open ? " 진입" : " 청산"}${res}`,
    };
  };

  const renderChart = (c) => {
    if (!c || !window.LightweightCharts) return;
    const first = !chart;
    if (first) buildChart();
    const at = (k) => c.bars.filter((b) => b[k] != null).map((b) => ({ time: b[0], value: b[k] }));
    series.candle.setData(c.bars.map((b) => ({ time: b[0], open: b[1], high: b[2], low: b[3], close: b[4] })));
    series.stoch.setData(at(5)); series.cci.setData(at(6)); series.rsi.setData(at(7)); series.score.setData(at(8));
    series.thr.forEach((p) => series.score.removePriceLine(p));
    series.thr = [
      [c.thr.B[0], GREEN, "B 롱"], [c.thr.B[1], RED, "B 숏"],
    ].map(([price, color, title]) => series.score.createPriceLine({ price, color, lineWidth: 1, lineStyle: 2,
      axisLabelVisible: true, title }));
    const marks = c.marks.map(marker).sort((a, b) => a.time - b.time);
    marks.push({ time: c.start, position: "aboveBar", color: css("--muted") || "#888", shape: "square", text: "기록 시작" });
    series.markers.setMarkers(marks.sort((a, b) => a.time - b.time));
    if (first) chart.timeScale().fitContent();
    $("btcml-legend").innerHTML = "위: 15분봉 캔들(KST) · 아래 칸: 모델 점수(점선 = 규칙 B 문턱) · Stoch · CCI · RSI. " +
      "● 롱, ▲▼ 숏 · 초록 = 롱 진입/숏 청산, 빨강 = 롱 청산/숏 진입. 기록 시작 전 점수는 참고값이에요.";
  };

  const load = () => fetch("/api/btc/ml").then((r) => r.json()).then((d) => {
    if (d.status === "pending") { $("btcml-meta").textContent = d.message; return; }
    renderTables(d);
    renderChart(d.chart);
  }).catch(() => { $("btcml-meta").textContent = "추적 기록을 불러오지 못했어요"; });

  load();
  setInterval(load, 60000);
})();

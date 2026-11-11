// 종목 분석 > SOXL·NVDA: 비트코인 화면과 같은 조건(SCR ① 15분, Stoch 9·CCI 20·RSI 21, W3, 3개 중 3개, 넘어옴)으로
// 미국 주식 무기한 선물을 보여준다(사용자 요청 2026-11-11). 보기만 하고 자동매매와 연결되지 않는다.
// 큰 그림·일봉 지표 분석 /api/soxl?symbol=, 봉 /api/btc/klines?symbol=. 종목마다 같은 틀을 상자 안에 그린다.
(() => {
  const $ = (id) => document.getElementById(id);
  const core = window.ScrLab;
  const U = window.MinervaUI;
  const V = window.ScrView;
  const { kst } = U;
  const GREEN = "#26a69a", RED = "#ef5350";
  const DP = 2;
  // box: 종목 분석 고르기 버튼이 여닫는 상자, id: 상자 안 요소 이름 앞머리
  const TICKERS = [
    { box: "an-sx", id: "sx", symbol: "SOXLUSDT", stock: "SOXL", base: "SOXX", what: "반도체 3배 ETF",
      short: "숏 타점 (뒤집은 차트 = SOXS 눌림)" },
    { box: "an-nv", id: "nv", symbol: "NVDAUSDT", stock: "NVDA", base: "NVDA", what: "엔비디아",
      short: "숏 타점 (20일선 되돌림 뒤 스토캐 꺾임)" },
  ];

  const n2 = (v) => U.num(v, 2);
  const sgn = (v) => (v == null ? "" : `${v > 0 ? "+" : ""}${n2(v)}%`);
  const ok = (b, yes, no) => (b ? `<span class="chip good">✓ ${yes}</span>` : `<span class="chip">✗ ${no}</span>`);
  const scrTxt = (f) => (!f ? "-" : `롱 ${f.long_level}단 · 숏 ${f.short_level}단 <span class="mini">(Stoch ${U.num(f.stoch, 0)} · CCI ${U.num(f.cci, 0)} · RSI ${U.num(f.rsi, 0)}`
    + `${f.last_long3 ? ` · 마지막 롱 3단 ${U.esc(f.last_long3.day)}` : ""}${f.last_short3 ? ` · 마지막 숏 3단 ${U.esc(f.last_short3.day)}` : ""})</span>`);

  // ---- 일봉 지표 분석(/api/soxl의 classic). lev가 1이면 환산 칸을 뺀다 ----
  const classicHtml = (c, t) => {
    if (!c) return '<p class="placeholder">계산 대기 중</p>';
    if (c.error) return `<p class="placeholder">${U.esc(c.error)}</p>`;
    const L = c.long_setup, S = c.short_setup, i = c.ind;
    const conv = c.lev !== 1;
    const px = (x) => `${n2(x.base)}${conv ? ` (${U.esc(c.stock)} ≈ ${n2(x.tgt)})` : ""}`;
    const lvRow = (x) => `<tr><td>${U.esc(x.name)}</td><td>${n2(x.base)} <span class="mini">${sgn(x.pct)}</span></td>${conv ? `<td>${x.tgt == null ? "-" : n2(x.tgt)}</td>` : ""}</tr>`;
    const prof = (c.profile || []).map((b) => `${n2(b.lo)}~${n2(b.hi)} (${U.num(b.pct, 1)}%)`).join(" · ") || "-";
    return `<div class="mini">${U.esc(c.base)} 일봉 ${U.esc(c.day)} 종가 ${n2(c.close)}${conv ? ` · ${U.esc(c.stock)} 같은 날 종가 ${n2(c.tgt_close)}` : ""} · Stoch ${U.num(i.stoch, 1)} (%D ${U.num(i.stoch_d, 1)}) · CCI ${U.num(i.cci, 1)} · RSI ${U.num(i.rsi, 1)} · 롱 ${i.long_level}단 / 숏 ${i.short_level}단</div>
      <h3 style="margin:10px 0 4px;font-size:14px">롱 타점 (${U.esc(c.base)}, 단타 데드캣) ${L.ok ? '<span class="chip good">타점 조건 충족</span>' : ""}</h3>
      <div>${ok(L.bb_touch, `볼밴 하단 터치 (${U.esc(L.bb_touch_day)})`, "최근 5봉 볼밴 하단 터치 없음")} ${ok(L.scr_level === 3, "SCR 3단 돌파", `SCR ${L.scr_level}단`)}</div>
      <div class="detail">목표: 10·20일선까지 반등 뒤 다시 하락 → ${L.targets.map((x) => `${U.esc(x.name)} ${px(x)}`).join(", ")}</div>
      <h3 style="margin:10px 0 4px;font-size:14px">${U.esc(t.short)} ${S.ok ? '<span class="chip warn">타점 조건 충족</span>' : ""}</h3>
      <div>${ok(S.near_ma20, "20일선 근처", "20일선에서 멂")} ${ok(S.stoch_hot, "스토캐 과매수", "스토캐 80 아래")} ${ok(S.stoch_turn, `스토캐 80 아래로 꺾여 마감 (${U.esc(S.turn_day)})`, "최근 2봉 꺾임 마감 없음")}</div>
      <div class="detail">20일선 ${px(S.ma20)}에서 스토캐가 과매수에서 꺾이면 숏 추세 유지 → 볼밴 하단 ${px(S.target)} 쪽.
        20일선 위 마감이면 숏 단기 위험, 50일선 ${px(S.trend_break)} 위면 숏 추세 이탈. 추세 쪽은 CCI·RSI가 극단을 안 만들어 스토캐만 봐요.</div>
      <div style="overflow-x:auto;margin-top:8px"><table class="acct"><thead><tr><th>선</th><th>${U.esc(c.base)}</th>${conv ? `<th>${U.esc(c.stock)} 환산</th>` : ""}</tr></thead>
        <tbody>${c.lines.map(lvRow).join("")}</tbody></table></div>
      <div class="detail">매물대(최근 60일 거래량 많은 두 칸): ${prof}</div>
      <div class="detail">장투(주·월봉 SCR 돌파 보고 진입) · 주봉 ${scrTxt(c.weekly)} · 월봉 ${scrTxt(c.monthly)}</div>
      <p class="mini">${U.esc(c.note)}</p>`;
  };

  const skeleton = (t) => `
    <section class="card">
      <h2>🧭 큰 그림 <span class="mini">${t.stock} · 4시간(선물)·일·주·월(미국 주가) · 마감 봉 기준 · 5분마다 갱신</span></h2>
      <div id="${t.id}-outlook" class="placeholder">불러오는 중…</div>
    </section>
    <section class="card">
      <h2>📊 ${t.base} 일봉 지표 분석 <span class="mini">${t.base} 일봉${t.base !== t.stock ? "(1배)" : ""} · Stoch 5 · CCI 20 · RSI 14 · 볼밴(20,2) · 이평선 · 매물대 · 검증 안 된 판독</span></h2>
      <div id="${t.id}-classic" class="placeholder">불러오는 중…</div>
    </section>
    <section class="card">
      <h2>차트 <span class="mini">${t.symbol} 선물 15분봉 7일 · SCR ① (비트코인과 같은 조건) · 보기만, 자동매매 없음</span></h2>
      <div class="mini">● 초록 = 롱 신호, ▼ 빨강 = 숏 신호. 아래: Stoch 9 · CCI 20 · RSI 21. 미국 장이 닫힌 밤·주말엔 선물 거래가 얇아요.</div>
      <div id="${t.id}-chart" style="height:640px;margin-top:6px"></div>
    </section>
    <section class="card">
      <h2>지금 지표 <span class="mini">SCR ① · 마감된 봉</span></h2>
      <div id="${t.id}-ind" class="placeholder">-</div>
    </section>`;

  const mount = (t) => {
    const box = $(t.box);
    if (!box) return;
    box.innerHTML = skeleton(t);
    let bars = null, chart = null, series = null, started = false;
    const visible = () => !box.hidden && $("tab-analysis") && !$("tab-analysis").hidden;

    const buildChart = () => {
      const LC = window.LightweightCharts;
      const el = $(`${t.id}-chart`);
      if (!LC || !el || !el.clientWidth) return false;
      ({ chart, series } = U.scrChart(el, { timeFormatter: (x) => V.fmt(x),
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
      $(`${t.id}-ind`).innerHTML = V.indHtml(core, bars, res, p, DP);
      if (!visible() || (!chart && !buildChart())) return;
      U.scrFill(series, bars, from, res, p, GREEN, RED);
      const marks = [];
      res.long.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "belowBar", color: GREEN, shape: "circle", text: "롱" }); });
      res.short.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "aboveBar", color: RED, shape: "arrowDown", text: "숏" }); });
      series.marks.setMarkers(marks.sort((a, b) => a.time - b.time));
    };

    const get = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    const load = () => Promise.all([get(`/api/soxl?symbol=${t.symbol}`), get(`/api/btc/klines?symbol=${t.symbol}&interval=15m&days=7`)])
      .then(([o, k]) => {
        try {
          $(`${t.id}-outlook`).innerHTML = o && o.outlook ? V.outlookHtml(o.outlook, DP)
            : `<p class="placeholder">${U.esc((o && o.message) || "큰 그림을 불러오지 못했어요")}</p>`;
        } catch (err) { console.error(err); }
        try { $(`${t.id}-classic`).innerHTML = classicHtml(o && o.classic, t); } catch (err) { console.error(err); }
        if (k && k.bars && k.bars.length) { bars = k.bars; bars.showFrom = k.show_from; }
        else if (!bars) $(`${t.id}-ind`).textContent = "봉을 받지 못했어요. 잠시 뒤 다시 열어 주세요.";
        try { recompute(); } catch (err) { console.error(err); }
      });

    // 처음 열 때 불러온다(안 보는 사람은 야후·바이낸스를 부르지 않게).
    const start = () => {
      if (!visible()) return;
      if (!started) { started = true; load(); setInterval(() => { if (visible()) load(); }, 60000); }
      else setTimeout(recompute, 50);
    };
    window.addEventListener("hashchange", () => setTimeout(start, 50));
    document.querySelectorAll('[data-pick="an"] button').forEach((b) => b.addEventListener("click", () => setTimeout(start, 50)));
    setTimeout(start, 0);
  };

  TICKERS.forEach(mount);
})();

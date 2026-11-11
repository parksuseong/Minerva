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

  // ---- 원문 방식(SOXX 일봉, /api/soxl의 classic) ----
  const n2 = (v) => U.num(v, 2);
  const sgn = (v) => (v == null ? "" : `${v > 0 ? "+" : ""}${n2(v)}%`);
  const ok = (b, yes, no) => (b ? `<span class="chip good">✓ ${yes}</span>` : `<span class="chip">✗ ${no}</span>`);
  const lvRow = (x) => `<tr><td>${U.esc(x.name)}</td><td>${n2(x.soxx)} <span class="mini">${sgn(x.pct)}</span></td><td>${x.soxl == null ? "-" : n2(x.soxl)}</td></tr>`;
  const scrTxt = (f, unit) => (!f ? "-" : `롱 ${f.long_level}단 · 숏 ${f.short_level}단 <span class="mini">(Stoch ${U.num(f.stoch, 0)} · CCI ${U.num(f.cci, 0)} · RSI ${U.num(f.rsi, 0)}`
    + `${f.last_long3 ? ` · 마지막 롱 3단 ${U.esc(f.last_long3.day)}` : ""}${f.last_short3 ? ` · 마지막 숏 3단 ${U.esc(f.last_short3.day)}` : ""})</span>`);
  const classicHtml = (c) => {
    if (!c) return '<p class="placeholder">계산 대기 중</p>';
    if (c.error) return `<p class="placeholder">${U.esc(c.error)}</p>`;
    const L = c.long_setup, S = c.short_setup, i = c.ind;
    const prof = (c.profile || []).map((b) => `${n2(b.lo)}~${n2(b.hi)} (${U.num(b.pct, 1)}%)`).join(" · ") || "-";
    return `<div class="mini">SOXX 일봉 ${U.esc(c.day)} 종가 ${n2(c.close)} · SOXL 같은 날 종가 ${n2(c.soxl_close)} · Stoch ${U.num(i.stoch, 1)} (%D ${U.num(i.stoch_d, 1)}) · CCI ${U.num(i.cci, 1)} · RSI ${U.num(i.rsi, 1)} · 롱 ${i.long_level}단 / 숏 ${i.short_level}단</div>
      <h3 style="margin:10px 0 4px;font-size:14px">롱 타점 (SOXX, 단타 데드캣) ${L.ok ? '<span class="chip good">원문 타점 조건 충족</span>' : ""}</h3>
      <div>${ok(L.bb_touch, `볼밴 하단 터치 (${U.esc(L.bb_touch_day)})`, "최근 5봉 볼밴 하단 터치 없음")} ${ok(L.scr_level === 3, "SCR 3단 돌파", `SCR ${L.scr_level}단`)}</div>
      <div class="detail">목표(원문): 10·20일선까지 반등 뒤 다시 하락 → ${L.targets.map((t) => `${U.esc(t.name)} ${n2(t.soxx)} (SOXL ≈ ${n2(t.soxl)})`).join(", ")}</div>
      <h3 style="margin:10px 0 4px;font-size:14px">숏 타점 (뒤집은 차트 = SOXS 눌림) ${S.ok ? '<span class="chip warn">원문 타점 조건 충족</span>' : ""}</h3>
      <div>${ok(S.near_ma20, "20일선 근처", "20일선에서 멂")} ${ok(S.stoch_hot, "스토캐 과매수(뒤집으면 과매도)", "스토캐 80 아래")} ${ok(S.stoch_turn, `스토캐 80 아래로 꺾여 마감 (${U.esc(S.turn_day)})`, "최근 2봉 꺾임 마감 없음")}</div>
      <div class="detail">20일선 ${n2(S.ma20.soxx)} (SOXL ≈ ${n2(S.ma20.soxl)})에서 스토캐가 과매수에서 꺾이면 숏 추세 유지 → 볼밴 하단 ${n2(S.target.soxx)} (SOXL ≈ ${n2(S.target.soxl)}) 쪽.
        20일선 위 마감이면 숏 단기 위험, 50일선 ${n2(S.trend_break.soxx)} (SOXL ≈ ${n2(S.trend_break.soxl)}) 위면 숏 추세 이탈. 추세 쪽은 CCI·RSI가 극단을 안 만들어 스토캐만 봐요.</div>
      <div style="overflow-x:auto;margin-top:8px"><table class="acct"><thead><tr><th>선</th><th>SOXX</th><th>SOXL 환산</th></tr></thead>
        <tbody>${c.lines.map(lvRow).join("")}</tbody></table></div>
      <div class="detail">매물대(최근 60일 거래량 많은 두 칸): ${prof}</div>
      <div class="detail">장투(원문: 주·월봉 SCR 돌파 보고 진입) · 주봉 ${scrTxt(c.weekly)} · 월봉 ${scrTxt(c.monthly)}</div>
      <p class="mini">${U.esc(c.note)}</p>`;
  };

  const get = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const load = () => Promise.all([get("/api/soxl"), get("/api/btc/klines?symbol=SOXLUSDT&interval=15m&days=7")])
    .then(([o, k]) => {
      try {
        $("sx-outlook").innerHTML = o && o.outlook ? V.outlookHtml(o.outlook, DP)
          : `<p class="placeholder">${U.esc((o && o.message) || "큰 그림을 불러오지 못했어요")}</p>`;
      } catch (err) { console.error(err); }
      try { $("sx-classic").innerHTML = classicHtml(o && o.classic); } catch (err) { console.error(err); }
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

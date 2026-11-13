// 종목 분석 > 🟡 바이낸스: 엔진이 지금 매매하는 종목 버튼 줄(/api/btc/traded, 새로 들어갈 수 있는 종목 + 든 종목,
// 사용자 2026-11-13 "바이낸스에는 현재거래하는것만"). BTC를 고르면 btcengine.js가 그리는 BTC 칸을 보이고,
// 다른 종목은 여기서 같은 틀(SCR ① 15분, Stoch 9·CCI 20·RSI 21, W3, 3개 중 3개, 넘어옴)로 차트·지금 지표를 그린다.
// 큰 그림·일봉 지표 분석은 /api/soxl이 아는 종목(SOXL·NVDA·ETH)만. 봉 /api/btc/klines?symbol=.
// 종목 상자는 처음 고를 때 만들고 그때부터 불러온다(안 보는 종목은 바이낸스를 부르지 않게). 보기만, 주문 없음.
(() => {
  const $ = (id) => document.getElementById(id);
  const core = window.ScrLab;
  const U = window.MinervaUI;
  const V = window.ScrView;
  const { kst } = U;
  const GREEN = "#26a69a", RED = "#ef5350";
  const SHORT_TXT = { SOXLUSDT: "숏 타점 (뒤집은 차트 = SOXS 눌림)" };
  const SHORT_DEFAULT = "숏 타점 (20일선 되돌림 뒤 스토캐 꺾임)";
  // 가격 소수 자리: 가격 크기로 고른다(BNB 600대 2자리, 0.5 같은 종목 4자리)
  const dpOf = (px) => (px == null ? 2 : px >= 1000 ? 1 : px >= 1 ? 2 : px >= 0.01 ? 4 : 6);

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
    ${t.analysis ? `<section class="card">
      <h2>🧭 큰 그림 <span class="mini">${U.esc(t.stock)} · ${t.crypto ? "4시간·일·주·월(바이낸스 선물)" : "4시간(선물)·일·주·월(미국 주가)"} · 마감 봉 기준 · 5분마다 갱신</span></h2>
      <div id="${t.id}-outlook" class="placeholder">불러오는 중…</div>
    </section>
    <section class="card">
      <h2>📊 ${U.esc(t.base)} 일봉 지표 분석 <span class="mini">${U.esc(t.base)} 일봉${t.base !== t.stock ? "(1배)" : ""} · Stoch 5 · CCI 20 · RSI 14 · 볼밴(20,2) · 이평선 · 매물대 · 검증 안 된 판독</span></h2>
      <div id="${t.id}-classic" class="placeholder">불러오는 중…</div>
    </section>` : ""}
    <section class="card">
      <h2>차트 <span class="mini">${U.esc(t.symbol)} 선물 15분봉 7일 · SCR ① (엔진과 같은 조건)${t.held ? " · 엔진이 지금 들고 있어요" : ""}</span></h2>
      <div class="mini">● 초록 = 롱 신호, ▼ 빨강 = 숏 신호. 아래: Stoch 9 · CCI 20 · RSI 21.${t.equity ? " 미국 장이 닫힌 밤·주말엔 선물 거래가 얇아요." : ""}</div>
      <div id="${t.id}-chart" style="height:640px;margin-top:6px"></div>
    </section>
    <section class="card">
      <h2>지금 지표 <span class="mini">SCR ① · 마감된 봉</span></h2>
      <div id="${t.id}-ind" class="placeholder">-</div>
    </section>`;

  // row: /api/btc/traded의 한 줄. 상자를 parent 안에 만들고 { box, start }를 돌려준다.
  const mount = (row, parent) => {
    const a = row.analysis || null;
    const name = row.name || row.symbol.replace(/USDT$/, "");
    const t = {
      symbol: row.symbol,
      id: `sv-${row.symbol.toLowerCase().replace(/[^a-z0-9]/g, "")}`,
      stock: (a && a.stock) || name,
      base: (a && a.base) || name,
      crypto: a ? !!a.crypto : row.underlying === "COIN",
      equity: a ? !a.crypto : row.underlying === "EQUITY",
      short: SHORT_TXT[row.symbol] || SHORT_DEFAULT,
      analysis: !!a,
      held: !!row.held,
    };
    const box = document.createElement("div");
    box.hidden = true;
    box.innerHTML = skeleton(t);
    parent.appendChild(box);
    const el = (k) => $(`${t.id}-${k}`);
    let bars = null, chart = null, series = null, started = false, dp = 2;
    const visible = () => !box.hidden && !$("an-bn").hidden && $("tab-analysis") && !$("tab-analysis").hidden;

    const buildChart = () => {
      const LC = window.LightweightCharts;
      const c = el("chart");
      if (!LC || !c || !c.clientWidth) return false;
      ({ chart, series } = U.scrChart(c, { timeFormatter: (x) => V.fmt(x),
        titles: ["Stoch 9 (20/80)", "CCI 20 (±100)", "RSI 21 (30/70)"], up: GREEN, down: RED }));
      series.marks = LC.createSeriesMarkers(series.candle, []);
      return true;
    };

    const recompute = () => {
      if (!bars || !core) return;
      const p = core.DEFAULTS;
      const res = core.signals(bars.map((b) => b[2]), bars.map((b) => b[3]), bars.map((b) => b[4]), p);
      let from = bars.findIndex((b) => b[0] >= (bars.showFrom || 0));
      if (from < 0) from = 0;
      el("ind").innerHTML = V.indHtml(core, bars, res, p, dp);
      if (!visible() || (!chart && !buildChart())) return;
      series.candle.applyOptions({ priceFormat: { type: "price", precision: dp, minMove: 10 ** -dp } });
      U.scrFill(series, bars, from, res, p, GREEN, RED);
      const marks = [];
      res.long.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "belowBar", color: GREEN, shape: "circle", text: "롱" }); });
      res.short.forEach((i) => { if (i >= from) marks.push({ time: kst(bars[i][0]), position: "aboveBar", color: RED, shape: "arrowDown", text: "숏" }); });
      series.marks.setMarkers(marks.sort((x, y) => x.time - y.time));
    };

    const get = (url) => fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    const sym = encodeURIComponent(t.symbol);
    const load = () => Promise.all([t.analysis ? get(`/api/soxl?symbol=${sym}`) : Promise.resolve(null),
      get(`/api/btc/klines?symbol=${sym}&interval=15m&days=7`)])
      .then(([o, k]) => {
        if (k && k.bars && k.bars.length) {
          bars = k.bars; bars.showFrom = k.show_from;
          dp = dpOf(bars[bars.length - 1][4]);
        } else if (!bars) el("ind").textContent = "봉을 받지 못했어요. 잠시 뒤 다시 열어 주세요.";
        if (t.analysis) {
          try {
            if (!bars && o && o.outlook) dp = dpOf(o.outlook.price);
            el("outlook").innerHTML = o && o.outlook ? V.outlookHtml(o.outlook, dp)
              : `<p class="placeholder">${U.esc((o && o.message) || "큰 그림을 불러오지 못했어요")}</p>`;
          } catch (err) { console.error(err); }
          try { el("classic").innerHTML = classicHtml(o && o.classic, t); } catch (err) { console.error(err); }
        }
        try { recompute(); } catch (err) { console.error(err); }
      });

    const start = () => {
      if (!visible()) return;
      if (!started) { started = true; load(); setInterval(() => { if (visible()) load(); }, 60000); }
      else setTimeout(recompute, 50);
    };
    return { box, start };
  };

  // ---- 바이낸스 종목 버튼 줄 ----
  const pane = $("an-bn"), row = $("bn-sym"), btcBox = $("bn-btc"), other = $("bn-other");
  if (!pane || !row || !btcBox || !other || !U || !V) return;
  const KEY = "minerva-bn-sym";
  const views = {};
  let list = null, cur = null, want = null, sig = "";
  const paneVisible = () => !pane.hidden && $("tab-analysis") && !$("tab-analysis").hidden;
  const saved = () => { try { return localStorage.getItem(KEY); } catch (e) { return null; } };
  const has = (s) => !!s && !!list && list.some((r) => r.symbol === s);

  const select = (sym, remember) => {
    cur = sym;
    if (remember) { try { localStorage.setItem(KEY, sym); } catch (e) { /* 저장 막혀도 전환은 된다 */ } }
    row.querySelectorAll("button").forEach((b) => {
      const on = b.dataset.sym === sym;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", String(on));
    });
    btcBox.hidden = sym !== "BTCUSDT";
    if (sym && sym !== "BTCUSDT" && !views[sym] && has(sym)) views[sym] = mount(list.find((r) => r.symbol === sym), other);
    Object.keys(views).forEach((s) => { views[s].box.hidden = s !== sym; });
    if (views[sym]) views[sym].start();
    window.dispatchEvent(new Event("minerva:bnsym"));
  };

  // 오늘 할 일의 #btc 링크 등에서 종목을 바로 고를 때
  window.minervaBnSym = (sym) => { if (list) { if (has(sym)) select(sym, false); } else want = sym; };

  const render = () => {
    if (!list.length) {
      row.innerHTML = '<span class="mini">지금 매매하는 종목이 없어요 (새로 들어갈 수 있는 종목도, 든 종목도 없음)</span>';
      select(null, false);
      return;
    }
    row.innerHTML = '<span class="mini">지금 매매하는 종목</span>' + list.map((r) => {
      const wr = r.winrate == null ? "" : ` <span class="mini">${U.num(r.winrate, 0)}%</span>`;
      const held = r.held ? ' <span class="chip">보유</span>' : r.watch ? ' <span class="chip">관심·매매 안 함</span>' : "";
      return `<button type="button" data-sym="${U.esc(r.symbol)}" aria-pressed="false">${U.esc(r.name || r.symbol)}${wr}${held}</button>`;
    }).join("");
    const pick = [want, cur, saved(), "BTCUSDT"].find(has) || list[0].symbol;
    want = null;
    select(pick, false);
  };

  row.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-sym]");
    if (b) select(b.dataset.sym, true);
  });

  const loadList = () => fetch("/api/btc/traded").then((r) => (r.ok ? r.json() : null)).catch(() => null).then((d) => {
    if (!d || d.status !== "ok" || !Array.isArray(d.symbols)) return; // 못 받으면 BTC 칸 그대로
    const next = JSON.stringify(d.symbols);
    if (next === sig) return;
    sig = next;
    list = d.symbols.filter((r) => r && typeof r.symbol === "string");
    try { render(); } catch (err) { console.error(err); }
  });

  let started = false;
  const start = () => {
    if (!paneVisible()) return;
    if (!started) { started = true; loadList(); setInterval(() => { if (paneVisible()) loadList(); }, 60000); }
    else if (views[cur]) views[cur].start();
  };
  window.addEventListener("hashchange", () => setTimeout(start, 50));
  document.querySelectorAll('[data-pick="an"] button, [data-tab="analysis"]').forEach((b) => b.addEventListener("click", () => setTimeout(start, 50)));
  setTimeout(start, 0);
})();

// 종목 분석 맨 위 "신호 가까운 순"(사용자 요청 2026-11-11 "누가 현재 신호 뜰 가능성이 높은지").
// 자동매매 네 종목의 SCR ① 15분(엔진과 같은 기본값)에서 롱·숏 각각 몇 개 지표가 최근 W봉 안에 넘어왔고,
// 남은 지표가 넘어오려면 다음 봉이 어디에 마감해야 하는지(ScrView.linePrice 어림값)로 줄 세운다.
// 확률이 아니라 거리 순서다. 화면 전용, 주문과 무관.
(() => {
  const $ = (id) => document.getElementById(id);
  const core = window.ScrLab;
  const U = window.MinervaUI;
  const V = window.ScrView;
  const SYMS = [
    { symbol: "BTCUSDT", name: "₿ BTC", dp: 1 },
    { symbol: "ETHUSDT", name: "⟠ ETH", dp: 2 },
    { symbol: "SOXLUSDT", name: "🔧 SOXL", dp: 2 },
    { symbol: "NVDAUSDT", name: "🟢 NVDA", dp: 2 },
  ];
  const FAR = 50; // 기준선 반대편이라 한 번 넘어갔다 와야 하는 지표 하나의 벌점(%)

  // 한 종목·한 방향의 거리. need = 남은 지표들, score가 작을수록 가깝다.
  const side = (bars, res, p, long) => {
    const n = bars.length - 1;
    const last = bars[n][4];
    const fired = (long ? res.long : res.short).includes(n);
    const items = [["Stoch", res.stoch, long ? p.stoch_lo : p.stoch_hi],
      ["CCI", res.cci, long ? p.cci_lo : p.cci_hi], ["RSI", res.rsi, long ? p.rsi_lo : p.rsi_hi]].map(([name, x, line]) => {
      const hit = core.recent(core.crossed(x, line, long), p.w)[n];
      const v = x[n];
      if (hit) return { name, hit: true };
      if (v == null) return { name, far: true, cost: FAR };
      const before = long ? v < line : v > line;
      if (!before) return { name, far: true, cost: FAR, v, line };
      const px = V.linePrice(core, bars, name, line, p);
      const d = px == null ? null : (px / last - 1) * 100;
      return { name, px, d, v, line, cost: d == null ? FAR : Math.abs(d) };
    });
    const k = items.filter((x) => x.hit).length;
    const cost = items.reduce((s, x) => s + (x.hit ? 0 : x.cost), 0);
    return { long, fired, k, items, score: fired ? -1 : (3 - k) * 10 + cost };
  };

  const needTxt = (r, dp) => r.items.filter((x) => !x.hit).map((x) => {
    if (x.far) return `${x.name} <span class="mini">(기준선 ${r.long ? "위" : "아래"}라 한 번 ${r.long ? "내려갔다 올라와야" : "올라갔다 내려와야"})</span>`;
    if (x.px == null) return `${x.name} <span class="mini">(±15% 밖)</span>`;
    return `${x.name} <b>${U.num(x.px, dp)}</b> <span class="mini">${r.long ? "위" : "아래"} 마감 (${x.d > 0 ? "+" : ""}${U.num(x.d, 2)}%)</span>`;
  }).join("<br>") || "-";

  const render = (rows, when) => {
    rows.sort((a, b) => a.score - b.score);
    const body = rows.map((r, i) => {
      const dir = r.long ? '<span class="up">롱</span>' : '<span class="down">숏</span>';
      const got = r.items.filter((x) => x.hit).map((x) => x.name).join("·") || "없음";
      const state = r.fired ? '<span class="chip good">지금 신호</span>' : `${r.k}/3 <span class="mini">(${got})</span>`;
      return `<tr><td>${i + 1}</td><td>${r.sym.name}</td><td>${dir}</td><td>${state}</td><td>${r.fired ? "-" : needTxt(r, r.sym.dp)}</td></tr>`;
    }).join("");
    $("near-list").innerHTML = `<div style="overflow-x:auto"><table class="acct"><thead><tr><th>#</th><th>종목</th><th>방향</th><th>충족</th><th>남은 지표 (다음 15분봉 마감 기준 어림값)</th></tr></thead>
      <tbody>${body}</tbody></table></div>
      <p class="mini">마지막 마감 봉 ${when} KST 기준. 순서 = 충족 개수 많은 순, 같으면 남은 지표까지 가격 거리가 짧은 순. 확률이 아니라 거리예요.
        신호는 세 지표가 모두 최근 3봉 안에 넘어와야 해서, 이미 넘어온 지표는 3봉이 지나면 빠져요. 같은 봉에 둘 이상 나면 엔진은 AI(실패 시 승률 순 BTC > NVDA > ETH > SOXL)로 하나를 골라요.</p>`;
  };

  const visible = () => $("tab-analysis") && !$("tab-analysis").hidden;
  const load = () => {
    if (!core || !V || !visible()) return;
    const p = core.DEFAULTS;
    Promise.all(SYMS.map((s) => fetch(`/api/btc/klines?symbol=${s.symbol}&interval=15m&days=3`)
      .then((r) => (r.ok ? r.json() : null)).catch(() => null)))
      .then((all) => {
        const rows = [];
        let when = "-";
        all.forEach((k, i) => {
          if (!k || !k.bars || k.bars.length < 60) return;
          const bars = k.bars;
          const res = core.signals(bars.map((b) => b[2]), bars.map((b) => b[3]), bars.map((b) => b[4]), p);
          [true, false].forEach((long) => rows.push(Object.assign(side(bars, res, p, long), { sym: SYMS[i] })));
          when = V.fmt(U.kst(bars[bars.length - 1][0]));
        });
        if (!rows.length) { $("near-list").textContent = "봉을 받지 못했어요. 잠시 뒤 다시 열어 주세요."; return; }
        try { render(rows, when); } catch (err) { console.error(err); }
      });
  };

  let started = false;
  const start = () => {
    if (!visible()) return;
    load();
    if (!started) { started = true; setInterval(load, 60000); }
  };
  window.addEventListener("hashchange", () => setTimeout(start, 50));
  document.querySelectorAll('[data-tab="analysis"]').forEach((b) => b.addEventListener("click", () => setTimeout(start, 50)));
  setTimeout(start, 0);
})();

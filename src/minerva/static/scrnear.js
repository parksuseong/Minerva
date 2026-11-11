// 종목 분석 맨 위 "신호 가까운 순"(사용자 요청 2026-11-11 "누가 현재 신호 뜰 가능성이 높은지").
// 자동매매 엔진이 15분마다 쓰는 /api/btc/near(엔진 종목 범위 = 거래량 상위 50, 없으면 네 종목)를 그린다:
// SCR ① 15분(엔진과 같은 값)에서 롱·숏 각각 몇 개 지표가 최근 W봉 안에 넘어왔고, 남은 지표가 넘어오려면
// 다음 봉이 어디에 마감해야 하는지(어림값)로 줄 세운다. 확률이 아니라 거리 순서다. 화면 전용, 주문과 무관.
(() => {
  const $ = (id) => document.getElementById(id);
  const U = window.MinervaUI;
  const TOP = 15;
  const FAR = 50; // 기준선 반대편이라 한 번 넘어갔다 와야 하는 지표 하나의 벌점(%)

  const name = (s) => (s.endsWith("USDT") ? s.slice(0, -4) : s);
  const dp = (px) => (px >= 1000 ? 1 : px >= 1 ? 2 : px >= 0.01 ? 4 : 6);
  const score = (r) => {
    if (r.fired) return -1;
    const cost = r.need.reduce((s, x) => s + (x.far || x.pct == null ? FAR : Math.abs(x.pct)), 0);
    return (3 - r.k) * 10 + cost;
  };

  const needTxt = (r) => r.need.map((x) => {
    const long = r.side === "long";
    if (x.far) return `${x.name} <span class="mini">(기준선 ${long ? "위" : "아래"}라 한 번 ${long ? "내려갔다 올라와야" : "올라갔다 내려와야"})</span>`;
    if (x.price == null) return `${x.name} <span class="mini">(±15% 밖)</span>`;
    return `${x.name} <b>${U.num(x.price, dp(r.price))}</b> <span class="mini">${long ? "위" : "아래"} 마감 (${x.pct > 0 ? "+" : ""}${U.num(x.pct, 2)}%)</span>`;
  }).join("<br>") || "-";

  const render = (data) => {
    const rows = (data.rows || []).map((r) => Object.assign({}, r, { score: score(r) }));
    rows.sort((a, b) => a.score - b.score || (b.winrate || 0) - (a.winrate || 0));
    const shown = rows.slice(0, TOP);
    const syms = new Set(rows.map((r) => r.symbol)).size;
    const body = shown.map((r, i) => {
      const dir = r.side === "long" ? '<span class="up">롱</span>' : '<span class="down">숏</span>';
      const got = r.hit.join("·") || "없음";
      const state = r.fired ? '<span class="chip good">지금 신호</span>' : `${r.k}/3 <span class="mini">(${got})</span>`;
      const wr = r.winrate == null ? "-" : `${U.num(r.winrate, 0)}%${r.low_sample ? ' <span class="mini">(표본 적음)</span>' : ""}`;
      const held = r.held ? ' <span class="chip">보유</span>' : "";
      return `<tr><td>${i + 1}</td><td>${U.esc(name(r.symbol))}${held}</td><td>${dir}</td><td>${wr}</td><td>${state}</td><td>${r.fired ? "-" : needTxt(r)}</td></tr>`;
    }).join("");
    const bar = rows.length ? Math.max(...rows.map((r) => r.bar)) : null;
    const when = bar ? window.ScrView.fmt(U.kst(bar)) : "-";
    const scope = data.source === "universe" ? `거래량 상위 ${syms}종목(${U.esc(data.universe_at || "")} 기준)` : `${syms}종목`;
    $("near-list").innerHTML = `<div style="overflow-x:auto"><table class="acct"><thead><tr><th>#</th><th>종목</th><th>방향</th><th>승률</th><th>충족</th><th>남은 지표 (다음 15분봉 마감 기준 어림값)</th></tr></thead>
      <tbody>${body}</tbody></table></div>
      <p class="mini">엔진 ${scope} 중 위 ${Math.min(TOP, rows.length)}줄. 마지막 마감 봉 ${when} KST, 엔진 계산 ${U.esc(data.updated_at || "-")}. 순서 = 지금 신호 먼저, 충족 개수 많은 순, 같으면 남은 지표까지 가격 거리가 짧은 순. 확률이 아니라 거리예요.
        신호는 세 지표가 모두 최근 ${data.w || 3}봉 안에 넘어와야 해서, 이미 넘어온 지표는 ${data.w || 3}봉이 지나면 빠져요. 승률은 종목별 백테스트(SCR ① 15분·3배·손절 3%, 최근 1년).
        같은 봉에 둘 이상 나면 엔진은 백테스트 승률 순 위 5개 중 AI가 하나를 골라요(실패 시 승률 1순위).</p>`;
  };

  const visible = () => $("tab-analysis") && !$("tab-analysis").hidden;
  const load = () => {
    if (!U || !window.ScrView || !visible()) return;
    fetch("/api/btc/near").then((r) => (r.ok ? r.json() : null)).catch(() => null).then((d) => {
      if (!d || d.status !== "ok" || !(d.rows || []).length) {
        $("near-list").textContent = (d && d.message) || "신호 거리를 받지 못했어요. 잠시 뒤 다시 열어 주세요.";
        return;
      }
      try { render(d); } catch (err) { console.error(err); }
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

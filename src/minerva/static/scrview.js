// 종목 분석의 비트코인·SOXL 화면이 같이 쓰는 조각: 큰 그림 카드, 지금 지표 표. common.js·scrlab_core.js 다음에 읽는다.
// dp = 가격 소수 자리(비트코인 0, SOXL 2).
window.ScrView = (() => {
  const U = window.MinervaUI;
  const { esc, pad, kst } = U;
  const num = (v, d = 1) => U.num(v, d);
  const fmt = (sec) => {
    const d = new Date(sec * 1000);
    return `${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  };

  // ---- 큰 그림(4시간·일·주·월 outlook) ----
  const WAY = { "상방": "good", "하방": "warn" };
  const sigAgo = (x, unit) => (x ? `${x.level}단 · ${x.bars_ago === 0 ? "마지막 봉" : `${x.bars_ago}${unit} 전`}` : "-");
  const UNIT_AGO = { "4h": "봉", "1d": "일", "1w": "주", "1M": "달" };
  const outlookHtml = (o, dp) => {
    if (!o || !o.summary) return '<p class="placeholder">큰 그림 계산 대기 중</p>';
    const s = o.summary;
    const lvList = (arr) => (arr || []).map((c) => `<div>${num(c.price, dp)} <span class="mini">(${c.pct > 0 ? "+" : ""}${num(c.pct, 2)}% · ${esc(c.what)})</span></div>`).join("") || "-";
    const tone = { up: "good", down: "warn" }[s.tone] || "";
    const rows = o.frames.map((f) => `<tr><td>${esc(f.name)}</td>
        <td><span class="chip ${WAY[f.direction] || ""}">${esc(f.direction)}</span> <span class="mini">${esc(f.zone)}</span></td>
        <td>${num(f.stoch, 0)} <span class="mini">${f.stoch_turn === "오름" ? "↑" : f.stoch_turn === "내림" ? "↓" : ""}</span> / ${num(f.cci, 0)} / ${num(f.rsi, 0)}</td>
        <td>${esc(s.ma[f.frame] || "")}</td>
        <td><span class="up">롱 ${sigAgo(f.last_long, UNIT_AGO[f.frame])}</span><br><span class="down">숏 ${sigAgo(f.last_short, UNIT_AGO[f.frame])}</span></td>
        <td>±${num(f.atr_pct, 1)}%</td></tr>`).join("");
    return `<div class="regime"><span class="chip ${tone}">${{ up: "위쪽 우세", down: "아래쪽 우세", mixed: "엇갈림" }[s.tone] || "-"}</span>
        <span class="mini">지금 ${num(o.price, Math.max(dp, 1))}</span></div>
      <div style="margin:6px 0"><b>${esc(s.verdict)}</b></div>
      ${(s.watch || []).length ? `<ul style="margin:4px 0 8px;padding-left:18px">${s.watch.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      <div style="display:flex;gap:16px;flex-wrap:wrap;margin:6px 0">
        <div style="flex:1;min-width:150px"><b>위 저항</b> <span class="mini">가까운 순</span>${lvList(s.resist)}</div>
        <div style="flex:1;min-width:150px"><b>아래 지지</b> <span class="mini">가까운 순</span>${lvList(s.support)}</div>
      </div>
      <div class="detail">보통 움직임(ATR 14): 하루 ${num(s.range_day.down, dp)} ~ ${num(s.range_day.up, dp)} · 한 주 ${num(s.range_week.down, dp)} ~ ${num(s.range_week.up, dp)}</div>
      <div style="overflow-x:auto;margin-top:6px"><table class="acct"><thead><tr><th>봉</th><th>방향</th><th>Stoch/CCI/RSI</th><th>이평선</th><th>최근 SCR 신호</th><th>봉 하나 폭</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="mini">${esc(o.note)} 방향 = CCI·RSI 중심선(둘 다 위 상방, 둘 다 아래 하방). 이평선: 4시간·일 20/50/200, 주·월 10/20/50. 저항·지지 = 이평선과 20봉 고저.</p>`;
  };

  // ---- 지금 지표 ----
  // 다음 15분봉이 꼬리 없이 X에 마감한다고 치고(고가=저가=종가=X), 지표가 기준선과 같아지는 X를 이분법으로 찾는다.
  // 세 지표 모두 X에 대해 단조라서 한 값이 나온다. 꼬리가 길면 Stoch·CCI가 달라지니 어림값이다.
  const linePrice = (core, bars, name, line, p) => {
    const h = bars.map((b) => b[2]), l = bars.map((b) => b[3]), c = bars.map((b) => b[4]);
    const at = (x) => {
      const hh = h.concat(x), ll = l.concat(x), cc = c.concat(x);
      const arr = name === "Stoch" ? core.stoch(hh, ll, cc, p.stoch_n)
        : name === "CCI" ? core.cci(hh, ll, cc, p.cci_n) : core.rsi(cc, p.rsi_n);
      return arr[arr.length - 1];
    };
    const last = c[c.length - 1];
    let lo = last * 0.85, hi = last * 1.15;
    const vlo = at(lo), vhi = at(hi);
    if (vlo == null || vhi == null || line < vlo || line > vhi) return null;
    for (let k = 0; k < 40; k++) {
      const mid = (lo + hi) / 2;
      if (at(mid) < line) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  };
  const indHtml = (core, bars, res, p, dp) => {
    const n = bars.length - 1;
    const last = bars[n][4];
    const priceTxt = (x, long, hit, below) => {
      if (x == null) return '<span class="mini">±15% 밖</span>';
      const d = (x / last - 1) * 100;
      const way = hit ? "" : below ? (long ? " 위로 마감하면" : " 아래로 마감하면")
        : (long ? " 아래로 내려갔다 다시 위로" : " 위로 올라갔다 다시 아래로");
      return `<b>${num(x, dp)}</b> <span class="mini">(${d > 0 ? "+" : ""}${num(d, 2)}%)${way}</span>`;
    };
    const rowsFor = (long) => [["Stoch", res.stoch, long ? p.stoch_lo : p.stoch_hi],
      ["CCI", res.cci, long ? p.cci_lo : p.cci_hi], ["RSI", res.rsi, long ? p.rsi_lo : p.rsi_hi]].map(([name, x, line]) => {
      const hit = core.recent(core.crossed(x, line, long), p.w)[n];
      const v = x[n];
      let how;
      if (hit) how = `<span class="chip good">✓ 최근 ${p.w}봉 안에 넘어옴</span>`;
      else if (v == null) how = "-";
      else if (long ? v < line : v > line) how = `기준선 ${long ? "아래" : "위"} ${num(Math.abs(v - line), 1)} — ${long ? "올라오면" : "내려오면"} 충족`;
      else how = `기준선까지 ${num(Math.abs(v - line), 1)} — ${long ? "아래로 내려갔다 올라와야" : "위로 올라갔다 내려와야"}`;
      const price = v == null ? "-" : priceTxt(linePrice(core, bars, name, line, p), long, hit, long ? v < line : v > line);
      return { name, v, line, hit, how, price };
    });
    const block = (long) => {
      const rows = rowsFor(long);
      const k = rows.filter((r) => r.hit).length;
      return `<h3 style="margin:8px 0 4px;font-size:14px">${long ? "롱" : "숏"} 신호까지 <span class="mini">${k}/3 충족 (3개 모두 최근 ${p.w}봉 안이어야 신호)</span></h3>
        <table class="acct"><tbody>${rows.map((r) => `<tr><td>${r.name}</td><td>${num(r.v, 1)}</td><td>기준 ${num(r.line, 0)}</td><td>${r.price}</td><td>${r.how}</td></tr>`).join("")}</tbody></table>`;
    };
    return `<div class="mini">마지막 마감 봉 ${fmt(kst(bars[n][0]))} KST · Stoch ${p.stoch_n} (${p.stoch_lo}/${p.stoch_hi}) · CCI ${p.cci_n} (${p.cci_lo}/${p.cci_hi}) · RSI ${p.rsi_n} (${p.rsi_lo}/${p.rsi_hi}) · 넘어옴</div>`
      + `<div class="mini">가격 칸: 다음 15분봉이 그 값에 마감하면 지표가 기준선에 닿아요(꼬리 없는 봉으로 친 어림값, 지금 종가 ${num(last, dp)} 기준).</div>`
      + block(true) + block(false);
  };

  return { fmt, outlookHtml, indHtml, linePrice };
})();

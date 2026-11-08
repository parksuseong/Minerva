// SCR 실험실 계산(화면 쪽). 파이썬 쌍둥이 strategy/scr_lab.py와 같은 순서로 계산한다 — 테스트가 둘을 맞춰 본다.
// 마감된 봉만 넣는다. 지표 값이 아직 없는 앞부분은 null.
(function (root) {
  const DEFAULTS = {
    stoch_n: 9, stoch_lo: 20, stoch_hi: 80,
    cci_n: 20, cci_lo: -100, cci_hi: 100,
    rsi_n: 21, rsi_lo: 30, rsi_hi: 70,
    w: 3, need: 3, mode: "cross",
  };

  function stoch(h, l, c, n) {
    const out = new Array(c.length).fill(null);
    for (let i = n - 1; i < c.length; i++) {
      let hh = -Infinity, ll = Infinity;
      for (let j = i - n + 1; j <= i; j++) { if (h[j] > hh) hh = h[j]; if (l[j] < ll) ll = l[j]; }
      const rng = hh - ll;
      out[i] = rng === 0 ? 50 : (c[i] - ll) / rng * 100;
    }
    return out;
  }

  function cci(h, l, c, n) {
    const tp = c.map((_, i) => (h[i] + l[i] + c[i]) / 3);
    const out = new Array(c.length).fill(null);
    for (let i = n - 1; i < c.length; i++) {
      let s = 0;
      for (let j = i - n + 1; j <= i; j++) s += tp[j];
      const m = s / n;
      let d = 0;
      for (let j = i - n + 1; j <= i; j++) d += Math.abs(tp[j] - m);
      const md = d / n;
      out[i] = md === 0 ? 0 : (tp[i] - m) / (0.015 * md);
    }
    return out;
  }

  function rsi(c, n) {
    const out = new Array(c.length).fill(null);
    if (c.length <= n) return out;
    const val = (ag, al) => (al === 0 ? 100 : 100 - 100 / (1 + ag / al));
    let ag = 0, al = 0;
    for (let i = 1; i <= n; i++) {
      const d = c[i] - c[i - 1];
      ag += Math.max(d, 0); al += Math.max(-d, 0);
    }
    ag /= n; al /= n;
    out[n] = val(ag, al);
    for (let i = n + 1; i < c.length; i++) {
      const d = c[i] - c[i - 1];
      ag = (ag * (n - 1) + Math.max(d, 0)) / n;
      al = (al * (n - 1) + Math.max(-d, 0)) / n;
      out[i] = val(ag, al);
    }
    return out;
  }

  function crossed(x, line, up) {
    const out = new Array(x.length).fill(false);
    for (let i = 1; i < x.length; i++) {
      const p = x[i - 1], v = x[i];
      if (p == null || v == null) continue;
      out[i] = up ? (p <= line && line < v) : (p >= line && line > v);
    }
    return out;
  }

  function inZone(x, line, below) {
    return x.map((v) => v != null && (below ? v < line : v > line));
  }

  function recent(flags, w) {
    return flags.map((_, i) => {
      for (let j = Math.max(0, i - w + 1); j <= i; j++) if (flags[j]) return true;
      return false;
    });
  }

  // 롱·숏 신호 봉 번호와 지표 값. 같은 봉에 둘 다 나올 수 있다(원본과 같음).
  function signals(h, l, c, params) {
    const p = Object.assign({}, DEFAULTS, params || {});
    const s = stoch(h, l, c, p.stoch_n | 0);
    const cc = cci(h, l, c, p.cci_n | 0);
    const r = rsi(c, p.rsi_n | 0);
    const w = p.w | 0, need = p.need | 0, zone = p.mode === "zone";
    const side = (long) => {
      const lines = [[s, long ? p.stoch_lo : p.stoch_hi], [cc, long ? p.cci_lo : p.cci_hi],
        [r, long ? p.rsi_lo : p.rsi_hi]];
      const hits = lines.map(([x, line]) => recent(zone ? inZone(x, line, long) : crossed(x, line, long), w));
      const on = c.map((_, i) => hits.filter((f) => f[i]).length >= need);
      const idx = [];
      on.forEach((v, i) => { if (v && !(i > 0 && on[i - 1])) idx.push(i); });
      return idx;
    };
    return { long: side(true), short: side(false), stoch: s, cci: cc, rsi: r };
  }

  // 신호 봉 종가에서 k봉 뒤 종가까지 움직임(%), 매매 방향 부호. 자료가 모자라면 null.
  function after(c, i, k, dir) {
    if (!(k >= 1) || k !== Math.floor(k) || i + k >= c.length) return null;
    return (c[i + k] / c[i] - 1) * 100 * dir;
  }

  // 롱↔숏 갈아타기 거친 추정(사용자 요청 2026-11-08 "손절 3% · 3배"). 신호 다음 봉 시가에 들어가고,
  // 반대 신호면 다음 봉 시가에 뒤집는다. 같은 방향 신호는 무시. 진입가 ±stop에 봉 고가·저가가 닿으면
  // 그 값(시가가 이미 넘었으면 시가)에 나가고 다음 신호까지 쉰다. 같은 봉에 롱·숏이 같이 나면 건너뛴다.
  // 수수료는 편도 fee × 레버리지(원금 대비). 슬리피지·펀딩비·청산은 없다.
  function switchSim(o, h, l, c, sigs, opt) {
    const stop = opt.stop, lev = opt.lev, fee = opt.fee;
    const at = new Map();
    sigs.forEach((s) => at.set(s.i, at.has(s.i) && at.get(s.i) !== s.dir ? 0 : s.dir));
    let eq = 1, peak = 1, mdd = 0, pos = 0, entry = 0, trades = 0, wins = 0, stops = 0;
    const close = (px) => {
      const net = lev * pos * (px / entry - 1) - lev * fee * (1 + px / entry);
      eq = Math.max(0, eq * (1 + net));
      trades += 1; if (net > 0) wins += 1;
      pos = 0;
    };
    const first = sigs.reduce((m, s) => Math.min(m, s.i + 1), c.length);
    for (let j = first; j < c.length; j++) {
      const d = at.get(j - 1);
      if (d && d !== pos) {
        if (pos) close(o[j]);
        pos = d; entry = o[j];
      }
      if (pos) {
        const sp = entry * (1 - pos * stop);
        if (pos > 0 ? l[j] <= sp : h[j] >= sp) {
          close(pos > 0 ? Math.min(o[j], sp) : Math.max(o[j], sp));
          stops += 1;
        }
      }
      const mark = pos ? eq * (1 + lev * pos * (c[j] / entry - 1)) : eq;
      if (mark > peak) peak = mark;
      if (peak > 0) mdd = Math.max(mdd, (1 - mark / peak) * 100);
    }
    const holding = pos !== 0;
    if (holding) close(c[c.length - 1]);
    return { trades, wins, stops, ret: (eq - 1) * 100, mdd, holding };
  }

  const api = { DEFAULTS, stoch, cci, rsi, crossed, inZone, recent, signals, after, switchSim };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ScrLab = api;
})(typeof self !== "undefined" ? self : this);

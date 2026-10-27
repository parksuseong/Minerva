import json, os, sys
from datetime import date
sys.path.insert(0, "/home/ec2-user/work/Minerva/src")
from minerva.strategy.indicators import Bar
from minerva.strategy.report import fetch_yahoo
D = os.path.dirname(os.path.abspath(__file__))
def load(sym):
    p = os.path.join(D, sym.replace("^", "_") + ".json")
    if not os.path.exists(p):
        bars = fetch_yahoo(sym, "max")
        json.dump([[b.day.isoformat(), b.open, b.high, b.low, b.close, getattr(b, "volume", 0)] for b in bars], open(p, "w"))
    rows = json.load(open(p))
    out = []
    for r in rows:
        if None in r[1:5]: continue
        out.append(Bar(date.fromisoformat(r[0]), r[1], r[2], r[3], r[4], r[5] or 0))
    return out
if __name__ == "__main__":
    for s in ["^VIX", "^VIX3M", "SPY", "QQQ", "SOXX"]:
        b = load(s); print(s, len(b), b[0].day, b[-1].day)

"""div_usable 진단 그림: chart_hit.png(사후 vs 현실 vs 무작위 적중, 한글 라벨).

사용: python3 docs/backtests/raw/div_usable_chart.py ~/.minerva/research/div-usable-2026-11-08
한글 글꼴: ~/.minerva/fonts/NanumGothic-*.ttf (Google Fonts OFL, 저장소 밖).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HIND, REAL, RAND = "#eb6834", "#2a78d6", "#a3a29c"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
FR = {"15": "15분", "30": "30분", "1h": "1시간", "4h": "4시간", "1d": "1일"}


def font() -> None:
    for p in sorted(Path("~/.minerva/fonts").expanduser().glob("NanumGothic-*.ttf")):
        font_manager.fontManager.addfont(str(p))
    plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams["axes.unicode_minus"] = False


def panel(ax, D, keys, metric, hind_metric, title):
    xs = range(len(keys))
    w = 0.26
    vals = []
    for i, k in enumerate(keys):
        r, h = D[k][metric]["pool"], D[k][hind_metric]["pool"]
        vals.append((h["hit"], r["hit"], r["rand"]))
        for j, (v, col) in enumerate(zip(vals[-1], (HIND, REAL, RAND))):
            x = i + (j - 1) * w
            ax.bar(x, 100 * v, w - 0.03, color=col, zorder=2)
            ax.text(x, 100 * v + 1, f"{100 * v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(50, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{FR[k.split('|')[0]]}\n오른쪽 {k.split('|')[1][1:]}봉" for k in keys], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("적중 %", color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    D = json.loads((out / "diag.json").read_text())
    font()
    fig, axs = plt.subplots(3, 1, figsize=(12, 13))
    low = [f"{f}|R{r}" for f in ("15", "30") for r in (1, 2, 3, 5)]
    high = [f"{f}|R{r}" for f in ("1h", "4h", "1d") for r in (2, 3, 5)]
    panel(axs[0], D, low, "hit_1h", "hind_1h",
          "① 15분·30분 다이버전스: 1시간 뒤 방향이 맞은 비율")  # fmt: skip
    panel(axs[1], D, high, "hit_24h", "hind_24h",
          "② 1시간·4시간·1일 다이버전스: 24시간 뒤 방향이 맞은 비율")  # fmt: skip
    panel(axs[2], D, low + high, "tb_0.01", "tbh_0.01",
          "③ 진입 후 신호 방향 +1%가 손절(피벗 극값 이탈)보다 먼저 온 비율")  # fmt: skip
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (HIND, REAL, RAND)]
    fig.legend(hs, ["사후(차트 라벨 자리 = 피벗 봉, 실제로는 불가능)", "현실(신호가 확정된 봉 마감에 진입)",
                    "무작위(같은 시기·같은 방향·같은 손절 거리)"],  # fmt: skip
               loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.text(0.01, 0.005, "BTCUSDT 선물 2020-03 ~ 2025-10, 세 등급(강·중·약) 전부. 점선 = 50%(동전 던지기).",
             fontsize=8, color=MUTED)  # fmt: skip
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(out / "chart_hit.png", dpi=120)
    print(out / "chart_hit.png")


if __name__ == "__main__":
    main()

"""One diachronic affect figure per author from the LLM annotations, genre-coloured.

EXPLORATORY: the LLM instrument only matches the majority baseline, so read the curve as a shape,
not a measurement. Output: .../sentiment_llm/<author>_llm_valence.png
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, stylo_experiment, author_meta, year_jitter

GENRE_COLOR = {"poetry": "#1f77b4", "article": "#d62728", "letter": "#7f7f7f"}


def plot(author):
    d = stylo_experiment(author, "sentiment_llm")
    df = pd.read_csv(d / f"{author}_llm_annotations.csv", encoding="utf-8-sig")
    df["v"] = pd.to_numeric(df["valence_score"], errors="coerce")
    df = df.dropna(subset=["v"])
    meta = author_meta(author)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for g, sub in df.groupby("genre"):
        c = GENRE_COLOR.get(g, "gray")
        xj = year_jitter(sub["year"])
        ax.scatter(xj, sub["v"], s=38, alpha=.7, color=c, label=f"{g} (n={len(sub)})", zorder=3)
        ym = sub.groupby("year")["v"].mean()                            # yearly-mean trend
        if len(ym) > 1:
            ax.plot(ym.index, ym.values, color=c, lw=1.4, alpha=.9, zorder=4)
    ax.axhline(0, color="k", lw=.6)
    if author == "alterman":
        ax.axvspan(1939, 1945, color="gray", alpha=.12, zorder=0)       # WWII
    for yr, lab in meta.get("events", []):
        ax.axvline(yr, color="k", ls="--", lw=.7, alpha=.45)
        ax.text(yr, 1.02, lab, fontsize=7, rotation=90, va="bottom", alpha=.6)
    ax.set_ylim(-1.05, 1.15)
    ax.set_ylabel("emotional charge  (LLM valence: +positive / -negative)")
    ax.set_xlabel("year")
    ax.set_title(f"{meta['display']} - emotional charge over the career "
                 f"(DictaLM-2.0; EXPLORATORY, instrument ≈baseline)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = d / f"{author}_llm_valence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[{author}] {len(df)} scored texts, years {int(df.year.min())}-{int(df.year.max())} -> {out}")


if __name__ == "__main__":
    for a in ["ahad_haam", "alterman"]:
        plot(a)

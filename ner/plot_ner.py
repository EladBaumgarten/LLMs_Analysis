"""Plot the top named entities per period."""
import sys
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
# Do NOT apply python-bidi get_display - matplotlib renders logical-order Hebrew correctly, and
# reshaping it double-reverses into gibberish.
matplotlib.rcParams["font.family"] = "Arial"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from authors_paths import for_author, current_author_name, author_meta
AUTHOR = current_author_name()
OUT = for_author(AUTHOR)["ner_out"]

LABELS = ["PER", "GPE", "ORG"]
PALETTE = ["#4C72B0", "#C44E52", "#55A868", "#8172B3"]


def main():
    df = pd.read_csv(OUT / "ner_entities.csv", encoding="utf-8-sig")
    periods = sorted(df["period"].unique())
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(periods)}
    n_essays = {p: df[df.period == p]["id"].nunique() for p in periods}

    fig, axes = plt.subplots(len(LABELS), len(periods), figsize=(6 * len(periods), 11), squeeze=False)
    for r, lab in enumerate(LABELS):
        for c, per in enumerate(periods):
            sub = df[(df.label == lab) & (df.period == per)]
            top = Counter(sub["phrase"]).most_common(8)[::-1]
            ax = axes[r][c]
            ax.barh([nm for nm, _ in top], [v for _, v in top], color=colors[per])
            ax.set_title(f"{lab} - {per} (n={n_essays[per]} essays)", fontsize=10)
            ax.tick_params(labelsize=9)

    fig.suptitle(f"{author_meta(AUTHOR)['display']} - top named entities by period", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "ner_by_period.png"
    fig.savefig(out, dpi=150)
    print("saved:", out)


if __name__ == "__main__":
    main()

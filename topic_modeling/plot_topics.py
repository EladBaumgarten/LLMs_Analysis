"""Plot the topic mixture over time from doc_topics_over_time.csv.

Topic labels are English on purpose: they are interpretations of the LDA topics, and matplotlib
does not render Hebrew RTL cleanly.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from authors_paths import for_author, current_author_name, author_meta
AUTHOR = current_author_name()
DERIVED = for_author(AUTHOR)["topic_out"]

TOPIC_LABELS = {
    "ahad_haam": {
        0: "Judaism / philosophy", 1: "Settlement / colonies", 2: "Tradition / community",
        3: "Modern education", 4: "Political Zionism", 5: "Zionist institutions",
    },
}.get(AUTHOR, {})


def main():
    df = pd.read_csv(DERIVED / "doc_topics_over_time.csv", encoding="utf-8-sig")
    topic_cols = [c for c in df.columns if c.startswith("t") and c[1:].isdigit()]

    df["bin"] = (df["year"] // 5 * 5).astype(int)
    trend = df.groupby("bin")[topic_cols].mean()
    ymax = trend.values.max()

    plt.figure(figsize=(11, 6))
    for c in topic_cols:
        plt.plot(trend.index, trend[c], marker="o", label=TOPIC_LABELS.get(int(c[1:]), c))
    for yr, lab in author_meta(AUTHOR)["events"]:
        plt.axvline(yr, ls="--", color="grey", lw=1)
        plt.text(yr + 0.4, ymax * 0.97, lab, color="grey", fontsize=8, rotation=90, va="top")
    plt.xlabel("Year (5-year bins)")
    plt.ylabel("Mean topic weight")
    plt.title(f"{author_meta(AUTHOR)['display']} - topic mixture over time (LDA, k={len(topic_cols)})")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = DERIVED / "topics_over_time.png"
    plt.savefig(out, dpi=150)
    print("saved:", out)
    print("\n5-year-bin trend:")
    print(trend.round(3).to_string())


if __name__ == "__main__":
    main()

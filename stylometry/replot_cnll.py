"""Re-render the CNLL figure locally from the Colab run's word table.

The Colab PNG is unusable for two reasons: some runtimes lack bidi shaping and render the Hebrew
tick labels reversed (a rendering problem - the CSV is correct either way), and `ftn2`, an
editorial footnote marker that survived cleaning, is perfectly period-correlated and tops the
per-word mean as a typesetting artifact rather than language.
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import stylo_experiment, author_meta

N_LATE, N_EARLY = 15, 10
ARTIFACT = re.compile(r"^ftn\d*$", re.I)          # footnote markers, see the docstring
HEB = re.compile(r"[֐-׿]")


def display(label: str, flip: bool) -> str:
    """Labels are stored logically; flip only if the renderer lacks bidi."""
    return label[::-1] if (flip and HEB.search(label)) else label


def plot(author: str, flip: bool):
    d = stylo_experiment(author, "perplexity_alm")
    cn = pd.read_csv(d / f"{author}_cnll_words.csv", encoding="utf-8-sig",
                     header=0, names=["word", "cnll"]).dropna()
    dropped = cn[cn.word.astype(str).str.match(ARTIFACT)]
    cn = cn[~cn.word.astype(str).str.match(ARTIFACT)]
    if len(dropped):
        print(f"  [{author}] dropped {len(dropped)} artifact token(s): "
              f"{', '.join(dropped.word.astype(str))}")

    n_chunks = len(pd.read_csv(d / f"{author}_oof_predictions.csv", encoding="utf-8-sig"))
    top = pd.concat([cn.nlargest(N_LATE, "cnll"), cn.nsmallest(N_EARLY, "cnll")])
    top = top.sort_values("cnll")                 # most early-characteristic at the bottom

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(top)), top.cnll.values,
            color=["#d62728" if v > 0 else "#1f77b4" for v in top.cnll])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([display(str(w), flip) for w in top.word])
    ax.axvline(0, color="k", lw=.6)
    ax.set_title(f"CNLL - words most period-distinct in {author_meta(author)['display']}'s texts\n"
                 f"all {n_chunks} out-of-fold chunks across 5 folds\n"
                 f"red = late-characteristic · blue = early-characteristic", fontsize=10)
    ax.set_xlabel("mean CNLL  (NLL_early − NLL_late), summed per word")
    fig.tight_layout()
    out = d / f"{author}_cnll_top.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [{author}] {len(cn)} words · {n_chunks} chunks -> {out}")


if __name__ == "__main__":
    flip = "--flip" in sys.argv          # pass --flip if the renderer shows Hebrew reversed
    print(f"bidi flip: {flip}")
    for a in ("ahad_haam", "alterman"):
        plot(a, flip)

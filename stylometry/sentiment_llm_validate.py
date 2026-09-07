"""Affect - local validation and plotting, run after the Colab annotator.

Place <author>_llm_annotations.csv in each author's sentiment_llm/ dir first. Scores the LLM
against the hand labels (sign accuracy, label match, Spearman) versus recomputed HeBERT and
majority baselines, and plots the diachronic valence curve per genre.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, stylo_experiment, author_meta, year_jitter
from sentiment_calibrate import parse_ground_truth, human_valence, to_class


def load_ann(author):
    d = stylo_experiment(author, "sentiment_llm")
    f = d / f"{author}_llm_annotations.csv"
    if not f.exists():
        print(f"  [skip] {f} not found - paste the Colab output here first.")
        return None
    df = pd.read_csv(f, encoding="utf-8-sig")
    df["valence_score"] = pd.to_numeric(df["valence_score"], errors="coerce")
    df["lbl_num"] = df["valence_label"].map(lambda s: human_valence(str(s))[0])
    return df


def validate_alterman(df):
    gt = parse_ground_truth()
    m = gt.merge(df, on="id", how="inner")
    print(f"\n=== Alterman LLM vs {len(m)} human labels ===")
    llm = m["valence_score"].fillna(m["lbl_num"]).values      # prefer the score, fall back to label
    y = m["human_val"].values
    sign_acc = np.mean([to_class(a) == to_class(b) for a, b in zip(llm, y)])
    label_match = np.mean(m["valence_label"].map(lambda s: human_valence(str(s))[1]) == m["human_label"])
    rho = spearmanr(llm, y).correlation
    print(f"  LLM valence: sign-acc {sign_acc:.2f} | 4-way label-match {label_match:.2f} | Spearman {rho:+.2f}")
    # Recompute the baselines from the same labels the LLM is scored on - hardcoding them let them
    # drift from every upstream fix.
    _maj = float(pd.Series([to_class(v) for v in y]).value_counts(normalize=True).max())
    _hb = np.nan
    _sc = stylo_experiment("alterman", "sentiment_diachronic") / "text_scores.csv"
    if _sc.exists():
        _s = pd.read_csv(_sc, encoding="utf-8-sig")
        _j = m[["id", "human_val"]].merge(_s[["id", "cls_valence"]], on="id", how="inner")
        if len(_j):
            _hb = float(np.mean([to_class(a) == to_class(b)
                                 for a, b in zip(_j.cls_valence, _j.human_val)]))
    print(f"  baselines (recomputed on these {len(m)} labels): "
          f"HeBERT sign-acc {_hb:.2f} | majority {_maj:.2f}")
    print(f"  parse_ok: {m['parse_ok'].mean():.0%}")
    cm = pd.crosstab(pd.Series([to_class(v) for v in y], name="human"),
                     pd.Series([to_class(v) for v in llm], name="LLM"))
    print("  confusion (human x LLM):\n   ", cm.to_string().replace("\n", "\n    "))
    _ref = _hb if _hb == _hb else _maj            # NaN-safe fallback to the majority baseline
    verdict = "BEATS" if sign_acc > _ref else "does NOT beat"
    print(f"  >>> GATE: LLM {verdict} the HeBERT baseline ({_ref:.2f}).")


def plot_both(anns):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = {"ahad_haam": {"article": "#2ca02c"},
              "alterman": {"poetry": "#1f77b4", "article": "#d62728"}}
    for author, df in anns.items():
        if df is None:
            continue
        for g, sub in df.dropna(subset=["valence_score"]).groupby("genre"):
            yj = year_jitter(sub["year"])
            ax.scatter(yj, sub["valence_score"], s=34, alpha=.8,
                       c=colors.get(author, {}).get(g, "gray"),
                       label=f"{author_meta(author)['display']} · {g}")
    ax.axhline(0, color="k", lw=.6)
    ax.axvspan(1939, 1945, color="gray", alpha=.10, zorder=0)
    ax.set_ylabel("LLM valence (+pos / -neg)"); ax.set_xlabel("Year")
    ax.set_title("Context-aware LLM affect valence over time (DictaLM-2.0), genre-aware")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = stylo_experiment("alterman", "sentiment_llm", create=True) / "llm_valence_diachronic.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\n  saved plot: {out}")


if __name__ == "__main__":
    anns = {a: load_ann(a) for a in ["ahad_haam", "alterman"]}
    if anns["alterman"] is not None:
        validate_alterman(anns["alterman"])
    plot_both(anns)

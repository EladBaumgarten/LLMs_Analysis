"""Affect - supervised calibration of the frozen sentiment tools against the manual labels.

Too few labels to fine-tune HeBERT, so a ridge model maps the frozen outputs [HeBERT valence,
intensity, NRC valence, NRC arousal] to the human valence label, validated leave-one-out. Tests
whether combining the two contradictory tools beats either one raw.

Ceiling to keep in mind: calibration fixes systematic bias, not missing signal - if both inputs
encode lexical loadedness rather than stance, no linear combination recovers stance.
"""
import sys, re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, stylo_experiment, author_meta, year_jitter

RES = Path(__file__).resolve().parent / "results"
GT_FILES = [RES / "sentiment_validation_poetry.md", RES / "sentiment_validation_prose.md"]
FEATURES = ["cls_valence", "cls_intensity", "lex_valence", "lex_intensity"]


def human_valence(cell):
    """Free-text Hebrew label -> (valence, class).

    Reads the label's HEAD first: a keyword scan alone scores "נייטרלי, נוטה לחיובי" as a full pole.
    """
    c = str(cell).lower()                     # labels code-switch between Hebrew and English
    head = re.split(r"[-,;—]", str(cell).strip())[0]      # the primary class is the head
    for scope in (head, cell):
        h = str(scope).lower()
        if "מעורב" in scope or "mixed" in h: return 0.0, "mixed"
        if "נייטרלי" in scope or "neutral" in h: return 0.0, "neutral"
        if "שלילי" in scope or "negativ" in h: return -1.0, "negative"
        if "חיובי" in scope or "positiv" in h: return 1.0, "positive"
    return None, None


def parse_ground_truth():
    rows, dropped = [], []
    for f in GT_FILES:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = [c.strip() for c in line.split("|")]
            ids = [c for c in cells if c.isdigit()]
            if not ids:
                continue
            tid = int(ids[0])
            val, lab = None, None
            for c in cells:                       # first cell carrying a valence keyword
                v, l = human_valence(c)
                if v is not None:
                    val, lab = v, l
                    break
            if lab:
                rows.append({"id": tid, "human_val": val, "human_label": lab})
            else:
                dropped.append((tid, " | ".join(c for c in cells if c)[:90]))
    # Report unparseable labels: dropping them silently made the n look complete.
    if dropped:
        print(f"  [ground truth] WARNING {len(dropped)} hand-labelled rows carry no canonical "
              f"valence keyword and are EXCLUDED from the n:")
        for tid, preview in dropped:
            print(f"      id {tid}: {preview}")
    return pd.DataFrame(rows).drop_duplicates("id")


def to_class(v, thr=0.33):
    return "pos" if v > thr else ("neg" if v < -thr else "neu")


def main(author="alterman"):
    P = for_author(author)
    scores = pd.read_csv(stylo_experiment(author, "sentiment_diachronic") / "text_scores.csv",
                         encoding="utf-8-sig")
    gt = parse_ground_truth()
    df = gt.merge(scores, on="id", how="inner")
    print(f"ground-truth texts matched: {len(df)}/{len(gt)}  "
          f"| label mix: {df.human_label.value_counts().to_dict()}")

    X, y = df[FEATURES].values, df["human_val"].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    loo = LeaveOneOut()
    pred = np.zeros(len(df))
    for tr, te in loo.split(X):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])

    def sign_acc(p): return float(np.mean([to_class(a) == to_class(b) for a, b in zip(p, y)]))
    baseline_majority = float((pd.Series([to_class(v) for v in y]).value_counts(normalize=True)).max())

    print("\n=== LOO valence recovery (target = human valence) ===")
    print(f"  calibrated (HeBERT+NRC): sign-acc {sign_acc(pred):.2f} | "
          f"Spearman {spearmanr(pred, y).correlation:+.2f} | MAE {np.mean(np.abs(pred-y)):.2f}")
    print(f"  raw HeBERT valence     : sign-acc {sign_acc(df.cls_valence.values):.2f} | "
          f"Spearman {spearmanr(df.cls_valence, y).correlation:+.2f}")
    # NRC valence spans ~[-0.01,+0.13], so at native scale nothing crosses the +/-0.33 band.
    # Threshold on its own median instead, so its neutral band is comparable to HeBERT's.
    nrc_centred = (df.lex_valence - df.lex_valence.median()).values
    nrc_scaled = nrc_centred / (np.abs(nrc_centred).max() or 1.0)     # -> [-1,1]
    print(f"  raw NRC valence        : sign-acc {sign_acc(nrc_scaled):.2f} | "
          f"Spearman {spearmanr(df.lex_valence, y).correlation:+.2f}")
    print(f"  majority-class baseline: sign-acc {baseline_majority:.2f}")

    cm = pd.crosstab(pd.Series([to_class(v) for v in y], name="human"),
                     pd.Series([to_class(v) for v in pred], name="calibrated_LOO"))
    print("\n  confusion (human rows x calibrated cols):")
    print("   ", cm.to_string().replace("\n", "\n    "))

    # Refit on every label, then score all texts for the diachronic plot.
    model.fit(X, y)
    scores["cal_valence"] = model.predict(scores[FEATURES].values)
    outdir = stylo_experiment(author, "sentiment_diachronic", create=True)
    scores.to_csv(outdir / "text_scores_calibrated.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = {"poetry": "#1f77b4", "article": "#d62728"}
    for g, sub in scores.groupby("genre_clean"):
        yj = year_jitter(sub["year"])
        ax.scatter(yj, sub["cal_valence"], c=colors.get(g, "gray"), s=42, label=g, zorder=3)
    ax.scatter(df["year"], df["human_val"], facecolors="none", edgecolors="k", s=90,
               label="human ground truth", zorder=4)
    ax.axhline(0, color="k", lw=.6)
    ax.axvspan(1939, 1945, color="gray", alpha=.12, zorder=0)
    for yr, lab in author_meta(author).get("events", []):
        ax.axvline(yr, color="k", ls="--", lw=.7, alpha=.5)
    ax.set_ylabel("Calibrated valence (+pos / -neg)")
    ax.set_xlabel("Year")
    ax.set_title(f"{author_meta(author)['display']} - CALIBRATED affect valence "
                 f"(HeBERT+NRC → human labels, genre-aware)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "sentiment_calibrated.png", dpi=140)
    plt.close(fig)
    print(f"\n  saved: {outdir/'text_scores_calibrated.csv'} , {outdir/'sentiment_calibrated.png'}")


if __name__ == "__main__":
    main()

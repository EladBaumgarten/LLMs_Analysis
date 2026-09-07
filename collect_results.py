"""Assemble the published evidence set from the pipeline outputs.

Everything under `figures/` and `docs/results/` is a copy or an aggregate of something the
pipeline already produced under `authors/<name>/*/output/`, which is not published because it is
large and regenerable. Running this after a re-run keeps the published set from lagging the
numbers it is supposed to evidence.

Large row dumps are published as top-N aggregates instead: raw NER rows and the full CNLL word
table together are ~1.5 MB and add nothing a reader can check by eye.

Run: python collect_results.py
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from authors_paths import for_author, stylo_experiment

AUTHORS = ["ahad_haam", "alterman"]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "docs" / "results"

NER_TOP = 15
SHAP_TOP = 40
CNLL_TOP = 25

# (destination stem, source path relative to the author's output tree)
FIGURE_MAP = [
    ("ner_by_period", "ner_out/ner_by_period.png"),
    ("topics_over_time", "topic_out/topics_over_time.png"),
    ("boosted_period_shap_top", "stylo/boosted_period/shap_top.png"),
    ("perplexity_alm_cnll_top", "stylo/perplexity_alm/{author}_cnll_top.png"),
    ("rolling_delta_centroid", "stylo/rolling/rolling_delta_centroid.png"),
    ("rolling_delta_first", "stylo/rolling/rolling_delta_first.png"),
    ("sentiment_llm_valence", "stylo/sentiment_llm/{author}_llm_valence.png"),
    ("sentiment_llm_valence_diachronic", "stylo/sentiment_llm/llm_valence_diachronic.png"),
    ("sentiment_diachronic", "stylo/sentiment_diachronic/sentiment_diachronic.png"),
    ("sentiment_diachronic_calibrated", "stylo/sentiment_diachronic/sentiment_calibrated.png"),
]

log = []


def resolve(author, rel):
    P = for_author(author)
    rel = rel.format(author=author)
    head, _, rest = rel.partition("/")
    base = {"ner_out": P["ner_out"], "topic_out": P["topic_out"], "stylo": P["stylo_out"]}[head]
    return base / rest


def copy_figures(author):
    out = FIGURES / author
    out.mkdir(parents=True, exist_ok=True)
    for stem, rel in FIGURE_MAP:
        src = resolve(author, rel)
        if src.exists():
            shutil.copyfile(src, out / f"{stem}.png")
            log.append(f"figure   {author}/{stem}.png")


def _write(df, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")
    log.append(f"table    docs/results/{name}  ({len(df)} rows)")


def topics(author):
    src = for_author(author)["topic_out"] / "topics.csv"
    if src.exists():
        _write(pd.read_csv(src, encoding="utf-8-sig"), f"topics-{author}.csv")


def topic_weights(author):
    src = for_author(author)["topic_out"] / "doc_topics_over_time.csv"
    if not src.exists():
        return
    d = pd.read_csv(src, encoding="utf-8-sig")
    tcols = [c for c in d.columns if c.startswith("t") and c[1:].isdigit()]
    d["decade"] = (d["year"] // 10 * 10).astype(int)
    g = d.groupby("decade")[tcols].mean().round(3).reset_index()
    g.insert(1, "n_articles", d.groupby("decade").size().values)
    _write(g, f"topic-weights-by-decade-{author}.csv")


def ner_top(author):
    src = for_author(author)["ner_out"] / "ner_entities.csv"
    if not src.exists():
        return
    d = pd.read_csv(src, encoding="utf-8-sig")
    rows = []
    for period in sorted(d["period"].unique()):
        for label in ("PER", "GPE", "LOC", "ORG"):
            sub = d[(d.period == period) & (d.label == label)]
            for rank, (phrase, n) in enumerate(Counter(sub["phrase"]).most_common(NER_TOP), 1):
                rows.append({"author": author, "period": period, "label": label,
                             "rank": rank, "entity": phrase, "mentions": n})
    _write(pd.DataFrame(rows), f"ner-top-entities-{author}.csv")


def cv_scores(author, experiment, stem):
    d = stylo_experiment(author, experiment)
    frames = []
    for f in sorted(d.glob("cv_scores_*.csv")):
        tag = f.stem[len("cv_scores_"):]
        variant = "drop2" if tag.endswith("_drop2") else "fix"
        df = pd.read_csv(f, encoding="utf-8-sig")
        df.insert(0, "scheme", tag[:-len("_drop2")] if variant == "drop2" else tag)
        df.insert(1, "variant", variant)
        frames.append(df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out.insert(0, "author", author)
        _write(out, f"{stem}-{author}.csv")


def shap_top(author):
    f = stylo_experiment(author, "boosted_period") / "shap_top_features.csv"
    if f.exists():
        d = pd.read_csv(f, encoding="utf-8-sig").head(SHAP_TOP)
        _write(d.round(4), f"shap-top-features-{author}.csv")


def copy_tables(author):
    for src, name in (
        (stylo_experiment(author, "direction") / "feature_direction_year_terciles.csv",
         f"feature-direction-{author}.csv"),
        (stylo_experiment(author, "period_scatter") / "period_scatter.csv",
         f"period-scatter-{author}.csv"),
        (stylo_experiment(author, "rolling") / "rolling_delta_first.csv",
         f"rolling-delta-first-{author}.csv"),
        (stylo_experiment(author, "rolling") / "rolling_delta_centroid.csv",
         f"rolling-delta-centroid-{author}.csv"),
    ):
        if src.exists():
            _write(pd.read_csv(src, encoding="utf-8-sig"), name)


def cnll_top(author):
    f = stylo_experiment(author, "perplexity_alm") / f"{author}_cnll_words.csv"
    if not f.exists():
        return
    d = pd.read_csv(f, encoding="utf-8-sig", header=0, names=["word", "cnll"]).dropna()
    d = d[~d.word.astype(str).str.match(r"^ftn\d*$", case=False)]      # editorial footnote markers
    top = pd.concat([d.nlargest(CNLL_TOP, "cnll").assign(characteristic_of="late"),
                     d.nsmallest(CNLL_TOP, "cnll").assign(characteristic_of="early")])
    top.insert(0, "author", author)
    _write(top.round(4), f"cnll-top-words-{author}.csv")


if __name__ == "__main__":
    for author in AUTHORS:
        copy_figures(author)
        topics(author)
        topic_weights(author)
        ner_top(author)
        cv_scores(author, "classical", "cv-scores-classical")
        cv_scores(author, "boosted_period", "cv-scores-boosted")
        shap_top(author)
        copy_tables(author)
        cnll_top(author)
    print("\n".join(log))
    print(f"\n{len(log)} artifacts published "
          f"({sum(1 for l in log if l.startswith('figure'))} figures, "
          f"{sum(1 for l in log if l.startswith('table'))} tables)")

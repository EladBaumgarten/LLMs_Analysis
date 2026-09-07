"""Track an author's topic mixture over time.

Scores each article against the trained LDA model and aggregates by the author's anchor event and
by decade, to separate gradual drift from a sharp shift. doc_lemmas is imported from LDA_prep so
the preprocessing matches what the model was trained on.
"""
import sys
from pathlib import Path
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel

from LDA_prep import doc_lemmas          # import-safe: LDA_prep guards its main()

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from authors_paths import for_author, current_author_name, author_meta
AUTHOR = current_author_name()
P = for_author(AUTHOR)
META = P["metadata"]
DERIVED = P["topic_out"]
ANCHOR = author_meta(AUTHOR)["anchor_year"]


def main():
    lda = LdaModel.load(str(DERIVED / "lda_model"))
    dictionary = Dictionary.load(str(DERIVED / "lda_dictionary"))
    K = lda.num_topics
    topic_cols = [f"t{t}" for t in range(K)]

    idx = pd.read_csv(META / "index.csv", encoding="utf-8-sig")
    essay_ids = idx.loc[idx["genre_clean"] == "article", "id"]

    rows = []
    for i in essay_ids:
        bow = dictionary.doc2bow(doc_lemmas(i))
        dist = dict(lda.get_document_topics(bow, minimum_probability=0))
        rows.append({"id": i, **{f"t{t}": round(dist.get(t, 0.0), 4) for t in range(K)}})
    df = pd.DataFrame(rows).merge(idx[["id", "title", "original_date"]], on="id")

    df = df[df["original_date"].notna()].copy()
    df["year"] = df["original_date"].astype(int)
    print("dated essays on timeline:", len(df), "| span:", df["year"].min(), "-", df["year"].max())

    df["period"] = df["year"].apply(lambda y: f"pre-{ANCHOR}" if y < ANCHOR else f"post-{ANCHOR}")
    print("\n=== mean topic weight by period ===")
    print(df.groupby("period")[topic_cols].mean().round(3).to_string())
    print("\nperiod counts:", df["period"].value_counts().to_dict())

    df["decade"] = (df["year"] // 10 * 10).astype(int)
    print("\n=== mean topic weight by decade ===")
    print(df.groupby("decade")[topic_cols].mean().round(3).to_string())

    df.to_csv(DERIVED / "doc_topics_over_time.csv", index=False, encoding="utf-8-sig")
    print("\nsaved: doc_topics_over_time.csv")


if __name__ == "__main__":
    main()

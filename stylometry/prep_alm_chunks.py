"""Local prep for Perplexity ALM - a per-author chunk-TEXT table.

Reuses boosted_period's chunker and labels each chunk early/late by the cutoff that splits total
chunks most evenly while keeping every article whole, so no article spans both periods (which would
leak across the train/test boundary downstream).

Run locally, then upload the CSVs to Colab. Output: .../perplexity_alm/<author>_chunks_text.csv
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, current_author_name, stylo_experiment
from boosted_period import load_tokens, chunk


def all_dated_articles(P):
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    df = idx[(idx["genre_clean"] == "article") & (idx["original_date"].notna())].copy()
    df["year"] = df["original_date"].astype(float).astype(int)
    return df[["id", "year"]].sort_values("year", kind="stable").reset_index(drop=True)


def chunk_text(ch):
    """Detokenize a chunk back to plain text for LM fine-tuning."""
    return " ".join(t["text"] for s in ch for t in s)


def balanced_cutoff(df):
    """Year boundary minimizing |early - late| chunks. early = year < cut. -> (cut, n_early, n_late)."""
    per_year = df.groupby("year").size().sort_index()
    total = per_year.sum()
    best = None
    for cut in per_year.index[1:]:
        early = int(per_year[per_year.index < cut].sum())
        late = total - early
        diff = abs(early - late)
        if best is None or diff < best[0]:
            best = (diff, int(cut), early, late)
    return best[1], best[2], best[3]


def build(author, target=400, min_tokens=200):
    P = for_author(author, create=True)
    arts = all_dated_articles(P)
    rows = []
    for _, r in arts.iterrows():
        for ci, ch in enumerate(chunk(load_tokens(P, r["id"]), target, min_tokens)):
            rows.append({"id": r["id"], "year": int(r["year"]), "chunk": ci,
                         "n_tok": sum(len(s) for s in ch), "text": chunk_text(ch)})
    X = pd.DataFrame(rows)
    cut, n_early, n_late = balanced_cutoff(X)
    X["period"] = X["year"].apply(lambda y: "early" if y < cut else "late")

    # Stamp the author into the data as well as the filename, so a file swap is detectable.
    X.insert(0, "author", author)
    out = stylo_experiment(author, "perplexity_alm", create=True) / f"{author}_chunks_text.csv"
    X.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"[{author}] chunks={len(X)} articles={X['id'].nunique()} "
          f"tok/chunk={X['n_tok'].mean():.0f}±{X['n_tok'].std():.0f}")
    print(f"  cutoff year={cut}  ->  early={n_early} ({X['year'][X.period=='early'].min()}-{cut-1})  "
          f"late={n_late} ({cut}-{X['year'][X.period=='late'].max()})")
    print(f"  saved: {out}")
    print(f"  sample early: {X[X.period=='early']['text'].iloc[0][:90]}...")
    print(f"  sample late : {X[X.period=='late']['text'].iloc[0][:90]}...")
    return X


if __name__ == "__main__":
    build(current_author_name())

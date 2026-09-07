"""Local prep for the LLM annotator - a per-author CSV of texts to annotate on Colab.

Text is the Dicta sentences joined; the date overrides and genre filter are imported from
sentiment_diachronic so both tracks share one definition.
Output: .../sentiment_llm/<author>_llm_input.csv
"""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, stylo_experiment
from sentiment_diachronic import DATE_OVERRIDE, GENRES

AUTHORS = ["ahad_haam", "alterman"]
MAX_CHARS = 4000                      # caps long essays, still enough context for stance


def clean_text(P, tid):
    doc = json.loads((P["dicta_out"] / f"{tid}.json").read_text(encoding="utf-8"))
    return " ".join(s["text"] for s in doc)


def build(author):
    P = for_author(author)
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    idx = idx[(idx["include_in_stylometry"] == True) & (idx["genre_clean"].isin(GENRES))].copy()
    idx["year"] = pd.to_numeric(idx["original_date"], errors="coerce")
    for tid, yr in DATE_OVERRIDE.items():
        idx.loc[idx["id"] == tid, "year"] = yr
    idx = idx[idx["year"].notna()].copy()
    idx["year"] = idx["year"].astype(int)
    rows = []
    for _, r in idx.sort_values("year").iterrows():
        f = P["dicta_out"] / f"{r['id']}.json"
        if not f.exists():
            continue
        t = clean_text(P, r["id"])
        rows.append({"author": author, "id": r["id"], "genre": r["genre_clean"],
                     "year": r["year"], "n_chars": len(t), "text": t[:MAX_CHARS]})
    df = pd.DataFrame(rows)
    out = stylo_experiment(author, "sentiment_llm", create=True) / f"{author}_llm_input.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[{author}] {len(df)} texts | {df.genre.value_counts().to_dict()} | "
          f"truncated(>{MAX_CHARS}c): {(df.n_chars > MAX_CHARS).sum()}")
    print(f"   -> {out}")


if __name__ == "__main__":
    for a in AUTHORS:
        build(a)

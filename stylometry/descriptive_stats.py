"""Descriptive corpus statistics, articles only, per author.

Career-wide by default; `--by-period` splits the DATED articles into the three year-boundary
periods instead, which is the time segmentation the frequency lists and the report both use.

Reuses features.py's POS sets and TTR definition so these numbers stay consistent with the
classifier pipeline. Re-run whenever the corpus or the Dicta output changes.
Output: docs/descriptive_stats.csv, docs/descriptive_stats_by_period.csv
"""
import sys, json, statistics as st
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, author_meta
from features import FUNC_POS, CONTENT_POS          # identical definitions, deliberately
from binning import stage_labels

AUTHORS = ["ahad_haam", "alterman"]
STAGE_ORDER = ["early", "middle", "late"]


def article_ids(P):
    """Articles this author actually WROTE.

    genre_clean alone is not enough: it admits texts he only translated, which would count as his
    own writing.
    """
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    keep = idx["genre_clean"] == "article"
    if "include_in_stylometry" in idx.columns:
        keep &= idx["include_in_stylometry"].astype(str).str.lower().isin(["true", "1", "1.0"])
    if "role" in idx.columns:
        keep &= idx["role"].fillna("author").ne("translator")
    return idx[keep]["id"].tolist()


def aggregate(P, ids):
    """The descriptive row for one set of article ids."""
    n_tok = n_punct = n_func = n_content = 0
    sent_lens, ttrs, lemmas = [], [], set()
    n_docs = 0
    for tid in ids:
        f = P["dicta_out"] / f"{tid}.json"
        if not f.exists():
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        n_docs += 1
        toks = [t for s in doc for t in s["tokens"]]
        n_tok += len(toks)
        sent_lens += [len(s["tokens"]) for s in doc]
        for t in toks:
            pos = t["morph"]["pos"]
            if pos == "PUNCT": n_punct += 1
            elif pos in FUNC_POS: n_func += 1
            elif pos in CONTENT_POS: n_content += 1
            lex = t.get("lex") or ""
            if lex and lex != "[BLANK]" and pos in CONTENT_POS:
                lemmas.add(lex)
        # Same [BLANK] exclusion as unique_lemmas above - two conventions in one row would skew TTR.
        content = [t["lex"] for t in toks
                   if t["morph"]["pos"] in CONTENT_POS
                   and (t.get("lex") or "") and t["lex"] != "[BLANK]"]
        if content:
            ttrs.append(len(set(content)) / len(content))
    return {
        "articles": n_docs,
        "tokens": n_tok,
        "mean_sent_len": round(st.mean(sent_lens), 1),
        "median_sent_len": int(st.median(sent_lens)),
        "punct_%": round(100 * n_punct / n_tok, 1),
        "funcword_%": round(100 * n_func / n_tok, 1),
        "func_content_ratio": round(n_func / n_content, 3),
        "unique_lemmas": len(lemmas),
        "mean_text_ttr": round(st.mean(ttrs), 3),
    }


def stats_for(author):
    P = for_author(author)
    return {"author": author_meta(author)["display"], **aggregate(P, article_ids(P))}


def stats_by_period(author, scheme="year_terciles"):
    """One row per period. Dated articles only - an undated article has no period."""
    P = for_author(author)
    st = stage_labels(author, scheme=scheme)
    rows = []
    for stage in STAGE_ORDER:
        g = st[st.stage == stage]
        if not len(g):
            continue
        rows.append({"author": author_meta(author)["display"], "period": stage,
                     "years": f"{g.year.min()}-{g.year.max()}",
                     **aggregate(P, g["id"].tolist())})
    return rows


if __name__ == "__main__":
    DOCS = Path(__file__).resolve().parent.parent / "docs"
    DOCS.mkdir(exist_ok=True)
    if "--by-period" in sys.argv:
        df = pd.DataFrame([r for a in AUTHORS for r in stats_by_period(a)])
        out = DOCS / "descriptive_stats_by_period.csv"
    else:
        df = pd.DataFrame([stats_for(a) for a in AUTHORS])
        out = DOCS / "descriptive_stats.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print("saved:", out)

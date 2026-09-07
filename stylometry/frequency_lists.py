"""Most-frequent items per time bucket - words, forms, structures, per author.

Five families per bucket, each a rate per 1,000 tokens: content lemmas, function-word lemmas,
morphological features, POS bigrams and dependency relations. Buckets are either the three
year-boundary periods or decades.

POS bigrams exclude any pair containing PUNCT - punctuation adjacency is already reported by the
dependency and punctuation families, and it would otherwise fill the whole top of the list.

Output: docs/S7-freq-<author>-{period,decade}.csv
"""
import sys, json
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from authors_paths import for_author, current_author_name
from features import FUNC_POS, CONTENT_POS
from binning import stage_labels
from descriptive_stats import article_ids

TOP_N = 30
KINDS = ["content", "func", "feats", "pos_bi", "deps"]
STAGE_ORDER = ["early", "middle", "late"]


def count_bucket(P, ids):
    """Counters for the five families over `ids`, plus the bucket's token total."""
    c = {k: Counter() for k in KINDS}
    n_tok = 0
    for tid in ids:
        f = P["dicta_out"] / f"{tid}.json"
        if not f.exists():
            continue
        for s in json.loads(f.read_text(encoding="utf-8")):
            toks = s["tokens"]
            n_tok += len(toks)
            for t in toks:
                pos = t["morph"]["pos"]
                lex = (t.get("lex") or "").strip()
                if lex and lex != "[BLANK]":
                    if pos in CONTENT_POS:
                        c["content"][lex] += 1
                    elif pos in FUNC_POS:
                        c["func"][lex] += 1
                for k, v in (t["morph"].get("feats") or {}).items():
                    c["feats"][f"{k}={v}"] += 1
                c["deps"][t.get("syntax", {}).get("dep_func") or "root"] += 1
            for a, b in zip(toks, toks[1:]):
                pa, pb = a["morph"]["pos"], b["morph"]["pos"]
                if pa != "PUNCT" and pb != "PUNCT":
                    c["pos_bi"][f"{pa} {pb}"] += 1
    return c, n_tok


def buckets_for(author, kind):
    """[(label, ids)] in chronological order. kind = 'period' | 'decade'."""
    P = for_author(author)
    if kind == "period":
        st = stage_labels(author, scheme="year_terciles")
        out = []
        for stage in STAGE_ORDER:
            g = st[st.stage == stage]
            if len(g):
                out.append((f"{stage} ({g.year.min()}-{g.year.max()})", g["id"].tolist()))
        return out
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    idx = idx[idx["id"].isin(article_ids(P)) & idx["original_date"].notna()].copy()
    idx["decade"] = (idx["original_date"].astype(float).astype(int) // 10 * 10).astype(int)
    return [(f"{d}-", g["id"].tolist()) for d, g in idx.groupby("decade")]


def build(author, kind):
    P = for_author(author)
    label = "לפי תקופה" if kind == "period" else "לפי עשור"
    rows = []
    for bucket, ids in buckets_for(author, kind):
        c, n_tok = count_bucket(P, ids)
        for family in KINDS:
            for item, n in c[family].most_common(TOP_N):
                rows.append({"author": author, "bucket_kind": label, "bucket": bucket,
                             "kind": family, "item": item,
                             "per_1000": round(1000 * n / n_tok, 2)})
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent.parent / "docs" / f"S7-freq-{author}-{kind}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[{author} | {kind}] {len(df)} rows, {df.bucket.nunique()} buckets -> {out.name}")
    return df


if __name__ == "__main__":
    authors = [current_author_name()] if "--author" in sys.argv else ["ahad_haam", "alterman"]
    for a in authors:
        for k in ("period", "decade"):
            build(a, k)

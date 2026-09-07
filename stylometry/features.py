import sys, json, statistics as st
from collections import Counter
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "corpus"))
from text_clean import clean_text, clean_paragraphs   # the same cleaning Dicta applied
from authors_paths import for_author, current_author_name, current_scheme, stylo_experiment
from binning import stage_labels

FUNC_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
PUNCT = {"comma": ",", "semicolon": ";", "colon": ":", "exclam": "!", "question": "?",
         "quote": '"', "geresh": "׳", "gershayim": "״", "maqaf": "־", "hyphen": "-"}

def load_doc(P, tid):
    d = json.loads((P["dicta_out"] / f"{tid}.json").read_text(encoding="utf-8"))
    raw = (P["corpus_stripped"] / f"{tid}.txt").read_text(encoding="utf-8")
    # Two cleanings, deliberately: punctuation() divides by Dicta's token count and so needs the
    # Dicta-matched STRING, while paragraph stats need the PARAGRAPH LIST (clean_text drops the
    # blank lines that separate them).
    return d, clean_text(raw), clean_paragraphs(raw)

def punctuation(raw, n_tok):
    return {f"pn_{name}": 1000 * raw.count(ch) / n_tok if n_tok else 0.0
            for name, ch in PUNCT.items()}

def phraseology(d, paras):
    toks = [t for s in d for t in s["tokens"]]
    words = [t["token"] for t in toks if t["morph"]["pos"] != "PUNCT"]
    sent_lens = [len(s["tokens"]) for s in d]
    para_lens = [len(p.split()) for p in paras if p.strip()]
    # [BLANK] is the tagger's lemmatization failure, not a lemma - it would count as vocabulary.
    content = [t["lex"] for t in toks
               if t["morph"]["pos"] in CONTENT_POS and t["lex"] and t["lex"] != "[BLANK]"]
    return {
        "ttr": len(set(content)) / len(content) if content else 0.0,
        "mean_word_len": st.mean([len(w) for w in words]) if words else 0.0,
        "mean_sent_len": st.mean(sent_lens) if sent_lens else 0.0,
        "std_sent_len": st.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0,
        "mean_para_len": st.mean(para_lens) if para_lens else 0.0,
        "std_para_len": st.pstdev(para_lens) if len(para_lens) > 1 else 0.0,
        "doc_len": len(toks),
    }

def lexical(d, vocab):
    toks = [t for s in d for t in s["tokens"]]
    n = len(toks)
    c = Counter(t["lex"] for t in toks
                if t["morph"]["pos"] in FUNC_POS and t["lex"] and t["lex"] != "[BLANK]")
    return {f"fw_{w}": (1000 * c.get(w, 0) / n if n else 0.0) for w in vocab}

def funcword_vocab(P, ids, k=50):
    c = Counter()
    for tid in ids:
        d, _, _ = load_doc(P, tid)
        # Same [BLANK] guard: it tracks archaic orthography, so it would leak era, not style.
        c.update(t["lex"] for s in d for t in s["tokens"]
                 if t["morph"]["pos"] in FUNC_POS and t["lex"] and t["lex"] != "[BLANK]")
    return [w for w, _ in c.most_common(k)]

def build(author, k=50):
    P = for_author(author, create=True)
    stages = stage_labels(author, scheme=current_scheme())
    vocab = funcword_vocab(P, stages["id"], k)
    rows = []
    for _, r in stages.iterrows():
        d, raw, paras = load_doc(P, r["id"])
        n_tok = sum(len(s["tokens"]) for s in d)
        feat = {"id": r["id"], "stage": str(r["stage"]), "year": int(r["year"])}
        feat.update(phraseology(d, paras))
        feat.update(punctuation(raw, n_tok))
        feat.update(lexical(d, vocab))
        rows.append(feat)
    X = pd.DataFrame(rows)
    X.to_csv(P["stylo_out"] / "feature_matrix.csv", index=False, encoding="utf-8-sig")
    return X

def build_articles_matrix(author, k=50):
    """Feature matrix over every dated article, unbinned - so experiments that bin by year cannot
    inherit a --scheme filter from the shared matrix."""
    P = for_author(author, create=True)
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    df = idx[(idx["genre_clean"] == "article") & (idx["original_date"].notna())].copy()
    df["year"] = df["original_date"].astype(float).astype(int)
    df = df.sort_values("year", kind="stable")
    vocab = funcword_vocab(P, df["id"], k)
    rows = []
    for _, r in df.iterrows():
        d, raw, paras = load_doc(P, r["id"])
        n_tok = sum(len(s["tokens"]) for s in d)
        feat = {"id": r["id"], "year": int(r["year"])}
        feat.update(phraseology(d, paras))
        feat.update(punctuation(raw, n_tok))
        feat.update(lexical(d, vocab))
        rows.append(feat)
    X = pd.DataFrame(rows)
    out = stylo_experiment(author, "rolling", create=True) / "feature_matrix_articles.csv"
    X.to_csv(out, index=False, encoding="utf-8-sig")
    return X


if __name__ == "__main__":
    X = build(current_author_name())
    print("matrix:", X.shape)
    print(X.groupby("stage", observed=True)[["mean_sent_len", "pn_comma", "ttr"]].mean())

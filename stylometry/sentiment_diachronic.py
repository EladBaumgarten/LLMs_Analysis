"""Diachronic affect axis - two instruments, two axes, genre-aware.

HeBERT (contextual, per sentence) and the NRC-VAD lexicon (bag-of-words, per lemma) each score
valence and intensity, so the two can be cross-checked. Poetry and articles are plotted as
SEPARATE series: the level gap between them is genre, not time.

Runs local on CPU - Alterman is by_permission and his text must not leave the machine.
Output: authors/<name>/stylometry/output/sentiment_diachronic/
"""
import sys, json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Hebrew crashes on Windows cp1252
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, current_author_name, stylo_experiment, year_jitter

# Local safetensors: this env's transformers refuses the hub .bin under torch<2.6.
_LOCAL = Path(__file__).resolve().parent / "models" / "heBERT_sentiment"
MODEL = str(_LOCAL) if (_LOCAL / "model.safetensors").exists() else "avichr/heBERT_sentiment_analysis"
LEXICON = Path(__file__).resolve().parent / "lexicons" / "Hebrew-NRC-VAD-Lexicon.txt"
GENRES = {"poetry", "article"}
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data" / "corpus"))
from text_clean import strip_nikud                  # the lexicon keys need the same cleaner
from features import CONTENT_POS                    # one definition of "content word" across tracks

# Scoped to this experiment only; index.csv is left unchanged.
DATE_OVERRIDE = {
    13089: 1945, 13090: 1946, 13091: 1945,   # Holocaust poems parked at 1968 by the BY-only rule
    13099: 1957,                             # a 1957 love poem misfiled into the WWII bin
    13816: 1969, 14672: 1969, 14674: 1969,   # tagged 1967/1970, published ~1969
}
# 13097 is dated 1941 but is a biblical poem, not a war poem - left in its bin, flagged here.

PREWAR = lambda y: y <= 1938
WARTIME = lambda y: 1939 <= y <= 1947


def load_index(P, author=None):
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    idx = idx[(idx["include_in_stylometry"] == True) & (idx["genre_clean"].isin(GENRES))].copy()
    idx["year"] = pd.to_numeric(idx["original_date"], errors="coerce")
    # Alterman ids only: an id collision would otherwise rewrite another author's years.
    if author in (None, "alterman"):
        for tid, yr in DATE_OVERRIDE.items():
            idx.loc[idx["id"] == tid, "year"] = yr
    idx = idx[idx["year"].notna()].copy()
    idx["year"] = idx["year"].astype(int)
    return idx[["id", "year", "genre_clean"]].sort_values("year").reset_index(drop=True)


def load_sentences(P, tid):
    """Return list of {text, lemmas} per Dicta sentence. lemmas = content-word lookup keys."""
    doc = json.loads((P["dicta_out"] / f"{tid}.json").read_text(encoding="utf-8"))
    out = []
    for s in doc:
        lemmas = []
        for t in s["tokens"]:
            if t["morph"]["pos"] not in CONTENT_POS:
                continue
            lex = (t.get("lex") or "").strip()
            if not lex or lex == "[BLANK]":
                seg = t.get("seg") or []
                lex = (seg[-1] if seg else t["token"]).strip()
            if lex:
                lemmas.append(lex)
        out.append({"text": s["text"], "lemmas": lemmas})
    return out


def load_lexicon():
    """Hebrew word -> (valence, arousal) in [0,1], averaged over the English sources mapping to it."""
    df = pd.read_csv(LEXICON, sep="\t", encoding="utf-8")
    df = df.dropna(subset=["Hebrew Word"])
    g = df.groupby("Hebrew Word")[["Valence", "Arousal"]].mean()
    # Keys ship with nikud, Dicta lemmas have none - normalising lifts coverage 33% -> 62%.
    out = {}
    for w, (v, a) in g.iterrows():
        k = strip_nikud(str(w).strip())
        if not k:
            continue
        out.setdefault(k, []).append((v, a))
    return {k: (float(np.mean([p[0] for p in ps])), float(np.mean([p[1] for p in ps])))
            for k, ps in out.items()}


def score_lexicon(sentences, lex):
    """Text-level lexicon valence (centred to [-.5,.5]) + arousal, plus coverage."""
    vs, as_, n_tok, n_hit = [], [], 0, 0
    for s in sentences:
        for w in s["lemmas"]:
            n_tok += 1
            if w in lex:
                v, a = lex[w]
                vs.append(v); as_.append(a); n_hit += 1
    if not vs:
        return dict(lex_valence=np.nan, lex_intensity=np.nan, lex_coverage=0.0, lex_hits=0)
    return dict(lex_valence=float(np.mean(vs)) - 0.5,
                lex_intensity=float(np.mean(as_)),
                lex_coverage=n_hit / max(n_tok, 1), lex_hits=n_hit)


def build_classifier():
    from transformers import pipeline
    import torch
    dev = 0 if torch.cuda.is_available() else -1
    clf = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL,
                   top_k=None, truncation=True, max_length=256, device=dev)
    print(f"  HeBERT loaded on {'GPU' if dev == 0 else 'CPU'}")
    return clf


def _probs(scores):
    """(p_neu, p_pos, p_neg) - labels may be words or LABEL_0/1/2 (order = neu, pos, neg)."""
    p = {"neutral": None, "positive": None, "negative": None}
    order = ["neutral", "positive", "negative"]
    for d in scores:
        lab = d["label"].lower()
        if "pos" in lab: p["positive"] = d["score"]
        elif "neg" in lab: p["negative"] = d["score"]
        elif "neu" in lab or "nat" in lab: p["neutral"] = d["score"]
        elif lab.startswith("label_"):
            p[order[int(lab.split("_")[1])]] = d["score"]
    return p["neutral"] or 0.0, p["positive"] or 0.0, p["negative"] or 0.0


def score_classifier(clf, sentences, batch=32):
    texts = [s["text"] for s in sentences if s["text"].strip()]
    if not texts:
        return dict(cls_valence=np.nan, cls_intensity=np.nan, n_sent=0), []
    results = clf(texts, batch_size=batch)
    per = []
    for txt, sc in zip(texts, results):
        neu, pos, neg = _probs(sc)
        per.append(dict(text=txt, p_neu=neu, p_pos=pos, p_neg=neg,
                        valence=pos - neg, intensity=1 - neu))
    val = float(np.mean([r["valence"] for r in per]))
    inten = float(np.mean([r["intensity"] for r in per]))
    return dict(cls_valence=val, cls_intensity=inten, n_sent=len(per)), per


def plot(df, outdir, meta):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"poetry": "#1f77b4", "article": "#d62728"}
    for g, sub in df.groupby("genre_clean"):
        c = colors.get(g, "gray")
        yj = year_jitter(sub["year"])
        a1.scatter(yj, sub["cls_valence"], c=c, s=45, label=f"{g} · HeBERT", zorder=3)
        a1.scatter(yj, sub["lex_valence"], facecolors="none", edgecolors=c, s=45,
                   marker="s", alpha=.55, label=f"{g} · NRC-VAD", zorder=2)
        a2.scatter(yj, sub["cls_intensity"], c=c, s=45, label=f"{g} · HeBERT", zorder=3)
        a2.scatter(yj, sub["lex_intensity"], facecolors="none", edgecolors=c, s=45,
                   marker="s", alpha=.55, label=f"{g} · NRC-VAD", zorder=2)
    a1.axhline(0, color="k", lw=.6)
    a1.axvspan(1939, 1945, color="gray", alpha=.12, zorder=0)
    a2.axvspan(1939, 1945, color="gray", alpha=.12, zorder=0)
    for yr, lab in meta.get("events", []):
        for ax in (a1, a2):
            ax.axvline(yr, color="k", ls="--", lw=.7, alpha=.5)
        a1.text(yr, a1.get_ylim()[1], f" {lab}", fontsize=8, va="top", rotation=90, alpha=.6)
    a1.set_ylabel("Valence  (+ positive / - negative)")
    a2.set_ylabel("Intensity  (emotional charge)")
    a2.set_xlabel("Year")
    a1.set_title(f"{meta['display']} - diachronic affect (genre-aware; HeBERT vs NRC-VAD). "
                 f"Poetry↔prose gap = genre, not time.")
    a1.legend(fontsize=7, ncol=2); a2.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = outdir / "sentiment_diachronic.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  saved figure: {p}")


def run(author):
    from authors_paths import author_meta
    P = for_author(author, create=True)
    outdir = stylo_experiment(author, "sentiment_diachronic", create=True)
    idx = load_index(P, author)
    print(f"[{author}] texts={len(idx)}  "
          f"{idx['genre_clean'].value_counts().to_dict()}  years {idx.year.min()}-{idx.year.max()}")

    lex = load_lexicon()
    print(f"  NRC-VAD Hebrew entries: {len(lex)}")
    clf = build_classifier()

    rows, sent_rows = [], []
    for i, r in idx.iterrows():
        sents = load_sentences(P, r["id"])
        cls, per = score_classifier(clf, sents)
        lx = score_lexicon(sents, lex)
        rows.append({**r.to_dict(), **cls, **lx})
        for k, pr in enumerate(per):
            sent_rows.append({"id": r["id"], "year": r["year"], "genre": r["genre_clean"],
                              "sent_idx": k, **pr})
        if (i + 1) % 25 == 0:
            print(f"    scored {i + 1}/{len(idx)}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "text_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sent_rows).to_csv(outdir / "sentence_scores.csv", index=False, encoding="utf-8-sig")
    print(f"  coverage (lexicon hits/content-tokens): mean {df['lex_coverage'].mean():.1%}")

    S = pd.DataFrame(sent_rows)
    print("\n  --- classifier sanity (HeBERT) ---")
    for lab, d in [("most negative", S.nsmallest(3, "valence")), ("most positive", S.nlargest(3, "valence"))]:
        print(f"  {lab}:")
        for _, s in d.iterrows():
            print(f"    v={s['valence']:+.2f} | {s['text'][:70]}")

    po = df[df["genre_clean"] == "poetry"]
    zoom = []
    for lab, m in [("pre-war (<=1938)", PREWAR), ("wartime (1939-47)", WARTIME)]:
        b = po[po["year"].apply(m)]
        if len(b):
            zoom.append({"bin": lab, "n": len(b),
                         "cls_valence": b["cls_valence"].mean(), "cls_intensity": b["cls_intensity"].mean(),
                         "lex_valence": b["lex_valence"].mean(), "lex_intensity": b["lex_intensity"].mean()})
    zdf = pd.DataFrame(zoom)
    zdf.to_csv(outdir / "ww2_poetry_zoom.csv", index=False, encoding="utf-8-sig")
    print("\n  --- WWII within-poetry zoom ---")
    print(zdf.to_string(index=False))

    plot(df, outdir, author_meta(author))
    return df


if __name__ == "__main__":
    run(current_author_name("alterman"))

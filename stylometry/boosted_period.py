"""Boosted-Period - style-only period classification on short chunks + SHAP.

Przystalski et al. (2025) method, relabelled: chunk (~400 tokens) -> style features -> LightGBM
grouped by article -> SHAP. Content lemmas and NER are excluded so topic cannot leak into a
period classifier.
Output: authors/<name>/stylometry/output/boosted_period/
"""
import sys, json
import statistics as st
from collections import Counter
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
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, current_author_name, current_scheme, stylo_experiment
from binning import stage_labels

FUNC_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
METACOLS = ["id", "chunk", "year", "n_tok", "stage"]


def load_tokens(P, tid):
    """List of sentences; each a list of token dicts with style-relevant fields only."""
    doc = json.loads((P["dicta_out"] / f"{tid}.json").read_text(encoding="utf-8"))
    sents = []
    for s in doc:
        raw = s["tokens"]
        out = []
        for t in raw:
            hi = t.get("syntax", {}).get("dep_head_idx", -1)
            head_pos = raw[hi]["morph"]["pos"] if isinstance(hi, int) and 0 <= hi < len(raw) else "ROOT"
            out.append({
                "pos": t["morph"]["pos"],
                "lex": t.get("lex") or "",
                "feats": t["morph"].get("feats") or {},
                "dep": t.get("syntax", {}).get("dep_func") or "root",
                "head_pos": head_pos,
                "text": t["token"],
            })
        sents.append(out)
    return sents


def chunk(sents, target=400, min_tokens=200):
    """Pack whole sentences until >= target tokens, so sentence length stays a usable feature."""
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        cur.append(s)
        cur_len += len(s)
        if cur_len >= target:
            chunks.append(cur)
            cur, cur_len = [], 0
    if cur and cur_len >= min_tokens:
        chunks.append(cur)
    return chunks


def features(ch, fw_vocab):
    """Style-only feature vector for one chunk (list of sentences). Rates per 1000 tokens."""
    toks = [t for s in ch for t in s]
    n = len(toks)
    f = {}
    fw = Counter(t["lex"] for t in toks if t["pos"] in FUNC_POS and t["lex"])
    for w in fw_vocab:
        f[f"fw_{w}"] = 1000 * fw.get(w, 0) / n
    for p, c in Counter(t["pos"] for t in toks).items():
        f[f"pos_{p}"] = 1000 * c / n
    pb = Counter((a["pos"], b["pos"]) for s in ch for a, b in zip(s, s[1:]))
    tot_pb = sum(pb.values()) or 1
    for (a, b), c in pb.items():
        f[f"posbi_{a}.{b}"] = 1000 * c / tot_pb
    for d, c in Counter(t["dep"] for t in toks).items():
        f[f"dep_{d}"] = 1000 * c / n
    for (d, h), c in Counter((t["dep"], t["head_pos"]) for t in toks).items():
        f[f"depbi_{d}.{h}"] = 1000 * c / n
    mr = Counter()
    for t in toks:
        for k, v in t["feats"].items():
            mr[f"{k}={v}"] += 1
    for kv, c in mr.items():
        f[f"mrf_{kv}"] = 1000 * c / n
    for mark, c in Counter(t["text"] for t in toks if t["pos"] == "PUNCT").items():
        f[f"pn_{mark}"] = 1000 * c / n
    sent_lens = [len(s) for s in ch]
    words = [t for t in toks if t["pos"] != "PUNCT"]
    # [BLANK] is the tagger's lemmatization failure, not a lemma - it would inflate phr_ttr.
    content = [t["lex"] for t in toks
               if t["pos"] in CONTENT_POS and t["lex"] and t["lex"] != "[BLANK]"]
    f["phr_mean_sent_len"] = st.mean(sent_lens) if sent_lens else 0.0
    f["phr_std_sent_len"] = st.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    f["phr_mean_word_len"] = st.mean([len(t["text"]) for t in words]) if words else 0.0
    f["phr_ttr"] = len(set(content)) / len(content) if content else 0.0
    return f


def build(author, target=400, min_tokens=200, k=50, cull=0.05, scheme=None):
    """Per-chunk style feature matrix over all dated articles, period-labelled."""
    P = for_author(author, create=True)
    scheme = scheme or current_scheme("terciles")
    stages = stage_labels(author, scheme=scheme, n=3)
    recs = []
    for _, r in stages.iterrows():
        for ci, ch in enumerate(chunk(load_tokens(P, r["id"]), target, min_tokens)):
            recs.append({"id": r["id"], "year": int(r["year"]), "stage": str(r["stage"]),
                         "chunk": ci, "ch": ch, "n_tok": sum(len(s) for s in ch)})
    # Same [BLANK] guard: it tracks archaic orthography, so it would leak era, not style.
    cnt = Counter(t["lex"] for rec in recs for s in rec["ch"] for t in s
                  if t["pos"] in FUNC_POS and t["lex"] and t["lex"] != "[BLANK]")
    fw_vocab = [w for w, _ in cnt.most_common(k)]
    rows = []
    for rec in recs:
        row = {"id": rec["id"], "chunk": rec["chunk"], "year": rec["year"],
               "n_tok": rec["n_tok"], "stage": rec["stage"]}
        row.update(features(rec["ch"], fw_vocab))
        rows.append(row)
    X = pd.DataFrame(rows).fillna(0.0)
    thresh = cull * len(X)
    keep = [c for c in X.columns if c not in METACOLS
            and (c.startswith("phr_") or (X[c] > 0).sum() >= thresh)]
    return X[METACOLS + keep]


def feature_groups(cols):
    feats = [c for c in cols if c not in METACOLS]
    return {
        "funcwords":   [c for c in feats if c.startswith("fw_")],
        "pos":         [c for c in feats if c.startswith("pos_") or c.startswith("posbi_")],
        "dependency":  [c for c in feats if c.startswith("dep_") or c.startswith("depbi_")],
        "morphology":  [c for c in feats if c.startswith("mrf_")],
        "punctuation": [c for c in feats if c.startswith("pn_")],
        "phraseology": [c for c in feats if c.startswith("phr_")],
        "combined":    feats,
    }


def _lgbm():
    return LGBMClassifier(n_estimators=200, num_leaves=15, max_depth=5, learning_rate=0.05,
                          subsample=0.8, subsample_freq=3, colsample_bytree=0.8,
                          class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)


def evaluate(author, X):
    """LightGBM per feature family, group-CV by article. numpy in: LightGBM rejects '=' in names."""
    groups = X["id"].to_numpy()
    y = X["stage"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for gname, cols in feature_groups(X.columns).items():
        Xv = X[cols].to_numpy()
        models = {"lgbm": _lgbm()}
        if gname == "combined":
            models["dummy"] = DummyClassifier(strategy="most_frequent")
        for mname, model in models.items():
            bal = cross_val_score(model, Xv, y, groups=groups, cv=cv, scoring="balanced_accuracy")
            f1 = cross_val_score(model, Xv, y, groups=groups, cv=cv, scoring="f1_macro")
            rows.append({"features": gname, "model": mname, "n_feats": len(cols),
                         "bal_acc": round(bal.mean(), 3), "bal_sd": round(bal.std(), 3),
                         "f1_macro": round(f1.mean(), 3)})
    return pd.DataFrame(rows)


def _per_class_shap(sv, n_classes):
    """Normalise shap_values() output (list | 2-D | 3-D) to a list of (n_rows, n_feats) arrays."""
    if isinstance(sv, list):
        return sv
    arr = np.asarray(sv)
    return [arr[:, :, k] for k in range(arr.shape[2])] if arr.ndim == 3 else [arr]


def shap_top(author, X, topn=15):
    """mean |SHAP| per feature per stage over HELD-OUT folds.

    In-sample SHAP ranks document identity, not period signal - hence the same grouping as evaluate().
    """
    cols = feature_groups(X.columns)["combined"]
    Xv, yv, groups = X[cols].to_numpy(), X["stage"].to_numpy(), X["id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    classes, acc, n = None, None, 0
    for tr, te in cv.split(Xv, yv, groups=groups):
        model = _lgbm().fit(Xv[tr], yv[tr])
        per = _per_class_shap(shap.TreeExplainer(model).shap_values(Xv[te]), len(model.classes_))
        m = np.stack([np.abs(p).mean(axis=0) for p in per])
        classes = model.classes_ if classes is None else classes
        acc = m * len(te) if acc is None else acc + m * len(te)    # weight folds by test size
        n += len(te)
    mean_abs = acc / n
    df = pd.DataFrame({str(cls): pd.Series(mean_abs[k], index=cols)
                       for k, cls in enumerate(classes)})
    df["overall"] = df.mean(axis=1)
    return df.sort_values("overall", ascending=False)


def plot_shap(author, sh, out, topn=15):
    top = sh.head(topn)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top["overall"], color="#1f77b4")
    ax.set_title(f"Boosted-Period - top style features by mean|SHAP| ({author})")
    ax.set_xlabel("mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out / "shap_top.png", dpi=130)
    plt.close(fig)


def run(author, target=400, scheme=None):
    P = for_author(author, create=True)
    scheme = scheme or current_scheme("terciles")
    X = build(author, target=target, scheme=scheme)
    out = stylo_experiment(author, "boosted_period", create=True)
    # scheme in the filename: stage labels are scheme-dependent, so one name would mislabel a re-run.
    X.to_csv(out / f"feature_matrix_chunks_{scheme}.csv", index=False, encoding="utf-8-sig")
    print(f"[{author} | scheme={scheme}] chunks={len(X)} articles={X['id'].nunique()} "
          f"tok/chunk={X['n_tok'].mean():.0f}±{X['n_tok'].std():.0f} "
          f"stages={ {k: int(v) for k, v in X['stage'].value_counts().items()} } "
          f"n_features={len([c for c in X.columns if c not in METACOLS])}")
    res = evaluate(author, X)
    res.to_csv(out / f"cv_scores_{scheme}.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    sh = shap_top(author, X)
    sh.round(4).to_csv(out / "shap_top_features.csv", encoding="utf-8-sig")
    plot_shap(author, sh, out)
    print("top style features (overall mean|SHAP|):", ", ".join(sh.head(8).index))
    return res


if __name__ == "__main__":
    author = current_author_name()
    if "--sweep" in sys.argv:
        print(f"=== {author}: chunk-size sensitivity (combined LGBM bal_acc) ===")
        for tgt in (300, 400, 600):
            X = build(author, target=tgt, min_tokens=tgt // 2)
            res = evaluate(author, X)
            c = res[(res.features == "combined") & (res.model == "lgbm")].iloc[0]
            print(f"  target={tgt}: bal_acc={c.bal_acc}±{c.bal_sd}  (chunks={len(X)})")
    else:
        run(author)

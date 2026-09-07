import sys
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, current_author_name, current_scheme, stylo_experiment

PARA_FEATS = ("mean_para_len", "std_para_len")


def feature_groups(X, drop_para=False):
    meta = {"id", "stage", "year"}
    features = [c for c in X.columns if c not in meta]

    # Sensitivity variant: whether Ben-Yehuda's line structure is trustworthy enough to keep the
    # paragraph features is a judgement, so the family can be reported both ways.
    if drop_para:
        features = [c for c in features if c not in PARA_FEATS]

    punctuation = [c for c in features if c.startswith("pn_")]
    lexical     = [c for c in features if c.startswith("fw_")]
    phraseology = [c for c in features
                   if not c.startswith("pn_") and not c.startswith("fw_")]

    return {
        "punctuation": punctuation,
        "lexical":     lexical,
        "phraseology": phraseology,
        "combined":    features,
    }

def evaluate(author):
    P = for_author(author)
    scheme = current_scheme()
    X = pd.read_csv(P["stylo_out"] / "feature_matrix.csv", encoding="utf-8-sig")

    # One matrix path serves every scheme, so verify the cached labels match --scheme before using it.
    from binning import stage_labels
    want = stage_labels(author, scheme=scheme)
    w_lbl = dict(zip(want["id"], want["stage"].astype(str)))
    g_lbl = dict(zip(X["id"], X["stage"].astype(str)))
    if set(g_lbl) != set(w_lbl) or any(g_lbl[i] != w_lbl[i] for i in g_lbl):
        raise SystemExit(
            f"\n  STALE MATRIX: feature_matrix.csv does not match scheme='{scheme}'.\n"
            f"    matrix : n={len(X)} stages={sorted(set(g_lbl.values()))} "
            f"years {X['year'].min()}-{X['year'].max()}\n"
            f"    scheme : n={len(want)} stages={sorted(set(w_lbl.values()))} "
            f"years {want['year'].min()}-{want['year'].max()}\n"
            f"  Rebuild first:  python stylometry/features.py --author {author} --scheme {scheme}\n")

    drop_para = "--drop-para" in sys.argv
    suffix = "_drop2" if drop_para else ""
    y = X["stage"]
    print(f"[{author} | scheme={scheme}] n={len(X)} | years {X['year'].min()}-{X['year'].max()} "
          f"| stages { {k: int(v) for k, v in X['stage'].value_counts().items()} }"
          f"{' | PHRASEOLOGY VARIANT: paragraph feats dropped' if drop_para else ''}")
    groups = feature_groups(X, drop_para=drop_para)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    models = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        "svm": make_pipeline(StandardScaler(),
                             LinearSVC(C=1.0, dual="auto", class_weight="balanced")),
    }
    rows = []
    for gname, cols in groups.items():
        for mname, model in models.items():
            bal = cross_val_score(model, X[cols], y, cv=cv, scoring="balanced_accuracy")
            f1 = cross_val_score(model, X[cols], y, cv=cv, scoring="f1_macro")
            acc = cross_val_score(model, X[cols], y, cv=cv, scoring="accuracy")
            rows.append({"features": gname, "model": mname, "n_feats": len(cols),
                         "bal_acc": round(bal.mean(), 3), "bal_sd": round(bal.std(), 3),
                         "f1_macro": round(f1.mean(), 3), "acc": round(acc.mean(), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(stylo_experiment(author, "classical", create=True) / f"cv_scores_{scheme}{suffix}.csv",
               index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    return res

if __name__ == "__main__":
    evaluate(current_author_name())

"""Direction of stylistic change - which features move, which way, and by how much.

Classifiers and SHAP answer how much a feature helps identify a period, never which way it moved.
Two measures: Cohen's d between the first and last stage (readable, but inherits --scheme), and
Spearman rho against the article's year (no binning at all, so no period label can be wrong).
Rank correlation because these are skewed rate features. With ~67 features tested at once the
p-values are BH-FDR corrected - quote q, not p.

Output: authors/<name>/stylometry/output/direction/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from authors_paths import for_author, current_author_name, current_scheme, stylo_experiment
from binning import stage_labels

# Feature names carry Hebrew (fw_כי) - cp1252 would kill the print after the CSV is written.
sys.stdout.reconfigure(encoding="utf-8")

META = {"id", "stage", "year"}


def family_of(col):
    return "punctuation" if col.startswith("pn_") else "lexical" if col.startswith("fw_") else "phraseology"


def cohens_d(a, b):
    """Signed standardized mean difference b - a, pooled sd. 0 when both groups are constant."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return 0.0 if pooled == 0 else (b.mean() - a.mean()) / pooled


def bh_fdr(p):
    """Benjamini-Hochberg q-values. Returns array aligned to the input order."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):     # largest p -> smallest
        prev = min(prev, p[i] * n / (n - rank + 1))
        q[i] = prev
    return q


def analyse(author):
    P = for_author(author)
    scheme = current_scheme()
    X = pd.read_csv(P["stylo_out"] / "feature_matrix.csv", encoding="utf-8-sig")

    # Same guard as classify.evaluate: one matrix path serves every scheme.
    want = stage_labels(author, scheme=scheme)
    w_lbl = dict(zip(want["id"], want["stage"].astype(str)))
    g_lbl = dict(zip(X["id"], X["stage"].astype(str)))
    if set(g_lbl) != set(w_lbl) or any(g_lbl[i] != w_lbl[i] for i in g_lbl):
        raise SystemExit(
            f"\n  STALE MATRIX: feature_matrix.csv does not match scheme='{scheme}'.\n"
            f"  Rebuild first:  python stylometry/features.py --author {author} --scheme {scheme}\n")

    feats = [c for c in X.columns if c not in META]
    stages = [s for s in ("early", "middle", "late") if s in set(X["stage"].astype(str))]
    by_stage = {s: X[X["stage"].astype(str) == s] for s in stages}
    first, last = stages[0], stages[-1]

    print(f"[{author} | scheme={scheme}] n={len(X)} articles | years {X['year'].min()}-{X['year'].max()}")
    for s in stages:
        g = by_stage[s]
        print(f"    {s:7s}: n={len(g):3d}  years {g['year'].min()}-{g['year'].max()}")

    rows = []
    for c in feats:
        rho, p = spearmanr(X["year"], X[c])
        row = {"feature": c, "family": family_of(c)}
        row.update({f"mean_{s}": round(by_stage[s][c].mean(), 4) for s in stages})
        row["delta_raw"] = round(by_stage[last][c].mean() - by_stage[first][c].mean(), 4)
        row["cohens_d"] = round(cohens_d(by_stage[first][c], by_stage[last][c]), 3)
        row["spearman_rho"] = round(float(rho), 3)
        row["p"] = float(p)
        rows.append(row)

    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p"].values)
    res["trend"] = np.where(res["q_bh"] >= 0.05, "n.s.",
                            np.where(res["spearman_rho"] > 0, "increases", "decreases"))
    res["p"] = res["p"].map(lambda v: float(f"{v:.3g}"))
    res["q_bh"] = res["q_bh"].map(lambda v: float(f"{v:.3g}"))
    res = res.reindex(res["spearman_rho"].abs().sort_values(ascending=False).index)

    out = stylo_experiment(author, "direction", create=True) / f"feature_direction_{scheme}.csv"
    res.to_csv(out, index=False, encoding="utf-8-sig")

    sig = res[res["trend"] != "n.s."]
    print(f"\n  {len(sig)}/{len(res)} features show a monotonic year trend at FDR q<.05 "
          f"({(sig['trend'] == 'increases').sum()} up, {(sig['trend'] == 'decreases').sum()} down)")
    print(f"  by family: { {k: int(v) for k, v in sig['family'].value_counts().items()} }")
    cols = ["feature", "family", f"mean_{first}", f"mean_{last}", "cohens_d", "spearman_rho", "q_bh", "trend"]
    print("\n  strongest monotonic movers:")
    print(sig.head(15)[cols].to_string(index=False))
    print(f"\n  -> {out}")
    return res


if __name__ == "__main__":
    analyse(current_author_name())

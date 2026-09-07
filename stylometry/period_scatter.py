"""Within-era vs between-era variance decomposition of the period fingerprint.

A low classifier score has two opposite causes that accuracy alone cannot tell apart: era
centroids sitting close together (a stable author), or chunks scattered within each era (an author
with no single era-style). Decomposing the variance separates them.

Runs on the boosted classifier's own chunk matrix. Distances are divided by sqrt(F) so they read
in Burrows's Delta units and compare across authors with different feature counts. The per-tercile
spread is the confound-breaker: a genuinely scattered author is scattered in ALL three eras, while
the time/genre confound scatters only the mixed early bin.

Output: authors/<name>/stylometry/output/period_scatter/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Hebrew crashes on Windows cp1252
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, stylo_experiment

METACOLS = ["id", "chunk", "year", "n_tok", "stage"]
STAGE_ORDER = ["early", "middle", "late"]

FAMILIES = {"funcwords": ("fw_",), "pos": ("pos_", "posbi_"),
            "dependency": ("dep_", "depbi_"), "morphology": ("mrf_",),
            "punctuation": ("pn_",), "phraseology": ("phr_",)}


def zscore(F: pd.DataFrame) -> np.ndarray:
    """Standardize each column corpus-wide. A constant column divides by 1, contributing 0 spread."""
    mu = F.mean(axis=0)
    sd = F.std(axis=0, ddof=0).replace(0.0, 1.0)
    return ((F - mu) / sd).to_numpy()


def _scatter(Z: np.ndarray, y: np.ndarray, stages: list[str], sil: bool = True) -> dict:
    """Between/within decomposition of a z-scored block. Distances are per-feature (÷sqrt(F))."""
    N, F = Z.shape
    cents = {g: Z[y == g].mean(axis=0) for g in stages}
    n_g = {g: int((y == g).sum()) for g in stages}
    ss_b = sum(n_g[g] * float(cents[g] @ cents[g]) for g in stages)
    ss_w = sum(float(((Z[y == g] - cents[g]) ** 2).sum()) for g in stages)
    ss_t = ss_b + ss_w
    within_rms = float(np.sqrt(ss_w / (N * F)))
    pair_d = {f"{a}-{b}": float(np.linalg.norm(cents[a] - cents[b]) / np.sqrt(F))
              for i, a in enumerate(stages) for b in stages[i + 1:]}
    between_mean = float(np.mean(list(pair_d.values())))
    return {"F": F, "eta2": ss_b / ss_t, "within_rms": within_rms,
            "between_mean": between_mean,
            "snr": between_mean / within_rms if within_rms else float("nan"),
            "silhouette": float(silhouette_score(Z, y, metric="euclidean")) if sil else None,
            "pair_d": pair_d,
            "per_tercile_rms": {g: float(np.sqrt(((Z[y == g] - cents[g]) ** 2).sum()
                                                 / (n_g[g] * F))) for g in stages}}


def decompose(author: str) -> dict:
    mpath = stylo_experiment(author, "boosted_period") / "feature_matrix_chunks.csv"
    if not mpath.exists():
        raise FileNotFoundError(f"{mpath} missing - run boosted_period.py for {author} first")
    X = pd.read_csv(mpath, encoding="utf-8-sig")
    feats = [c for c in X.columns if c not in METACOLS]
    y = X["stage"].to_numpy()
    stages = [s for s in STAGE_ORDER if s in set(y)]

    combined = _scatter(zscore(X[feats]), y, stages)
    families = {}
    for fam, prefixes in FAMILIES.items():
        cols = [c for c in feats if c.startswith(prefixes)]
        if cols:
            families[fam] = _scatter(zscore(X[cols]), y, stages, sil=False)

    return {"author": author, "N": len(X), "n_g": {g: int((y == g).sum()) for g in stages},
            **combined, "families": families}


def report(rows: list[dict]):
    for r in rows:
        out = stylo_experiment(r["author"], "period_scatter", create=True)
        flat = {"author": r["author"], "N": r["N"], "F": r["F"],
                "eta2": round(r["eta2"], 3), "silhouette": round(r["silhouette"], 3),
                "snr": round(r["snr"], 3), "within_rms": round(r["within_rms"], 3),
                "between_mean": round(r["between_mean"], 3),
                **{f"between_{k}": round(v, 3) for k, v in r["pair_d"].items()},
                **{f"within_{k}": round(v, 3) for k, v in r["per_tercile_rms"].items()},
                **{f"snr_{fam}": round(d["snr"], 3) for fam, d in r["families"].items()}}
        pd.DataFrame([flat]).to_csv(out / "period_scatter.csv", index=False, encoding="utf-8-sig")

    print("\n=== Period fingerprint: within-era vs between-era scatter ===")
    print(f"{'author':12} {'N':>4} {'F':>4} {'eta^2':>6} {'silhouette':>10} {'SNR':>6} "
          f"{'within':>7} {'between':>8}")
    for r in rows:
        print(f"{r['author']:12} {r['N']:>4} {r['F']:>4} {r['eta2']:>6.3f} "
              f"{r['silhouette']:>10.3f} {r['snr']:>6.3f} {r['within_rms']:>7.3f} "
              f"{r['between_mean']:>8.3f}")
    print("\nBetween-era centroid distances (per-feature RMS-z, ~Delta units):")
    for r in rows:
        print(f"  {r['author']:12} " + "  ".join(f"{k}={v:.3f}" for k, v in r["pair_d"].items()))
    print("\nPer-tercile within-era spread (confound check - is the EARLY bin the scattered one?):")
    for r in rows:
        print(f"  {r['author']:12} " + "  ".join(f"{k}={v:.3f}"
              for k, v in r["per_tercile_rms"].items()))

    fams = list(rows[0]["families"].keys())
    print("\nPer-family SNR (between-era ÷ within-era; higher = eras more separable in that channel):")
    print(f"{'family':12} " + " ".join(f"{r['author']:>10}" for r in rows))
    for fam in fams:
        print(f"{fam:12} " + " ".join(f"{r['families'][fam]['snr']:>10.3f}" for r in rows))
    print("\nPer-family within-era spread (is Alterman more scattered WITHIN eras than Ahad Ha'am?):")
    print(f"{'family':12} " + " ".join(f"{r['author']:>10}" for r in rows))
    for fam in fams:
        print(f"{fam:12} " + " ".join(f"{r['families'][fam]['within_rms']:>10.3f}" for r in rows))


if __name__ == "__main__":
    authors = (["ahad_haam", "alterman"] if "--author" not in sys.argv
               else [sys.argv[sys.argv.index("--author") + 1]])
    report([decompose(a) for a in authors])

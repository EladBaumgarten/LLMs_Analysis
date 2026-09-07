"""Rolling stylometry (Eder 2016), repurposed as self-distance over a career.

Instead of sliding a window over one text to catch an authorship takeover, slide it over the
author's year-ordered articles and measure each window's Burrows's Delta from a fixed reference.
A flat curve is a uniform author; a rising one is a drifting author.

Input is a scheme-independent matrix over ALL dated articles, so no --scheme filter can truncate
the timeline. Feature set is the S4 classifier's, so the two methods stay comparable.
Output: authors/<name>/stylometry/output/rolling/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, current_author_name, author_meta, stylo_experiment

META = {"id", "stage", "year"}


def zscore(F: pd.DataFrame) -> pd.DataFrame:
    """Delta step 1 - one common scale. A constant column divides by 1, contributing 0, not NaN."""
    mu = F.mean(axis=0)
    sd = F.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (F - mu) / sd


def rolling_windows(df: pd.DataFrame, window: int, step: int):
    """Chronological windows; a short trailing window is dropped so every point averages the same n."""
    n = len(df)
    for start in range(0, n - window + 1, step):
        yield start, df.iloc[start:start + window]


def delta_distance(window_vec: np.ndarray, reference_vec: np.ndarray) -> float:
    """Burrows's Delta - Manhattan distance between z-scored vectors, normalised by #features."""
    return float(np.mean(np.abs(window_vec - reference_vec)))


def rolling_delta(author: str, window: int = 15, step: int = 3,
                  reference: str = "first") -> pd.DataFrame:
    """Rolling Delta over dated articles. reference='first' = drift from youth, 'centroid' = from
    the career mean."""
    mpath = stylo_experiment(author, "rolling", create=True) / "feature_matrix_articles.csv"
    # Staleness guard: rebuild when the DATA or the feature CODE is newer than the cache.
    from features import build_articles_matrix
    import features as _feat, text_clean as _tc
    _P = for_author(author)
    _srcs = (list(_P["dicta_out"].glob("*.json")) + [_P["metadata"] / "index.csv"]
             + [Path(_feat.__file__), Path(_tc.__file__)])
    _newest = max((p.stat().st_mtime for p in _srcs if p.exists()), default=0.0)
    if not mpath.exists():
        build_articles_matrix(author)
    elif "--rebuild" in sys.argv or _newest > mpath.stat().st_mtime:
        why = "forced" if "--rebuild" in sys.argv else "inputs or feature code newer than cache"
        print(f"  [rolling] {why} -> rebuilding feature matrix")
        build_articles_matrix(author)
    X = pd.read_csv(mpath, encoding="utf-8-sig")
    X = X.sort_values("year", kind="stable").reset_index(drop=True)

    feats = [c for c in X.columns if c not in META]
    Z = zscore(X[feats])

    if len(X) < window:
        raise ValueError(f"{author}: only {len(X)} texts < window {window}")

    if reference == "first":
        reference_vec = Z.iloc[:window].mean(axis=0).to_numpy()
    elif reference == "centroid":
        reference_vec = Z.mean(axis=0).to_numpy()
    else:
        raise ValueError(f"unknown reference: {reference}")

    rows = []
    for start, win in rolling_windows(X, window, step):
        zwin = Z.iloc[win.index].to_numpy()
        window_vec = zwin.mean(axis=0)
        rows.append({
            "center_year": float(win["year"].median()),
            "start_year": int(win["year"].min()),
            "end_year": int(win["year"].max()),
            "n": len(win),
            "delta": delta_distance(window_vec, reference_vec),
        })

    res = pd.DataFrame(rows)
    out = stylo_experiment(author, "rolling", create=True)
    res.to_csv(out / f"rolling_delta_{reference}.csv", index=False, encoding="utf-8-sig")
    return res


def plot(author: str, res: pd.DataFrame, reference: str = "first") -> Path:
    """Delta vs. center_year, with dashed vlines at the author's anchor events."""
    meta = author_meta(author)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(res["center_year"], res["delta"], marker="o", ms=4, lw=1.6, color="#1f77b4")
    for yr, label in meta.get("events", []):
        if res["center_year"].min() <= yr <= res["center_year"].max():
            ax.axvline(yr, ls="--", lw=1, color="#888")
            ax.text(yr, ax.get_ylim()[1], f" {label}", rotation=90,
                    va="top", ha="left", fontsize=8, color="#555")
    ref_txt = "earliest window" if reference == "first" else "career mean"
    ax.set_title(f"Rolling stylometry - {meta['display']} "
                 f"(Delta from {ref_txt}; window={res['n'].iloc[0]} texts)")
    ax.set_xlabel("window center year")
    ax.set_ylabel("Burrows's Delta")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = stylo_experiment(author, "rolling") / f"rolling_delta_{reference}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def summarize(author: str, res: pd.DataFrame) -> dict:
    """Curve level, spread and slope.

    With reference="first", delta[0] is 0 by construction, so `range` scales with the window COUNT.
    Use `std` or the `*_excl_ref` values for cross-author claims, never `range`.
    """
    yrs = res["center_year"].to_numpy()
    d = res["delta"].to_numpy()
    slope = float(np.polyfit(yrs, d, 1)[0]) if len(res) > 1 else 0.0
    structural = bool(len(d) and abs(float(d[0])) < 1e-12)
    d_ex, y_ex = (d[1:], yrs[1:]) if structural and len(d) > 1 else (d, yrs)
    slope_ex = float(np.polyfit(y_ex, d_ex, 1)[0]) if len(d_ex) > 1 else 0.0
    return {"mean": round(float(d.mean()), 3), "std": round(float(d.std()), 3),
            "min": round(float(d.min()), 3), "max": round(float(d.max()), 3),
            "range": round(float(d.max() - d.min()), 3),
            "structural_zero": structural,
            "n_windows": int(len(d)),
            "max_excl_ref": round(float(d_ex.max()), 3),
            "std_excl_ref": round(float(d_ex.std()), 3),
            "slope_per_yr": round(slope, 4),
            "slope_excl_ref": round(slope_ex, 4)}


if __name__ == "__main__":
    reference = "centroid" if "--centroid" in sys.argv else "first"
    author = current_author_name()
    res = rolling_delta(author, reference=reference)
    print(f"author: {author} | reference: {reference} | windows: {len(res)}")
    print(res.to_string(index=False))
    png = plot(author, res, reference)
    print("summary:", summarize(author, res))
    print("saved:", png)

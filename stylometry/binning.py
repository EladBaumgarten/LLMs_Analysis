"""Stage binning - tag each dated article early/middle/late, per Gómez-Adorno et al. (2018)."""
import sys
from itertools import combinations
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authors_paths import for_author, current_author_name, current_scheme, author_meta


def year_boundary_bins(years, n=3):
    """Cut `years` into n bins on YEAR boundaries, as close to equal-count as possible.

    The `terciles` scheme forces exactly-equal counts, which means cutting inside a tied year and
    breaking the tie by ROW ORDER - 48.9% of Alterman's labels vs 15.6% of Ahad Ha'am's came from
    row position rather than date. Here a label is always a fact about the date; the cost is that
    bins are only approximately equal. Returns (labels, cut_years).
    """
    yrs = pd.Series(years).astype(int)
    uniq = sorted(yrs.unique())
    if len(uniq) < n:
        raise ValueError(f"cannot make {n} year-boundary bins from {len(uniq)} distinct years")
    target = len(yrs) / n
    best, best_cost = None, None
    for cuts in combinations(uniq[1:], n - 1):              # cut = first year of each later bin
        edges = [uniq[0]] + list(cuts) + [uniq[-1] + 1]
        sizes = [int(((yrs >= edges[i]) & (yrs < edges[i + 1])).sum()) for i in range(n)]
        if min(sizes) == 0:
            continue
        cost = sum((s - target) ** 2 for s in sizes)
        if best_cost is None or cost < best_cost:
            best, best_cost = cuts, cost
    if best is None:
        raise ValueError("no valid year-boundary split")
    labels = ["early", "middle", "late"][:n]
    edges = [uniq[0]] + list(best) + [uniq[-1] + 1]
    out = pd.Series(index=yrs.index, dtype=object)
    for i, lab in enumerate(labels):
        out[(yrs >= edges[i]) & (yrs < edges[i + 1])] = lab
    return out, list(best)


def stage_labels(author, scheme="split", n=3):
    """Dated articles, each tagged with a stage.
    'split'         -> 2 classes at the author's split_year;
    'terciles'      -> n equal-count stages (the Gómez-Adorno replication);
    'year_terciles' -> n stages cut on year boundaries, no row-order tie-breaking."""
    P = for_author(author)
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    df = idx[(idx["genre_clean"] == "article") & (idx["original_date"].notna())].copy()
    df["year"] = df["original_date"].astype(int)
    if scheme == "terciles":
        labels = ["early", "middle", "late"][:n]
        # rank() makes values unique so qcut can still yield equal-count bins across shared years.
        df["stage"] = pd.qcut(df["year"].rank(method="first"), q=n, labels=labels)
    elif scheme == "split":
        cut = author_meta(author)["split_year"]
        df["stage"] = df["year"].apply(lambda y: "early" if y < cut else "late")
    elif scheme == "year_terciles":
        df["stage"], cuts = year_boundary_bins(df["year"], n=n)
        df.attrs["cut_years"] = cuts
    elif scheme == "late_terciles":
        cut = author_meta(author)["split_year"]
        df = df[df["year"] >= cut].copy()
        labels = ["early", "middle", "late"][:n]
        df["stage"] = pd.qcut(df["year"].rank(method="first"), q=n, labels=labels)
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return df[["id", "year", "stage"]].reset_index(drop=True)


if __name__ == "__main__":
    author = current_author_name()
    d = stage_labels(author, scheme=current_scheme())
    print(f"author: {author} | dated articles: {len(d)}")
    print("stage counts + year span:")
    for stage, g in d.groupby("stage", observed=True):
        print(f"  {stage:7s}: n={len(g):3d}  years {g['year'].min()}-{g['year'].max()}")

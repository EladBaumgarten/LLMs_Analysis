"""Single source of truth for per-author I/O locations, so switching authors is one --author flag.

    authors/<name>/corpus/{stripped,nikud}/<id>.txt
    authors/<name>/metadata/index.csv (+ index.json)
    authors/<name>/{dicta,ner,topic_modeling,stylometry}/output/
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def current_author_name(default: str = "ahad_haam") -> str:
    """Author for this run: `--author X` on the CLI, else $AUTHOR, else the default."""
    if "--author" in sys.argv:
        return sys.argv[sys.argv.index("--author") + 1]
    return os.environ.get("AUTHOR", default)


AUTHOR_META = {
    # split_year = the 2-class cutoff; Alterman's 1960 lands in his criticism/prose gap.
    "ahad_haam": {"display": "Ahad Ha'am", "anchor_year": 1897, "split_year": 1897,
                  "events": [(1897, "1897 Congress"), (1914, "WWI")]},
    "alterman":  {"display": "Alterman", "anchor_year": 1948, "split_year": 1960,
                  "events": [(1948, "Independence"), (1967, "Six-Day War")]},
}


def author_meta(name: str) -> dict:
    return AUTHOR_META.get(name, {"display": name, "anchor_year": None,
                                  "split_year": None, "events": []})


def current_scheme(default: str = "split") -> str:
    """Binning scheme: `--scheme X`, else $SCHEME, else the default. See binning.stage_labels."""
    if "--scheme" in sys.argv:
        return sys.argv[sys.argv.index("--scheme") + 1]
    return os.environ.get("SCHEME", default)


def for_author(name: str, *, create: bool = False) -> dict[str, Path]:
    """Return the canonical I/O dirs for `name`. Set create=True to mkdir them."""
    base = ROOT / "authors" / name
    dirs = {
        "base": base,
        "corpus_stripped": base / "corpus" / "stripped",
        "corpus_nikud": base / "corpus" / "nikud",
        "metadata": base / "metadata",
        "dicta_out": base / "dicta" / "output",
        "ner_out": base / "ner" / "output",
        "topic_out": base / "topic_modeling" / "output",
        "stylo_out": base / "stylometry" / "output",
    }
    if create:
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
    return dirs


def year_jitter(years, width: float = 0.8):
    """Spread same-year points horizontally without inventing dates.

    Normalised by group size, so the spread stays inside +/-width/2 however many texts share a year
    - an un-normalised offset pushed Alterman's 1969 points past his death year.
    """
    g = years.groupby(years)
    n = g.transform("size")
    return years + (g.cumcount() - (n - 1) / 2) / (n - 1).clip(lower=1) * width


def stylo_experiment(name: str, experiment: str, *, create: bool = False) -> Path:
    """Per-experiment results subdir. feature_matrix.csv stays at the root: it is the shared input."""
    d = for_author(name)["stylo_out"] / experiment
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d

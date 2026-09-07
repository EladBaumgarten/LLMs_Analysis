"""Corpus builder - Ahad Ha'am (Ben-Yehuda author id 23), from the public-domain dump.

Reads data/pseudocatalogue.csv plus data/p23{,_Nikud}/, and writes the curated corpus and metadata
index under authors/ahad_haam/. Idempotent - safe to re-run.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")   # Hebrew crashes on Windows cp1252

ROOT = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(ROOT.parent.parent))
from authors_paths import for_author as _for_author       # noqa: E402
_P = _for_author("ahad_haam")
AH_META = _P["metadata"]
AH_STRIPPED = _P["corpus_stripped"]
AH_NIKUD = _P["corpus_nikud"]
PROJECT_DATA = ROOT.parent.parent / "data"
DATA = PROJECT_DATA
CATALOG = DATA / "pseudocatalogue.csv"
SRC_STRIPPED = DATA / "p23"
SRC_NIKUD = DATA / "p23_Nikud"

OUT_STRIPPED = AH_STRIPPED
OUT_NIKUD = AH_NIKUD
OUT_META = AH_META

AUTHOR = "אחד העם"
GENRE_PREFIX = "Translation missing: he."


def main() -> None:
    for d in (OUT_STRIPPED, OUT_NIKUD, OUT_META):
        d.mkdir(parents=True, exist_ok=True)

    cat = pd.read_csv(CATALOG, encoding="utf-8")
    ahad = cat[cat["path"].str.startswith("/p23/")].copy()
    ahad = ahad.sort_values("ID").reset_index(drop=True)

    ahad["genre_clean"] = ahad["genre"].str.replace(GENRE_PREFIX, "", regex=False)

    is_translator = (
        ahad["translators"].fillna("").str.contains(AUTHOR)
        & ~ahad["authors"].fillna("").str.contains(AUTHOR)
    )
    ahad["role"] = is_translator.map({True: "translator", False: "author"})

    ahad["include_in_stylometry"] = (ahad["role"] == "author") & (
        ahad["genre_clean"] != "reference"
    )

    ahad["original_date"] = ""
    ahad["publication_date"] = ""
    ahad["date_basis"] = "undated"
    ahad["ip_status"] = "public_domain"

    ahad["text_stripped_path"] = ahad["ID"].map(
        lambda i: f"authors/ahad_haam/corpus/stripped/{i}.txt"
    )
    ahad["text_nikud_path"] = ahad["ID"].map(lambda i: f"authors/ahad_haam/corpus/nikud/{i}.txt")

    missing = []
    for _id in ahad["ID"]:
        for src_dir, out_dir in ((SRC_STRIPPED, OUT_STRIPPED), (SRC_NIKUD, OUT_NIKUD)):
            src = src_dir / f"m{_id}.txt"
            dst = out_dir / f"{_id}.txt"
            if not src.exists():
                missing.append(str(src))
                continue
            shutil.copyfile(src, dst)

    cols = [
        "ID", "title", "authors", "translators", "role", "include_in_stylometry",
        "genre", "genre_clean", "original_language", "source_edition",
        "original_date", "publication_date", "date_basis", "ip_status",
        "text_stripped_path", "text_nikud_path", "path",
    ]
    index = ahad[cols].rename(columns={"ID": "id"})
    index.to_csv(OUT_META / "index.csv", index=False, encoding="utf-8-sig")
    index.to_json(
        OUT_META / "index.json", orient="records", force_ascii=False, indent=2
    )

    n_stripped = len(list(OUT_STRIPPED.glob("*.txt")))
    n_nikud = len(list(OUT_NIKUD.glob("*.txt")))
    empty = [p.name for p in OUT_STRIPPED.glob("*.txt") if p.stat().st_size == 0]

    print("=== S1 corpus build - Ahad Ha'am (p23) ===")
    print(f"catalog rows for /p23/      : {len(ahad)}")
    print(f"files written  (stripped)   : {n_stripped}")
    print(f"files written  (nikud)      : {n_nikud}")
    print(f"missing source files        : {len(missing)}  {missing if missing else ''}")
    print(f"empty output files          : {len(empty)}  {empty if empty else ''}")
    print()
    print("role split:")
    print(ahad["role"].value_counts().to_string())
    print()
    print("genre split (all 163):")
    print(ahad["genre_clean"].value_counts().to_string())
    print()
    print(f"included in stylometry      : {int(ahad['include_in_stylometry'].sum())}")
    print("excluded rows:")
    excl = ahad[~ahad["include_in_stylometry"]][["ID", "title", "role", "genre_clean"]]
    print(excl.to_string(index=False))
    print()
    print("index head:")
    print(index[["id", "title", "role", "genre_clean", "text_stripped_path"]].head(6).to_string(index=False))


if __name__ == "__main__":
    main()

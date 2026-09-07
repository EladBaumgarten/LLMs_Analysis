import sys
import re
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(ROOT.parent.parent))
from authors_paths import for_author as _for_author       # noqa: E402
_P = _for_author("ahad_haam")
AH_META = _P["metadata"]
AH_STRIPPED = _P["corpus_stripped"]
AH_NIKUD = _P["corpus_nikud"]
PROJECT_DATA = ROOT.parent.parent / "data"

idx = pd.read_csv(AH_META / "index.csv", encoding="utf-8-sig")


YEAR_RE = r'תר[א-ת]?["״][א-ת]'

def essay_year(text_id):
    """Publication-year token from the essay's end-note.

    'נדפס' wins over the last token in the tail, which can be a year cited in a footnote.
    """
    txt = (AH_STRIPPED / f"{text_id}.txt").read_text(encoding="utf-8")
    txt = txt.split("את הטקסט")[0]                       # drop the Ben-Yehuda footer
    for m in re.finditer("נדפס", txt):                   # the publication note is authoritative
        found = re.search(YEAR_RE, txt[m.start(): m.start() + 40])
        if found:
            return found.group()
    matches = re.findall(YEAR_RE, txt[-500:])            # fallback: last token in the tail
    return matches[-1] if matches else None


MANUAL_YEAR = {
    1153: 1893,
    3292: 1904,
    4500: 1902,
    7867: 1898,
}

NEEDS_MANUAL = {13933}

idx["title_year"] = idx["title"].str.extract(r"(\d{4})")
idx["heb_year_tok"] = idx["id"].map(essay_year)

dated = idx["title_year"].notna() | idx["heb_year_tok"].notna()
print("total texts     :", len(idx))
print("dated (either)  :", dated.sum())
print("still undated   :", (~dated).sum())
print()
print("sample tail tokens:")
print(idx.loc[idx["heb_year_tok"].notna(), ["id", "title", "heb_year_tok"]].head(12).to_string(index=False))

WORK_MIN, WORK_MAX = 1885, 1930

VALUES = {'א':1,'ב':2,'ג':3,'ד':4,'ה':5,'ו':6,'ז':7,'ח':8,'ט':9,
          'י':10,'כ':20,'ל':30,'מ':40,'נ':50,'ס':60,'ע':70,'פ':80,'צ':90,
          'ק':100,'ר':200,'ש':300,'ת':400}

def heb_to_gregorian(tok):
    if not isinstance(tok, str):
        return None
    value = sum(VALUES[c] for c in tok if c in VALUES)
    return 5000 + value - 3760

def resolve_year(row):
    if pd.notna(row["title_year"]):
        return int(row["title_year"]), "letter_title"
    if isinstance(row["heb_year_tok"], str):
        return heb_to_gregorian(row["heb_year_tok"]), "essay_pub_note"
    return None, "undated"

idx[["year", "date_basis"]] = idx.apply(resolve_year, axis=1, result_type="expand")

for tid, yr in MANUAL_YEAR.items():
    idx.loc[idx["id"] == tid, ["year", "date_basis"]] = [yr, "external_verified"]

idx.loc[idx["id"].isin(NEEDS_MANUAL), ["year", "date_basis"]] = [pd.NA, "undated"]

suspect = idx["year"].notna() & ~idx["year"].between(WORK_MIN, WORK_MAX)
idx.loc[suspect, "date_basis"] = "suspect_out_of_range"

print("year range:", idx["year"].min(), "-", idx["year"].max())
print(idx["date_basis"].value_counts().to_string())
print("\nspot checks (expect 548≈1889, 10≈1892-93):")
print(idx[idx["id"].isin([548, 10, 1224])][["id","title","heb_year_tok","year"]].to_string(index=False))
print("\nsuspect (out of range) - inspect these:")
print(idx[suspect][["id","title","heb_year_tok","year"]].to_string(index=False))
print("\nstill undated:")
print(idx[idx["date_basis"]=="undated"][["id","title","genre_clean"]].to_string(index=False))

def all_year_tokens(text_id):
    txt = (AH_STRIPPED / f"{text_id}.txt").read_text(encoding="utf-8")
    txt = txt.split("את הטקסט")[0]
    return re.findall(r'תר[א-ת]?["״][א-ת]', txt)

und = idx[idx["date_basis"] == "undated"]
has_any = und["id"].map(lambda i: len(all_year_tokens(i)) > 0)
print("undated with a תר-token SOMEWHERE:", int(has_any.sum()), "/", len(und))

for tid in [3292, 4500]:
    txt = (AH_STRIPPED / f"{tid}.txt").read_text(encoding="utf-8").split("את הטקסט")[0]
    print(f"\n--- {tid} tokens:", all_year_tokens(tid))
    print(repr(txt[-300:]))

idx["original_date"] = idx["year"].astype("Int64")

# This overwrites the LIVE index, including the dating every later result rests on - so back it up
# and require an explicit flag rather than writing as a side effect of inspection.
import time as _time
if "--write-index" in _sys.argv:
    _bk = AH_META / f"index.backup-{_time.strftime('%Y%m%d-%H%M%S')}.csv"
    if (AH_META / "index.csv").exists():
        pd.read_csv(AH_META / "index.csv", encoding="utf-8-sig").to_csv(
            _bk, index=False, encoding="utf-8-sig")
        print(f"[date_corpus] backed up existing index -> {_bk.name}")
    idx.to_csv(AH_META / "index.csv", index=False, encoding="utf-8-sig")
    idx.to_json(AH_META / "index.json", orient="records", force_ascii=False, indent=2)
    print("[date_corpus] index.csv rewritten")
else:
    print("[date_corpus] DRY RUN - index NOT written. Pass --write-index to persist.")
print("saved:", int(idx["original_date"].notna().sum()), "dated /", len(idx))

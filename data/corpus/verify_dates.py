"""Data-quality check on the auto-extracted essay dates.

Re-extracts the year anchored to 'נדפס' ("published in") and flags every essay where it disagrees
with the stored value, which was taken from the last year token in the tail and can be a footnote
citation rather than the publication year.
"""
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

VALUES = {'א':1,'ב':2,'ג':3,'ד':4,'ה':5,'ו':6,'ז':7,'ח':8,'ט':9,
          'י':10,'כ':20,'ל':30,'מ':40,'נ':50,'ס':60,'ע':70,'פ':80,'צ':90,
          'ק':100,'ר':200,'ש':300,'ת':400}
YEAR_RE = r'תר[א-ת]?["״][א-ת]'

def heb_to_greg(tok):
    return 5000 + sum(VALUES[c] for c in tok if c in VALUES) - 3760

def nidpas_year(text_id):
    """Year token found within 40 chars after an occurrence of 'נדפס'."""
    txt = (AH_STRIPPED / f"{text_id}.txt").read_text(encoding="utf-8")
    txt = txt.split("את הטקסט")[0]
    for m in re.finditer("נדפס", txt):
        window = txt[m.start(): m.start() + 40]
        found = re.search(YEAR_RE, window)
        if found:
            return heb_to_greg(found.group()), found.group()
    return None, None

idx = pd.read_csv(AH_META / "index.csv", encoding="utf-8-sig")
essays = idx[idx["date_basis"] == "essay_pub_note"].copy()

rows = []
for _, r in essays.iterrows():
    ny, ntok = nidpas_year(r["id"])
    rows.append((r["id"], r["title"], int(r["original_date"]), ny, ntok))
chk = pd.DataFrame(rows, columns=["id", "title", "stored", "nidpas", "nidpas_tok"])

no_nidpas = chk["nidpas"].isna().sum()
agree = ((chk["stored"] == chk["nidpas"])).sum()
disagree = chk[chk["nidpas"].notna() & (chk["stored"] != chk["nidpas"])]

print(f"essays checked      : {len(chk)}")
print(f"no 'נדפס' anchor    : {no_nidpas}  (kept last-token value)")
print(f"agree with stored   : {agree}")
print(f"DISAGREE (suspects) : {len(disagree)}")
print()
print(disagree.sort_values("stored").to_string(index=False))

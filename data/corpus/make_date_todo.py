"""Export the texts still needing a manual date to a fill-in CSV.

Rows are those with date_basis in {undated, suspect_out_of_range}; fill 'year' and 'source', then
merge back into index.csv.
"""
import sys
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
META = AH_META

idx = pd.read_csv(META / "index.csv", encoding="utf-8-sig")

need = idx[idx["date_basis"].isin(["undated", "suspect_out_of_range"])].copy()
todo = need[["id", "title", "genre_clean", "role", "date_basis", "original_date"]].rename(
    columns={"original_date": "auto_guess"}
)
todo["year"] = ""
todo["source"] = ""
todo = todo.sort_values("id")

todo.to_csv(META / "undated_todo.csv", index=False, encoding="utf-8-sig")
print(f"wrote {len(todo)} rows -> data/metadata/undated_todo.csv")
print(todo[["id", "title", "date_basis", "auto_guess"]].to_string(index=False))

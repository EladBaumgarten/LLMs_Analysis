"""Build a manual-dating worksheet for Alterman.

The API returns collected-edition years, not original composition dates, so those are dated by
hand. Emits an RTL .xlsx with an empty `original_date` column, grouped by edition year so
same-volume texts sit together.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT))
from authors_paths import for_author  # noqa: E402

P = for_author("alterman")
YEAR = re.compile(r"\b(19[0-7]\d)\b")


def first_body_line(text_id) -> str:
    """First non-empty line that isn't the 'title מאת נתן אלתרמן' header line."""
    f = P["corpus_stripped"] / f"{text_id}.txt"
    lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    body = [ln for ln in lines if "מאת נתן אלתרמן" not in ln]
    return (body[0] if body else "")[:60]


def hint_from_text(text_id):
    """A year token near the top of the text, if any - a hint to verify, often absent."""
    head = P["corpus_stripped"].joinpath(f"{text_id}.txt").read_text(encoding="utf-8")[:1500]
    m = YEAR.search(head)
    return int(m.group(1)) if m else ""


def main():
    idx = pd.read_csv(P["metadata"] / "index.csv", encoding="utf-8-sig")
    ws = pd.DataFrame({
        "id": idx["id"],
        "title": idx["title"],
        "genre": idx["genre_clean"],
        "first_line": idx["id"].map(first_body_line),
        "edition_year": idx["edition_date"],
        "hint_from_text": idx["id"].map(hint_from_text),
        "url": idx["id"].map(lambda i: f"https://benyehuda.org/read/{i}"),
        "original_date": "",
        "notes": "",
    })
    ws = ws.sort_values(["edition_year", "id"], na_position="last").reset_index(drop=True)

    out = P["metadata"] / "dating_worksheet.xlsx"
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as xl:
            ws.to_excel(xl, index=False, sheet_name="dating")
            sh = xl.sheets["dating"]
            sh.sheet_view.rightToLeft = True
            sh.freeze_panes = "A2"
            widths = {"A": 8, "B": 34, "C": 10, "D": 40, "E": 12, "F": 12, "G": 34, "H": 14, "I": 24}
            for col, w in widths.items():
                sh.column_dimensions[col].width = w
        made = out
    except Exception as e:
        made = P["metadata"] / "dating_worksheet.csv"
        ws.to_csv(made, index=False, encoding="utf-8-sig")
        print(f"(xlsx failed: {e}; wrote CSV instead)")

    print(f"wrote: {made}")
    print(f"rows: {len(ws)}  (fill the 'original_date' column)")
    print("edition_year groups (texts per volume-year):")
    print(ws["edition_year"].value_counts(dropna=False).sort_index().to_string())
    print("\nrows already carrying a text hint:", (ws["hint_from_text"] != "").sum())


if __name__ == "__main__":
    main()

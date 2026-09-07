import sys, json
from pathlib import Path
from collections import Counter
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from authors_paths import for_author, current_author_name, author_meta
AUTHOR = current_author_name()
P = for_author(AUTHOR, create=True)
DICTA = P["dicta_out"]
META = P["metadata"]
OUT = P["ner_out"]; OUT.mkdir(parents=True, exist_ok=True)
ANCHOR = author_meta(AUTHOR)["anchor_year"]

KEEP = {"PER", "GPE", "LOC", "ORG"}

import re

def entities(text_id):
    d = json.loads((DICTA / f"{text_id}.json").read_text(encoding="utf-8"))
    out = []
    for s in d:
        txt = s["text"]
        for e in s.get("ner_entities", []):
            if e.get("label") not in KEEP:
                continue
            ph = txt[e["start"]:e["end"]]
            ph = re.sub(r'\s*([\"״׳])\s*', r"\1", ph).strip()
            if len(ph) < 2 or "[UNK]" in ph:
                continue
            out.append((e["label"], ph))
    return out

def main():
    idx = pd.read_csv(META / "index.csv", encoding="utf-8-sig")
    essays = idx[(idx["genre_clean"] == "article") & (idx["original_date"].notna())].copy()
    essays["period"] = essays["original_date"].apply(lambda y: f"pre-{ANCHOR}" if y < ANCHOR else f"post-{ANCHOR}")

    rows = []
    for _, r in essays.iterrows():
        for lab, ph in entities(r["id"]):
            rows.append({"id": r["id"], "year": int(r["original_date"]),
                         "period": r["period"], "label": lab, "phrase": ph})
    ner = pd.DataFrame(rows)
    ner.to_csv(OUT / "ner_entities.csv", index=False, encoding="utf-8-sig")
    print("dated essays:", len(essays), "| entities:", len(ner))

    for lab in ["PER", "GPE", "ORG"]:
        print(f"\n=== top {lab} by period ===")
        for period in sorted(ner["period"].unique()):
            top = Counter(ner[(ner.label == lab) & (ner.period == period)]["phrase"]).most_common(10)
            print(f"  {period}: {top}")

if __name__ == "__main__":
    main()

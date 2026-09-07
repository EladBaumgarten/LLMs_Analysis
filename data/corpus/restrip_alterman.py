"""Re-strip Alterman's corpus from the local nikud files with the maqaf-preserving cleaner.

No API call. Overwrites the stripped texts and reports the diff against the previous version.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from authors_paths import for_author
from text_clean import strip_nikud

MAQAF = "־"
OTHER = {"׀": "paseq", "׃": "sof-pasuq", "׆": "nun-hafukha"}

P = for_author("alterman")
nik, strp = P["corpus_nikud"], P["corpus_stripped"]

total = changed = maqaf_restored = 0
other_restored = {k: 0 for k in OTHER}
sample = []
for f in sorted(nik.glob("*.txt")):
    body = f.read_text(encoding="utf-8")
    new = strip_nikud(body)
    of = strp / f.name
    old = of.read_text(encoding="utf-8") if of.exists() else ""
    total += 1
    if new != old:
        changed += 1
        maqaf_restored += new.count(MAQAF) - old.count(MAQAF)
        for k in OTHER:
            other_restored[k] += new.count(k) - old.count(k)
        if len(sample) < 3 and MAQAF in new:
            sample.append((f.stem, old[:60], new[:60]))
    of.write_text(new, encoding="utf-8")

print(f"re-stripped: {total} texts | changed: {changed} | maqaf restored: {maqaf_restored}")
print("other non-combining chars restored:",
      {OTHER[k]: v for k, v in other_restored.items() if v})
for tid, o, n in sample:
    print(f"  {tid}: OLD {o!r}\n        NEW {n!r}")

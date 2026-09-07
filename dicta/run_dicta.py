import re, sys
from pathlib import Path
import json, torch
from transformers import AutoTokenizer, AutoModel


sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from authors_paths import for_author, current_author_name
P = for_author(current_author_name(), create=True)
STRIPPED = P["corpus_stripped"]
META = P["metadata"]

# The cleaner is shared, not local: a second partial copy once drifted out of sync with this one.
sys.path.insert(0, str(ROOT.parent / "data" / "corpus"))
from text_clean import clean_paragraphs as clean_text   # noqa: E402

def split_sentences(paras):
    text = " ".join(paras)
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]

DEVICE = "cuda"          # small GPU: keep the batch small, fall back to CPU per text below
tok = AutoTokenizer.from_pretrained("dicta-il/dictabert-joint", trust_remote_code=True)
model = AutoModel.from_pretrained("dicta-il/dictabert-joint", trust_remote_code=True)
model.eval()
model.to(DEVICE)

OUT = P["dicta_out"]
OUT.mkdir(parents=True, exist_ok=True)

def analyze(text_id, style="json", batch_size=4):
    raw = (STRIPPED / f"{text_id}.txt").read_text(encoding="utf-8")
    sents = split_sentences(clean_text(raw))
    result = []
    with torch.no_grad():
        for i in range(0, len(sents), batch_size):
            result.extend(model.predict(sents[i:i+batch_size], tok, output_style=style))
    (OUT / f"{text_id}.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result

from tqdm import tqdm
import pandas as pd

ids = pd.read_csv(META / "index.csv", encoding="utf-8-sig")["id"].tolist()
done, failed, stale = 0, [], 0
for tid in tqdm(ids, desc="Dicta"):
    # Staleness guard: skipping on .exists() alone makes a re-strip a silent no-op.
    _out, _src = OUT / f"{tid}.json", STRIPPED / f"{tid}.txt"
    if _out.exists():
        if not (_src.exists() and _src.stat().st_mtime > _out.stat().st_mtime):
            done += 1
            continue
        stale += 1
    try:
        analyze(tid)
        done += 1
    except Exception as e:                       # OOM -> retry this text on CPU
        try:
            torch.cuda.empty_cache()
            model.to("cpu")
            analyze(tid)
            done += 1
        except Exception as e2:
            failed.append((tid, f"gpu:{str(e)[:50]} | cpu:{str(e2)[:50]}"))
        finally:
            model.to(DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

print(f"\ndone: {done}/{len(ids)} | re-analysed as stale: {stale} | failed: {len(failed)}")
for tid, err in failed:
    print("  FAIL", tid, err)

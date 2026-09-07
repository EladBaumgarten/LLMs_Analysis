# Affect - LLM ANNOTATOR - RUN ON GOOGLE COLAB (GPU).
# Annotates WHOLE texts with an instruction-tuned Hebrew LLM, to test whether full context
# (negation, irony, stance) recovers human charge where the surface tools failed.
#
# SETUP on Colab:
#   !pip -q install "transformers>=4.40" accelerate bitsandbytes
#   upload ahad_haam_llm_input.csv + alterman_llm_input.csv  (from prep_llm_annotation.py)
#   %run sentiment_llm_colab.py
# Outputs: <author>_llm_annotations.csv  (download + paste back).
import re, json, sys
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "dicta-il/dictalm2.0-instruct"
INPUT_FILES = {"ahad_haam": "ahad_haam_llm_input.csv", "alterman": "alterman_llm_input.csv"}
RUN_AUTHORS = ["ahad_haam", "alterman"]
MAX_NEW = 192

# DictaLM-2.0 takes NO system prompt, so the whole rubric goes in the user turn. Strict-JSON
# instruction so the answer parses.
RUBRIC = (
    "אתה חוקר ספרות המנתח מטען רגשי. קרא את הקטע והחזר אך ורק אובייקט JSON תקין אחד - "
    "ללא טקסט לפני/אחרי, ללא הסברים, וללא markdown. קבע כל שדה לפי הקטע עצמו בלבד (אל תעתיק את "
    "מילות הסכימה). המפתחות בעברית, לפי הסכימה (החלף כל <...> בערך אמיתי שנקבע מהקטע):\n"
    '{"valence_label": "<חיובי | שלילי | נייטרלי | מעורב>", '
    '"valence_score": <מספר עשרוני בסולם דו־קוטבי מ־(מינוס 1.0) עד (פלוס 1.0): ערך שלילי '
    'לקטע של עצב/אבל/זעם/ביקורת, 0 לניטרלי, ערך חיובי לקטע שמח או מלא־תקווה. אל תימנע מערכים שליליים>, '
    '"intensity": <מספר שלם בין 0 (מנותק) ל-3 (עצים מאוד)>, '
    '"register": "<המשלב הרטורי של הקטע במילה אחת: לירי/פולמוסי/אירוני/אלגי/הגותי/מעריך/תיאורי>", '
    '"charge_change": <true אם המטען הרגשי משתנה לאורך הקטע, אחרת false>, '
    '"justification": "<משפט קצר אחד בעברית, ללא מרכאות>"}\n'
    "שים לב לאירוניה, מטאפורה ומשמעות כפולה, ולהבחנה בין העמדה הרגשית של הכותב לבין אוצר המילים. "
    "השתמש בתווית 'שלילי' ובערך שלילי כאשר רוב הקטע מבטא עצב, אבל, זעם או ביקורת - גם אם הלשון עשירה, "
    "ציורית או חגיגית.\n\n"
    "הקטע:\n{TEXT}\n\nJSON:"
)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="auto")
model.eval()
print("loaded", MODEL)

KEYS = ["valence_label", "valence_score", "intensity", "register", "charge_change", "justification"]


def parse_json(s):
    """Strict JSON on the first balanced block, then a per-key regex fallback for truncated or
    quote-broken output. Returns a possibly-partial dict."""
    s = s.replace("```json", " ").replace("```", " ")
    i = s.find("{")
    if i >= 0:
        depth = 0
        for j in range(i, len(s)):
            depth += (s[j] == "{") - (s[j] == "}")
            if depth == 0:
                try:
                    return json.loads(s[i:j + 1])
                except Exception:
                    break
    out = {}                                    # regex fallback
    def grab(key, pat, cast=str):
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            v = m.group(1).strip().strip('"').strip("'").strip()
            try: out[key] = cast(v)
            except Exception: out[key] = v
    grab("valence_label", r'valence_label\D*?(positive|negative|neutral|mixed|[֐-׿]+)')
    grab("valence_score", r'valence_score"?\s*[:=]\s*"?\s*(-?\d+\.?\d*)', float)
    grab("intensity",     r'intensity"?\s*[:=]\s*"?\s*(\d)', int)
    grab("register",      r'register\D*?([֐-׿][֐-׿ \-]*)')
    grab("charge_change", r'charge_change"?\s*[:=]\s*"?\s*(true|false)')
    grab("justification", r'justification"?\s*[:=]\s*"?\s*([^"\n}]+)')
    return out


@torch.no_grad()
def annotate(text):
    msg = [{"role": "user", "content": RUBRIC.replace("{TEXT}", text)}]   # replace: the rubric has literal {}
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
    if not torch.is_tensor(ids):            # some versions return a BatchEncoding
        ids = ids["input_ids"]
    ids = ids.to(model.device)
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def run_author(author):
    df = pd.read_csv(INPUT_FILES[author], encoding="utf-8-sig")
    out_path = f"{author}_llm_annotations.csv"
    print(f"\n=== {author}: annotating {len(df)} texts -> {out_path} ===")
    rows = []
    for n, (_, r) in enumerate(df.iterrows()):
        raw = annotate(str(r["text"]))
        obj = parse_json(raw) or {}
        ok = ("valence_label" in obj) or ("valence_score" in obj)
        rows.append({"author": author, "id": r["id"], "genre": r["genre"], "year": r["year"],
                     **{k: obj.get(k) for k in KEYS}, "parse_ok": ok, "raw": raw[:300]})
        if (n + 1) % 20 == 0:                        # partial save: the GPU session can drop
            pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  {n + 1}/{len(df)}  saved partial | parse_ok {pd.DataFrame(rows).parse_ok.mean():.0%}")
    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  DONE {out_path} | parse_ok {res.parse_ok.mean():.0%} | "
          f"labels {res.valence_label.value_counts(dropna=False).to_dict()}")
    return res


if __name__ == "__main__":
    for a in RUN_AUTHORS:
        run_author(a)
    print("\n================ DONE - download *_llm_annotations.csv and paste back ================")

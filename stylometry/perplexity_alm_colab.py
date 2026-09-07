# Perplexity ALM - RUN ON GOOGLE COLAB (GPU); not run locally.
# Huang, Murakami & Grieve (2025), adapted so the "candidates" are TIME PERIODS within one author.
# Fine-tune one model per period, assign a held-out chunk to the period whose model is less
# surprised by it.
#
# Run: upload <author>_chunks_text.csv (from prep_alm_chunks.py),
#      pip install -q transformers accelerate datasets, then %run this file.
import os, math, json, random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling)
from datasets import Dataset

BASE_MODEL = "Norod78/hebrew-gpt_neo-small"
CHUNK_FILES = {"ahad_haam": "ahad_haam_chunks_text.csv",
               "alterman":  "alterman_chunks_text.csv"}
OUT_DIR   = "."
MAX_LEN   = 1024         # Hebrew BPE expands ~2x; 512 truncated half the chunk.
K_FOLDS   = 5            # a true partition: every chunk tested exactly once.
EPOCHS    = 5
LR        = 2e-5
BATCH     = 4            # long sequences, small GPU
SEED      = 42
N_BOOT    = 2000

device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", device, "| base:", BASE_MODEL, "| folds:", K_FOLDS)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def fresh_model():
    m = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    m.config.pad_token_id = tokenizer.pad_token_id
    return m.to(device)


def fine_tune(texts):
    """Further-pretrain a fresh base model on `texts` (one chunk = one example)."""
    ds = Dataset.from_dict({"text": list(texts)})
    ds = ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
                batched=True, remove_columns=["text"])
    args = TrainingArguments(
        output_dir="_alm_tmp", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH,
        learning_rate=LR, weight_decay=0.01, logging_steps=25, save_strategy="no",
        report_to="none", fp16=(device == "cuda"), seed=SEED)
    model = fresh_model()
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)).train()
    model.eval()
    return model


@torch.no_grad()
def perplexity(model, text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    loss = model(**enc, labels=enc["input_ids"]).loss
    return float(torch.exp(loss))


@torch.no_grad()
def word_cnll(model_e, model_l, text):
    """Word-level CNLL: group BPE sub-tokens into words, sum NLL per word for each model."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    ids = enc["input_ids"]

    def per_tok(model):
        logp = torch.log_softmax(model(**enc).logits[:, :-1, :], dim=-1)
        return (-logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]).tolist()

    ne, nl = per_tok(model_e), per_tok(model_l)
    toks = tokenizer.convert_ids_to_tokens(ids[0])[1:]        # aligned with ne/nl (predict pos i>0)
    words, cur, ce, cl = [], [], 0.0, 0.0
    for t, e, l in zip(toks, ne, nl):
        if t.startswith("Ġ") and cur:                     # 'Ġ' marks a new word
            words.append((tokenizer.convert_tokens_to_string(cur).strip(), ce - cl))
            cur, ce, cl = [], 0.0, 0.0
        cur.append(t); ce += e; cl += l
    if cur:
        words.append((tokenizer.convert_tokens_to_string(cur).strip(), ce - cl))
    return words


def balanced_train_texts(tr_df, rng):
    """Equal-sized training text per period.

    argmin-PPL has no threshold and decision margins are only ~8%, so a model trained on more text
    would win on volume alone.
    """
    by = {p: tr_df[tr_df.period == p].text.tolist() for p in ("early", "late")}
    n = min(len(by["early"]), len(by["late"]))
    out = {}
    for p, txt in by.items():
        if len(txt) > n:
            idx = rng.choice(len(txt), size=n, replace=False)
            txt = [txt[i] for i in sorted(idx)]
        out[p] = txt
    return out, n


def bootstrap_ci(pred_df, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI on pooled out-of-fold balanced accuracy.

    Resamples ARTICLES, not chunks: chunks of one article are near-duplicates, so treating them as
    independent would overstate the sample size.
    """
    rng = np.random.default_rng(seed)
    arts = pred_df["id"].unique()
    by_art = {a: g for a, g in pred_df.groupby("id")}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(arts, size=len(arts), replace=True)
        s = pd.concat([by_art[a] for a in pick], ignore_index=True)
        if s.true.nunique() < 2:
            continue
        stats.append(balanced_accuracy_score(s.true, s.pred))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi), float(np.std(stats))


def run_author(author):
    df = pd.read_csv(CHUNK_FILES[author], encoding="utf-8-sig").reset_index(drop=True)
    if "author" in df.columns:
        assert set(df.author.unique()) == {author}, \
            f"FILE/AUTHOR MISMATCH: {CHUNK_FILES[author]} contains {set(df.author.unique())}"
    print(f"\n=== {author}: {len(df)} chunks | {dict(df.period.value_counts())} | "
          f"{df.id.nunique()} articles | base PPL check…")

    base = fresh_model(); base.eval()
    base_ppl = np.mean([perplexity(base, t)
                        for t in df.text.sample(min(30, len(df)), random_state=SEED)])
    del base; torch.cuda.empty_cache()
    print(f"    un-fine-tuned base mean PPL (30 chunks) = {base_ppl:.1f}")

    rng = np.random.default_rng(SEED)
    cv = StratifiedGroupKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    per_fold, oof = [], []
    cnll_agg = defaultdict(list)
    n_cnll_chunks = 0

    for k, (tr, te) in enumerate(cv.split(df, df["period"], groups=df["id"])):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        texts, n_bal = balanced_train_texts(tr_df, rng)
        models = {}
        for period in ("early", "late"):
            print(f"  fold {k}: fine-tuning {period}-ALM on {len(texts[period])} chunks "
                  f"(balanced to {n_bal})…")
            models[period] = fine_tune(texts[period])

        rows = []
        for _, r in te_df.iterrows():
            pe = perplexity(models["early"], r.text)
            pl = perplexity(models["late"], r.text)
            rows.append({"fold": k, "id": r.id, "true": r.period,
                         "ppl_early": pe, "ppl_late": pl,
                         "pred": "early" if pe < pl else "late"})
        sdf = pd.DataFrame(rows)
        oof.append(sdf)
        bal = balanced_accuracy_score(sdf.true, sdf.pred)
        per_fold.append({"fold": k, "n_test": len(sdf), "bal_acc": bal})
        print(f"  fold {k}: balanced period-acc = {bal:.3f}  (n_test={len(sdf)})")

        # Accumulate over every fold's held-out chunks. Iterate te_df directly - chunk ids are not
        # unique, so joining on `id` would score one article's first chunk over and over.
        for t in te_df.text.tolist():
            n_cnll_chunks += 1
            for word, c in word_cnll(models["early"], models["late"], t):
                if len(word) >= 2:
                    cnll_agg[word].append(c)      # c>0 surprises the EARLY model => late-characteristic

        del models; torch.cuda.empty_cache()

    oof = pd.concat(oof, ignore_index=True)
    assert len(oof) == len(df), f"partition broken: {len(oof)} scored vs {len(df)} chunks"
    oof.to_csv(os.path.join(OUT_DIR, f"{author}_oof_predictions.csv"), index=False,
               encoding="utf-8-sig")

    pooled = balanced_accuracy_score(oof.true, oof.pred)
    lo, hi, boot_sd = bootstrap_ci(oof)
    res = pd.DataFrame(per_fold)
    res.to_csv(os.path.join(OUT_DIR, f"{author}_classification_scores.csv"), index=False)

    mat = {f"{tp}_test|{mp}_ALM": float(oof[oof.true == tp][f"ppl_{mp}"].mean())
           for tp in ("early", "late") for mp in ("early", "late")}
    drift_late = (mat["late_test|early_ALM"] - mat["late_test|late_ALM"]) / mat["late_test|late_ALM"]
    drift_early = (mat["early_test|late_ALM"] - mat["early_test|early_ALM"]) / mat["early_test|early_ALM"]
    drift = float((drift_late + drift_early) / 2)

    summary = {"author": author, "n_chunks": int(len(df)), "n_articles": int(df.id.nunique()),
               "bal_acc_pooled": float(pooled), "ci_lo": lo, "ci_hi": hi,
               "boot_sd": boot_sd, "fold_sd": float(res.bal_acc.std()),
               "drift_score": drift, "base_ppl": float(base_ppl),
               **{k: float(v) for k, v in mat.items()}}
    print(f"  => pooled bal_acc {pooled:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"(article bootstrap) | drift {drift:.3f}")

    cn = pd.Series({w: float(np.mean(v)) for w, v in cnll_agg.items() if len(v) >= 2}
                   ).sort_values(ascending=False)
    cn.to_csv(os.path.join(OUT_DIR, f"{author}_cnll_words.csv"), encoding="utf-8-sig")
    top = pd.concat([cn.head(15), cn.tail(10)])[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(top)), top.values,
            color=["#d62728" if v > 0 else "#1f77b4" for v in top.values])
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index)
    ax.set_title(f"CNLL - words most period-distinct in {author}'s texts\n"
                 f"all {n_cnll_chunks} out-of-fold chunks across {K_FOLDS} folds · "
                 f"red = late-characteristic · blue = early-characteristic")
    ax.set_xlabel("mean CNLL  (NLL_early − NLL_late), summed per word")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, f"{author}_cnll_top.png"), dpi=130)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    summaries = [run_author(a) for a in CHUNK_FILES]
    out = pd.DataFrame(summaries)
    out.to_csv(os.path.join(OUT_DIR, "drift_summary.csv"), index=False)
    print("\n================ SUMMARY (paste this back) ================")
    print(out[["author", "n_chunks", "n_articles", "bal_acc_pooled", "ci_lo", "ci_hi",
               "drift_score", "base_ppl"]].to_string(index=False))
    a, b = summaries[0], summaries[1]
    print(f"\ncross-author gap = {a['bal_acc_pooled'] - b['bal_acc_pooled']:+.3f}"
          f"  ({a['author']} vs {b['author']})")
    print("CIs overlap?  ",
          "YES -> ordering NOT significant" if not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"])
          else "NO -> ordering significant at ~95%")

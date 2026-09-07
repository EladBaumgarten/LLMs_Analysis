import sys
import json
import csv
from pathlib import Path
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from authors_paths import for_author, current_author_name
P = for_author(current_author_name(), create=True)
META = P["metadata"]
OUT = P["dicta_out"]
DERIVED = P["topic_out"]
DERIVED.mkdir(parents=True, exist_ok=True)

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}
STOP = {"[BLANK]", "מר", 'ד"ר', "ע'", "ידיעה", "מחבר"}


def doc_lemmas(text_id):
    d = json.loads((OUT / f"{text_id}.json").read_text(encoding="utf-8"))
    return [t["lex"] for s in d for t in s["tokens"]
            if t["morph"]["pos"] in CONTENT_POS and t["lex"] and t["lex"] not in STOP]


def main():
    idx = pd.read_csv(META / "index.csv", encoding="utf-8-sig")
    ids = idx.loc[idx["genre_clean"] == "article", "id"].tolist()
    docs = [doc_lemmas(i) for i in ids]
    print("essay documents:", len(docs), "| avg content lemmas/doc:", sum(map(len, docs)) // len(docs))

    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(d) for d in docs]
    print("vocab after filter:", len(dictionary))

    # processes=1 avoids a Windows multiprocessing spawn bug.
    print("\n=== coherence sweep ===")
    scores = {}
    for k in range(4, 21, 2):
        m = LdaModel(corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        cm = CoherenceModel(model=m, texts=docs, dictionary=dictionary,
                            coherence="c_v", processes=1)
        scores[k] = cm.get_coherence()
        print(f"k={k:2d}  c_v={scores[k]:.4f}")
    best_k = max(scores, key=scores.get)
    print("best k by coherence:", best_k)

    lda = LdaModel(corpus, id2word=dictionary, num_topics=best_k, passes=10, random_state=42)
    print(f"\n=== topics (k={best_k}) ===")
    for i in range(lda.num_topics):
        print(f"topic {i}:", "  ".join(w for w, _ in lda.show_topic(i, topn=10)))

    lda.save(str(DERIVED / "lda_model"))
    dictionary.save(str(DERIVED / "lda_dictionary"))
    with open(DERIVED / "topics.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "top_words"])
        for i in range(lda.num_topics):
            w.writerow([i, " ".join(word for word, _ in lda.show_topic(i, topn=12))])
    print("\nsaved: lda_model, lda_dictionary, topics.csv")


if __name__ == "__main__":
    main()

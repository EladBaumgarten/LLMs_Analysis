# Two-Author Hebrew Stylometry: Ahad Ha'am and Natan Alterman

Ben-Gurion University of the Negev, Department of Computer Science
Topics in Natural Language Processing (Spring 2026) - mini-project
Author: Elad Baumgarten · Lecturer: Menachem Adler

A computational study of how an author's writing changes over a career, and how a
**consistent** author differs from a **reality-driven** one.

## The question

**Ahad Ha'am** (Asher Ginzberg, 1856-1927) is an essayist famous for ideological
consistency. **Natan Alterman** (1910-1970) is a poet and topical journalist whose
writing follows current events. The study asks, for each author separately and only
then in comparison:

1. **Style** - is the stylistic fingerprint (function words, sentence length,
   morphology, syntax) stable over time, or does it change?
2. **Content** - is the distribution of topics stable, or do subjects rise and fade?
3. **The relation between the layers** - do content and style move together, or is
   one stable while the other shifts?
4. **Events** - do changes line up with formative events (gradual drift vs. a break)?

Because the two authors write in different genres, the comparison is never between
raw feature values ("who writes longer sentences"). Each author is measured **against
himself over time**, and only the two amounts of change are compared.

## Corpus

| | Ahad Ha'am | Alterman |
|---|:---:|:---:|
| texts where he is the author | 162 | 196 |
| articles (the comparison spine) | 131 | 174 |
| **dated** articles - what every result rests on | **96** | **174** |
| tokens | 448,989 | 283,949 |
| mean tokens per text | 3,427 | 1,632 |
| mean / median sentence length | 31.4 / 26 | 22.6 / 17 |
| punctuation % · function-word % | 18.0 · 22.7 | 17.5 · 23.5 |
| unique lemmas · mean per-text TTR | 8,438 · 0.464 | 10,640 · 0.582 |

The full title lists are in [docs/texts/](docs/texts/): every text per author, and
separately the dated articles the analysis actually rests on.

Source: **Project Ben-Yehuda**. The texts themselves are **not** in this repository -
they are rebuilt by the stage-A scripts. What is published is the metadata index the
pipeline reads, plus the derived statistics and results - see
[Data availability](#data-availability).

## Pipeline

Ten stages. Each stage names its purpose, its tool, and its result.

### A - Building the corpus

*Purpose:* assemble both corpora, keeping only items where the author is the author
(translations and edited volumes filtered out).
*Tools:* Ben-Yehuda public-domain dump (Ahad Ha'am), Ben-Yehuda REST API (Alterman);
a shared Hebrew text cleaner used by both, so preprocessing cannot become a
cross-author confound.
*Code:* `data/corpus/build_corpus.py`, `build_alterman_api.py`, `text_clean.py`
*Result:* 162 texts (Ahad Ha'am), 196 (Alterman). One further item in his index is a text he
only translated, and it is excluded everywhere.

### B - Dating

*Purpose:* every text needs its original date of writing or publication, otherwise it
cannot be tied to the events of its time.
*Code:* `data/corpus/date_corpus.py`, `verify_dates.py`, `build_dating_worksheet.py`
*Result:* Ahad Ha'am - 96 of 131 articles carry an original date (26.7% loss).
Alterman - all 174 are dated, but 76 of them show a source conflict (the same text
published in several places; 80 across all 196 of his texts) and are marked
lower-confidence.

### C - Automatic linguistic analysis

*Purpose:* produce the layer every later style feature is computed on.
*Tool:* **Dicta** `dicta-il/dictabert-joint` - part of speech, lemma, syntactic
function, and named entities per word.
*Code:* `dicta/run_dicta.py`
*Result:* a JSON analysis per text. No style feature is read off raw text; all of them
are defined over lemmas, POS tags, and dependency labels.

### D - Descriptive statistics and frequency lists

*Purpose:* describe both corpora in measurable terms.
*Code:* `stylometry/descriptive_stats.py` (career-wide, and `--by-period`),
`stylometry/frequency_lists.py` → `docs/descriptive_stats*.csv`, `docs/S7-freq-*.csv`
*Result:* Ahad Ha'am's sentences are ~40% longer than Alterman's - an essay against a
newspaper column. The sharpest movements are in **function words**, not content words:
Ahad Ha'am's *כי* falls 42% (10.81 → 6.32 per 1,000 words) while *של* nearly doubles
(6.12 → 11.45); Alterman's *אלא* rises steadily (3.37 → 5.53).
*Methodological result:* the two corpora are shaped differently. Ahad Ha'am spreads
over forty continuous years; 143 of Alterman's 174 articles fall in 1967-1970. Even
under equal change, change is easier to detect in the author spread over more time.
This caveat qualifies every result below.

### E - Period classifier

*Purpose:* if a text's period can be identified from style alone, then early and late
style differ.
*Method:* Gómez-Adorno et al. (2018) - three feature families (phraseology,
punctuation, function words), Logistic Regression and SVM, cross-validated. Content
words and named entities are deliberately excluded so the classifier learns style, not
subject. All frequencies normalized per 1,000 words.
*Code:* `stylometry/features.py`, `binning.py`, `classify.py`
*Result* (balanced accuracy; three periods, chance = 0.333):

| feature family | Ahad Ha'am | Alterman |
|---|:---:|:---:|
| function words | 0.603 | 0.463 |
| punctuation | 0.576 | 0.439 |
| phraseology | 0.547 | 0.565 |
| **combined** | **0.648** | **0.567** |

Under the alternative binning: 0.650 vs 0.554. In a two-period split (chance = 0.5):
0.795 vs 0.890 - but for Alterman that split separates literary criticism from
political commentary, i.e. it measures a *register* change. Restricted to his late
political articles only, the score drops to **0.534**.

*Conclusion:* both authors changed, but Ahad Ha'am's periods are more separable -
0.648 vs 0.567. The initial hypothesis is not supported; the direction is the opposite
of what was expected. The breakdown also shows two different regimes: Ahad Ha'am's
time signal is spread over all three channels, while Alterman's sits almost entirely
in phraseology (combined 0.567 ≈ phraseology alone 0.565).

### F - Measuring along the career

*Purpose:* distinguish **gradual drift** from a **sharp break**.
*Method:* Eder (2016) rolling stylometry, adapted - instead of detecting an author
change inside one text, a window slides over the author's chronologically ordered
articles and measures his distance from his own starting point.
*Code:* `stylometry/rolling.py`
*Result:* 28 windows (Ahad Ha'am), 54 (Alterman). Spread around the mean is identical
(0.129 vs 0.129) and the yearly rate of change is near-identical (0.0157 vs 0.0148).
The difference is the **shape** of the curve: Ahad Ha'am climbs continuously across
the whole dated range; Alterman rises sharply until 1967 and is flat afterwards.
*Caveat:* windows are measured in texts, not years. The median window for Alterman
spans **0 years** (against 4 for Ahad Ha'am), so much of his "step" is an artifact of
how his dates are distributed, not a stylistic break.

### G - Distinguishing periods a second way, and asking *what* changed

*Purpose:* repeat stage E with a different method, and add feature interpretation.
*Method:* Przystalski (2025), adapted - articles split into short chunks,
gradient-boosted trees, SHAP for interpretation. Cross-validation is grouped by source
article, so chunks from one article never straddle train and test.
*Code:* `stylometry/boosted_period.py`, `period_scatter.py`, `direction.py`
*Result:* 820 chunks from 95 articles (Ahad Ha'am), 682 from 172 (Alterman). Balanced
accuracy **0.628 vs 0.456**; alternative binning 0.584 vs 0.507. SHAP - computed on
held-out chunks - points to punctuation and *כי* for Ahad Ha'am, and to syntactic
structures, *אלא*, and mean sentence length for Alterman.
*Conclusion:* converges with stage E, and the four discriminating markers it names are
exactly the four visible in the raw frequency counts of stage D - the classifier is
gripping real, countable differences.

### H - Content: topics and entities

*Purpose:* is the content stable, or do subjects rise and fade?
*Tools:* **LDA** topic model (gensim, k=6 for both authors) and Dicta's named-entity
recognition.
*Code:* `topic_modeling/LDA_prep.py`, `topics_over_time.py`, `ner/ner_over_time.py`
*Result:* Alterman moves sharply - the literary topic falls 0.409 → 0.116 → 0.035
across the 1950s/60s/70s, while foreign policy and the Arab world rise from zero to
0.386. (His pre-1960s decades hold only a handful of articles each, so the early end of
that curve is thin.) The literary critic becomes a political commentator. Ahad Ha'am also moves, but
in an orderly way inside one project: the Jewish-moral topic declines 0.511 → 0.396 →
0.265 → 0.136, while the two Zionist topics rise 0.289 → 0.527 → 0.782, with the turn
in the decade after the First Zionist Congress. Entities agree: before 1897 he cites
contemporaries and polemicists; after, Maimonides (170) and real places in the Land of
Israel.
*Conclusion:* on the **content** layer the original hypothesis does hold.

### I - AI track: authorial language models

*Purpose:* attack the same question with a completely different instrument, one that
cannot separate style from content.
*Method:* Huang, Murakami & Grieve (2025), adapted - instead of attributing a text to
one of several authors, attribute it to one of two periods of the *same* author. A copy
of a Hebrew base language model is fine-tuned per period; an unseen text goes to the
period whose model is less surprised by it (lower perplexity).
*Tool:* `Norod78/hebrew-gpt_neo-small`. DictaBERT was rejected (masked, not generative
- the method needs next-token probability) and DictaLM 7B was beyond available compute.
5 folds × 2 periods × 2 authors = 20 fine-tuned models; training sets balanced across
periods.
*Code:* `stylometry/prep_alm_chunks.py`, `perplexity_alm_colab.py`, `replot_cnll.py`
*Result:* balanced accuracy **0.716** (Ahad Ha'am) vs **0.696** (Alterman), chance =
0.5, with 95% article-bootstrap confidence intervals [0.649, 0.780] and [0.616, 0.779].
Base perplexity before fine-tuning: 96.1 and 98.3.
*Conclusion:* both authors drift clearly - both intervals sit far above chance. But the
intervals overlap almost entirely, so **this track does not separate the two authors**.
When measurement is restricted to style, Ahad Ha'am's periods are more identifiable;
once content is admitted, the gap closes. Alterman's time signal leans heavily on his
changing subjects - an independent confirmation of stage H.

### J - Affect: does the emotional charge shift?

*Purpose:* test a third layer - does the emotional tone of the writing change over the
years, and in which direction? Run on Alterman only, the author whose writing responds
to events.
*Method:* four instruments arranged as a **ladder of context**, each seeing more than
the last, so that agreement is real cross-validation and failure is diagnostic.
*Ground truth:* 31 hand-labelled texts, of which 29 carry a label the scorer can map to a
class and are the ones every number below is computed on. The two it drops are reported by
name when the script runs, rather than silently.
*Code:* `stylometry/sentiment_diachronic.py`, `prep_llm_annotation.py`,
`sentiment_llm_colab.py`, `sentiment_llm_validate.py`, `sentiment_calibrate.py`

| instrument | context it sees | sign accuracy (3-way) | Spearman |
|---|---|:---:|:---:|
| **majority baseline** | - | **0.48** | - |
| DictaLM-2.0 | whole text (≤4,000 chars) | 0.48 | −0.01 |
| HeBERT | one sentence | 0.45 | +0.18 |
| NRC-VAD lexicon | one word | 0.38 | +0.14 |
| calibration layer (ridge, LOO) | four numbers | 0.38 | −0.20 |

*Result:* **no instrument beats the majority baseline** - not even the one that reads
the whole text. The instruments also contradict each other in *sign*: HeBERT reads
almost everything as negative, the lexicon as mildly positive. DictaLM does not collapse
onto one class (15 negative / 13 neutral / 1 positive, against a human 12 / 14 / 3), yet
its rank correlation with the human labels is **−0.01**, i.e. zero.

*Conclusion - the central result of this stage:* this is a **validity** failure, not a
technical one. The tools reliably measure something real - how emotionally **loaded the
vocabulary** is - but not what was asked, the writer's **stance** toward what he
describes. In *בעבור נעלים*, where Alterman praises residents for refusing a humiliating
handout, the charge is measured as maximally negative: every word describes poverty and
humiliation, while the approval lives in the writer's choice to present the act as
praiseworthy, and is present in none of his words. More context cannot fix this, because
extending context supplies *scope* (what the negation applies to) and not a *value
standard*. More data would not fix it either: a validity failure is repaired by changing
the training target, not by enlarging the sample.

## Classical track vs. AI track

*Where they agree:* both authors' writing changed over their careers, far from chance,
in every method - the finding does not depend on any one algorithm.

*Where they disagree:* **which of the two changed more.** In the classical track, where
only style is measured, Ahad Ha'am is more separable in all six configurations tested.
In the AI track, where the language model reads style and content together, the
difference disappears. These are two measurements of different things, so there is
no contradiction - and the gap between them is what makes it possible to estimate how
much of each author's change sits in style and how much in content.

Four further differences:

- *Field of view.* The classical track deliberately removes content words and named
  entities. A language model cannot remove anything - it assigns a probability to every
  word present.
- *Statistical power.* Only the AI track yields a genuine confidence interval, because
  it resamples at the article level. The classical track's spread comes from repeated
  refits on one fixed sample and cannot support a significance claim.
- *Sensitivity to preprocessing.* A tokenization bug fixed mid-project (affecting a few
  percent of tokens) barely moved the classical features - each is an average over tens
  of thousands of words - but noticeably moved the language model, because bad splits
  create rare forms and a probabilistic model struggles with rare forms. Robustness to
  preprocessing noise is a property of *how* a method computes.
- *Nature of the failures.* Classical failures are confounds - time mixed with register,
  noisy period labels - and they can be located, measured, and fixed, as they were. The
  AI-track failure in stage J is different in kind: the tool measured something reliably,
  just not the intended thing.

## Conclusions

### Answers to the four questions

**1. Style - stable or changing?** Both authors changed, clearly and far from chance.
Between them, Ahad Ha'am's periods are the more separable, in all six configurations
tested. The initial hypothesis - that the author known for consistency would also be
the stylistically stable one - is not supported.

**2. Content - stable or changing?** Both move, but not in the same way. Alterman's
literary topic falls 0.409 → 0.035 while foreign policy rises from zero to 0.386: the
literary critic becomes a political commentator. Ahad Ha'am's Jewish-moral topic falls
0.511 → 0.136 while the two Zionist topics rise 0.289 → 0.782. The difference is not
the size of the movement but its **kind**: Alterman's crosses domains and follows
external events, Ahad Ha'am's stays inside one intellectual project and follows its
development. In that sense the hypothesis does hold - but not as "a moving author
versus a fixed one".

**3. The relation between the layers - do they move together?** No. Ahad Ha'am's
content is broadly stable while his style shifts across the whole timeline; Alterman's
content shifts while his style is relatively stable *within* a register. Each author's
time signal also rides a different channel: for Ahad Ha'am punctuation and function
words - unconscious habits - and for Alterman syntax and sentence length.

**4. Evolution or break?** Ahad Ha'am: **evolution**. His distance curve climbs
continuously over thirty-three years with no marked step, and his measurement windows
are evenly spread, so the smooth shape is not an artifact of sampling. On the content
layer, though, his frame of reference does change around the First Zionist Congress.
Alterman: a **concentrated transition**, not a wandering one - but it cannot be called
a dated stylistic break, because its position is fixed by the shape of the corpus and
coincides exactly with his switch from literary criticism to political journalism.
What an independent test does support is the other half of the claim: inside the late
political register, his style is relatively stable.

### From a dichotomy to axes

The study began by assuming two types - a stable author and a changing one - and the
data forced that framing to be abandoned. Neither author is stable or changing in
himself. Each is stable on one axis and moving on another, and in opposite directions.
Ahad Ha'am holds a coherent world of ideas together and changes how he writes;
Alterman keeps a steady fingerprint inside a given register and changes his subjects
and his register. **The meaningful distinction is not how much an author changed, but
on which axis he changed.**

- *Ahad Ha'am's consistency is intellectual, not linguistic.* The historical claim
  about him is partly confirmed: his world of ideas really is coherent. But his manner
  of writing drifted over the career more than that of the author considered dynamic.
  What he wrote persisted; how he wrote did not.
- *Alterman's dynamism is one of content and register, not of style.* His single large
  movement is one crossing between kinds of writing, with topics changing above it in
  step with reality. Within one register his style is fairly stable. The first half of
  that claim needs a caveat - the crossing sits exactly where the corpus becomes dense,
  so its apparent sharpness is partly a property of the data. The stability inside the
  late register was measured separately and does not depend on it.

### The answer depends on what the instrument is allowed to see

Tools restricted to style separate the two authors; a tool that also spans content does
not. There is no contradiction, because the two are not measuring the same thing - and
the **difference between them** is exactly what allows the question to be decomposed
into its two layers. That is how a methodological inconvenience became an instrument.

### Limits that belong beside every conclusion

The gap between the authors is consistent in direction, but its size depends on
methodological choices and it is **not statistically significant**. The two corpora are
also shaped differently: Alterman's journalistic writing is concentrated into four years,
so part of the gap reflects how much time there was to measure across, not style alone. The lesson
for a cleaner comparison is to choose authors whose corpora have a similar shape.

### What the AI track taught

The two AI experiments came out opposite, and the contrast is itself a finding. Language
models trained to identify a text's period scored far above chance for both authors,
while the attempt to classify emotional charge never beat guessing the most common
label - not even the tool that read whole texts. The difference is not sophistication:
the largest and newest of the instruments is the one that failed. It is **task
definition**. "When was this written" has a single answer fixed in the source data.
"What is this text's emotional charge" does not: human readers themselves spread across
a range of readings.

So the tools taught one lesson and the task another. About the tools: their failure was
one of **validity, not accuracy**. They reliably measured a real quantity - how loaded
the vocabulary is - and in doing so carried the assumptions of the material they were
trained on, where harsh words signal a negative stance. Moral and literary prose breaks
that assumption systematically, describing hardship precisely in order to praise the
person who meets it. About the task: its definition has to be examined *before* the
instrument is chosen, and a property that is not sufficiently defined does not become
measurable through a stronger model.

What is left at the end is not only an answer about two authors, but a more precise
statement of the question: not which of the two changed more, but **on which axis each
of them changed, and what a measuring instrument needs in order to see it**.

## Repository layout

```
authors/<author>/metadata/    index.csv / index.json - the pipeline's canonical input
data/corpus/                  corpus building, cleaning, dating
dicta/                        DictaBERT-joint morphological + syntactic analysis
stylometry/                   feature extraction, classifiers, rolling, affect
topic_modeling/               LDA topics over time
ner/                          named entities over time
docs/                         descriptive statistics and frequency lists (CSV)
docs/results/                 per-experiment result tables - the evidence behind every figure
docs/texts/                   title lists per author: all texts, and the ones used
figures/<author>/             the plots used in the report
authors_paths.py              per-author path resolution; every script takes --author
collect_results.py            rebuilds figures/ and docs/results/ from the pipeline outputs
```

## Where the evidence lives

Every headline number in this README can be checked against a file, without re-running anything:

- `docs/descriptive_stats.csv` · `docs/descriptive_stats_by_period.csv` - corpus statistics,
  career-wide and per period.
- `docs/S7-freq-<author>-{period,decade}.csv` - the most frequent content lemmas, function words,
  morphological features, POS bigrams and dependency relations per time bucket, per 1,000 tokens.
- `docs/results/cv-scores-classical-<author>.csv` · `cv-scores-boosted-<author>.csv` - every
  classifier score, broken down by feature family, binning scheme and variant, with the feature
  count of each family.
- `docs/results/shap-top-features-<author>.csv` - which features carried the period signal.
- `docs/results/feature-direction-<author>.csv` - and which way each of them moved, as Cohen's d
  and as a rank correlation against the year, with BH-FDR corrected q-values.
- `docs/results/topics-<author>.csv` · `topic-weights-by-decade-<author>.csv` - the LDA topic word
  lists and their weight per decade.
- `docs/results/ner-top-entities-<author>.csv` - the most-mentioned people, places and
  organisations per period: the place axis.
- `docs/results/rolling-delta-*-<author>.csv` - the curves behind the rolling figures.
- `docs/results/cnll-top-words-<author>.csv` - the words the period language models found most
  distinguishing.
- `docs/results/period-scatter-<author>.csv` - within-era versus between-era variance.

`python collect_results.py` rebuilds this set and `figures/` from the pipeline outputs, so the
published evidence cannot silently lag a re-run.

Sources for every method and model are in [REFERENCES.md](REFERENCES.md).

## Running it

Python 3.11 via conda; a GPU is needed for stages C and I (the latter was run on Colab).

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate stylometry
```

Both files pin the versions this project was actually run with. For GPU, install `torch`
from pytorch.org instead of from the pinned CPU build; everything except stages C and I
runs on CPU.

**One file has to be downloaded separately.** Stage J reads the Hebrew NRC-VAD lexicon,
which is not redistributed here because NRC lexicons are free for research but need NRC's
permission to republish. Get it from the NRC VAD Lexicon page
(https://saifmohammad.com/WebPages/nrc-vad.html - the distribution includes translations
into about a hundred languages, Hebrew among them) and save it as
`stylometry/lexicons/Hebrew-NRC-VAD-Lexicon.txt`. The loader expects a tab-separated file
with the columns `English Word`, `Valence`, `Arousal`, `Dominance`, `Hebrew Word`, with
valence and arousal in [0, 1]. Only stage J needs it; every other stage runs without it.

Every analysis script takes `--author {ahad_haam,alterman}` and, where periods are
involved, `--scheme`:

```bash
python data/corpus/build_corpus.py                            # A - Ahad Ha'am corpus
python data/corpus/build_alterman_api.py                      # A - Alterman (needs an API key)
python dicta/run_dicta.py             --author ahad_haam      # C
python stylometry/descriptive_stats.py --author ahad_haam     # D
python stylometry/features.py         --author ahad_haam      # E - build the matrix first
python stylometry/classify.py         --author ahad_haam      # E
python stylometry/rolling.py          --author ahad_haam      # F
python stylometry/boosted_period.py   --author ahad_haam      # G
python topic_modeling/topics_over_time.py --author ahad_haam  # H
python ner/ner_over_time.py           --author ahad_haam      # H
python stylometry/prep_alm_chunks.py  --author ahad_haam      # I - then fine-tune on Colab
python stylometry/frequency_lists.py                          # D - frequency lists
python stylometry/descriptive_stats.py --by-period            # D - time segmentation
python collect_results.py                                     # refresh figures/ + docs/results/
```

The Ben-Yehuda API key is read from an environment file that is not committed.

## Data availability

Neither author's running text is in this repository, and neither is the Dicta analysis
derived from it - together they are roughly 700 MB, and Alterman's half could not be
published in any case. What is published is everything needed to rebuild them and to
check the results:

- **`authors/<author>/metadata/index.csv`** - the canonical index the whole pipeline
  reads: text ids, titles, genres, and the dating layer, including the manually verified
  composition years that no script can regenerate.
- **`data/pseudocatalogue.csv`** - the Ben-Yehuda dump catalogue that
  `build_corpus.py` reads.
- **`docs/`, `docs/results/`, `figures/`** - the statistics, result tables and plots.

Also not published: the Hebrew NRC-VAD lexicon that stage J reads. See Running it for
where to download it.

**To rebuild Ahad Ha'am's corpus** (public domain): download the
`projectbenyehuda/public_domain_dump` into `data/p23/` and `data/p23_Nikud/`, then run
`data/corpus/build_corpus.py`.
**To rebuild Alterman's** (`by_permission`, not redistributable): run
`data/corpus/build_alterman_api.py` with a Ben-Yehuda API key.
Then `dicta/run_dicta.py` regenerates the linguistic analysis for both, and the stage
scripts regenerate everything downstream.

## License

Code is MIT; the report, figures and derived tables are all rights reserved and citable
with attribution. Neither author's running text is distributed here, and nothing in this
repository grants rights in the underlying Ben-Yehuda texts. See [LICENSE](LICENSE).

## Known limitations

- **Time and genre are confounded for Alterman.** His Ben-Yehuda corpus is bimodal -
  early poetry, late prose - so the comparison runs on an articles↔articles spine, and
  period binning is done within genre.
- **The date loss is one-sided.** Ahad Ha'am loses 26.7% of his articles to missing dates
  and Alterman 0%. The dropped rows have no date field at all, so it cannot be tested
  whether the loss is temporally random.
- **The classical track reports no confidence intervals.** Its spread comes from repeated
  refits on a fixed sample, which is not sampling uncertainty.
- **Stage J produced no finding about the author** - only about the instruments and the
  conditions under which they can be used.

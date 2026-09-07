# References

## Methods implemented in this project

**Gómez-Adorno, H., Posadas-Duran, J.-P., Ríos-Toledo, G., Sidorov, G., & Sierra, G. (2018).**
*Stylometry-based Approach for Detecting Writing Style Changes in Literary Texts.*
Computación y Sistemas 22(1).
→ Stage E. Three feature families (phraseology, punctuation, function words) and a
chronological-stage classifier. Adopted with the period labels taken from the article dates
instead of from named literary periods.

**Eder, M. (2016).** *Rolling stylometry.*
Digital Scholarship in the Humanities 31(3), 457-469. https://doi.org/10.1093/llc/fqv010
→ Stage F. Repurposed: instead of sliding a window over one text to detect an authorship
takeover, the window slides over the author's chronologically ordered articles and measures his
distance from his own starting point.

**Przystalski, K., Argasiński, J. K., Grabska-Gradzińska, I., & Ochab, J. K. (2025).**
*Stylometry recognizes human and LLM-generated texts in short samples.*
Expert Systems With Applications 296, article 129001. https://doi.org/10.1016/j.eswa.2025.129001
→ Stage G. Short-sample chunking, gradient-boosted trees and SHAP, relabelled from
human-vs-LLM to early-vs-late period within one author.

**Huang, W., Murakami, A., & Grieve, J. (2025).**
*Attributing authorship via the perplexity of authorial language models.*
PLOS ONE 20(7): e0327081. https://doi.org/10.1371/journal.pone.0327081
Reference code and data: https://github.com/Weihang-Huang/ALMs
→ Stage I. Authorial language models, with the candidates changed from several authors to two
periods of the same author.

**Alsudais, A., & Tchalian, H. (2016).**
*Corpus Periodization Framework to Periodize a Temporally Ordered Text Corpus.*
Twenty-second Americas Conference on Information Systems (AMCIS), San Diego.
→ Examined and **rejected**. It infers period boundaries from changes in writing volume and
vocabulary across a large single-subject news corpus; for one author, yearly volume reflects what
survived of his writings rather than a shift in discourse, and the yearly counts are too small to
carry a boundary decision. Reasoning is in the report's literature review.

## Supporting methods

**Burrows, J. (2002).** *'Delta': a Measure of Stylistic Difference and a Guide to Likely
Authorship.* Literary and Linguistic Computing 17(3), 267-287.
→ The distance metric inside the rolling analysis.

**Eder, M., Rybicki, J., & Kestemont, M. (2016).** *Stylometry with R: A Package for Computational
Text Analysis.* The R Journal 8(1), 107-121.
→ The reference implementation of rolling stylometry, reimplemented here in Python.

**Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003).** *Latent Dirichlet Allocation.*
Journal of Machine Learning Research 3, 993-1022.
→ The topic model behind stage H, via gensim.

**Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions.*
Advances in Neural Information Processing Systems 30 (NIPS 2017).
→ SHAP, used in stage G to ask which features carried the period signal.

**Mohammad, S. M. (2018).** *Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance
for 20,000 English Words.* Proceedings of the 56th Annual Meeting of the Association for
Computational Linguistics (ACL), Melbourne.
→ The NRC-VAD lexicon; the Hebrew translation of it is the bottom rung of the stage-J ladder.
Not redistributed in this repository; download it from https://saifmohammad.com/WebPages/nrc-vad.html
(see Running it in the README for the expected path and columns).

## Models

| Model | Role |
|---|---|
| `dicta-il/dictabert-joint` | Morphology, syntax and NER for every text - stage C, and the base of every style feature |
| `Norod78/hebrew-gpt_neo-small` | Base model fine-tuned per period in stage I |
| `dicta-il/dictalm2.0-instruct` | Whole-text affect annotation, the top rung of the stage-J ladder |
| `avichr/heBERT_sentiment_analysis` | Sentence-level sentiment, the middle rung of stage J |

DictaBERT was rejected for stage I: the method needs next-token probability and DictaBERT is a
masked model. DictaLM 7B was rejected for the same stage - fine-tuning it twenty times was beyond
the available compute.

## Data

**Project Ben-Yehuda** - https://benyehuda.org (public API v1).
Ahad Ha'am is authority id 23 and is public domain; Alterman is authority id 533 and is
`by_permission`. Neither corpus is republished here - both are rebuilt by the stage-A
scripts. See Data availability in the README.

## Libraries

scikit-learn (classifiers, cross-validation) · LightGBM (boosted trees) · SHAP · gensim (LDA) ·
transformers / PyTorch (Dicta, HeBERT, DictaLM, GPT-Neo) · pandas · matplotlib · SciPy

# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** NaN Sense  
**Team Members:** Hemanvitha Pullela Pullela, Avani Gupta, Chandrima Hazra  
**Institution:** National Institute of Technology (NIT), Jamshedpur  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
We use a multi-key inverted-index blocking stage (nine key families, ranked by a small learned
"meta-blocking" LightGBM model rather than raw similarity) to cut a pool of ~10M source-2/3 records
down to a top-30 candidate set per S1 entity at 94.3% recall, followed by a LightGBM classifier over ~60
pairwise string/rank features and a decision rule tuned directly for macro F0.5 (not accuracy or a
fixed 0.5 threshold). Offline 5-fold cross-validated F0.5 is **0.9598**. Our main identified risk is
generalization to France, which never appears in training — a country-holdout check (train on one
country, test on the other) shows F0.5 dropping from ~0.94–0.97 in-domain to ~0.84–0.94 out-of-domain,
which motivates a second stage (in progress) using a multilingual bi-encoder for retrieval and a
transformer/LLM reranker for the pairs the first-stage model is least confident about.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training ground truth showed: 5.6% of S1 entities are true singletons (no match), the
mean is 3.5 matches per S1 (up to 10), every S2/S3 record belongs to at most one S1 entity (verified:
0 records appear in more than one S1's match list), and 100% of true matches share the same `country`
label — so blocking can safely restrict candidates to the same country without losing recall.
Name noise includes legal-suffix variants (Pvt/Private, Corp/Corporation), DBA/FKA aliases, businesses
transliterated into Devanagari/Kannada/Tamil script, scrambled-letter typos, and names derived from a
website domain (e.g. `victorylaboratories.com`) rather than the legal name. Address noise includes
missing components, leading zeros on house numbers (`00272` vs `272`), landmark references ("Near SBI
ATM"), native-script state names, and French addresses carrying a region or department name that is
not part of the street address and needs to be set aside rather than matched against.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier, with a second (in-progress) neural retrieval/reranking stage.
**Core Innovation:** (1) a learned meta-blocking ranker — instead of a fixed similarity formula, a
LightGBM model trained on per-key-type evidence (which key families matched, and their IDF weight)
decides which candidates are worth keeping, measurably beating a raw-similarity ranking at the same
candidate budget; (2) an F0.5-aware decision rule that, per entity, either applies a tuned probability
threshold or directly selects the subset of candidates maximizing *expected* F0.5 under the model's own
calibrated probabilities — both combined with a one-candidate-per-record consistency constraint learned
directly from the ground truth structure (see 2.1).

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used (9 families, all restricted to same-country pairs):** house number + next
  street token; two consecutive numeric address tokens; street-name bigram following a number; a
  single rare name token; the first two name tokens (sorted); the whole compacted name; a name token
  paired with a house number; a rare alphabetic address word; a pair of the two rarest address words
  in a record. Keys appearing in more than 100–300 S2/S3 records (per key family) are dropped to keep
  candidate volume manageable. A LightGBM ranker scores every retrieved candidate from its per-key-type
  evidence, and the top 30 per S1 are kept.
- **Candidate pairs generated:** 62.5M (train, all 2.2M S1 entities); 49.6M (test, all 1.73M S1 entities).
- **How true matches were not lost:** recall was measured directly against ground truth on a held-out
  10% sample of training S1 entities *before* any modeling decisions were made. Top-30 recall is
  **94.25%** with the learned ranker (94.26% on all 2.2M training S1 entities), versus 93.45% with a
  raw summed-similarity ranking at the same candidate budget — confirming the ranker is worth the
  extra training step. At top-50 the learned ranker reaches 94.9% (US 96.8%, India 92.0%) versus
  94.2% for summed similarity (US 96.6%, India 90.7%); per-country recall at top-30 was not broken
  out. This recall figure is the hard ceiling on our final F0.5; it is the main thing a
  further blocking iteration (e.g. dense multilingual retrieval, in progress) would improve.

---

## 4. Matching Model

**Features used (~60 total):**
- Name features: rapidfuzz ratio / partial-ratio / token-sort / token-set ratios, Jaro-Winkler
  similarity, IDF-weighted token Jaccard and overlap, first-token match, acronym match, exact-match
  and substring-containment flags, length difference, legal-suffix agreement, best score across the
  record's alias/trade-name variants.
- Address features: the same string-similarity family applied to the normalized address core, exact
  postcode match, house-number-set Jaccard and conflict flags, first-house-number match, landmark-
  phrase overlap, region (state/French department) agreement.
- Other (relative/rank features): each candidate's blocking score rank and gap within its own S1's
  candidate list, and — separately — within the list of *other* S1 entities competing for that same
  candidate record. The second kind lets the model use "no one else wants this record" as evidence.

**Model type:** LightGBM binary classifier (gradient-boosted decision trees), trained with 5-fold
cross-validation grouped by S1 entity (an entity's candidates never split across folds).
**Threshold selection method:** tuned directly for macro F0.5 on out-of-fold predictions. We compared
a fixed probability threshold against a per-entity "expected-F0.5" selector (choosing the top-k
candidates, for k starting at 0, that maximize expected F_beta under the model's calibrated
probabilities), each with and without a one-candidate-per-record consistency constraint. The best
configuration was a **0.70 probability threshold combined with one-to-one assignment**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), 5-fold out-of-fold:** **0.9598** overall (US 0.971, India 0.943).
- **Country-holdout check (proxy for the unseen France segment):** training on one country and
  evaluating on the other drops F0.5 from ~0.94–0.97 (in-domain) to **0.841–0.944** (out-of-domain),
  confirming real generalization risk for France, which appears only in the test set.
- **Common false positives (wrong merges):** candidates sharing a common or generic business name
  (e.g. "Primary Care Group", which recurs 253 times across S1 alone) combined with an address that
  is similar but not identical.
- **Common false negatives (missed matches):** business names with no lexical relationship to their
  true match (domain-derived names, heavily scrambled typos, or a completely different trade name)
  where the address also lacks a distinguishing house number — these can fall outside the top-30
  candidate set entirely and are the main driver of the blocking-recall gap in Section 3.

---

## 6. Conclusion
A learned-ranker blocking stage plus a LightGBM classifier tuned directly for macro F0.5 reaches 0.96
offline, with US/India performing well in-domain. The clearest remaining weakness is generalization to
an unseen country, quantified by our country-holdout check rather than assumed — this is the basis for
the multilingual dense-retrieval and transformer/LLM-reranking stage described in the appendix, which
we expect to primarily help the France segment and the harder India cases where names transliterate
across scripts.

---

## Appendix

### A. Code Artefacts
The complete, runnable pipeline ships under `code/business_entity_resolution/` (`src/`, `README.md`,
`requirements.txt`). Entry points, in order: `src/prep.py` (normalise + build blocking keys for a
split), `src/blocking.py --train-ranker` (train the meta-blocking ranker and report candidate recall),
`src/train.py` (candidate search, feature engineering, cross-validated LightGBM, F0.5 decision-rule
tuning), and `src/run_pipeline.py` (scores the test set and writes `output/matching_results.tsv` +
`output/candidate_pairs.tsv`, then runs the official validator). `src/check_ids.py` is a lightweight
extra check that every output ID exists in the test set. `src/dense.py`, `src/cross_encoder.py` and
`src/stack.py` implement the second-stage neural retrieval/reranking work referenced above (GPU-side
training happens on Kaggle; see `kaggle/KAGGLE_SETUP.md` and `kaggle/THREE_TRACKS.md` in the repository).
Full reproduction steps with exact commands are in `README.md`.

### B. Additional Results
*(to be expanded once the second-stage retrieval/reranking results are in)*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

# Business Entity Resolution — full account of the approach, start to end

Amazon ML Challenge 2026 · team NaN Sense · status as of 26 Sep 2026 (deadline 27 Sep 23:59)

---------------------------------------------------------------------------------------------------------------
## 0. The problem in one paragraph

Three sources describe businesses. Source 1 (S1) is a clean, deduplicated reference; Sources 2 and 3 (S2, S3)
contain noisy copies of those businesses plus unrelated records. For every S1 business we must list all S2/S3
records that are the same business. Scoring is **F0.5 computed per S1 business and averaged** — precision counts
twice as much as recall, and an S1 with no true match scores 1.0 only if we predict nothing for it.

| | S1 | S2 + S3 | labels |
|---|---|---|---|
| train (US + India) | 2.2 M | 10.3 M | yes |
| test (US 38%, India 47%, **France 15%**) | 1.73 M | 9.97 M | no |

France never appears in training. Models must be MIT or Apache-2.0 and ≤ 8B parameters. 5 leaderboard
submissions per day.

---------------------------------------------------------------------------------------------------------------
## 1. What the data turned out to be (measured before building anything)

- **It is synthetic, and the generator is the same in every country.** The number of true matches per S1 has
  the same distribution in US and India to three decimals: 5.6% have none, mean 3.46, up to 10.
- **Each S2/S3 record belongs to at most one S1**; 26% of S2/S3 records belong to none (copies of businesses
  that have no S1 record). Matches never cross countries.
- **How copies differ from their S1** (noise the generator applies): typos, accents added ("Màison"),
  UPPER/lower case, abbreviations (Street→St, Rue→R., Private→Pvt), address parts reordered or dropped,
  "##"/"No."/"N°" before numbers, leading zeros, a missing address (4.4% of copies are name-only), a completely
  new invented name at the same address (4.1%), "X DBA Y" / "fka" aliases, website-style names
  ("victorylaboratories.com"), legal forms rewritten or bracketed ("[LTD.]"), honorific prefixes (India:
  Shri/Sri/Dr/Smt/M/s), suffix words ("Center", "Services"), and **house numbers corrupted on purpose**
  (a digit dropped 909→90, a letter added 2143→2143B).
- **Look-alikes the generator creates** (the hard negatives): the same street with another house number, another
  street in the same city with a similar name, a different name at the same address, and "sibling" names where
  one business word is swapped (Anilesh Enterprises vs Anilesh Developers, Tap Fab Private vs Tap Fab Public).
- **Some labels are impossible to get right**: identical name + address labelled as different businesses;
  a random new name at the same address labelled "match" in one case and "no match" in another; name-only
  copies whose name is shared by several S1s. This caps the achievable score below 1.0.

---------------------------------------------------------------------------------------------------------------
## 2. Phase 1 — the first working pipeline: key blocking + LightGBM (run1, leaderboard 0.948)

### Why this shape
Comparing every S1 with every S2/S3 record in its country is ~10^12 pairs. The standard entity-resolution
recipe is: generate a small candidate list per S1 ("blocking"), score each candidate pair with a classifier,
then turn scores into a match list.

### 2.1 Normalisation (src/normalize.py, src/prep.py)
Every name and address is cleaned into comparable fields: lower-case, accents removed, non-Latin scripts
transliterated (Hindi, Kannada … → Latin), legal forms mapped to one form (Pvt/Private, Ltd/Limited, SARL…),
common name words canonicalised (Intl→international), street types abbreviated one consistent way
(US, Indian and French: rue/r, avenue/av, allée/all …), house numbers extracted, postcodes extracted, landmark
phrases ("Near SBI ATM") split off, and regions/states/French departments moved to a separate low-weight field
(sources disagree about giving the region vs the department). DBA/fka aliases and website names are split into a
second "alias" name. 12.5 M records normalise in ~4 minutes in parallel.

### 2.2 Key blocking (src/blocking.py)
Each record gets several kinds of keys, and two records become candidates if they share a key within the same
country: house number + next street word, two consecutive numbers, a street-name bigram, a single rare name
token, the first two name tokens, the whole compact name, name token + house number, a rare address word, and a
pair of the record's rarest address words. Keys shared by too many records are dropped. A small LightGBM
"meta-blocking" ranker scores each candidate from which key types it shares, and the top 30 per S1 are kept.
**Recall: 94.3%** of true matches reach the candidate list.

### 2.3 The matcher (src/features.py, src/train.py)
For every candidate pair, ~60 features: fuzzy name similarity (ratio, partial, token-sort, token-set,
Jaro-Winkler), IDF-weighted name/address word overlap and the rarest unmatched word, acronym match, legal-form
agreement/conflict, address similarities, postcode / house-number / number-set agreement, landmark and region
agreement, lengths, the blocking score, **rank/gap of this candidate among the S1's other candidates**, and
**candidate-side competition** (how many S1s retrieved this record, and how this S1 ranks among them).
No country one-hot — every feature is a similarity, so it could in principle transfer to France.
LightGBM is trained on 200k sampled S1s (5.7 M pairs) with 5-fold cross-validation grouped by S1.

### 2.4 The decision rule (src/decide.py)
A pair is a match if its probability ≥ threshold **and** the S1 is the record's best-scoring S1 ("one-to-one":
each record goes to at most one S1, which the data guarantees). The threshold is tuned for macro F0.5
(0.70); an "expected-F0.5" per-S1 rule was also tried and was equal.

**Result:** validation F0.5 0.9598 → **leaderboard 0.948**.
**Lesson from the loss breakdown:** of the 0.040 lost, 0.022 was blocking (true matches never retrieved) and
0.019 the classifier. Even a perfect classifier on these candidates could only reach 0.978 — below the leader.
So recall of the candidate stage had to be fixed first.

---------------------------------------------------------------------------------------------------------------
## 3. Phase 2 — dense retrieval to fix recall (run2, leaderboard 0.970)

### What
A bi-encoder (intfloat/multilingual-e5-small, MIT, 118M) fine-tuned on Kaggle with a contrastive loss
(CachedMultipleNegativesRankingLoss = InfoNCE with in-batch negatives + one hard negative per pair) on 1.39M
(S1, true copy, hard non-copy) triples from S1s held out of every evaluation. All 12.5M records are embedded;
each S1's 20 nearest same-country records are retrieved by cosine similarity on the GPU.

### How it was combined
Candidates = key top-30 ∪ dense top-10 (+6 candidates per S1 on average). The retriever's cosine, its rank and
the gap to the S1's best cosine became three new LightGBM features.

### Result
| | key only | key + dense |
|---|---|---|
| candidate recall | 0.943 | **0.996** |
| validation F0.5 | 0.9598 | **0.9829** (US 0.983, India 0.983) |
| leaderboard | 0.948 | **0.970** |

India, previously the weak country, jumped ~4 points. Candidate recall is no longer a limit.

---------------------------------------------------------------------------------------------------------------
## 4. Phase 3 — cross-encoder rerankers (run5, leaderboard 0.9742)

### What
A cross-encoder reads both records together ("name | address" of each) and outputs a match score — the Ditto
approach from the entity-matching literature. Scoring all 60M test pairs would take days, so only pairs the
LightGBM is unsure about (probability 0.01–0.99, ~5% of pairs) are re-scored. Its score is then fed with the
LightGBM probability into a small stacking LightGBM, and the threshold is re-tuned.

### Models tried (all trained out-of-fold, 3 folds)
| reranker | AUC on the uncertain pairs | LightGBM on same pairs | stacked validation F0.5 |
|---|---|---|---|
| XLM-R base (MIT, Kaggle T4) | 0.939–0.945 | 0.939–0.940 | ≈ stage 1 (not used) |
| **Qwen3-0.6B + LoRA** (Apache, Jarvis A100, 4.2 h) | **0.958–0.959** | 0.939–0.940 | **0.9830 → 0.9868** |

**Result:** run5 = LightGBM + Qwen stacked → **leaderboard 0.9742** (a gain of only +0.0012 over run3, versus
+0.003 expected from US/India — see §5: Qwen hurt France).

---------------------------------------------------------------------------------------------------------------
## 5. Phase 4 — discovering that France is the entire remaining gap

### 5.1 Reading France off the leaderboard
With validated US/India scores and known country shares, the leaderboard reveals France:
LB ≈ 0.383·US + 0.468·India + 0.150·France.

| run | US + India share | leaderboard | implied France |
|---|---|---|---|
| run1 | 0.813 | 0.948 | ~0.90 |
| run2 | 0.836 | 0.970 | ~0.89 |
| run3 | 0.836 | 0.973 | ~0.91 |
| run5 | 0.840 | 0.9742 | ~0.90 |

US/India are at ~0.987 on validation and their remaining errors are mostly the impossible labels of §1.
**Every further point must come from France**; 0.985 on the leaderboard needs France ≈ 0.97.

### 5.2 France-specific fixes already in the submissions
- **Stricter threshold for countries absent from training** (0.95 instead of 0.75; the code never names France —
  it checks which countries appear in training). Borderline French matches were mostly look-alikes: same street
  with another number, or a different street with a similar city-based name. run3 **+0.003** (France ≈ +2 pts).
  0.98 (run4) was no better.
- **The retriever's cosine is unreliable on an unseen country.** A retriever trained on US only finds 99.2% of
  India's matches when India was in training but only 78.6% when it wasn't (dense top-10), and the matcher
  over-trusts the cosine. For France, a matcher **without the three retriever features** is used
  (candidates from the retriever are kept). This is run6, not yet on the leaderboard.
- **Qwen is kept off France:** trained on US and tested on India it scores AUC 0.785 (1.7B: 0.830) against
  LightGBM's 0.939. Fine-tuned transformers transfer worst of all components.

### 5.3 A labelled stand-in for France (the key experimental tool)
France has no labels, so every idea is measured on a simulation: retriever + matcher trained on **US only**,
scored on **India** with the true labels. One trap had to be removed: 77% of India's unseen-country misses are
native-script pairs (Hindi/Kannada names), which France does not have. The **France-like test** keeps only
India S1s whose record and all true matches are in Latin script (26,379 S1s):

| | F0.5 on France-like India |
|---|---|
| in-domain (India in training) | 0.9755 |
| unseen (US-trained) | **0.901** — the same gap as France's ~0.90 |
| reverse direction, India-trained → US | 0.952 |

### 5.4 Where the unseen-country loss actually is
- It is on **ordinary pairs** where name and address both look similar: the unseen model loses 4,047 true
  matches and adds 2,784 false matches there, while the in-domain model gets them right.
- 97% of those misses and 71% of those false matches sit in the 0.01–0.99 probability band (4.7% of pairs).
- Looking at them: the in-domain model has learned **country-specific word conventions** — which extra words are
  just copy noise (India: Dr / Sri / Shri / Smt / M/s prefixes, a "Center" suffix, a dropped word) and which mark
  a sibling business (Enterprises↔Developers, Private↔Public, an inserted "Exports", Development↔Projects).
  A model that has never seen a country's labels does not know that country's lists — and neither do we for
  France (is "Groupe" noise? is "Participations" a sibling marker?).

---------------------------------------------------------------------------------------------------------------
## 6. Everything tried for France and rejected (each measured on the unseen-country simulation)

| idea | how | result |
|---|---|---|
| drop retriever features | matcher without cos/rank/gap | **+2.6** on full India — kept (run6) |
| remove feature groups | 7 groups, both directions (US→India, India→US) | ±0.5, nothing helps |
| per-country quantile normalisation | every feature → percentile within its country | worse (0.831 vs 0.847) |
| simpler trees / heavy regularisation | 15 leaves, L2 = 20, feature fraction 0.5 | equal or worse |
| monotone constraints | "more similar ⇒ likelier match" enforced | worse (0.889 vs 0.901) |
| **synthetic target-country data** | emulated the generator (src/synth.py) on the target's own S1 text: 1.38M labelled copies + 0.66M labelled look-alikes, through the real pipeline | 0.901 → 0.899 (no gain; synthetic-only model 0.843) |
| LightGBM self-training | retrain on target pseudo-labels, 2 rounds | +0.9 on full India; France-like rerun pending |
| zero/few-shot LLM judge | Qwen2.5-7B-Instruct (Apache), not fine-tuned | AUC 0.62 vs 0.66 for the unseen LightGBM on the same hard pairs |
| unique-name rule for name-only copies | assign to the only same-name S1 | +0.00001 (model already does it) |
| "agreement with the S1's other copies" | collective feature | AUC 0.62 alone, +0.003 |

---------------------------------------------------------------------------------------------------------------
## 7. Current direction — a token-aware reranker for the unseen country

The only remaining idea that targets the measured cause (target-country word conventions in the uncertain band)
is a model that **sees the words** and **learns from the target country without labels**. Research basis:
DADER (domain adaptation for entity resolution, SIGMOD 2022: GRL, MMD, InvGAN+KD; "MMD is more stable"),
DANN gradient reversal (Ganin et al. 2016), Noisy Student (Xie et al. 2020) and DGER (domain generalisation for
ER, 2024).

**Model (src/da_encoder.py):** XLM-R base (MIT) cross-encoder, mean-pooled, reading "name | address" of both
records — the country string is deliberately left out. Losses:
1. match loss (BCE) on labelled source-country pairs;
2. **alignment** between source and target representations — MMD (multi-kernel) or a country classifier behind a
   gradient-reversal layer (DANN);
3. **teacher–student**: the LightGBM teacher's *confident* decisions on the target country (probability ≥ 0.97 as
   one-to-one matches, ≤ 0.03 as non-matches) are pseudo-labels (96.6% correct on the simulation), with word
   dropout on inputs so the student generalises rather than copies. The student is meant to learn which extra
   words co-occur with true copies in the target country, and apply that to the uncertain pairs.

**Experiment running on Kaggle (France-like simulation, source = US labels, target = India unlabelled):**
A plain cross-encoder · B + MMD · C + GRL · D teacher–student · E teacher–student + MMD.

**Gate:** a variant only proceeds if it beats its LightGBM teacher on the Latin-only target pairs. Then it is
trained on the France bundle (US + India labels, 812k uncertain French pairs, 298k French pseudo-labels), its
scores are fused with the LightGBM for France, and the result becomes the next submission.
If no variant passes, the honest ceiling of this system is ~0.975–0.977.

**On the use of test data:** the France bundle uses the *text* of test records (never labels). That is
transductive unsupervised domain adaptation, and it will be described as such in the methodology document.

---------------------------------------------------------------------------------------------------------------
## 8. Open questions for review
1. The unseen-country gap sits on ordinary-looking pairs. What would you try that is not in §6?
2. How can a target country's copy-noise words vs sibling words be learned without labels, other than
   pseudo-labels?
3. Would mdeberta-v3-base (MIT) transfer better than XLM-R here, and why?

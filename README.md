# Amazon ML Challenge 2026: Business Entity Resolution

**Best public leaderboard score: 0.988441 (macro F0.5)**, run33.

Contributor: [@hemanvithapullela0456](https://github.com/hemanvithapullela0456)

---

## Executive Summary

The task: given three sources of business records (`entity_id, business_name, business_address, country`), list for
every Source-1 (S1) record the Source-2/Source-3 records that are copies of the same business. The metric is
**macro F0.5 per S1**, so a wrong match costs about twice as much as a missed one, and an S1 with no true copies
scores 1 only if nothing is predicted for it.

The data is synthetic: every record is a distorted copy of a hidden entity (typos, accents, casing, script switches
to Hindi/Kannada/Tamil, reordered words, dropped addresses, made-up aliases such as "Zephbrix", and deliberate
decoys). Train covers the **US and India** (2.2M S1s, 10.3M S2/S3 records, 7.6M true pairs); test adds **France**,
which never appears in training (1.73M S1s in total).

Our best solution is a **retrieve → score → rerank → stack → decide** funnel:

- key blocking and a fine-tuned embedding retriever find the candidates;
- a LightGBM scores every candidate pair;
- four transformer rerankers re-read the uncertain pairs;
- a LightGBM stack combines everything;
- France, with no labels, gets its own **self-training** pipeline with label-free decoy detection.

The largest single late gain came from finding a **train/test mismatch bug in the stack**. The final model had
been trained with the stack's own out-of-fold predictions in the `prob` column but applied to stage-1 probabilities
on test. Fixing it lifted the leaderboard from 0.986314 to **0.988441**.

---

## 1. Best Performing Approach

### 1.1 Architecture Overview
- **Normalization:** names split into core name / legal form / alias / acronym; addresses into core / house numbers
  / postcode / region.
- **Candidate generation:**
  - key blocking (inverted index + character TF-IDF, top 30 per S1, ranked by a small LightGBM);
  - a fine-tuned **multilingual-e5-small** dense retriever (top 10 per S1);
  - candidate recall is **0.996**.
- **Stage 1:** LightGBM over ~60 string, key, embedding and candidate-competition features.
- **Stage 2 (US/India), four cross-encoder rerankers on the uncertain band:**
  - Qwen3-0.6B (LoRA);
  - Qwen3-1.7B (LoRA);
  - mDeBERTa-v3;
  - a **one-to-set** mDeBERTa that reads an S1 together with its 10 candidates.
- **Stack:** LightGBM over stage 1 + rerankers + graph/consensus + generator-move + same-name owner + co-location
  features.
- **France (unseen country):**
  - a dedicated LightGBM with label-free **word-role** features;
  - **teacher–student** mDeBERTa trained on the pipeline's own confident French predictions;
  - fusion with the one-to-set model;
  - label-free **decoy rules**.
- **Decision:** threshold (US/India 0.725, France 0.90), then **one-to-one** assignment (each record goes to its
  highest-scoring S1).

### 1.2 Model Architecture Details
```
          S1 records                          S2 / S3 records
              |                                      |
              +-------------- normalize -------------+
                                  |
                 +----------------+------------------+
                 |                                   |
        key blocking (top-30)          dense retriever (e5-small, top-10)
                 +----------------+------------------+
                                  |
                   candidate pairs (~35 per S1, recall 0.996)
                                  |
                  STAGE 1: LightGBM, ~60 features -> p1
                                  |
        +-------------------------+----------------------------+
        | US / India                                           | France (no labels)
        |                                                      |
        | uncertain band (0.01 < p1 < 0.99):                   | LightGBM (no dense) + word roles
        |   Qwen3-0.6B LoRA   Qwen3-1.7B LoRA                  | teacher-student mDeBERTa students
        |   mDeBERTa-v3       one-to-set mDeBERTa              | + one-to-set model (rank fusion 0.4)
        |                                                      | vocabulary-swap decoys x0.5
        | STACK (LightGBM):                                    | same-name owner rule
        |   p1 + reranker scores/ranks/gaps                    | shifted house numbers kept conservative
        |   + graph/consensus + generator moves                |
        |   + owner model (name-only) + co-location            |
        |                                                      |
        | threshold 0.725                                      | threshold 0.90
        +-------------------------+----------------------------+
                                  |
                   one-to-one assignment -> matching_results.tsv
```

### 1.3 Key Technical Innovations

**Candidate generation that survives heavy noise**
- Keys: name words, squashed core name, house number + street, acronyms, aliases split from "X doing business as Y".
- The dense retriever catches copies that share almost no text with the S1 (aliases, script switches).
  It added **+0.022** on the leaderboard (0.948 → 0.970).

**Rerankers and the one-to-set model**
- Pairwise cross-encoders read the S1 and the candidate as one input.
- The one-to-set model reads `[CLS] S1 [SEP] [CAND] c1 [SEP] … [CAND] c10`, with a scoring head on every `[CAND]`
  token. It compares each candidate with the S1 **and with its siblings** (band AUC 0.964, the best single
  reranker).
- All rerankers are trained on 3 S1 folds, so every training pair has an out-of-fold score.

**Stack features that encode the data generator**
- **Graph/consensus:** agreement of an uncertain candidate with the S1's confident copies.
- **Generator moves:** house-number shift size, dropped address, script change, alias word, alias clusters.
- **Same-name owner model:** when several S1s share a name, an address-less copy tends to keep its owner's legal
  form, casing and punctuation. A LightGBM ranks the members of each same-name group.
- **Co-location:** counts of S1s sharing an address or a name, name-word rarity, and competing S1s at the record's
  address. A made-up alias at an S1's exact address is **99.6% true** when that S1 is alone there and **~50%**
  when 2–3 S1s share it.

**France without labels**
- **Word roles:** words over-represented in French S2/S3 names vs S1 names are generator insertions (Groupe, & Fils,
  Services); the others are real vocabulary (Ecole, Club, Santé).
- **Vocabulary-swap decoys:** same address, stem kept, one real vocabulary word swapped
  ("Bordeaux Ecole SARL" → "Bordeaux Ehpad SARL"). These are demoted (×0.5): **+0.003** on the leaderboard.
- **Teacher–student self-training:**
  - pseudo-labels: 150k confident positives, 149k negatives, and 40k demoted decoys as hard negatives;
  - the students are mDeBERTa with 10% word dropout;
  - band AUC rose from 0.853 to 0.956.

**Stack train/test consistency fix**
- The final stack model is now trained on the same stage-1 `prob` it receives at test time.
- Pairs the set model never scored keep their stage-1 probability, because the stack never saw such pairs above
  0.01 in training.

### 1.4 Training Configuration
```
Stage 1 (LightGBM)       objective=binary  lr=0.05  num_leaves=63  min_data_in_leaf=40
                         feature_fraction=0.8  bagging=0.8  lambda_l2=1  grouped 5-fold CV by S1
                         trained on a 200k-S1 sample of train
Rerankers                3 CE folds over 130.8k S1s with uncertain pairs (0.01 < p1 < 0.99)
  one-to-set model       microsoft/mdeberta-v3-base, 2 epochs, batch 16, lr 3e-5, bf16, BCE per candidate
Stack (LightGBM)         lr=0.05  num_leaves=31  min_data_in_leaf=200  feature_fraction=0.9
                         5-fold CV by S1 with early stopping; final model rounds = 1.1 x mean best iteration
France students          mdeberta-v3-base, 1 epoch, batch 32, lr 2e-5, word dropout 0.1, seeds 42 and 7
France fusion            0.1 x LightGBM + 0.9 x students, then rank fusion with the one-to-set model (w=0.4)
Decision                 US/India t=0.725, France t=0.90, one-to-one
Hardware                 CPU (16 threads, 16 GB RAM) + one A100 40GB (Jarvis Labs) / Kaggle GPUs
```

### 1.5 Data Processing Pipeline
1. **Text normalization:** lowercase, strip accents and punctuation, standardize abbreviations (Road → rd,
   Rue → rue), split legal forms (Pvt Ltd, LLC, SARL…), aliases ("t/a", "f/k/a", "doing business as") and acronyms.
2. **Address parsing:** core address tokens, house/unit numbers, postcode, landmark, region. Order-free address
   keys are used for co-location counts.
3. **Missing data:** records with an empty address (about 4.4% of true copies) are handled as **name-only**
   pairs, with their own owner model and rules.
4. **Out-of-fold everything:** stage 1, rerankers and stack are all scored out-of-fold, so validation matches
   what the model sees on test.

### 1.6 Why This Approach Succeeded
1. **High recall first:** keys plus embeddings find 99.6% of true copies before any scoring.
2. **Expensive models only where needed:** most pairs are obvious; rerankers read only the uncertain band.
3. **Features that mirror the generator:** moves, owner, decoy and co-location features encode how copies and
   decoys are created.
4. **Label-free adaptation to France:** word roles plus self-training replace the labels France does not have.
5. **Checking test against validation:** comparing score distributions per stage-1 bin exposed the stack bug,
   the largest late improvement.

---

## 2. Other Approaches Summary

### 2.1 Stage 1 only (LB 0.948–0.973)
- **Approaches:** key blocking + LightGBM (0.948); + dense retriever (0.970); stricter France threshold (0.973).
- **Key features:** fast, strong on in-domain US/India text.
- **Limitations:** misses subtle differences (one swapped word, shifted house number); weak on France.

### 2.2 Rerankers with the buggy stack (LB 0.974–0.9797)
- **Approaches:** Qwen3-0.6B stack (0.974214); + one-to-set, mDeBERTa, graph features, Qwen3-1.7B (0.979683).
- **Key features:** validation gains of +0.007–0.010 per stack S1.
- **Limitations:** only a fraction reached the leaderboard, later traced to the `prob`-column bug.

### 2.3 France pipeline evolution (LB 0.977–0.986)
- **Approaches:** dedicated LightGBM + word roles + teacher–student (0.977405); mDeBERTa students (0.979557);
  set-model fusion (0.981046); decoy demotion (0.984004); name-only rules (0.985773); students retrained on cleaned
  pseudo-labels (0.986314).
- **Key features:** everything learned without French labels.
- **Limitations:** every step needed a leaderboard upload to verify; simulations misled twice.

### 2.4 Tried and rejected
- **Transfer from US/India text alone:** graph token model, entity bi-encoder, LLM judge (Qwen2.5-7B) all at band
  AUC 0.62–0.71 on French-like data.
- **Domain adaptation:** adversarial (DANN/GRL) and MMD domain adaptation collapsed.
- **Decision-layer changes:** veto of low-confidence matches (−0.0006 on LB), per-S1 "no match" abstention,
  expected-F0.5 selection instead of a threshold.
- **Stack variants:** XLM-R as a 5th reranker, separate thresholds per pair type, stack averaging, and slower
  learning rates were all flat on validation.

---

## 3. Performance Analysis

### 3.1 Leaderboard Progress (all uploads)
| Rank | Run | Public LB | Approach |
|---|---|---|---|
| 1 | run33 | **0.988441** | Stack bug fix + co-location corrector |
| 2 | run31 | 0.986314 | France students retrained on cleaned pseudo-labels |
| 3 | run25 | 0.985773 | All vocabulary swaps + name-only rules + move-feature stack |
| 4 | run28 | 0.985652 | Owner model v1 + France threshold 0.84 (regressed) |
| 5 | run21 | 0.984004 | France vocabulary-swap decoy demotion |
| 6 | run13 | 0.981046 | One-to-set fusion in France (w=0.4) |
| 7 | run12 | 0.980703 | One-to-set fusion in France (w=0.2) |
| 8 | run11 | 0.979683 | + Qwen3-1.7B reranker |
| 9 | run10 | 0.979557 | mDeBERTa students, set model + graph features |
| 10 | run7 | 0.977405 | Dedicated France pipeline |
| 11 | run8 | 0.976795 | France veto (regressed) |
| 12 | run5 | 0.974214 | Qwen3-0.6B reranker stack |
| 13 | run4 | 0.973302 | France threshold 0.98 |
| 14 | run3 | 0.973 | France threshold 0.95 |
| 15 | run2 | 0.970 | + dense retriever |
| 16 | run1 | 0.948 | Key blocking + LightGBM |

Built after run33 (validated on labelled data, not scored at the time of writing):
- **run35:** set-model scores for all folds, stack trained on 3× the S1s.
- **run36:** co-location features inside the stack (validation 0.98616 → 0.98636).

### 3.2 Key Insights
1. **Recall is the foundation:** the dense retriever was the largest single jump (+0.022).
2. **France gains came from finding generator decoys,** not from bigger models.
3. **Validation must match test:** the stack's gains were real on validation but never fully reached the
   leaderboard until the train/test feature mismatch was fixed.
4. **The remaining US/India loss is mostly information-limited:** name-only copies whose exact name is shared by
   5+ businesses are close to a coin flip.
5. **Small labelled-validation gains transfer** once validation and test are consistent (run33 matched its estimate).

---

## 4. Technical Lessons Learned

### 4.1 What Worked
1. Dense retrieval on top of key blocking (+0.022).
2. Rerankers on the uncertain band only, stacked with generator-aware features.
3. Label-free word roles and decoy rules for the unseen country (+0.003 from one rule).
4. Teacher–student self-training on confident pseudo-labels, with decoys as hard negatives.
5. Comparing test and validation score distributions per stage-1 bin, which exposed the stack bug (+0.0021).

### 4.2 What Didn't Work
1. Any model trained only on US/India text when applied to French text.
2. Adversarial domain adaptation (collapsed).
3. Decisions validated only on a France simulation (the veto and a lower threshold both lost on the leaderboard).
4. Refitting a stack and replacing *all* its test scores when new features concern only a subset of pairs.
5. Hand-tuned per-S1 abstention and expected-F0.5 selection (both below a global threshold).

---

## 5. Conclusion
Entity resolution on noisy synthetic data rewarded a funnel that maximizes recall cheaply and then spends model
capacity only on uncertain pairs. The unseen country was handled without labels, through word roles, decoy
detection and self-training. The last and largest improvement came from engineering discipline rather than a new
model: checking that the stack sees the same inputs on test as in training.

**Key success factors:**
1. Two complementary candidate generators (keys and embeddings).
2. Generator-aware features (moves, owner, decoys, co-location).
3. Out-of-fold scoring at every stage.
4. Label-free adaptation for France.
5. Systematic test-vs-validation checks.

---

## Appendix

### A. Model Comparison Summary
| Component | Model | Role | Measured effect |
|---|---|---|---|
| Dense retriever | multilingual-e5-small (fine-tuned) | Candidates | LB +0.022 |
| Stage 1 | LightGBM, ~60 features | Score all pairs | CV 0.983 (US/India) |
| Rerankers | Qwen3-0.6B / 1.7B (LoRA), mDeBERTa-v3 | Re-read uncertain pairs | Stack fold 0.976 → 0.983 |
| One-to-set | mDeBERTa-v3-base over S1 + 10 candidates | List-level reranking | Band AUC 0.964 |
| Owner model | LightGBM per same-name group | Name-only ownership | Stack fold 0.983 → 0.986 |
| Co-location | LightGBM features | Shared-address ambiguity | Validation +0.0008 |
| France students | mDeBERTa teacher–student | Unseen-country reranking | Band AUC 0.853 → 0.956 |
| Decoy rules | Label-free word roles | France look-alikes | LB +0.003 |

### B. Technical Specifications
**Winning model (run33)**
- Stage 1 + 4 rerankers + two LightGBM stacks: an address-pair stack and a name-only owner stack.
- Co-location corrector on uncertain address pairs.
- France pipeline as in run31.
- Models used are MIT/Apache-2.0 and ≤ 8B parameters; no external data; no hand-labelling of test data.

**Performance metrics**
- Public leaderboard: **0.988441**.
- US/India labelled validation (stack S1s): 0.98565 → 0.98637 with the corrector.

### C. Repository Layout and Reproduction
```
src/        all pipeline code (config.py holds paths, blocking caps and LightGBM params)
jarvis/     GPU scripts for Jarvis Labs (France students)
kaggle/     GPU setup for Kaggle (dense retriever, rerankers)
aws/        AWS setup notes
results/    findings log, handoff summaries, leaderboard notes
tools/      helper scripts
```
Main steps, in order (dataset in `student_resource/dataset/`, or set `ER_DATA_DIR`):
```bash
pip install -r requirements.txt
python src/prep.py --split train && python src/prep.py --split test        # normalize + blocking keys
python src/blocking.py --split train --sample 0.1 --train-ranker            # meta-blocking ranker
#   dense retriever on GPU: src/dense.py (see kaggle/KAGGLE_SETUP.md)
python src/train.py                                                         # stage 1 (OOF on a 200k-S1 sample)
python src/run_pipeline.py                                                  # stage-1 test scores
#   rerankers on GPU: src/cross_encoder.py (3 folds), src/set_encoder.py --val_fold {0,1,2}
python src/graph_feats.py --split train && python src/graph_feats.py --split test
python src/owner.py --split train && python src/owner.py --split test
python src/coloc.py --split train && python src/coloc.py --split test
python src/stack.py --tag _set,_qwen,_mdeb,_q17 --graph --moves --restrict --out_prefix fix_
python src/stack.py --tag _set,_qwen,_mdeb,_q17 --graph --moves --owner --restrict --out_prefix fix_
python src/fix_noset.py <stack scores> <out>     # for both stacks
python src/make_hyb.py --moves <moves> --owner <owner> --tag _hybfix
python src/coloc_corr.py --src test_scored_stage2_hybfix.parquet --out_tag _hybfixc
#   France: src/word_roles.py, src/unseen_model.py, src/da_encoder.py (GPU), src/fuse_v3.py --mode france,
#           src/fuse_set_france.py --w_set 0.4, src/decoy_vocab.py, src/owner_apply.py, src/merge_students_shift.py
python src/run_pipeline.py --reuse --stage2 --stage2_tag _hybfixc --unseen_scores <france scores> --unseen_t 0.9
```
`run_pipeline.py` writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, runs the official
validator, and archives every submission as `output/runs/runN_matching_results.tsv`.

The full experiment log is in [results/FINDINGS_SUMMARY.md](results/FINDINGS_SUMMARY.md).

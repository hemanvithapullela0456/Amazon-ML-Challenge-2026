# Amazon ML Challenge 2026: entity resolution, full handoff (27 Sep 2026, ~12:30 IST)

Self-contained summary of everything tried, what the leaderboard said, and what is still open.
Goal: public leaderboard > 0.988. Best so far: **0.985773 (run25)**. Leaderboard top: ~0.993.
Deadline 27 Sep 23:59 IST. 2 uploads left today. ~₹450 of A100 time (Jarvis, ₹84/hr) plus Kaggle notebooks.

---

## 1. Task and data
- Three sources of business records: `entity_id, business_name, business_address, country`. No other fields.
- For every Source-1 (S1) record, list the Source-2/Source-3 records that are copies of the same business.
- **Metric:** macro F0.5 per S1, so precision weighs 2×. An S1 with no true copies scores 1 only if we predict nothing, else 0.
- **Train:** US and India with labels. 2.2M S1s, 10.3M S2/S3 records, 7.64M true pairs, 3.46 copies per S1 on average.
- **Test:** US 663k, India 810k and **France 259k** S1s. France never appears in train. Leaderboard weights are roughly 0.38 US / 0.47 India / 0.15 France.
- Rules:
  - models must be MIT/Apache-2.0 and ≤ 8B parameters;
  - no external data;
  - no hand-labelling of test data;
  - unlabelled test text may be used (transductive use is documented).
- The public leaderboard scores a subset of test; the private leaderboard scores the rest.
- **The data is synthetic.** Records are distorted copies of latent entities:
  - noise: typos, accents, case, brackets, script switches;
  - address dropped in ~4.4% of true copies;
  - name replaced by a made-up alias in ~13% of copies (e.g. "Zephbrix"); aliases often repeat within one entity;
  - decoys: the same name with the house number shifted, the same address with another name, and in France "vocabulary swaps".
- The one-to-one constraint holds: each S2/S3 record belongs to at most one S1.
- **About 20% of test S1s are withheld** (0.172–0.181 S1s per S2/S3 record in test vs 0.214 in train). About 40% of test S2/S3 records have no owner in S1, versus 26% in train.

## 2. Current pipeline (run25 = best)
1. **Candidates:**
   - key blocking with a LightGBM meta-ranker (top-30);
   - a fine-tuned multilingual-e5-small dense retriever (top-10);
   - recall 0.996 on US/India.
2. **Stage 1:** LightGBM, ~60 string, key and candidate-competition features. Trained on a 200k-S1 sample.
3. **US/India stage 2:** LightGBM stack over the uncertain band. Inputs:
   - rerankers: Qwen3-0.6B LoRA, mDeBERTa-v3 and Qwen3-1.7B LoRA cross-encoders;
   - a one-to-set mDeBERTa (S1 plus its top-10 candidates in one input, band AUC 0.964);
   - graph/consensus features and generator-move features.
   The stack is trained on one reranker fold of about 44k S1s.
4. **Decision:** threshold 0.725, then one-to-one (each record goes to its highest-scoring S1).
5. **France** (the unseen country):
   - base model: LightGBM without dense features, plus word-role features learned from unlabelled France records;
   - its uncertain band is re-ranked by mDeBERTa teacher–student models trained on French pseudo-labels;
   - that is rank-fused with the set model (weight 0.4);
   - **vocabulary-swap decoys** get prob × 0.5 (`src/decoy_vocab.py`);
   - ambiguous name-only copies (exact name shared by ≥ 2 S1s) get prob × 0.5, and unique-name name-only copies are raised to ≥ 0.95;
   - threshold 0.90.

## 3. Leaderboard history (every upload)
| Run | Change | Public LB |
|---|---|---|
| run1 | key blocking + LightGBM | 0.948 |
| run2 | + dense retrieval | 0.970 |
| run3 / run4 | stricter France threshold (0.95 / 0.98) | 0.973 / 0.973302 |
| run5 | + Qwen-0.6B reranker stack (US/India) | 0.974214 |
| run7 | France: LightGBM + teacher–student reranker fusion | 0.977405 |
| run8 | France: + veto of the lowest 2% of confident matches | 0.976795 (veto hurt) |
| run10 | France: mDeBERTa students, no veto; US/India: + set model + graph | 0.979557 |
| run11 | + Qwen3-1.7B in the stack | 0.979683 |
| run12 / run13 | France fused with the set model, weight 0.2 / 0.4 | 0.980703 / 0.981046 |
| run21 | run13 + France vocabulary-swap decoys demoted | 0.984004 |
| run25 | + all vocabulary swaps + name-only ambiguity rules + US/India move-feature stack | **0.985773** |
| run28 | run25 + owner model (v1) + France threshold 0.84 | 0.985652 (worse, see §6) |

Lesson: France changes gave the big jumps (+0.003, +0.0018). US/India additions gave about +0.0001 each.

## 4. Implied score split
- Test US/India stage-1 probability distributions match train's out-of-fold ones, so US/India on test is probably ≈ 0.988 (full-sample validation).
- Then France ≈ (0.9858 − 0.85 × 0.988) / 0.15 ≈ **0.966**.
- To reach 0.988 overall, France must reach about 0.978–0.988. US/India headroom that we can validate is about +0.0004 on the leaderboard.
- **Label-free France recall deficit:**
  - predicted copies / expected copies (3.46 per S1) = 0.947 for France vs 0.970–0.973 for US/India, about 22k missing France copies;
  - France has about 38k near-miss records (best S1 prob 0.3–0.9, not predicted, not a demoted decoy) vs about 14k at the US/India rate;
  - by type: same name at a different address 17.4k, name variants 10.9k, aliases at the S1's address 6.6k (4k of them at p 0.8–0.9), name-only 3.5k.
  - Many of these are real copies, but there are no labels to tell which.

## 5. What worked (with evidence)
- Dense retrieval (+0.022 LB) and the rerankers/stack for US/India.
- **France teacher–student students** (mDeBERTa on the teacher's confident French pseudo-labels, with word dropout): band AUC 0.853 → 0.956; LB +0.0022.
- **One-to-set cross-encoder:** transfers to an unseen country (trained on US only: simulation AUC 0.951). LB +0.0010 and then +0.00034 as a France fusion.
- **France vocabulary-swap decoys:**
  - what they are: copies at the S1's address whose name keeps the stem but swaps an ordinary word ("Bordeaux Ecole SARL" → "Bordeaux Ehpad SARL"), scored ~0.99 by the unseen-country model;
  - word roles learned label-free: words over-represented in S2/S3 vs S1 names are generator insertions, the rest are vocabulary words;
  - demotion gave LB +0.003 and then more; the label-free estimate under-predicted the gain by ~1.8×.
- **Label-free "only-match" signature:** a decoy is its S1's only predicted match 5–10% of the time, a true copy 1.5–2%. Only compare classes defined by name/address features within one probability bin; across bins it is confounded.
- **Same-name owner model** (new, `src/owner.py`):
  - the finding: name-only copies whose name is shared by m S1s are not 1/m random. The generator keeps the owner's legal form (Corp / LLC / Inc / Pvt Ltd / SARL), casing and punctuation;
  - a LightGBM over each same-name group finds the owner well above chance;
  - it must be trained with ~20% of train S1s hidden (as in test), outputting an unconditional P(owner);
  - on the stack validation fold: 0.98339 → 0.98501;
  - **but that fold is enriched with hard S1s** (baseline 0.983 vs 0.988 full). The change touches 3.3% of fold S1s vs 0.9% of test S1s, so the realistic LB gain is about +0.0003–0.0005;
  - built as **run29** (only US/India name-only pairs changed; France identical to run25). **Not uploaded.**

## 6. run28 post-mortem (why +0.0027 was predicted and −0.0001 happened)
1. **Owner model v1 assumed the owner is always in the group.** True in train (97.7%), false in test (~20% of S1s withheld). Now fixed by training with hidden S1s.
2. **Refitting the stack with new features flipped 17.8k test decisions on pairs WITH addresses.** The new features only concern name-only pairs, and on validation those address pairs barely moved. Uncertain address pairs are ~3× more common per S1 on test. **Rule: when a feature concerns a subset of pairs, replace only that subset** (run29 does this).
3. **The stack validation fold overstates per-S1 gains by ~3.5×.** Scale fold gains by ~0.25–0.3 for the LB.
4. **France threshold 0.90 → 0.84** was only simulation-validated, and the simulation has no hidden owners.

## 7. What failed or was flat (measured)
- **Anything trained only on US/India text collapses on the unseen country** (band AUC):

| Model | Band AUC |
|---|---|
| Graph token model | 0.63 |
| Entity bi-encoder | 0.67 |
| Move-only teacher | 0.71 |
| LLM judge, Qwen2.5-7B zero/few-shot | 0.62 |

- Adversarial domain adaptation (DANN/GRL, MMD): collapsed.
- Synthetic-generator emulation: no gain.
- **On the France simulation, all tied or lost against the student fusion:**
  - veto (gained on the simulation, lost on LB);
  - transitive consistency;
  - selective override;
  - graph features on top of the fusion;
  - student ensembles across architectures;
  - teacher–student round 2.
- **Set-model variants:**
  - vocabulary scrambling 0.954; XLM-R-large 0.953; a second seed 0.941;
  - an ensemble reaches AUC 0.963, but F0.5 barely moves (0.9503 vs 0.9492);
  - set model trained on teacher-labelled French lists: mostly copies the teacher.
- Wider retrieval: only 13% of the simulation's missing copies are in the dense top-30.
- Count-aware "expected F0.5" selection: worse than a global threshold.
- Per-S1 "has a copy" model: AUC 0.9993, but only +0.00007.
- "No match" abstention: +0.004 on the simulation, but it fires 4× less on France.
- Tabular self-training (older simulation): Latin-only F0.5 0.904 → 0.911 in round 1. Never combined with the current France recipe.
- Feature ablations for transfer: no single dropped feature group helps much. Monotone constraints and quantile normalisation hurt.
- Hidden-owner simulation on stage 1: −0.0003 only.
- Owner-model extras: a record-side "superset" count (S1s containing the record's rarest tokens) and the owner's sibling copies' legal forms added nothing.
- ID / file-order leak check: nothing.
- Searched France for more decoy classes: word-role edit signatures × same-address × house-number relations are clean.
  - France's same-name pairs at a different address in p 0.84–0.90 show the decoy only-match signature (7.5% vs 2.7%).
  - France predicted-copy count distributions per source match US/India, and so does the alias share.

## 8. Remaining US/India loss (validation fold after the owner model; oracle gains on the fold, divide by ~3.5 for LB)
| Missed class | Oracle gain on the fold |
|---|---|
| Ambiguous exact-name name-only copies | +0.0046 (mostly genuine 50/50: same name and legal form, different states) |
| Noisy-name name-only copies (truncated or typo'd names) | +0.0036 |
| Copies with addresses: missed | +0.0024 (mostly made-up aliases at the S1's address) |
| Copies with addresses: false positives | +0.0016 (214 of 279 belong to another S1, often at the same address) |

## 9. Open ideas, ranked (none validated)
1. **France recall.** The deficit (~22k copies, §4) is the clearest label-free signal.
   - Candidates: aliases at the S1's address with p 0.8–0.9 (4k, only-match 2.7%, looks true), name-only near-misses, name variants.
   - Needs one diagnostic upload (e.g. run29 + only the alias/name-only additions) because France has no labels.
2. **In-country France pipeline from pseudo-labels:**
   - fine-tune a France dense retriever on confident France pairs (decoys as hard negatives);
   - retrain the France LightGBM with dense features on pseudo-labels, then a France stack.
   - Cost: ~4–5 GPU hours. Validation only via the Latin-India simulation, which has misled twice.
3. A stronger ≤ 8B multilingual listwise reranker (e.g. Qwen 7–8B LoRA) trained on US+India, then teacher–student on France. Band rankers are already at AUC 0.944–0.956 vs 0.964 in-domain, so the headroom may be small.
4. US/India: stack on more reranker folds (needs GPU re-scoring of train). Learning curve ≈ +0.0005 per doubling on stage 1; maybe +0.0003 LB.

## 10. Files
Code is in `src/`:
- `run_pipeline.py`: decisions and output;
- `stack.py`: US/India stack; new options `--owner`, `--owner_file`, `--owner_val`, `--out_prefix`;
- `owner.py` / `owner_apply.py`: owner model and France rule;
- `decoy_vocab.py`: France vocabulary decoys and the name-only rules;
- `fuse_set_france.py`, `fuse_v3.py`: France fusion;
- `sim_*.py`: France simulation.

Runs: `output/runs/runN_matching_results.tsv`, logged in `output/runs/RUNS.md`. Candidates ready (all pass the validator):
- **run29** (safe, est. ≈ 0.9861);
- run26 / run27 (superseded, contain the run28 problems).

The longer running log is `results/FINDINGS_SUMMARY.md`.

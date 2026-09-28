# Amazon ML Challenge 2026 — findings summary (as of 27 Sep, 05:30 IST)

## Task
- For each Source-1 (S1) business record, list all matching Source-2/Source-3 records (copies of the same business).
- Metric: macro F0.5 per S1, precision-weighted. An S1 with no true copies scores 1 only if we predict nothing.
- Train: US and India. Test: US, India, and **France (15% of test S1s, absent from training)**.
- Leaderboard weights: ≈ 0.383·US + 0.468·India + 0.150·France.
- Rules: models must be MIT/Apache-2.0 and ≤ 8B parameters. No external data. Hand-labelling test data is not allowed. Unlabelled test text is used transductively (documented).
- Deadline: 27 Sep 23:59. **3 uploads left.** About ₹400 of A100 time left (₹84/hr).
- Top of the leaderboard: ~0.991. **Our best: 0.985773 (run25).** Next goal: 0.988.

## Pipeline
1. **Candidates:** key blocking with a LightGBM ranker (top-30), unioned with a fine-tuned multilingual-e5-small dense retriever (top-10). Recall 0.996 on US/India.
2. **Stage 1:** LightGBM, about 60 string, key and competition features. Trained on a 200k-S1 sample of 2.2M train S1s.
3. **US/India stage 2:** LightGBM stack over the uncertain band (probability 0.01–0.99), using:
   - Qwen3-0.6B LoRA, mDeBERTa-v3 and Qwen3-1.7B LoRA cross-encoders;
   - a **one-to-set mDeBERTa** (S1 plus its top-10 candidates in one input, per-candidate head, band AUC 0.964);
   - entity-consensus (graph) features and generator-move features.
4. **Decision:** threshold plus one-to-one (each S2/S3 record goes to at most one S1).
5. **France:** an "unseen-country" LightGBM (no dense features, plus word-role features learned from unlabelled records).
   - Its uncertain band is re-ranked by **mDeBERTa teacher–student students**: the teacher's confident pseudo-labels on French records, with word dropout.
   - Then **rank-fused with the one-to-set model**, mapped back onto LightGBM's probability quantiles.
   - France decision threshold 0.90.

## Leaderboard history

| Run | Change | Score |
|---|---|---|
| run1 | key blocking + LightGBM | 0.948 |
| run2 | + dense retrieval | 0.970 |
| run3–4 | stricter France threshold | 0.973 |
| run5 | + Qwen-0.6B reranker (US/India) | 0.9742 |
| run7 | France: LightGBM + XLM-R students fusion | 0.977405 |
| run8 | France: mDeBERTa student + veto of the lowest 2% of confident matches | 0.976795 (the veto hurt) |
| run10 | France: LightGBM 0.1 + 2-seed mDeBERTa students 0.9, **no veto**; US/India: + set model + consensus features | 0.979557 |
| run11 | + Qwen3-1.7B in the US/India stack | 0.979683 |
| run12 | France fused with the set model at weight 0.2 | 0.980703 |
| run13 | set-model weight 0.4 | 0.981046 |
| run21 | run13 + France vocabulary-swap decoys demoted (23.8k predictions) | 0.984004 |
| run25 | run21 + all vocabulary swaps + France/US/India ambiguous name-only demoted + US/India move stack + France unique-name name-only promoted | **0.985773 (best)** |

**Lesson:** France changes gave the big leaderboard moves; US/India additions gave +0.0001 each.

## France-like simulation
- **Setup:** Latin-script India S1s, scored by a matcher and retriever trained on US only, with true labels.
- **It is calibrated for the set model:** its gains predicted the leaderboard gains (leaderboard ≈ 0.18 × the simulation's France gain).
- **Its limit:** it overrated the veto, which gained on the simulation but lost on the leaderboard.
- **Current France recipe on the simulation:** 0.9492 (weight 0.4, threshold 0.84). An in-country model on the same S1s scores 0.9755.
- **France-simulation loss breakdown:**

| Source of loss | Share |
|---|---|
| S1s with no true copy but a prediction | 22.5% |
| False positives only | 19.7% |
| Copies retrieved but not chosen (median probability 0.72) | 17.1% |
| Predicted nothing | 14.4% |
| Copies never retrieved (candidate recall only 0.972) | 20.7% |

## US/India findings (validation, true labels)
- **Stack validation (held-out S1s with uncertain pairs):** 0.98313. Full-sample validation ≈ 0.988.
- **Stack loss breakdown:**

| Source of loss | Share |
|---|---|
| Copies retrieved but missed (median probability 0.31) | 52% |
| Predicted nothing | 18% |
| False positives | 17% |
| Copies not retrieved | 8% |
| S1s with no copy but a prediction | 3% |

- **82% of missed copies are name-only records (address dropped).**
- For name-only records whose name exactly equals the S1's, **P(true) ≈ 1/m**, where m is the number of S1s in the dataset with that name:

| m | P(true) |
|---|---|
| 1 | 0.97 |
| 2 | 0.51 |
| 3 | 0.31 |
| 4 | 0.26 |
| 5–6 | 0.19–0.23 |

- So ownership among same-name S1s is effectively random, and with F0.5, predicting a ~50% candidate loses on average. **Most of the remaining US/India loss looks irreducible.**
- Other checks:
  - A per-S1 "has a copy" decision model has AUC 0.9993, but adds only +0.00007 over the stack.
  - Learning curve for stage 1: 40k → 80k → 160k S1s gives 0.98180 → 0.98261 → 0.98309, about half the gain per doubling.
  - The hidden-owner simulation (19% of S1s withheld, as test statistics suggest) changes the score by only ±0.0003.

## Generator structure (the data is synthetic)
Records are distorted copies of latent entities:
- **Noise:** accents, typos, case changes, brackets, script switches (Devanagari, Bengali, Telugu, …).
- **Address dropped:** 4.4% of true copies.
- **Name replaced by a made-up alias:** 12.9% of true copies share no name word with their S1.
- **Aliases repeat within an entity:** when an entity has ≥ 2 alias copies, they share the same alias 56.7% of the time. Aliases are nearly unique (median 1 record).
- **Decoys:** the same name with the house number shifted by a few, or the same address with another name.
- **Copies per (S1, source):** usually 1–4.

## What worked
- mDeBERTa teacher–student students for France: band AUC 0.853 → 0.956, leaderboard +0.0022 without the veto.
- **One-to-set cross-encoder:**
  - the best single reranker;
  - **it transfers to an unseen country:** trained on US only, it reaches band AUC 0.951 on the simulation;
  - fused into France: +0.0010 and then +0.00034 on the leaderboard.
- Consensus/graph features (+0.0003 in the stack), move features (+0.0002), and each extra reranker (+0.0001–0.0002) for US/India.

## What failed or was flat (measured)
- **Anything trained only on US/India text collapses on the unseen country** (band AUC):

| Model | Band AUC |
|---|---|
| Graph token model | 0.63 |
| Entity bi-encoder | 0.67 |
| Move-only teacher | 0.71 |
| Moves + probability context | 0.864 (vs LightGBM 0.853) |

- **Adversarial domain adaptation (DANN/GRL, MMD):** collapsed.
- **Synthetic-generator emulation:** no gain.
- **LLM judge (Qwen2.5-7B zero/few-shot):** AUC 0.62.
- **On the France simulation, all tied or lost against the student fusion:**
  - veto;
  - transitive (triangle) consistency;
  - selective high-confidence override (0.906–0.927 vs 0.941);
  - graph features on top of the fusion;
  - student ensembles across architectures;
  - teacher–student round 2.
- **Set-model variants on the simulation:**
  - vocabulary scrambling (delexicalised training): AUC 0.954;
  - XLM-R-large: AUC 0.953;
  - a second seed: AUC 0.941;
  - **ensemble of original + scrambling + large: AUC 0.963, but fused F0.5 only 0.9503 vs 0.9492.** Better ranking no longer turns into F0.5.
- **Set model trained on teacher-labelled French lists:** mostly copies the teacher (flips 1.3% of decisions vs 4.3%).
- **Wider retrieval:** only 13% of the simulation's missing copies are in the dense top-30.
- **Count-aware "expected F0.5" selection:** worse (0.906–0.941).
- **"No match" abstention:**
  - on the simulation, 0.9492 → 0.9534 (cross-validated);
  - but it fires 4× less on France, because France's candidate lists look much more confident than the simulation's (top probability 0.963 vs 0.866; 3.5 vs 2.4 candidates above the threshold).

## Open questions and hypotheses
- **The split of the leaderboard gap is unknown.** If US/India on test equals validation (0.988), France ≈ 0.935. But France's features look closer to in-domain, so US/India on test may be lower than on validation. No label-free way to split it has been found. Diagnostic uploads are too noisy to settle it, because the S1s with no true copies score 1.0 whenever we predict nothing, and their share is unknown.
- **The top team at 0.9906 must be ≥ 0.99 on both US/India and France.** That implies either a way to resolve name-only ownership that we haven't found (the ID/file-order leak check found nothing), or a much stronger France model.

## New (27 Sep, 04:00–05:00): France vocabulary-swap decoys
- **What:** France copies at the S1's address whose name keeps the stem but carries a different ordinary business
  word ("Bordeaux Ecole SARL" → "Bordeaux Ehpad SARL", "Tourcoing Maternelle" → "Tourcoing Amicale",
  "nmb amis" → "nmb fetes"). The unseen-country model scored ~40k of them ≈ 0.99 (≈ 3.8% of France predictions,
  > 10% of France S1s).
- **Why they are decoys:**
  - US train: same-address one-word swaps to a real vocabulary word are 23–57% true; swaps to generator-inserted
    words (services, center, partners …) are 99.9% true. France names (city/initials + generic word) make the
    decoy kind common.
  - Word roles, label-free, from the country's own records: inserted words are over-represented in S2/S3 names vs
    S1 names (France: participations, holding, distribution, associes, international, developpement, groupe,
    services, fils, france); vocabulary words are not (club, ecole, amicale, comite, ehpad, sante …).
  - **"Only-match" signature** (checked with train labels): a decoy lands on an S1 regardless of its copy count,
    so it is the S1's only match ~5–10% of the time; a true copy ~1.4–1.8%. France vocabulary-swap predictions:
    4.5% overall, 6% below prob 0.999 (other France predictions in the same bins: 2.7%). Mixture fits give
    ~60–76% false. Negative control: the US/India stack's vocabulary-swap predictions show no signature (2.0–2.9%).
  - Caveat: comparing France *probability bands* with this signature is confounded (France probabilities fall
    when an S1 has few sibling copies), so only compare classes defined by name/address features, within a bin.
- **Change:** `src/decoy_vocab.py` multiplies prob by 0.5 for France pairs whose copy name adds a vocabulary word
  (prob < 0.999): 23.8k predictions removed, 247 records reassigned by one-to-one. France predicted singletons
  0.053 → 0.059 and mean matches 3.41 → 3.32, now equal to US/India (0.058–0.059 / 3.35–3.36).
  Expected ≈ +0.0015–0.0025 on the leaderboard.
- Also: France name-only copies whose exact name is shared by ≥ 2 France S1s (train P(true) ≈ 1/m) — 4.2k
  predictions demoted with `--nameonly_m 2` (≈ +0.0002).

## Ready-to-upload candidates (all validator-PASS, in `output/runs/`)
- **run21:** run13 exactly + France vocabulary-decoy demotion. **Leaderboard 0.984004 (+0.00296 over run13; estimate was +0.0017, so the decoy share is higher than the only-match estimate).**
- **run22:** run21 + US/India move-feature stack + France ambiguous name-only demotion.
- **run23:** run22 with *all* vocabulary swaps demoted (no 0.999 cap; ≈ break-even by the estimate) — use only if
  run21 gains ≥ +0.002 over run13 (that would mean the decoy share is higher than estimated).
- **run19:** run13 + France threshold 0.84 + move features. Estimated ≈ 0.9816.
- **run20:** run19 + "no match" abstention on 740 France S1s. Estimated ≈ 0.9817–0.982.
- run14–run18: weight and ensemble variants. The simulation predicts run15 (weight 0.6) loses.

## Status after run25 (05:30) — for a second opinion
- **Implied split:** stage-1 probability distributions on test match training out-of-fold for US and India, and no
  US/India class shows the decoy signature, so US/India on test is probably ≈ validation (≈ 0.988). Then France ≈
  (0.985773 − 0.851·0.988)/0.15 ≈ **0.966** (was ≈ 0.935 at run13).
- **To reach 0.988** with US/India fixed at 0.988, France would need ≈ 0.988 (in-domain level). Alternatively
  +0.0026 on US/India (validation stack is saturated: best label-validated tweaks give +0.0001–0.0003).
- **Our label-free estimates under-predicted both France uploads by ~1.8×** (run21 est +0.0017 → +0.0030; run25 est
  +0.0010 → +0.0018). So the France model's errors are larger/more systematic than the "only-match" signature shows.
- **run25 changes, in detail (all in `src/decoy_vocab.py`):**
  - France: pairs whose copy name adds an ordinary vocabulary word (word roles from the country's own S1 vs S2/S3
    name frequencies; country name treated as an inserted word) → prob × 0.5 (39.5k predictions removed).
  - France and US/India: name-only copies (empty address) whose exact `name_core` equals this S1's and is shared by
    ≥ 2 S1s of the country → prob × 0.5 (train P(true) ≈ 1/m; US/India checked on labelled fold: +0.00029).
  - France: name-only copies whose exact `name_core` belongs to exactly one France S1 → prob ≥ 0.95 (train P ≈ 0.97;
    ~4.5k added).
  - US/India: stack with generator-move features (`_set_qwen_mdeb_q17_graph_moves`).
- **Searched and found clean (no further decoy class):** every France word-role edit signature × same-address ×
  house-number relation; France below-threshold classes (only invented-alias copies at the same address,
  ~2.2k pairs at p 0.7–0.9, look true: ≈ +0.00004); US/India test by word-role and house-number classes; France
  exact-copy retrieval (0.05% missed); share of S2/S3 records assigned (France 59.0%, US 58.4%, India 57.6%).
- **Diagnostic caveats:** the only-match signature is confounded across probability bands (France probabilities drop
  when an S1 has few sibling copies; US shifted-house-number true copies also show 7% only-match at 0.92 precision),
  so only compare classes defined by name/address features, within the same probability bin.

### Ideas not yet tried (ranked by my expectation)
1. **Use the France errors we now know about to retrain France's model.** Pseudo-label France with run25's
   decisions (confident positives; vocabulary-swap decoys and same-name-other-address pairs as hard negatives) and
   train a France-only LightGBM/stack on France features. The prior teacher–student round used run7-era labels,
   which contained the ~40k decoys as positives.
2. **"Wrong owner" errors in France:** templated names give many same-name S1s in one city; copies at another
   address are assigned by 1:1 on probability. A record-side model (which of the same-name S1s owns this copy)
   using address similarity only among same-name S1s could fix these; the only-match signature cannot see them.
3. **France name-only copies with noisy names (m = 0):** 22.5k best-for-record pairs, only 674 predicted. The US/
   India stack's biggest miss bucket is the same; France may be under-predicting even more.
4. **Recalibrate France probabilities per S1 context:** France probabilities fall when an S1 has few sibling copies
   (set-model fusion / rank features), which leaves real copies of low-copy S1s at 0.9–0.999 and may push some
   below 0.9.
5. US/India: more stack training data (stack uses one reranker fold ≈ 44k S1s; stage 1 uses 200k of 2.2M S1s;
   learning curve ≈ +0.0005 per doubling) — needs GPU re-scoring, ≈ +0.0003 LB.

## New (27 Sep, after run25): same-name owner choice — the name-only ambiguity is partly resolvable
- **Finding:** name-only copies (empty address) whose name is shared by m >= 2 S1s were treated as P(true) ~ 1/m.
  But the generator keeps the owner's **legal form** (Corp / LLC / Inc / PC / Pvt Ltd / SARL ...), casing and
  punctuation. A LightGBM over each same-name group (`src/owner.py`, trained on train truth, 5 folds by record)
  picks the owner with top-1 0.37 vs 1/m 0.22 over all groups; where it is >= 0.8 sure (24k groups) it is 96% right.
  Remaining ~50/50 groups are genuinely indistinguishable (same name and legal form, different states).
- **US/India:** owner features (`own_pn`, group size, exact-name flag, legal-form matches) added to the stage-2 stack
  (`stack.py --owner`): labelled stack fold **0.98313 -> 0.98563** (+0.0025; run25's halving rule gave +0.0003).
  Oracle headroom left in this class: +0.0046 (mostly irreducible 50/50 groups).
- **France:** label-free rule `src/owner_apply.py`: exact-name ambiguous copies -> own_pn; noisy-name ones ->
  min(1, prob*m)*own_pn (rule on the US/India fold: 0.98481). France-like simulation: +0.0009 at t 0.90.
- **France threshold:** on the simulation, with the owner rule, t 0.84 scores 0.95159 vs 0.94738 at t 0.90. On France
  test, the pairs 0.84-0.90 are alias copies at the S1's address (3.1k, only-match 2.7%), name-only (0.8k, 2.8%),
  name variants (1.0k, 3.7%) and same-name copies at a DIFFERENT address (0.9k, **7.5%** = decoy signature) ->
  the latter keep t 0.90.
- Tried, no gain: record-side "superset" count (S1s containing the record's rarest tokens): stack fold 0.98563 either
  way. Owner's sibling-copy legal forms: mostly already captured by own_pn.
- **Candidates:** run26 = owner stack + France owner rule (t 0.90), estimate ~0.9876-0.988. run27 = run26 + France
  t 0.84. **run28 = run26 + France t 0.84 except same-name/different-address pairs (recommended), estimate ~0.988-0.9885.**

## Post-mortem of run28 (LB 0.985652, below run25 0.985773) and run29
- **Test hides ~20% of S1s:** test has 0.172-0.181 S1s per S2/S3 record vs 0.214 in train, so ~20% of the owners
  of test copies are not in S1. The first owner model assumed the owner is always in the same-name group (true in
  train, 97.7%). Fixed: `owner.py` now trains with 20% of unscored train S1s hidden and outputs an unconditional
  P(owner). (On the fold the old model barely suffered from this, so it is not the whole story.)
- **Refit noise on pairs validation cannot see:** adding features to the stack changes its trees everywhere. On test,
  US/India pairs WITH addresses flipped 17.8k decisions (12.9k lost / 4.9k gained); on the fold such pairs barely
  moved. Uncertain address pairs are ~3x more common per S1 on test than on the fold (hidden owners leave more
  ambiguous records). Lesson: when a new feature only concerns a subset of pairs, only replace that subset.
- **The stack fold overstates per-S1 gains ~3.5x:** it holds S1s with uncertain pairs (baseline 0.983 vs ~0.988
  full). run29's change touches 3.3% of fold S1s but 0.9% of test US/India S1s. Scale fold gains by ~0.25-0.3 for LB.
- France threshold 0.84 (run27/28) was only simulation-validated; the simulation has no hidden owners either.
- **run29** = run25 exactly, except US/India name-only pairs take the hidden-owner-aware owner stack's scores
  (fold: 0.98339 -> 0.98501; only name-only pairs change: 13.3k US/India S1s; France identical to run25).
  Realistic LB estimate: run25 + ~0.0003-0.0005 ≈ 0.9861.

# Amazon ML Challenge 2026 — handoff v2 (27 Sep 2026, ~19:00 IST)

Supersedes the "what's next" parts of `results/HANDOFF_SUMMARY.md` (read that first for task, pipeline, and history up to run29).
Deadline 27 Sep 23:59 IST. **Best public LB: 0.986314 (run31).** Goal was > 0.988.
Uploads left: 2 (as of 19:00). Jarvis credit left: ~₹56 (A100 at ₹84/hr).

---

## 1. Leaderboard results

| Run | Content | Public LB |
|---|---|---|
| run25 | previous best (see v1 handoff) | 0.985773 |
| run30 | run29 + owner model trained **without** hiding S1s, plus entity-sibling features; France owner rule | not uploaded |
| run31 | run30 + France students retrained on cleaned pseudo-labels; shifted-house-number pairs keep run30 scores | **0.986314 (best)** |
| run32 | run31 + 1,944 France alias/acronym copies at the S1's exact house number, prob 0.80–0.90, promoted to 0.901 | built, not uploaded (estimate +0.0001–0.0003) |

All files are in `output/runs/runN_matching_results.tsv`, logged in `output/runs/RUNS.md`, and pass the validator.
The final zip must contain the chosen run's `matching_results.tsv` plus `output/candidate_pairs.tsv`.

---

## 2. Findings from 27 Sep (new, with evidence)

### 2.1 Test is NOT missing ~20% of S1s (the v1 handoff assumption was wrong)
- Name-only (empty-address) S2/S3 records per S1:
  - test: US 0.166, India 0.138, France 0.166;
  - train: US 0.168, India 0.139.
- Name-only records are ~98% true copies (train), so their owners are present in test S1.
- The extra test records (5.76 vs 4.67 S2/S3 records per S1) carry addresses and are unrelated to any S1 (best stage-1 prob < 0.01). They do not affect decisions.
- Consequences:
  - run29's hidden-S1 owner model was needlessly conservative. Retrained with `--hide 0`: fold 0.98500 → 0.98565.
  - Don't simulate hidden owners.

### 2.2 US/India: where the remaining loss is (labelled stack fold, 43.6k S1s, after owner stack)
- Fold oracle gains:
  - exact-name name-only copies with the name shared by m ≥ 2 S1s: +0.0052 (recall 0.20, precision 0.80);
  - noisy-name name-only copies, m ≥ 2: +0.0031;
  - noisy-name, m = 1: +0.0012;
  - all address pairs: +0.0040 (recall 0.993, precision 0.998).
- Full 200k-sample stage-1 oracle: name-only pairs +0.0084, address pairs +0.0077.
- Same-name ownership cues tested on train labels:
  - legal forms are mostly per-copy generator noise (one entity's copies carry LP / Corp / Ltd randomly);
  - owner model with legal form, casing and script: top-1 0.36 vs chance 0.22;
  - adding entity-sibling features (per-source copy counts, script mix of the member's copies, legal/name matches) gives top-1 0.41, but confident picks barely grow (p ≥ 0.8: 22.3k → 23.0k correct);
  - a member with fewer same-source address copies owns the copy 55–65% of the time; members with zero copies own it 25%;
  - **no ID or file-order leak:** a LightGBM on ID digits, row positions, differences and moduli gives within-group top-1 0.421 vs chance 0.416;
  - no sibling raw-name identity: a name-only copy's raw name equals one of the owner's other copies' names 6% vs 4% for competitors.
- Decision layer is saturated: a lower threshold for S1s with no confident copy gives +0.00017 on the fold.
- US/India test ≈ validation:
  - model-implied expected F0.5 on test US 0.9839, India 0.9829 vs train 0.9842 (actual 0.9829);
  - predicted copies per S1 are the same (3.36);
  - test has 1.1–1.7× more low/mid-band address pairs per S1 (the extra decoy-like records), ≈ −0.0005 at most.

### 2.3 France: every label-free check came back clean
- **Added words:**
  - train US/India: same-address insertions of Center/Services/Service (/Partners) are 97–99% true; Group/Holdings/Enterprises/Trading/Associates/Ventures/Industries/International/Sons/Brothers/country name are ~0% true even at the same address;
  - France groupe, developpement, france, fils and associes behave like TRUE insertions: the S1s-with-≥2 share is 1.2–2.6%, like services (1.8%), vs single-record decoys at 0.0–0.1%; only-match rate 2.1–2.2%, the same as all France predictions;
  - France holding, participations, international and distribution are the decoy words and are already ~never predicted;
  - **the in-domain US/India stack is wrong on French text** (it rejects Groupe/Développement insertions and accepts vocabulary swaps). Don't use it for France.
- **Vocabulary swaps:** train confirms same-address vocab-to-vocab swaps are ~0% true (true "swaps" are typos, abbreviations, plural/singular). The existing France demotion is right; "center" is a real vocabulary word in France (10.5k S1 names).
- **Acronyms:** France predicts 62 per 1,000 S1s at the same house number vs US 2.8. The codes exactly match S1 initials (skipping "(France)"), and the ≥2-per-S1 share scales with the rate. Train precision for acronyms at the same address is 0.994. They look real.
- **House numbers:**
  - France "digit-drop" candidates are mostly same-name businesses on OTHER streets (templated names), correctly low;
  - France predicts far fewer different-number copies than US (24 vs 314 per 1,000), mostly because US different-number copies are unit-number/PO-box artifacts and digit drops.
- **Near-threshold mass is thin:** only ~11k best-for-record France pairs in 0.7–0.9. Even if all were true, adding them is worth ≈ +0.0005 LB, so thresholds cannot close the gap.
- **Retrieval:** ~24k France S2/S3 records are in no candidate list; ~1k match an S1 address exactly, several clearly real (aliases, acronyms, domains). Worth ≈ +0.00003.
- **Label-free decoy-share estimator (size-biased copy-count mixture):** fails validation on train (estimates 0.35 where the true FP rate is 0.003). Don't use it.
- **Implied split:** US/India ≈ 0.988–0.989 on test, so France ≈ 0.97. France's loss seems spread over its ~850k confident predictions, and no label-free signal found it.

### 2.4 France students retrained (run31)
- **Bundle `work/da_france_v5`:** v3 source/target/texts plus new `tgt_pseudo`:
  - 150k positives from run30's final France decisions (≥ 0.97, best for record);
  - 149k negatives (≤ 0.03, half in 0.001–0.03);
  - the 40,006 demoted vocabulary-swap decoys as hard negatives.
- **Training:** `da_encoder.py --model microsoft/mdeberta-v3-base --da none --pseudo_w 1.0 --aug 0.1 --epochs 1 --bs 32 --lr 2e-5`, seeds 42 and 7, on a Jarvis A100 (~29 min per seed).
- **Outputs:** `work/fr4_c42_*`, `work/fr4_c7_*`.
- **Fusion (identical to run25's recipe, verified to reproduce run25's France scores exactly):**
  - `fuse_v3.py --mode france --lgbm test_scored_unseen_roles.parquet --students c42,c7 --w_lgbm 0.1`
  - → `fuse_set_france.py --w_set 0.4`
  - → `decoy_vocab.py --max_prob 2 --nameonly_m 1000000 --promote_unique 0.95`
  - → `owner_apply.py --owner owner_test_unseen_sib0.parquet`
  - → `src/merge_students_shift.py`: pairs with a different house number keep run30's scores.
- **Why the shift filter:** the new students also raised ~1.4k same-name shifted-number pairs, and train P(true) for small shifts is 0.34–0.51.
- **Result:** France +3,497 / −458 decisions vs run30, mostly aliases at the S1's exact address. LB 0.986314.

---

## 3. Open ideas (none validated; realistic sizes)
1. **Upload run32** (+0.0001–0.0003). If it gains, extend the alias promotion to 0.70 (~580 more pairs).
2. **US/India stack on 3× more S1s.** Only the set model is missing out-of-fold scores beyond CE fold 0. It needs 2 more set-model trainings on GPU, then `stack.py` without `--restrict`. Learning curve suggests ≈ +0.0002 LB. 3–4 h, risky tonight.
3. **Second France teacher–student round** from run31 labels. Earlier rounds on the simulation tied, expect small.
4. **Same-name name-only ownership** is the largest pool (≈ +0.004–0.007 on US/India if solved), but every cue found so far is weak. A better idea is needed, not more of the same features.

---

## 4. Practical notes
- **Jarvis (A100):**
  - driven through its Jupyter server with `jr.py`, in the session scratchpad (a copy is at `C:\Users\heman\AppData\Local\Temp\claude\c--Users-heman-Amazon-ML-Challenge-2026\4f9253ed-...\scratchpad\jr.py`); the token is in `jarvis_token.txt` next to it;
  - host: `7193145164011.notebooksn.jarvislabs.net`;
  - usage: `python jr.py <host> run "<cmd>" <timeout>` / `upload <local> <remote>` / `fetch <remote> <local>`;
  - in Git Bash set `MSYS_NO_PATHCONV=1`, or remote `/home/...` paths get rewritten to Windows paths;
  - **the client retries on network errors: never launch a long job through `run` without `setsid`/`nohup` AND a guard, or a retry starts a duplicate and kills the first when its kernel is closed**;
  - pausing wipes everything outside `/home` (pip packages too); work lives in `/home/s2`;
  - Windows-made zips have backslash paths; extract with a script that replaces `chr(92)` with `/` (`/home/s2/fixzip.py`).
- **Kaggle:** the key in `~/.kaggle/kaggle.json` (user heyyhema) is rejected by the API (authentication fails).
- **Disk:** C: was full at one point; the user has since freed space.
- **Code added today:**
  - `src/owner.py`: `--sib`, `--sib_pairs`, `--hide 0`;
  - `src/merge_students_shift.py`;
  - `src/exp_owner_sib.py`, `src/exp_kmix.py` (experiments);
  - `jarvis/run_students.sh`.
- **Key work files:**
  - `work/owner_*_sib0.parquet`;
  - `work/v2_test_scored_stage2_set_qwen_mdeb_q17_graph_moves_owner.parquet`;
  - `work/test_scored_stage2_hyb2.parquet` (US/India scores used by run30–32);
  - `work/test_scored_unseen_c_shiftkeep.parquet` (run31 France);
  - `work/test_scored_unseen_c_alias80.parquet` (run32 France).
- **Rebuild a run:** `python src/run_pipeline.py --reuse --stage2 --stage2_tag _hyb2 --unseen_scores ../work/<france file> --unseen_t 0.9 --note "..."`, run from `src/`.
- **None of today's code is committed to git.**

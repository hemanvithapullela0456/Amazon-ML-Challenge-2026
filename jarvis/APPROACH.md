# Model roadmap: what to try, in order

Scoring is precision-heavy (macro F0.5), and France appears only in the test set. A model has to **read** the two records,
not just count overlapping characters. Transformers do that, and multilingual ones help with French.

| Phase | Approach | Where | Why | Expected gain |
|---|---|---|---|---|
| 0 | LightGBM on string features (done) | Kaggle CPU | strong, fast baseline | baseline |
| **1** | **Cross-encoder fine-tuning** (`microsoft/mdeberta-v3-base`, MIT): reads "name \| address" of both records together → match score | Jarvis GPU | learns abbreviations, transliterations and landmark phrases that fuzzy scores miss; multilingual, so it covers French | usually the biggest jump |
| **2** | **Stacking:** cross-encoder score (+ its rank/gap within the S1's candidates) becomes an extra LightGBM feature → re-tune the F0.5 decision | CPU | combines both; the decision rule stays calibrated | +1–3 pts over phase 1 alone |
| 3 | Bigger / LLM reranker via LoRA: `Qwen/Qwen2.5-1.5B` or `Qwen2.5-7B` (Apache-2.0, ≤ 8B) as a sequence classifier | Jarvis A100 | more world knowledge (brand/DBA names); only if phase 2 plateaus | maybe +0.5–2 |
| 4 | Better blocking with dense retrieval (`intfloat/multilingual-e5-base`, MIT) | GPU | only if `blocking.py` shows recall < ~98% | raises the ceiling |

License notes: mDeBERTa-v3 (MIT), XLM-R (MIT), multilingual-e5 (MIT), Qwen2.5-0.5B/1.5B/7B (Apache-2.0).
**Qwen2.5-3B is NOT Apache**, so avoid it.

## How the pieces connect
```
train.py (stage 1 LightGBM)  ->  work/train_oof.pkl, model.txt
cross_encoder.py             ->  scores the top-K pairs per S1 (by stage-1 prob)
                                 train: K-fold out-of-fold scores | test: average of fold models
                                 -> work/ce_train.pkl, work/ce_test.pkl
train.py (again)             ->  picks up the ce_* files automatically as features (stage 2)
run_pipeline.py              ->  output/*.tsv
```
Pairs outside the top-K get no cross-encoder score (NaN), which LightGBM handles natively. That keeps GPU time bounded:
cost ≈ (#S1 × K) pairs per pass.

## Experiment log (fill in as you go)
| Run | Setup | OOF F0.5 | Public LB |
|---|---|---|---|
| baseline | similarity threshold | | |
| lgb v1 | stage 1 | | |
| ce v1 | mdeberta-v3-base, 3 folds, 2 epochs, K=10 | | |
| stack v1 | lgb + ce | | |

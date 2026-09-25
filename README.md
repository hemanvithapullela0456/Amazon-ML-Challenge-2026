# Business Entity Resolution — Amazon ML Challenge 2026

For each Source-1 business, find the matching Source-2/3 records. Scored by macro F0.5 per S1 entity.
Dataset: 12.5M train / 11.7M test records (US, India; test also has France, unseen in training).

## Pipeline
```
prep (src/prep.py)              normalise text + build blocking keys for a whole split -> work/<split>_norm/
      -> blocking (src/blocking.py)     inverted-index key search + a small learned "meta-blocking" ranker
      -> features (src/features.py)     ~60 pairwise similarity / rank features
      -> train (src/train.py)           LightGBM, grouped 5-fold CV, F0.5 decision-rule tuning (src/decide.py)
      -> run_pipeline (src/run_pipeline.py)   scores the test set, writes output/*.tsv, runs the validator
```
Optional stage 2 (GPU, `src/dense.py` + `src/cross_encoder.py` + `src/stack.py`): a fine-tuned multilingual
bi-encoder adds candidates the key search misses, and a cross-encoder / small LLM reranks pairs the LightGBM
model is unsure about; `stack.py` blends its score back in and re-tunes the decision rule.

## Reproduce
```bash
pip install -r requirements.txt
# dataset expected at student_resource/dataset/{train,test}/  (or set ER_DATA_DIR, e.g. D:\student_resource\dataset)

python src/eda.py                          # data checks (one-to-one, same-country assumptions)
python src/prep.py --split train           # normalise + build blocking keys (~4 min)
python src/prep.py --split test            # same for test (~6 min)
python src/blocking.py --split train --sample 0.1 --train-ranker   # train the meta-blocking ranker + recall report
python src/train.py --country-check        # candidate search + features + CV + decision tuning -> work/model.txt
python src/run_pipeline.py                 # test predictions -> output/*.tsv (+ runs the official validator)
python src/check_ids.py                    # lightweight check that every output ID exists in the test set
```
Settings (blocking key caps, one-to-one assignment, LightGBM params, seed) live in `src/config.py`.
`ER_WORK_DIR` / `ER_OUT_DIR` override where intermediate files and the final output are written.

Memory note: `run_pipeline.py` processes the test set in batches (`--chunk`, default 15,000 businesses)
and checkpoints each batch to `work/test_scored_parts/`, so an interrupted run resumes instead of restarting.

Running on AWS: see [aws/AWS_SETUP.md](aws/AWS_SETUP.md). Running the GPU stage on Kaggle: see
[kaggle/KAGGLE_SETUP.md](kaggle/KAGGLE_SETUP.md) and [kaggle/THREE_TRACKS.md](kaggle/THREE_TRACKS.md).

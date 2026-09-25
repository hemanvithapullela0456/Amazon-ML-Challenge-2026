# Three Kaggle accounts, three model families

Every track is judged with the **same data split and the same metric**, so the comparison is fair and fast:
- Retrieval (A): recall of true matches on the laptop's held-out 10% of train S1 (`python src/dense.py eval`).
- Reranking (B, C): the same validation S1s (CE fold 0 of the uncertain pairs), scored by
  `python src/stack.py --suffix <tag>`, which prints **stage-1 F0.5 → stage-2 F0.5** on exactly those S1s.
- France stand-in (B, C): train on **US only**, validate on **India only** (`--train_country us --eval_country india`).

| Account | Track | Model (license) | Fixes which weakness | Research basis |
|---|---|---|---|---|
| **A** | Dense retrieval (bi-encoder), unioned with key blocking | `intfloat/multilingual-e5-small` (MIT, 118M) | recall ceiling (94.3%), non-Latin names | R-SupCon (Peeters & Bizer 2022), DeepBlocker (2021), Sparkly (2023) |
| **B** | Cross-encoder reranker (Ditto-style) | `FacebookAI/xlm-roberta-base` (MIT, 278M) | classifier errors on hard pairs | Ditto (Li et al. 2021); cross-encoders beat bi-encoders for matching |
| **C** | Small LLM reranker, LoRA + classification head | `Qwen/Qwen3-0.6B` (Apache-2.0) | unseen country (France) | fine-tuned LLM matchers transfer better under distribution shift (EDBT 2025; arXiv 2409.08185, 2607.24688) |

Laptop (CPU, in parallel): stage-1 test submission + "collective" features (a candidate that resembles the S1's
other confident matches is likely a match too; safe because stage-1 precision is > 0.9).

---
## Uploads (once)
- `er-code` dataset: the `src` folder (re-upload as a new version after code changes).
- `amazon-ml`: already uploaded (only account A needs it).
- `er-dense-input` (account A): `work\dense_export` from `python src\dense.py export`.
- `er-ce-bundle` (accounts B, C): `work\ce_bundle` from `python src\cross_encoder.py --export`.
Share each dataset with your other two Kaggle accounts (dataset → Settings → Sharing), so you upload only once.

Notebook settings for all three: **GPU T4 x2**, **Internet On**. Long runs: **Save Version → Save & Run All**.

---
## Account A — dense retrieval
Follow `kaggle/KAGGLE_SETUP.md` (cells 1–3). Download `work/dense_train.parquet` → laptop `work\` →
`python src\dense.py eval`.

---
## Accounts B and C — rerankers
### Cell 1 (both)
```python
!pip install -q -U "transformers>=4.51" peft polars unidecode sentencepiece
import os, glob, shutil
shutil.copytree(os.path.dirname(glob.glob("/kaggle/input/**/cross_encoder.py", recursive=True)[0]), "/kaggle/working/src", dirs_exist_ok=True)
BUNDLE = os.path.dirname(glob.glob("/kaggle/input/**/train_pairs.parquet", recursive=True)[0])
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)
os.environ["ER_WORK_DIR"] = "/kaggle/working/work"
print(BUNDLE, os.listdir(BUNDLE)); !nvidia-smi --query-gpu=name,memory.total --format=csv
```

### Account B — cell 2 (smoke, ~3 min) then cell 3 (commit)
```python
M = "FacebookAI/xlm-roberta-base"
!cd /kaggle/working && python src/cross_encoder.py --bundle {BUNDLE} --out_dir {OUT} --model {M} --smoke --bs 32
```
```python
!cd /kaggle/working && python src/cross_encoder.py --bundle {BUNDLE} --out_dir {OUT} --model {M} --eval_only --epochs 2 --bs 32 --lr 3e-5 --max_train 600000 --tag _xlmr
!cd /kaggle/working && python src/cross_encoder.py --bundle {BUNDLE} --out_dir {OUT} --model {M} --eval_only --epochs 2 --bs 32 --lr 3e-5 --max_train 600000 --train_country us --eval_country india --tag _xlmr_us2in
```

### Account C — cell 2 (smoke) then cell 3 (commit)
```python
M = "Qwen/Qwen3-0.6B"
!cd /kaggle/working && python src/cross_encoder.py --bundle {BUNDLE} --out_dir {OUT} --model {M} --lora --smoke --bs 16 --lr 2e-4
```
```python
!cd /kaggle/working && python src/cross_encoder.py --bundle {BUNDLE} --out_dir {OUT} --model {M} --lora --eval_only --epochs 1 --bs 16 --lr 2e-4 --max_train 300000 --tag _qwen
!cd /kaggle/working && python src/cross_encoder.py --bundle {BUNDLE} --out_dir {OUT} --model {M} --lora --eval_only --epochs 1 --bs 16 --lr 2e-4 --max_train 300000 --train_country us --eval_country india --tag _qwen_us2in
```
Smoke must end with `saved scores ...` and a finite loss (if the loss is `nan`, fp16 overflowed: re-run with
`--model Qwen/Qwen2.5-0.5B`).

### Back on the laptop
Download `out/ce_train_eval_*.parquet` into `work\`, then for each tag:
```powershell
python src\stack.py --suffix _eval_xlmr
python src\stack.py --suffix _eval_xlmr_us2in
python src\stack.py --suffix _eval_qwen
python src\stack.py --suffix _eval_qwen_us2in
```
Record the numbers in `kaggle/EXPERIMENTS.md`. Only the winner gets a full run (all folds + test scoring).

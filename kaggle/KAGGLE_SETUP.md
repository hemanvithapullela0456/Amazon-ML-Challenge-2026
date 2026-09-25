# Kaggle track: dense retrieval (fine-tuned multilingual bi-encoder)

What this does and why: see `src/dense.py` docstring and the plan. The laptop owns keys, labels and evaluation;
Kaggle only needs the raw data (already uploaded as `amazon-ml`), the code, and a tiny export.

## 0. On the laptop (after `train.py` has finished)
```powershell
python src\dense.py export          # -> work\dense_export\triples.parquet + train_queries.parquet (small)
```
Zip for upload, then create **two private Kaggle datasets** (kaggle.com → Create → New Dataset):
- `er-code`: the `src` folder
- `er-dense-input`: the `work\dense_export` folder

## 1. Notebook settings
New Notebook → Add Input: `amazon-ml`, `er-code`, `er-dense-input`.
Settings → **Accelerator: GPU T4 x2**, **Internet: On**, Persistence: off.
For long steps use **Save Version → Save & Run All** so they run in the background (up to 12 h).

## 2. Cells
### Cell 1 — setup
```python
!pip install -q -U "sentence-transformers>=3.0" polars unidecode
import os, glob, shutil
shutil.copytree(os.path.dirname(glob.glob("/kaggle/input/**/dense.py", recursive=True)[0]), "/kaggle/working/src", dirs_exist_ok=True)
DATA = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/train_source1.tsv", recursive=True)[0]))
EXP = os.path.dirname(glob.glob("/kaggle/input/**/triples.parquet", recursive=True)[0])
os.environ.update(ER_DATA_DIR=DATA, ER_WORK_DIR="/kaggle/working/work")
os.makedirs("/kaggle/working/work", exist_ok=True)
EMB = "/kaggle/temp"   # big embedding files live here (not saved as output)
os.makedirs(EMB, exist_ok=True)
print(DATA, EXP); !nvidia-smi --query-gpu=name,memory.total --format=csv
```

### Cell 2 — smoke test (~5 min). Must print a falling loss and finish.
```python
!cd /kaggle/working && python src/dense.py train --export_dir {EXP} --out_dir /kaggle/working/smoke --smoke
!cd /kaggle/working && python src/dense.py embed --split test --model_path /kaggle/working/smoke/dense_model --emb_dir {EMB} --smoke
!cd /kaggle/working && python src/dense.py knn --split test --emb_dir {EMB} --out_dir /kaggle/working/smoke --smoke
```

### Cell 3 — real run (commit / "Save & Run All"): train, then train-split retrieval for evaluation
```python
!cd /kaggle/working && python src/dense.py train --export_dir {EXP} --out_dir /kaggle/working/work
!cd /kaggle/working && python src/dense.py embed --split train --emb_dir {EMB}
!cd /kaggle/working && python src/dense.py knn --split train --emb_dir {EMB} --queries {EXP}/train_queries.parquet
!rm -f {EMB}/emb_train.npy
```
Outputs: `work/dense_model/` (reuse as a dataset later) and `work/dense_train.parquet` (~100 MB).

## 3. Back on the laptop — the decision gate
Download `dense_train.parquet` into `work\`, then:
```powershell
python src\dense.py eval
```
It prints recall of dense top-K, key top-30, and their union, per country. Continue to the test split and
the merge only if the union beats key-only (94.3%) by about 1 point or more, or India clearly improves.

## 4. Test split (only after the gate passes)
```python
!cd /kaggle/working && python src/dense.py embed --split test --emb_dir {EMB}
!cd /kaggle/working && python src/dense.py knn --split test --emb_dir {EMB} --k 20
```

## Optional: France-transfer proxy
Train on US clusters only (`--country us`), then run `eval`: India recall shows how the model does on a country
it never saw.

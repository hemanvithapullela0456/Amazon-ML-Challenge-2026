# Jarvislabs: cross-encoder only

Only the cross-encoder runs on the GPU. Everything else (blocking, features, LightGBM, stacking, writing the
submission) runs on the laptop. You upload a small **bundle** (uncertain pairs + the text of the records they
involve) and download two small score files.

## 0. On the laptop, before starting any instance
```powershell
python src\train.py                       # stage 1 (already done if work\train_oof.parquet exists)
python src\run_pipeline.py                # stage-1 submission + work\test_scored.parquet
python src\cross_encoder.py --export      # -> work\ce_bundle\  (prints its size)
```
Zip two things:
- `work\ce_bundle` → `ce_bundle.zip`
- `src` + `jarvis` + `requirements.txt` → `er_code.zip`

## 1. Start the instance (only when both zips are ready)
Dashboard → Templates → **PyTorch** → GPU **A100 (40 GB, ₹84/hr)** → storage 50 GB → Launch → open **JupyterLab**.

## 2. Upload + install (JupyterLab → drag both zips into the file browser, then Terminal)
```bash
cd ~ && unzip -o er_code.zip -d er && unzip -o ce_bundle.zip -d ~
ls ~/ce_bundle                          # train_pairs / test_pairs / train_texts / test_texts .parquet
cd ~/er && pip install -q -r jarvis/requirements-gpu.txt
```

## 3. Run (inside tmux, so closing the browser doesn't kill it)
```bash
tmux new -s er
bash jarvis/run_gpu.sh smoke     # must finish with "saved ..." and show a CE AUC
bash jarvis/run_gpu.sh ce        # real run; progress lines show pairs/s and time per fold
```
Detach with `Ctrl+b`, `d`; reattach with `tmux attach -t er`.

## 4. Download and PAUSE
In JupyterLab, right-click `ce_bundle/ce_train.parquet` and `ce_bundle/ce_test.parquet` → Download.
**Pause the instance immediately.**
Put both files in the laptop's `work\` folder, then:
```powershell
python src\stack.py                          # prints stage-1 vs stage-2 F0.5 on validation
python src\run_pipeline.py --reuse --stage2  # writes output\ and runs the validator
```
Submit stage 2 only if `stack.py` shows it beats stage 1.

## Cost control
- The smoke test prints throughput (pairs/s). Multiply by the pair counts from `--export` to estimate the real run
  before starting it. If it would exceed ~3 h, add `CE_ARGS="--folds 2 --epochs 1 --max_train 800000"`.
- Nothing is debugged on the running instance: if smoke fails, pause, copy the error to Claude, fix locally.

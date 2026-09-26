"""Fine-tune a transformer cross-encoder on the pairs stage-1 is unsure about. Needs a GPU.

Inputs (produced by train.py and run_pipeline.py):
  work/train_oof.parquet  (i, j, label, prob, fold)  +  work/train_norm/  (raw text)
  work/test_scored.parquet (i, j, prob)              +  work/test_norm/
Pairs sent to the model: stage-1 prob in [--lo, --hi], plus each S1's top --k pairs if --k > 0.

    python src/cross_encoder.py --model microsoft/mdeberta-v3-base --folds 3 --epochs 2
    python src/cross_encoder.py --smoke          # 2-minute end-to-end test on a few thousand pairs
Writes work/ce_train.parquet (out-of-fold scores) and work/ce_test.parquet (fold-averaged): i, j, ce.
Then:  python src/stack.py
"""
import argparse
import os
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

try:  # GPU-side dependencies; not needed for --export on the laptop
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError:
    torch = None

import config as C
from prep import load_norm


def export_bundle(a):
    """LOCAL (CPU): write the small upload bundle work/ce_bundle/ for the GPU box:
    selected train/test pairs + the text of only the records they reference."""
    out = os.path.join(C.WORK_DIR, "ce_bundle")
    os.makedirs(out, exist_ok=True)
    for split, fname in [("train", "train_oof.parquet"), ("test", "test_scored.parquet")]:
        if not os.path.exists(os.path.join(C.WORK_DIR, fname)):
            print(f"{split}: {fname} not found - skipped (evaluation-only bundle)")
            continue
        pairs = select_pairs(pl.read_parquet(os.path.join(C.WORK_DIR, fname)), a.lo, a.hi, a.k)
        norm = (load_norm(split, columns=["business_name", "business_address", "country", "country_key"])
                .with_row_index("r").with_columns(pl.col("r").cast(pl.UInt32)))
        pairs = pairs.join(norm.select(pl.col("r").alias("i"), "country_key"), on="i", how="left")
        cols = ["i", "j", "prob", "country_key"] + (["label"] if split == "train" else [])
        pairs.select(cols).write_parquet(os.path.join(out, f"{split}_pairs.parquet"))
        need = pl.concat([pairs["i"], pairs["j"]]).unique()
        norm = norm.join(need.to_frame("r"), on="r", how="inner")
        norm.select("r", (pl.col("business_name").str.strip_chars() + " | " + pl.col("business_address").str.strip_chars()
                          + " | " + pl.col("country")).alias("text")).write_parquet(os.path.join(out, f"{split}_texts.parquet"))
        print(f"{split}: {pairs.height} pairs, {len(need)} records")
    mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
    print(f"bundle -> {out} ({mb:.0f} MB): upload this folder to the GPU machine")


def rec_texts(bundle, split):
    t = pl.read_parquet(os.path.join(bundle, f"{split}_texts.parquet"))
    return dict(zip(t["r"].to_list(), t["text"].to_list()))


def select_pairs(df, lo, hi, k):
    sel = (pl.col("prob") >= lo) & (pl.col("prob") <= hi)
    if k > 0:
        sel = sel | (pl.col("prob").rank("ordinal", descending=True).over("i") <= k)
    return df.filter(sel)


def native_bf16():
    """True only on Ampere or newer. torch.cuda.is_bf16_supported() also counts *emulated* bf16, which is
    True on a T4 but far slower than fp16."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def load_model(a, dev):
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # T4 (Kaggle) has no bf16: keep fp32 master weights and train under fp16 autocast there
    bf16 = native_bf16()
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=1, torch_dtype=torch.bfloat16 if (a.lora and bf16) else torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    if a.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                                                 target_modules="all-linear"))
        model.print_trainable_parameters()
    return tok, model.to(dev)


def make_loader(tok, texts, ia, ja, y, bs, max_len, shuffle, decoder=False):
    idx = np.arange(len(ia))
    if not shuffle:  # length-sorted batches for fast inference
        idx = np.argsort([len(texts[a]) + len(texts[b]) for a, b in zip(ia, ja)], kind="stable")

    def collate(batch):
        if decoder:  # decoder tokenizers add no separator between a text pair, so spell the pair out
            enc = tok([f"Record A: {texts[ia[n]]}\nRecord B: {texts[ja[n]]}\nSame business?" for n in batch],
                      truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        else:
            enc = tok([texts[ia[n]] for n in batch], [texts[ja[n]] for n in batch], truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
        yy = torch.tensor(y[batch] if y is not None else np.zeros(len(batch)), dtype=torch.float32)
        return enc, yy, torch.tensor(batch)
    return DataLoader(idx.tolist(), batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=2)


def predict(model, loader, n, dev, amp):
    model.eval()
    out = np.zeros(n, np.float32)
    with torch.no_grad():
        for enc, _, idx in loader:
            with torch.autocast("cuda", dtype=amp, enabled=dev == "cuda"):
                logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1)
            out[idx.numpy()] = logits.float().cpu().numpy()
    model.train()
    return out


class NotLearning(RuntimeError):
    """The loss is still at chance level at the guard step (XLM-R occasionally never takes off)."""


def train_one(a, texts, tr, dev, lr=None):
    from collections import deque
    lr = lr or a.lr
    tok, model = load_model(a, dev)
    amp = torch.bfloat16 if dev == "cuda" and native_bf16() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda" and amp == torch.float16)
    y = tr["label"].to_numpy().astype(np.float32)
    loader = make_loader(tok, texts, tr["i"].to_numpy(), tr["j"].to_numpy(), y, a.bs, a.max_len, True, a.lora)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = max(1, len(loader) * a.epochs // a.accum)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    recent = deque(maxlen=200)
    model.train()
    for ep in range(a.epochs):
        t, tot = time.time(), 0.0
        for step, (enc, yb, _) in enumerate(loader):
            with torch.autocast("cuda", dtype=amp, enabled=dev == "cuda"):
                logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.squeeze(-1)
            loss = loss_fn(logits.float(), yb.to(dev)) / a.accum
            scaler.scale(loss).backward()
            if (step + 1) % a.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
            cur = loss.item() * a.accum
            tot += cur
            recent.append(cur)
            if ep == 0 and a.guard_step > 0 and step == a.guard_step and sum(recent) / len(recent) > a.guard_loss:
                raise NotLearning(f"mean loss over the last {len(recent)} steps is {sum(recent) / len(recent):.3f} "
                                  f"at step {step} (chance level is about 0.69)")
            if step % 500 == 0:
                print(f"  ep {ep} step {step}/{len(loader)} loss {tot / (step + 1):.4f} "
                      f"({time.time() - t:.0f}s, {(step + 1) * a.bs / max(time.time() - t, 1e-6):.0f} pairs/s)", flush=True)
    return tok, model, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=160, help="tokens per pair (native-script names use many)")
    ap.add_argument("--lo", type=float, default=0.01, help="send pairs with stage-1 prob in [lo, hi]")
    ap.add_argument("--hi", type=float, default=0.99)
    ap.add_argument("--k", type=int, default=0, help="also send each S1's top-k pairs")
    ap.add_argument("--max_train", type=int, default=1_500_000, help="cap on training pairs")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--export", action="store_true", help="LOCAL: build work/ce_bundle for upload, then exit")
    ap.add_argument("--bundle", default=os.path.join(C.WORK_DIR, "ce_bundle"))
    ap.add_argument("--out_dir", default="", help="where to write scores (default: the bundle dir)")
    ap.add_argument("--eval_only", action="store_true",
                    help="train on CE folds 1..F-1, score ONLY the common validation fold 0; no test scoring")
    ap.add_argument("--train_country", default="", help="train only on this country_key (transfer test)")
    ap.add_argument("--eval_country", default="", help="validate only on this country_key")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _xlmr, _qwen")
    ap.add_argument("--only_folds", default="",
                    help="comma list of CE folds to run, e.g. '1' or '0,2' (one fold per process/account)")
    ap.add_argument("--test_folds", default="",
                    help="comma list of folds whose model scores the test pairs (default: every fold run); "
                         "'0' scores test once instead of once per fold - 1/3 of the test cost for slow models")
    ap.add_argument("--guard_step", type=int, default=1500,
                    help="if the loss is still at chance at this step, restart the fold at 0.6x lr (0 = off)")
    ap.add_argument("--guard_loss", type=float, default=0.685)
    a = ap.parse_args()
    if a.export:
        return export_bundle(a)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(C.SEED)
    t0 = time.time()
    out_dir = a.out_dir or a.bundle
    os.makedirs(out_dir, exist_ok=True)

    tr_pairs = pl.read_parquet(os.path.join(a.bundle, "train_pairs.parquet"))
    te_path = os.path.join(a.bundle, "test_pairs.parquet")
    score_test = os.path.exists(te_path) and not a.eval_only
    te_pairs = pl.read_parquet(te_path) if score_test else None
    if a.smoke:
        keep = tr_pairs["i"].unique().sort().head(1500)
        tr_pairs = tr_pairs.filter(pl.col("i").is_in(keep))
        te_pairs = te_pairs.head(3000) if te_pairs is not None else None
        a.folds, a.epochs = 2, 1

    # CE folds by S1 over the FULL bundle, so every account/model validates on the same S1s
    ids = tr_pairs["i"].unique().sort().to_numpy()
    fold_of = dict(zip(ids.tolist(), (np.random.default_rng(C.SEED).permutation(len(ids)) % a.folds).tolist()))
    tr_pairs = tr_pairs.with_columns(pl.col("i").replace_strict(fold_of, return_dtype=pl.Int32).alias("cefold"))
    print(f"train pairs {tr_pairs.height} ({tr_pairs['label'].sum()} pos) | "
          f"test pairs {te_pairs.height if te_pairs is not None else 0} | device {dev}", flush=True)
    tr_texts = rec_texts(a.bundle, "train")
    te_texts = rec_texts(a.bundle, "test") if te_pairs is not None else None

    ce_tr = np.full(tr_pairs.height, np.nan, np.float32)
    ce_te = np.zeros(te_pairs.height, np.float32) if te_pairs is not None else None
    if a.eval_only:
        folds = [0]
    elif a.only_folds:
        folds = sorted(int(x) for x in a.only_folds.split(","))
    else:
        folds = list(range(a.folds))
    test_folds = [int(x) for x in a.test_folds.split(",")] if a.test_folds else folds
    test_folds = [f for f in folds if f in test_folds]
    if te_pairs is not None and not test_folds:
        # this process trains folds that do not score the test set (another process does): skip test scoring
        print(f"no test scoring in this process (test folds {a.test_folds}, trained folds {folds})", flush=True)
        te_pairs = None
    for f in folds:
        t = time.time()
        tr = tr_pairs.filter(pl.col("cefold") != f)
        if a.train_country:
            tr = tr.filter(pl.col("country_key") == a.train_country)
        if tr.height > a.max_train:  # subsample whole S1 groups
            keep = tr["i"].unique().sample(fraction=a.max_train / tr.height, seed=C.SEED)
            tr = tr.filter(pl.col("i").is_in(keep))
        va_sel = pl.col("cefold") == f
        if a.eval_country:
            va_sel = va_sel & (pl.col("country_key") == a.eval_country)
        va_mask = tr_pairs.select(va_sel).to_series().to_numpy()
        va = tr_pairs.filter(va_mask)
        print(f"fold {f}: train {tr.height} / val {va.height}", flush=True)
        for attempt, lr in enumerate([a.lr, a.lr * 0.6]):
            try:
                tok, model, amp = train_one(a, tr_texts, tr, dev, lr)
                break
            except NotLearning as e:
                print(f"fold {f}: NOT LEARNING at lr {lr:g}: {e}", flush=True)
                if dev == "cuda":
                    torch.cuda.empty_cache()
                if attempt == 1:
                    raise
                print(f"fold {f}: restarting at lr {a.lr * 0.6:g}", flush=True)
        vl = make_loader(tok, tr_texts, va["i"].to_numpy(), va["j"].to_numpy(), None, a.bs * 4, a.max_len, False, a.lora)
        tv = time.time()
        ce_tr[va_mask] = predict(model, vl, va.height, dev, amp)
        print(f"fold {f}: inference {va.height / max(time.time() - tv, 1e-6):.0f} pairs/s", flush=True)
        y = va["label"].to_numpy()
        if 0 < y.sum() < len(y):
            print(f"fold {f}: CE AUC {roc_auc_score(y, ce_tr[va_mask]):.4f} | stage-1 AUC on same pairs "
                  f"{roc_auc_score(y, va['prob'].to_numpy()):.4f} ({time.time() - t:.0f}s)", flush=True)
        if te_pairs is not None and f in test_folds:
            tl = make_loader(tok, te_texts, te_pairs["i"].to_numpy(), te_pairs["j"].to_numpy(), None,
                             a.bs * 4, a.max_len, False, a.lora)
            ce_te += predict(model, tl, te_pairs.height, dev, amp) / len(test_folds)
            print(f"fold {f}: test scored ({time.time() - t:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    suffix = ("_smoke" if a.smoke else "") + ("_eval" if a.eval_only else "") + a.tag
    tr_pairs.select("i", "j").with_columns(pl.Series("ce", ce_tr)).write_parquet(os.path.join(out_dir, f"ce_train{suffix}.parquet"))
    if te_pairs is not None:
        te_pairs.select("i", "j").with_columns(pl.Series("ce", ce_te)).write_parquet(os.path.join(out_dir, f"ce_test{suffix}.parquet"))
    print(f"saved scores with suffix '{suffix}' in {out_dir} ({time.time() - t0:.0f}s) -> download into work/ on the laptop")


if __name__ == "__main__":
    main()

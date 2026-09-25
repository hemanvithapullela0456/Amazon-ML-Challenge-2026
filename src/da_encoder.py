"""Domain-adapted cross-encoder (DADER / DANN style) for a country that has no labels. Needs a GPU.

Labelled source pairs (training countries) + UNLABELLED target pairs (the unseen country). Each step takes one
source batch and one target batch through the same encoder (XLM-R, mean-pooled):
  match loss  : BCE on source pairs only
  alignment   : --da mmd  multi-kernel MMD between source and target pooled features (DADER: "more stable")
                --da grl  country classifier behind a gradient-reversal layer (DANN), lambda ramped 0 -> 1
                --da none no alignment (baseline)
Input is "name | address" for both records; the country string is left out (a trivial country cue).
Target labels, when present (the India simulation), are used ONLY to print AUC at the end.
Teacher-student (Noisy Student): with --pseudo_w > 0 the student also learns from the teacher's pseudo-labels on
the target country (tgt_pseudo.parquet: confident teacher decisions), so it picks up the target's own conventions;
--aug drops random input words (the student's noise) so it generalises instead of copying the teacher.

    python src/da_encoder.py --bundle <dir> --out <dir> --da mmd --lam 1.0
bundle: src_pairs.parquet (ka, kb, label), tgt_pairs.parquet (ka, kb[, label, p_lgbm]), texts.parquet (k, text)
-> <out>/tgt_scores_<da>.parquet (ka, kb, ce)
"""
import argparse
import itertools
import os
import time

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class DAModel(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.match = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(h, 1))
        self.dom = torch.nn.Sequential(torch.nn.Linear(h, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1))

    def pooled(self, enc):
        out = self.enc(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1)


def mmd(x, y, scales=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Multi-kernel (Gaussian) MMD^2 between two batches of features, bandwidths relative to the median distance."""
    z = torch.cat([x, y])
    d = torch.cdist(z, z).pow(2)
    med = d.detach().median().clamp(min=1e-6)
    k = sum(torch.exp(-d / (s * med)) for s in scales)
    n = len(x)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()


def drop_words(s, p, rng):
    w = s.split()
    keep = [x for x in w if x == "|" or rng.random() >= p]
    return " ".join(keep) if len(keep) > 1 else s


def loader(tok, texts, ka, kb, bs, max_len, shuffle, aug=0.0):
    idx = np.arange(len(ka))
    if not shuffle:
        idx = np.argsort([len(texts[a]) + len(texts[b]) for a, b in zip(ka, kb)], kind="stable")

    rng = np.random.default_rng(len(ka))

    def collate(batch):
        A = [texts[ka[n]] for n in batch]
        Bt = [texts[kb[n]] for n in batch]
        if aug > 0:
            A = [drop_words(x, aug, rng) for x in A]
            Bt = [drop_words(x, aug, rng) for x in Bt]
        enc = tok(A, Bt, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        return enc, torch.tensor(batch)
    return DataLoader(idx.tolist(), batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=2, drop_last=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    ap.add_argument("--da", choices=["none", "mmd", "grl"], default="mmd")
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the alignment loss (GRL: max reversal)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--max_src", type=int, default=400_000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pseudo_w", type=float, default=0.0, help="weight of the teacher pseudo-label loss (0 = off)")
    ap.add_argument("--aug", type=float, default=0.0, help="input word-dropout rate for the student (Noisy Student)")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(42)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    src = pl.read_parquet(os.path.join(a.bundle, "src_pairs.parquet"))
    tgt = pl.read_parquet(os.path.join(a.bundle, "tgt_pairs.parquet"))
    tx = pl.read_parquet(os.path.join(a.bundle, "texts.parquet"))
    texts = dict(zip(tx["k"].to_list(), tx["text"].to_list()))
    ps = None
    if a.pseudo_w > 0:
        ps = pl.read_parquet(os.path.join(a.bundle, "tgt_pseudo.parquet"))
        if a.smoke:
            ps = ps.sample(3000, seed=1)
    if src.height > a.max_src:
        src = src.sample(a.max_src, seed=42)
    if a.smoke:
        src, tgt = src.sample(3000, seed=1), tgt.sample(min(3000, tgt.height), seed=1)
    print(f"source {src.height} pairs ({src['label'].mean():.3f} pos) | target {tgt.height} pairs (unlabelled for training) "
          f"| da={a.da} lam={a.lam}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = DAModel(a.model).to(dev)
    sk, sb, sy = src["ka"].to_list(), src["kb"].to_list(), src["label"].to_numpy().astype(np.float32)
    tk, tb = tgt["ka"].to_list(), tgt["kb"].to_list()
    sl = loader(tok, texts, sk, sb, a.bs, a.max_len, True, a.aug)
    if ps is not None:
        pk, pb, py = ps["ka"].to_list(), ps["kb"].to_list(), ps["pl_label"].to_numpy().astype(np.float32)
        pl_loader = loader(tok, texts, pk, pb, a.bs, a.max_len, True, a.aug)
        print(f"teacher pseudo-labels: {ps.height} ({py.mean():.3f} positive), weight {a.pseudo_w}, word dropout {a.aug}", flush=True)
    tl = loader(tok, texts, tk, tb, a.bs, a.max_len, True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = len(sl) * a.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    bce = torch.nn.BCEWithLogitsLoss()
    model.train()
    step = 0
    for ep in range(a.epochs):
        t, lm, la = time.time(), 0.0, 0.0
        tgt_iter = itertools.cycle(tl)
        ps_iter = itertools.cycle(pl_loader) if ps is not None else None
        for n, (enc_s, idx_s) in enumerate(sl):
            enc_t, _ = next(tgt_iter)
            p = step / max(steps, 1)
            with torch.autocast("cuda", dtype=torch.float16):
                hs = model.pooled({k: v.to(dev) for k, v in enc_s.items()})
                logits = model.match(hs).squeeze(-1)
                loss_m = bce(logits.float(), torch.from_numpy(sy[idx_s.numpy()]).to(dev))
                loss_a = torch.zeros((), device=dev)
                if a.da != "none":
                    ht = model.pooled({k: v.to(dev) for k, v in enc_t.items()})
                    if a.da == "mmd":
                        loss_a = mmd(hs.float(), ht.float())
                    else:
                        lam = a.lam * (2 / (1 + np.exp(-10 * p)) - 1)
                        h = GradReverse.apply(torch.cat([hs, ht]).float(), lam)
                        d = model.dom(h).squeeze(-1)
                        dy = torch.cat([torch.zeros(len(hs)), torch.ones(len(ht))]).to(dev)
                        loss_a = bce(d, dy)
            loss = loss_m + (a.lam * loss_a if a.da == "mmd" else loss_a)
            if ps_iter is not None:
                enc_p, idx_p = next(ps_iter)
                with torch.autocast("cuda", dtype=torch.float16):
                    lp = model.match(model.pooled({k: v.to(dev) for k, v in enc_p.items()})).squeeze(-1)
                loss = loss + a.pseudo_w * bce(lp.float(), torch.from_numpy(py[idx_p.numpy()]).to(dev))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
            step += 1
            lm += loss_m.item(); la += loss_a.item()
            if n % 500 == 0:
                print(f"  ep {ep} step {n}/{len(sl)} match {lm / (n + 1):.4f} align {la / (n + 1):.4f} "
                      f"({time.time() - t:.0f}s, {(n + 1) * a.bs / max(time.time() - t, 1e-6):.0f} src pairs/s)", flush=True)

    model.eval()
    out = np.zeros(tgt.height, np.float32)
    with torch.no_grad():
        for enc, idx in loader(tok, texts, tk, tb, a.bs * 4, a.max_len, False):
            with torch.autocast("cuda", dtype=torch.float16):
                out[idx.numpy()] = model.match(model.pooled({k: v.to(dev) for k, v in enc.items()})).squeeze(-1).float().cpu().numpy()
    res = tgt.select("ka", "kb").with_columns(pl.Series("ce", out))
    tag = f"{a.da}{'_ts' if ps is not None else ''}{'_smoke' if a.smoke else ''}"
    res.write_parquet(os.path.join(a.out, f"tgt_scores_{tag}.parquet"))
    if "label" in tgt.columns:
        from sklearn.metrics import roc_auc_score
        y = tgt["label"].to_numpy()
        msg = f"TARGET AUC ce {roc_auc_score(y, out):.4f}"
        if "p_lgbm" in tgt.columns:
            msg += f" | lgbm (same pairs) {roc_auc_score(y, tgt['p_lgbm'].to_numpy()):.4f}"
            if "latin" in tgt.columns:
                m = tgt["latin"].to_numpy()
                msg += f" | LATIN-ONLY ce {roc_auc_score(y[m], out[m]):.4f} lgbm {roc_auc_score(y[m], tgt['p_lgbm'].to_numpy()[m]):.4f}"
        print(msg, flush=True)
    print(f"saved {a.out}/tgt_scores_{tag}.parquet ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

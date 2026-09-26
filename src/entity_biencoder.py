"""Entity bi-encoder: embed every record so that all distorted copies of one hidden entity land together.

The data is synthetic: each S1/S2/S3 record is a distorted copy of a latent entity (noise, dropped address,
name replaced by an alias, script switch), and decoys are near-copies of a DIFFERENT entity (same name, shifted
house number, same address with another name). Training is entity-level supervised contrastive learning:
positives = any two records of the same entity (S1-copy AND copy-copy), negatives = other entities in the batch
+ one decoy per entity (a high-scoring false candidate of that S1). Language-agnostic in intent: the model must
learn the generator's moves, which are the same in every country.
Refs: Bayesian ER with distortion models (Steorts 2016; d-blink, Marchant et al. 2021), SupCon (Khosla 2020),
E5 (Wang 2022; intfloat/multilingual-e5-base, MIT).

LOCAL  python src/entity_biencoder.py --export          -> work/bi_bundle/{groups,texts}.parquet
GPU    python src/entity_biencoder.py --bundle b_bi --train_country us --out out_bi --tag sim   (France-like sim)
       python src/entity_biencoder.py --bundle b_bi --out out_bi --tag full                    (US+India, holds out val)
Scoring (GPU, after training): cosine for the pair files given with --score name=pairs.parquet:texts.parquet
"""
import argparse
import os
import time

import numpy as np
import polars as pl

import config as C


def export():
    from blocking import load_records, truth_pairs
    from prep import load_norm
    out = os.path.join(C.WORK_DIR, "bi_bundle")
    os.makedirs(out, exist_ok=True)
    G = pl.read_parquet(os.path.join(C.WORK_DIR, "train_G.parquet")).select("i", "country_key")
    t = truth_pairs(load_records("train"), "train", G["i"].to_numpy())
    oof = pl.read_parquet(os.path.join(C.WORK_DIR, "train_oof.parquet"), columns=["i", "j", "label", "prob"])
    negs = (oof.filter(~pl.col("label")).with_columns(pl.col("prob").rank("ordinal", descending=True).over("i").alias("r"))
            .filter(pl.col("r") <= 5).group_by("i").agg(pl.col("j").alias("negs")))
    # validation S1s = CE fold 0 of the set model (same rule as set_encoder.py) -> never trained on
    sl = pl.read_parquet(os.path.join(C.WORK_DIR, "set_bundle", "train_lists.parquet"))
    ids = sl["i"].unique().sort().to_numpy()
    f0 = set(ids[np.random.default_rng(C.SEED).permutation(len(ids)) % 3 == 0].tolist())
    g = (G.join(t.group_by("i").agg(pl.col("j").alias("copies")), on="i", how="left").join(negs, on="i", how="left")
         .with_columns(pl.col("i").is_in(list(f0)).alias("val")))
    g.write_parquet(os.path.join(out, "groups.parquet"))
    need = pl.concat([g["i"], g["copies"].explode().drop_nulls(), g["negs"].explode().drop_nulls()]).unique()
    norm = load_norm("train", columns=["business_name", "business_address"]).with_row_index("r").with_columns(pl.col("r").cast(pl.UInt32))
    (norm.join(need.cast(pl.UInt32).to_frame("r"), on="r").select(
        "r", (pl.col("business_name").fill_null("").str.strip_chars() + " | " + pl.col("business_address").fill_null("").str.strip_chars()).alias("text"))
     .write_parquet(os.path.join(out, "texts.parquet")))
    print(f"groups {g.height} (val {int(g['val'].sum())}) | records {len(need)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--bundle", default="b_bi")
    ap.add_argument("--out", default="out_bi")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--train_country", default="")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128, help="entities per batch")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--score", action="append", default=[], help="name=pairs.parquet:texts.parquet (pairs: ka,kb or i,j)")
    a = ap.parse_args()
    if a.export:
        return export()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda"
    torch.manual_seed(C.SEED); rng = np.random.default_rng(C.SEED)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    g = pl.read_parquet(os.path.join(a.bundle, "groups.parquet")).filter(~pl.col("val"))
    if a.train_country:
        g = g.filter(pl.col("country_key") == a.train_country)
    g = g.filter(pl.col("copies").list.len() > 0)
    tx = pl.read_parquet(os.path.join(a.bundle, "texts.parquet"))
    text = dict(zip(tx["r"].to_list(), tx["text"].to_list()))
    groups = [([i] + c, n or []) for i, c, n in zip(g["i"].to_list(), g["copies"].to_list(), g["negs"].to_list())]
    print(f"training entities {len(groups)} ({a.train_country or 'all countries'}) | texts {len(text)}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)

    def embed(texts):
        enc = tok(["query: " + s for s in texts], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        return F.normalize((h.float() * m).sum(1) / m.sum(1), dim=-1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * (len(groups) // a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    step, run = 0, []
    for ep in range(a.epochs):
        order = rng.permutation(len(groups))
        for s in range(0, len(order) - a.bs + 1, a.bs):
            batch = [groups[k] for k in order[s:s + a.bs]]
            A, P, N = [], [], []
            for mem, neg in batch:
                x, y = rng.choice(len(mem), 2, replace=False) if len(mem) > 1 else (0, 0)
                A.append(text[mem[x]]); P.append(text[mem[y]])
                N.append(text[neg[rng.integers(len(neg))]] if neg else text[mem[x]])
            ea, ep_, en = embed(A), embed(P), embed(N)
            has_neg = torch.tensor([bool(n) for _, n in batch], device=dev)
            logits = torch.cat([ea @ ep_.T, (ea * en).sum(-1, keepdim=True).masked_fill(~has_neg.unsqueeze(-1), -1e4)], 1) / a.temp
            lab = torch.arange(len(batch), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy((ep_ @ ea.T) / a.temp, lab))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            run.append(loss.item()); step += 1
            if step % 200 == 0:
                print(f"  ep {ep} step {step}/{steps} loss {np.mean(run[-200:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    torch.save(model.state_dict(), os.path.join(a.out, f"bi_{a.tag}.pt"))

    @torch.no_grad()
    def embed_all(keys, texts, bs=1024):
        order = np.argsort([len(s) for s in texts])
        out = torch.zeros(len(keys), model.config.hidden_size)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            out[idx] = embed([texts[k] for k in idx]).cpu()
        return out

    for spec in a.score:
        name, rest = spec.split("=", 1)
        pp, tp = rest.split(":")
        pairs = pl.read_parquet(pp)
        ca, cb = ("ka", "kb") if "ka" in pairs.columns else ("i", "j")
        tt = pl.read_parquet(tp)
        kcol = "k" if "k" in tt.columns else "r"
        need = set(pairs[ca].to_list()) | set(pairs[cb].to_list())
        tt = tt.filter(pl.col(kcol).is_in(list(need)))
        keys = tt[kcol].to_list()
        E = embed_all(keys, tt["text"].fill_null("").to_list())
        ix = {k: n for n, k in enumerate(keys)}
        xa = torch.tensor([ix[k] for k in pairs[ca].to_list()]); xb = torch.tensor([ix[k] for k in pairs[cb].to_list()])
        cos = (E[xa] * E[xb]).sum(-1).numpy()
        res = pairs.select(ca, cb).with_columns(pl.Series("ce", cos.astype(np.float32)))
        res.write_parquet(os.path.join(a.out, f"{a.tag}_{name}.parquet"))
        msg = ""
        if "label" in pairs.columns and pairs["label"].n_unique() > 1:
            from sklearn.metrics import roc_auc_score
            y = pairs["label"].to_numpy()
            sel = pairs["latin"].to_numpy() if "latin" in pairs.columns else np.ones(len(y), bool)
            msg = f" | AUC {roc_auc_score(y[sel], cos[sel]):.4f}" + (
                f" vs LightGBM {roc_auc_score(y[sel], pairs['p_lgbm'].to_numpy()[sel]):.4f}" if "p_lgbm" in pairs.columns else "")
        print(f"scored {name}: {res.height} pairs{msg} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

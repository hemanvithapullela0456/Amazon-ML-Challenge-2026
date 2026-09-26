"""One-to-set cross-encoder (Set-Encoder / GNEM-style): ONE mDeBERTa reads an S1 and its whole candidate list,
    [CLS] S1 [SEP] [CAND] c1 [SEP] [CAND] c2 [SEP] ... [CAND] c10 [SEP]
so attention compares every candidate with the S1 AND with its sibling copies (the graph signal, learned),
and a head on each [CAND] token scores that candidate.
Refs: GNEM (WWW'21, one-to-set matching), HierGAT+ (SIGMOD'22), Set-Encoder (ECIR'25, inter-passage attention).

LOCAL  python src/set_encoder.py --export            -> work/set_bundle/{train,test}_lists.parquet + texts
GPU    python src/set_encoder.py --bundle b_set --out_dir out_set [--smoke]
       trains on CE folds 1..2 (same S1 folds as the pairwise rerankers), scores fold 0 (validation) and test.
"""
import argparse
import os
import time

import numpy as np
import polars as pl

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError:
    torch = None

import config as C

K, LO, HI = 10, 0.01, 0.99
S1_TOK, C_TOK = 48, 40


# ------------------------------------------------------------------ export (local, CPU)
def lists_from(scores):
    """S1s with at least one uncertain pair; list = its top-K candidates with prob > LO."""
    o = scores.filter(pl.col("prob") > LO)
    band_s1 = o.filter(pl.col("prob") < HI)["i"].unique()
    o = o.filter(pl.col("i").is_in(band_s1.implode()))
    o = o.with_columns(pl.col("prob").rank("ordinal", descending=True).over("i").alias("r")).filter(pl.col("r") <= K)
    return o


def export(a):
    from prep import load_norm
    out = os.path.join(C.WORK_DIR, "set_bundle")
    os.makedirs(out, exist_ok=True)
    for split in ("train", "test"):
        norm = (load_norm(split, columns=["business_name", "business_address", "country_key"]).with_row_index("r")
                .with_columns(pl.col("r").cast(pl.UInt32)))
        if split == "train":
            o = lists_from(pl.read_parquet(os.path.join(C.WORK_DIR, "train_oof.parquet"), columns=["i", "j", "prob", "label"]))
        else:
            sc = pl.read_parquet(os.path.join(C.WORK_DIR, "test_scored.parquet"))
            ck = norm.select(pl.col("r").alias("i"), "country_key")
            unseen = set(ck["country_key"].unique().to_list()) - set(
                pl.read_parquet(os.path.join(C.WORK_DIR, "train_G.parquet"))["country_key"].unique().to_list())
            un_ids = ck.filter(pl.col("country_key").is_in(list(unseen)))["i"]
            fr = pl.read_parquet(os.path.join(C.WORK_DIR, a.unseen_scores)).select("i", "j", "prob")
            sc = pl.concat([sc.filter(~pl.col("i").is_in(un_ids.implode())),
                            fr.with_columns(pl.col("prob").cast(sc["prob"].dtype))])
            o = lists_from(sc)
            print(f"test: unseen countries {sorted(unseen)} use {a.unseen_scores}")
        o = o.join(norm.select(pl.col("r").alias("i"), "country_key"), on="i", how="left")
        o.drop("r").write_parquet(os.path.join(out, f"{split}_lists.parquet"))
        need = pl.concat([o["i"], o["j"]]).unique()
        (norm.join(need.to_frame("r"), on="r", how="inner")
         .select("r", (pl.col("business_name").fill_null("").str.strip_chars() + " | "
                       + pl.col("business_address").fill_null("").str.strip_chars()).alias("text"))
         .write_parquet(os.path.join(out, f"{split}_texts.parquet")))
        print(f"{split}: {o['i'].n_unique()} S1 lists, {o.height} candidates, {len(need)} records", flush=True)
    mb = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
    print(f"-> {out} ({mb:.0f} MB)")


# ------------------------------------------------------------------ model (GPU)
class SetModel(nn.Module if torch else object):
    def __init__(self, name, n_tok):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name, torch_dtype=torch.float32)   # checkpoint is fp16: NaNs
        self.enc.resize_token_embeddings(n_tok)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, ids, att, cpos, cmask):
        hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state          # B x T x H
        c = torch.gather(hs, 1, cpos.unsqueeze(-1).expand(-1, -1, hs.size(-1)))     # B x K x H
        cls = hs[:, :1].expand_as(c)
        return self.head(torch.cat([c, cls], -1)).squeeze(-1).masked_fill(~cmask, 0.0)


def encode_texts(tok, texts):
    """record id -> token ids (no specials), tokenized once"""
    keys = list(texts)
    enc = tok([texts[k] for k in keys], add_special_tokens=False, truncation=True, max_length=S1_TOK)["input_ids"]
    return dict(zip(keys, enc))


SCR = {"p": 0.0, "tok_p": 0.7, "pool": None, "keep": None}   # vocabulary scrambling (training only)


def scramble_setup(tok, p, tok_p=0.7):
    """Delexicalised training (McDonald et al. 2011 style): per list, alphabetic sub-words are replaced by random
    sub-words CONSISTENTLY across the S1 and all its candidates, so the model sees which records share tokens but
    not which words they are. Digits, punctuation and special tokens are kept."""
    vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
    alpha = np.array([n for n, t in enumerate(vocab) if t and t.strip("▁").isalpha() and len(t.strip("▁")) >= 2])
    SCR.update(p=p, tok_p=tok_p, pool=alpha, keep=set(n for n, t in enumerate(vocab) if not (t and t.strip("▁").isalpha())))


def build(tok, cand_id, toks, s1, cands, rng=None):
    order = np.arange(len(cands)) if rng is None else rng.permutation(len(cands))
    recs = [toks[s1][:S1_TOK]] + [toks[cands[k]][:C_TOK] for k in order]
    if rng is not None and SCR["p"] > 0 and rng.random() < SCR["p"]:
        uniq = {t for r in recs for t in r if t not in SCR["keep"]}
        mp = {t: int(SCR["pool"][rng.integers(len(SCR["pool"]))]) for t in uniq if rng.random() < SCR["tok_p"]}
        recs = [[mp.get(t, t) for t in r] for r in recs]
    ids = [tok.cls_token_id] + recs[0] + [tok.sep_token_id]
    pos = []
    for r in recs[1:]:
        pos.append(len(ids))
        ids += [cand_id] + r + [tok.sep_token_id]
    return ids, pos, order


def make_loader(tok, cand_id, toks, lists, bs, shuffle, seed=0):
    def collate(batch):
        rng = np.random.default_rng(seed + batch[0]) if shuffle else None
        seqs, poss, orders, ys = [], [], [], []
        for n in batch:
            s1, cands, y = lists[n]
            ids, pos, order = build(tok, cand_id, toks, s1, cands, rng)
            seqs.append(ids); poss.append(pos); orders.append(order); ys.append(np.asarray(y, np.float32)[order] if y is not None else None)
        T = max(map(len, seqs))
        ids = torch.full((len(batch), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), T), dtype=torch.long)
        cpos = torch.zeros((len(batch), K), dtype=torch.long)
        cmask = torch.zeros((len(batch), K), dtype=torch.bool)
        yy = torch.zeros((len(batch), K))
        for b, (s, p) in enumerate(zip(seqs, poss)):
            ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
            cpos[b, :len(p)] = torch.tensor(p); cmask[b, :len(p)] = True
            if ys[b] is not None:
                yy[b, :len(p)] = torch.tensor(ys[b])
        return ids, att, cpos, cmask, yy, torch.tensor(batch), orders
    idx = np.arange(len(lists))
    if not shuffle:   # length-sorted for fast inference
        idx = np.argsort([len(l[1]) for l in lists], kind="stable")
    return DataLoader(idx.tolist(), batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=4)


def to_lists(df, with_label):
    g = df.group_by("i", maintain_order=True).agg("j", *(["label"] if with_label else []))
    labels = g["label"].to_list() if with_label else [None] * g.height
    return list(zip(g["i"].to_list(), g["j"].to_list(), labels))


def score(model, loader, lists, dev):
    model.eval()
    rows_i, rows_j, rows_s = [], [], []
    with torch.no_grad():
        for ids, att, cpos, cmask, _, idx, orders in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(ids.to(dev), att.to(dev), cpos.to(dev), cmask.to(dev)).float().cpu().numpy()
            for b, n in enumerate(idx.tolist()):
                s1, cands, _ = lists[n]
                for slot, k in enumerate(orders[b]):
                    rows_i.append(s1); rows_j.append(cands[k]); rows_s.append(lo[b, slot])
    model.train()
    return pl.DataFrame({"i": rows_i, "j": rows_j, "ce": rows_s}).with_columns(
        pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32), pl.col("ce").cast(pl.Float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--unseen_scores", default="test_scored_unseen_fused.parquet")
    ap.add_argument("--bundle", default="b_set")
    ap.add_argument("--out_dir", default="out_set")
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--pseudo", default="", help="teacher-labelled target lists (i, j, pl_label; null = context only)")
    ap.add_argument("--pseudo_n", type=int, default=120_000, help="target lists sampled into training")
    ap.add_argument("--test_country", default="", help="score only test lists of this country_key")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scramble", type=float, default=0.0, help="share of training lists whose vocabulary is scrambled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_country", default="", help="train only on this country's lists (transfer test)")
    ap.add_argument("--test_lists", default="test_lists.parquet")
    ap.add_argument("--test_texts", default="test_texts.parquet")
    a = ap.parse_args()
    if a.export:
        return export(a)
    dev = "cuda"
    torch.manual_seed(C.SEED + a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    tr = pl.read_parquet(os.path.join(a.bundle, "train_lists.parquet")).sort("i", "prob", descending=[False, True])
    ids = tr["i"].unique().sort().to_numpy()   # same S1 folds as cross_encoder.py (3 folds, SEED)
    fold_of = dict(zip(ids.tolist(), (np.random.default_rng(C.SEED).permutation(len(ids)) % 3).tolist()))
    tr = tr.with_columns(pl.col("i").replace_strict(fold_of, return_dtype=pl.Int32).alias("cefold"))
    te = None if a.no_test else pl.read_parquet(os.path.join(a.bundle, a.test_lists)).sort("i", "prob", descending=[False, True])
    if te is not None and a.test_country:
        te = te.filter(pl.col("country_key") == a.test_country)
    if a.smoke:
        tr = tr.filter(pl.col("i").is_in(ids[:3000]))
        te = te.filter(pl.col("i").is_in(te["i"].unique().sort().head(2000).implode())) if te is not None else None
        a.epochs = 1
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.add_special_tokens({"additional_special_tokens": ["[CAND]"]})
    cand_id = tok.convert_tokens_to_ids("[CAND]")
    if a.scramble > 0:
        scramble_setup(tok, a.scramble)
        print(f"vocabulary scrambling on {a.scramble:.0%} of training lists ({len(SCR['pool'])} replacement sub-words)", flush=True)
    texts = pl.read_parquet(os.path.join(a.bundle, "train_texts.parquet"))
    toks = encode_texts(tok, dict(zip(texts["r"].to_list(), texts["text"].to_list())))
    fit = tr.filter(pl.col("cefold") != 0)
    if a.train_country:
        fit = fit.filter(pl.col("country_key") == a.train_country)
    fit_l = to_lists(fit, True)
    if a.pseudo:
        ps = pl.read_parquet(a.pseudo)
        keep = ps["i"].unique().sample(min(a.pseudo_n, ps["i"].n_unique()), seed=C.SEED)
        tl = pl.read_parquet(os.path.join(a.bundle, a.test_lists)).filter(pl.col("i").is_in(keep.implode()))
        ps = tl.select("i", "j", "prob").join(ps, on=["i", "j"], how="left").sort("i", "prob", descending=[False, True])
        ps = ps.with_columns(pl.col("pl_label").fill_null(float("nan")))
        tt = pl.read_parquet(os.path.join(a.bundle, a.test_texts)).filter(
            pl.col("r").is_in(pl.concat([ps["i"], ps["j"]]).unique().implode()))
        # target records live in the TEST row space: offset their ids so they never collide with train rows
        OFF = 1 << 30
        toks.update({k + OFF: v for k, v in encode_texts(tok, dict(zip(tt["r"].to_list(), tt["text"].to_list()))).items()})
        ps = ps.with_columns((pl.col("i").cast(pl.Int64) + OFF).alias("i"), (pl.col("j").cast(pl.Int64) + OFF).alias("j")).rename({"pl_label": "label"})
        pl_lists = to_lists(ps, True)
        fit_l = fit_l + pl_lists
        print(f"+ {len(pl_lists)} teacher-labelled target lists ({int(np.nansum(ps['label'].to_numpy()))} positive labels, "
              f"{int(np.isnan(ps['label'].to_numpy()).sum())} context-only candidates)", flush=True)
    val_l = to_lists(tr.filter(pl.col("cefold") == 0), True)
    print(f"train lists {len(fit_l)} | validation lists {len(val_l)} | tokenized ({time.time() - t0:.0f}s)", flush=True)

    model = SetModel(a.model, len(tok)).to(dev)
    loader = make_loader(tok, cand_id, toks, fit_l, a.bs, True, seed=1000 * a.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(loader)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    step, run = 0, []
    for ep in range(a.epochs):
        for bids, att, cpos, cmask, yy, _, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = model(bids.to(dev), att.to(dev), cpos.to(dev), cmask.to(dev))
            yd = yy.to(dev)
            m = cmask.to(dev) & ~torch.isnan(yd)
            loss = (lossf(lo.float(), torch.nan_to_num(yd)) * m).sum() / m.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            run.append(loss.item()); step += 1
            if step % 500 == 0:
                print(f"  ep {ep} step {step}/{steps} loss {np.mean(run[-500:]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    vs = score(model, make_loader(tok, cand_id, toks, val_l, a.bs * 4, False), val_l, dev)
    vs.write_parquet(os.path.join(a.out_dir, f"set_val{a.tag}.parquet"))
    from sklearn.metrics import roc_auc_score
    v = vs.join(tr.select("i", "j", "label", "prob"), on=["i", "j"])
    band = v.filter((pl.col("prob") > LO) & (pl.col("prob") < HI))
    print(f"validation (fold 0): set-model AUC on band {roc_auc_score(band['label'], band['ce']):.4f} | "
          f"stage-1 AUC on same pairs {roc_auc_score(band['label'], band['prob']):.4f} ({time.time() - t0:.0f}s)", flush=True)
    if te is None:
        return
    texts = pl.read_parquet(os.path.join(a.bundle, a.test_texts))
    toks = encode_texts(tok, dict(zip(texts["r"].to_list(), texts["text"].to_list())))
    te_l = to_lists(te, False)
    ts = score(model, make_loader(tok, cand_id, toks, te_l, a.bs * 4, False), te_l, dev)
    ts.write_parquet(os.path.join(a.out_dir, f"set_test{a.tag}.parquet"))
    print(f"test scored: {ts.height} candidates in {len(te_l)} lists -> {a.out_dir}/set_test.parquet ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""Dense retrieval track: fine-tuned multilingual bi-encoder for candidate generation (retrieve-and-rerank).

Research basis: supervised contrastive learning over entity clusters (R-SupCon, Peeters & Bizer 2022),
in-batch + mined hard negatives (MultipleNegativesRanking / DPR), dense blocking unioned with sparse keys
(DeepBlocker 2021, Sparkly 2023). Base model: intfloat/multilingual-e5-small (MIT, 118M).

Record index r = row order of S1, S2, S3 files concatenated (same as prep.py), so ids are shared
between the laptop (keys, labels) and Kaggle (GPU) without moving text around.

Subcommands
  LAPTOP (CPU):
    export  : training triples (S1, true match, hard non-match from key blocking) as record indices,
              for S1 NOT in the LightGBM sample or the blocking-eval sample; plus the train query list.
    eval    : recall of dense top-K vs key top-K vs their union on the blocking-eval sample.
  KAGGLE (GPU):
    train   : fine-tune the bi-encoder on the exported triples.
    embed   : encode every record of a split -> fp16 .npy
    knn     : per-country exact top-K S2/S3 for S1 queries -> dense_<split>.parquet (i, j, cos, dense_rank)
"""
import argparse
import os
import sys
import time

import numpy as np
import polars as pl

import config as C

MODEL = "intfloat/multilingual-e5-small"
EXPORT_DIR = os.path.join(C.WORK_DIR, "dense_export")
SMOKE_S1, SMOKE_X = 4_000, 30_000  # smoke-test subset: S1 rows, and S2 / S3 rows each


# ------------------------------------------------------------------ shared
def raw_records(split, data_dir=C.DATA_DIR):
    """Raw records in record-index order (identical to prep.py)."""
    from prep import read_source
    parts = []
    for s in (1, 2, 3):
        df = read_source(os.path.join(data_dir, split, f"{split}_source{s}.tsv"))
        df = df.select([pl.col(c).fill_null("") for c in ["entity_id", "business_name", "business_address", "country"]])
        parts.append(df.with_columns(pl.lit(s, dtype=pl.Int8).alias("source")))
    return pl.concat(parts).with_row_index("r").with_columns(pl.col("r").cast(pl.UInt32))


def texts_of(recs, prefix="query: "):
    """E5 models expect a 'query: ' prefix for symmetric similarity; BGE-M3 wants none (pass ''). The prefix
    must be the same at training and embedding time. Raw text keeps native scripts."""
    return (prefix + recs["business_name"].str.strip_chars() + " | "
            + recs["business_address"].str.strip_chars()).to_list()


def splits_train(s1_rows):
    """(lgbm_sample, eval_sample) - reproduce train.py and blocking.main exactly."""
    g = pl.read_parquet(os.path.join(C.WORK_DIR, "train_G.parquet"))["i"].to_numpy()
    perm = np.random.default_rng(C.SEED).permutation(s1_rows)
    return g, np.sort(perm[: int(len(s1_rows) * 0.1)])


# ------------------------------------------------------------------ laptop: export
def cmd_export(a):
    from blocking import build_keys, candidates, load_ranker, load_records, truth_pairs
    t = time.time()
    recs = load_records("train")
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    lgbm, ev = splits_train(s1)
    pool = np.setdiff1d(s1, np.union1d(lgbm, ev))
    rng = np.random.default_rng(C.SEED + 7)
    anchors = np.sort(rng.choice(pool, min(a.n_s1, len(pool)), replace=False))
    truth = truth_pairs(recs, "train", anchors)
    keys = build_keys(recs)
    pairs, _ = candidates(keys, anchors, k=a.hard_k, ranker=load_ranker())
    del keys
    neg = (pairs.join(truth, on=["i", "j"], how="anti")          # top-ranked NON-matches
           .sort("i", "blk_rank").group_by("i", maintain_order=True).agg(pl.col("j").head(a.hard_k).alias("negs")))
    tri = truth.join(neg, on="i", how="left")
    # one hard negative per positive, sampled from the S1's top non-matches (null when none)
    n = tri.height
    tri = tri.with_columns(pl.col("negs").list.len().fill_null(0).alias("nn"))
    pick = (rng.random(n) * np.maximum(tri["nn"].to_numpy(), 1)).astype(np.int64)
    tri = tri.with_columns(pl.Series("pick", pick)).with_columns(
        pl.when(pl.col("nn") > 0).then(pl.col("negs").list.get(pl.col("pick"), null_on_oob=True)).otherwise(None).alias("n"))
    cc = recs.select(pl.col("r").alias("i"), "country_key")
    tri = tri.join(cc, on="i").select(pl.col("i").alias("a"), pl.col("j").alias("p"), "n", "country_key")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    tri.write_parquet(os.path.join(EXPORT_DIR, "triples.parquet"))
    q = np.union1d(lgbm, ev)
    pl.DataFrame({"r": q.astype(np.uint32)}).write_parquet(os.path.join(EXPORT_DIR, "train_queries.parquet"))
    print(f"triples: {tri.height} from {len(anchors)} S1 (hard negative for {tri['n'].is_not_null().mean():.3f}) | "
          f"train queries: {len(q)} | -> {EXPORT_DIR} ({time.time() - t:.0f}s)")
    print(tri.group_by("country_key").len())


# ------------------------------------------------------------------ kaggle: train
def cmd_train(a):
    from datasets import Dataset, DatasetDict
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
    # one GPU only: HF Trainer would otherwise wrap the model in DataParallel on "T4 x2", which does not
    # play well with the cached contrastive loss (and would silently double the batch size)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    t = time.time()
    recs = raw_records("train", a.data_dir)
    tri = pl.read_parquet(os.path.join(a.export_dir, "triples.parquet")).filter(pl.col("n").is_not_null())
    if a.country:
        tri = tri.filter(pl.col("country_key") == a.country)
    if a.max_triples and tri.height > a.max_triples:
        tri = tri.sample(a.max_triples, seed=C.SEED)
    if a.smoke:
        tri = tri.sample(min(20_000, tri.height), seed=C.SEED)
    # text only for the records the triples reference (~2.5M of 12.5M): far less RAM and time
    need = pl.concat([tri["a"], tri["p"], tri["n"]]).unique().to_frame("r")
    sub = recs.join(need, on="r", how="inner")
    texts = dict(zip(sub["r"].to_list(), texts_of(sub, a.prefix)))
    del recs, sub, need
    # one dataset per country -> every batch is single-country (realistic in-batch negatives)
    dsd = {}
    for (c,), g in tri.group_by("country_key"):
        dsd[c] = Dataset.from_dict({"anchor": [texts[x] for x in g["a"].to_list()],
                                    "positive": [texts[x] for x in g["p"].to_list()],
                                    "negative": [texts[x] for x in g["n"].to_list()]})
    print({c: len(d) for c, d in dsd.items()}, f"({time.time() - t:.0f}s)", flush=True)
    del texts
    model = SentenceTransformer(a.model)
    model.max_seq_length = a.max_len
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=a.mini_bs)
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(a.out_dir, "ckpt"), num_train_epochs=1, per_device_train_batch_size=a.bs,
        learning_rate=a.lr, warmup_ratio=0.05, fp16=True, bf16=False, logging_steps=100, save_strategy="no",
        batch_sampler=BatchSamplers.NO_DUPLICATES, multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        report_to="none", dataloader_num_workers=2, seed=C.SEED)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=DatasetDict(dsd), loss=loss)
    trainer.train()
    path = os.path.join(a.out_dir, "dense_model")
    model.save(path)
    print(f"saved {path} ({time.time() - t:.0f}s)")


# ------------------------------------------------------------------ kaggle: embed + knn
def cmd_embed(a):
    import torch
    from sentence_transformers import SentenceTransformer
    t = time.time()
    recs = raw_records(a.split, a.data_dir)
    os.makedirs(a.emb_dir, exist_ok=True)
    rows_path = os.path.join(a.emb_dir, f"emb_{a.split}_rows.npy")
    if a.smoke:
        # a small MIXED subset (the file lists all S1 rows first, so head() alone would contain no S2/S3
        # records to search). Embeddings are stored compactly; the rows file maps position -> record index.
        recs = pl.concat([recs.filter(pl.col("source") == 1).head(SMOKE_S1),
                          recs.filter(pl.col("source") == 2).head(SMOKE_X),
                          recs.filter(pl.col("source") == 3).head(SMOKE_X)])
        np.save(rows_path, recs["r"].to_numpy())
    elif os.path.exists(rows_path):
        os.remove(rows_path)  # stale smoke file: a full run is indexed by record index directly
    texts = texts_of(recs, a.prefix)
    model = SentenceTransformer(a.model_path, device="cuda").half()
    model.max_seq_length = a.max_len
    path = os.path.join(a.emb_dir, f"emb_{a.split}.npy")
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), model.get_sentence_embedding_dimension()))
    order = np.argsort([len(x) for x in texts], kind="stable")  # length-sorted batches = less padding
    step = 200_000
    with torch.inference_mode():
        for s in range(0, len(order), step):
            idx = order[s:s + step]
            e = model.encode([texts[x] for x in idx], batch_size=a.bs, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
            out[idx] = e.astype(np.float16)
            print(f"  {s + len(idx)}/{len(order)} ({time.time() - t:.0f}s, {(s + len(idx)) / (time.time() - t):.0f}/s)", flush=True)
    out.flush()
    print(f"embedded {len(texts)} -> {path} ({time.time() - t:.0f}s)")


def cmd_knn(a):
    import torch
    t = time.time()
    recs = raw_records(a.split, a.data_dir).select("r", "source", "country")
    recs = recs.with_columns(pl.col("country").str.strip_chars().str.to_lowercase().alias("country"))
    emb = np.load(os.path.join(a.emb_dir, f"emb_{a.split}.npy"), mmap_mode="r")
    rows_path = os.path.join(a.emb_dir, f"emb_{a.split}_rows.npy")
    if os.path.exists(rows_path):  # smoke run: embeddings exist only for a subset of the records
        rows = np.load(rows_path)
        pos = pl.DataFrame({"r": rows.astype(np.uint32), "pos": np.arange(len(rows), dtype=np.int64)})
        recs = recs.join(pos, on="r", how="inner")
    else:
        recs = recs.with_columns(pl.col("r").cast(pl.Int64).alias("pos"))
    if a.queries:
        q_set = pl.read_parquet(a.queries).select(pl.col("r").cast(pl.UInt32))
    else:
        q_set = recs.filter(pl.col("source") == 1).select("r")
    out = []
    for (c,), g in recs.group_by("country"):
        xg = g.filter(pl.col("source") != 1).sort("pos")
        qg = g.join(q_set, on="r", how="inner").sort("pos")
        if xg.height == 0 or qg.height == 0:
            continue
        x_r, x_pos = xg["r"].to_numpy(), xg["pos"].to_numpy()
        q_r, q_pos = qg["r"].to_numpy(), qg["pos"].to_numpy()
        X = torch.from_numpy(np.ascontiguousarray(emb[x_pos])).cuda()
        k = min(a.k, len(x_r))
        for s in range(0, len(q_r), a.chunk):
            Q = torch.from_numpy(np.ascontiguousarray(emb[q_pos[s:s + a.chunk]])).cuda()
            v, ix = torch.topk(Q @ X.T, k, dim=1)
            qq = q_r[s:s + a.chunk]
            out.append(pl.DataFrame({
                "i": np.repeat(qq, k).astype(np.uint32), "j": x_r[ix.cpu().numpy().ravel()].astype(np.uint32),
                "cos": v.float().cpu().numpy().ravel(), "dense_rank": np.tile(np.arange(1, k + 1, dtype=np.uint16), len(qq))}))
        print(f"  {c}: {len(q_r)} queries x {len(x_r)} records ({time.time() - t:.0f}s)", flush=True)
        del X
        torch.cuda.empty_cache()
    if not out:
        sys.exit("no dense pairs produced: no country had both queries and S2/S3 records")
    res = pl.concat(out)
    path = os.path.join(a.out_dir, f"dense_{a.split}.parquet")
    res.write_parquet(path)
    print(f"{res.height} dense pairs -> {path} ({time.time() - t:.0f}s)")


# ------------------------------------------------------------------ laptop: eval
def cmd_eval(a):
    from blocking import build_keys, candidates, load_ranker, load_records, truth_pairs
    recs = load_records("train")
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    _, ev = splits_train(s1)
    truth = truth_pairs(recs, "train", ev)
    n_true = truth.height
    dense = pl.read_parquet(a.dense).filter(pl.col("i").is_in(ev))
    keys = build_keys(recs)
    key, _ = candidates(keys, ev, k=30, ranker=load_ranker())
    del keys
    cc = recs.select(pl.col("r").alias("i"), "country_key")

    def rec(p, name):
        hit = p.select("i", "j").unique().join(truth, on=["i", "j"])
        per = hit.join(cc, on="i").group_by("country_key").len()
        tot = truth.join(cc, on="i").group_by("country_key").len()
        by = {r[0]: r[1] / r[2] for r in per.join(tot, on="country_key").iter_rows()}
        print(f"{name:<28} recall {hit.height / n_true:.4f} | " + " | ".join(f"{k} {v:.4f}" for k, v in sorted(by.items()))
              + f" | cands/S1 {p.select('i', 'j').unique().height / len(ev):.1f}")

    for kd in (5, 10, 20, 30):
        rec(dense.filter(pl.col("dense_rank") <= kd), f"dense top-{kd}")
    rec(key, "key top-30")
    for kd in (5, 10, 20):
        rec(pl.concat([key.select("i", "j"), dense.filter(pl.col("dense_rank") <= kd).select("i", "j")]),
            f"UNION key-30 + dense-{kd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "train", "embed", "knn", "eval"])
    ap.add_argument("--data_dir", default=C.DATA_DIR)
    ap.add_argument("--export_dir", default=EXPORT_DIR)
    ap.add_argument("--out_dir", default=C.WORK_DIR)
    ap.add_argument("--emb_dir", default=C.WORK_DIR)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prefix", default="query: ", help="text prepended to each record: 'query: ' for e5, '' for bge-m3")
    ap.add_argument("--model_path", default=os.path.join(C.WORK_DIR, "dense_model"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--queries", default="", help="parquet with column r (default: all S1)")
    ap.add_argument("--dense", default=os.path.join(C.WORK_DIR, "dense_train.parquet"))
    ap.add_argument("--n_s1", type=int, default=400_000)
    ap.add_argument("--hard_k", type=int, default=5)
    ap.add_argument("--country", default="", help="train on one country only (France-transfer proxy)")
    ap.add_argument("--max_triples", type=int, default=1_500_000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--mini_bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=128, help="queries per GPU matmul (lower = less GPU memory)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    {"export": cmd_export, "train": cmd_train, "embed": cmd_embed, "knn": cmd_knn, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()

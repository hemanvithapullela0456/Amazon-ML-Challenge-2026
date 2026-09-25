"""Scalable candidate generation: inverted-index joins on blocking keys + a learned ranker.

Key families (prep.KEY_TYPES plus two address-word families built here from token rarity):
  h,p,s,n,b,c,m : see prep.KEY_TYPES
  a             : a rare alphabetic address word (tolerates house-number typos / reordering)
  t             : a pair of the 3 rarest address words of the record (e.g. "laurie"+"cedar")
Only same-country records are joined. Keys shared by more than CAPS[type] S2/S3 records are dropped.
Candidates are ranked by a small LightGBM "meta-blocking" model over per-key-type evidence
(falls back to summed IDF when no ranker is trained) and the top K per S1 are kept.

Usage:
  python src/blocking.py --split train --sample 0.1 --train-ranker   (train ranker + recall report)
  python src/blocking.py --split train --sample 0.1                  (recall report with saved ranker)
"""
import argparse
import math
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
from normalize import STREET_WORDS
from prep import KEY_TYPES, load_norm

KEY_TYPES = {**KEY_TYPES, "a": "rare address word", "t": "pair of rarest address words"}
KT = list(KEY_TYPES)
CAPS = {"h": 300, "p": 300, "s": 300, "n": 200, "b": 200, "c": 200, "m": 300, "a": 100, "t": 100}
RANKER = os.path.join(C.WORK_DIR, "blk_ranker.txt")
_STREET = sorted(STREET_WORDS)


def _kt(code):
    return pl.lit(KT.index(code), dtype=pl.UInt8)


def load_records(split):
    df = load_norm(split, columns=["entity_id", "source", "country_key", "keys", "addr_core"])
    df = df.with_row_index("r").with_columns(pl.col("r").cast(pl.UInt32))
    codes = {c: n for n, c in enumerate(sorted(df["country_key"].unique().to_list()))}
    return df.with_columns(pl.col("country_key").replace_strict(codes, return_dtype=pl.UInt8).alias("cc"))


def _prep_keys(df):
    """(r, cc, kh, kt) for the key lists built in prep."""
    return (df.select("r", "cc", "keys").explode("keys").drop_nulls("keys")
            .select("r", "cc", pl.col("keys").hash().alias("kh"),
                    pl.col("keys").str.slice(0, 1).replace_strict(KT, list(range(len(KT))),
                                                                  return_dtype=pl.UInt8).alias("kt")))


def _addr_word_keys(recs, batch=2_000_000):
    """Rare address word keys (a) and pairs of the 3 rarest address words (t)."""
    parts = []
    for a in range(0, recs.height, batch):
        parts.append(recs.slice(a, batch)
                     .select("r", "cc", pl.col("addr_core").str.split(" ").alias("w")).explode("w")
                     .filter(pl.col("w").str.len_chars() >= 4, pl.col("w").str.contains("^[a-z]+$"),
                             ~pl.col("w").is_in(_STREET))
                     .unique(["r", "w"]).select("r", "cc", pl.col("w").hash().alias("wh")))
    w = pl.concat(parts)
    df = w.group_by("cc", "wh").agg(pl.len().alias("wdf"))
    w = w.join(df, on=["cc", "wh"])
    single = w.select("r", "cc", pl.col("wh").alias("kh"), _kt("a").alias("kt"))
    top3 = (w.with_columns(pl.col("wdf").rank("ordinal").over("r").alias("rk")).filter(pl.col("rk") <= 3)
            .select("r", "cc", "wh"))
    pairs = (top3.join(top3.select("r", pl.col("wh").alias("wh2")), on="r").filter(pl.col("wh") < pl.col("wh2"))
             .select("r", "cc", pl.struct("wh", "wh2").hash().alias("kh"), _kt("t").alias("kt")))
    return pl.concat([single, pairs])


def build_keys(recs, batch=1_000_000):
    """Usable keys of ALL records with IDF weights (keys too common in the S2/S3 pool are dropped)."""
    t = time.time()
    k = pl.concat([_prep_keys(recs.slice(a, batch)) for a in range(0, recs.height, batch)]
                  + [_addr_word_keys(recs)])
    src = recs.select("r", "source")
    k = k.join(src, on="r")
    n_x = int((recs["source"] != 1).sum())
    freq = k.filter(pl.col("source") != 1).group_by("cc", "kh", "kt").agg(pl.len().alias("df"))
    caps = pl.DataFrame({"kt": list(range(len(KT))), "cap": [CAPS[x] for x in KT]},
                        schema={"kt": pl.UInt8, "cap": pl.UInt32})
    freq = (freq.join(caps, on="kt").filter(pl.col("df") <= pl.col("cap"))
            .with_columns((pl.lit(math.log(n_x)) - pl.col("df").cast(pl.Float32).log()).cast(pl.Float32).alias("w"))
            .select("cc", "kh", "kt", "w"))
    k = k.join(freq, on=["cc", "kh", "kt"], how="inner")
    print(f"keys: {k.height} usable key rows for {recs.height} records ({time.time() - t:.0f}s)", flush=True)
    return k


BLK_FEATS = [f"w_{t}" for t in KT] + [f"n_{t}" for t in KT] + ["w_sum", "w_max", "n_keys"]


def _pair_evidence(m):
    return m.group_by("i", "j").agg(
        *[pl.col("w").filter(pl.col("kt") == n).sum().alias(f"w_{t}") for n, t in enumerate(KT)],
        *[(pl.col("kt") == n).sum().cast(pl.Float32).alias(f"n_{t}") for n, t in enumerate(KT)],
        pl.col("w").sum().alias("w_sum"), pl.col("w").max().alias("w_max"),
        pl.len().cast(pl.Float32).alias("n_keys"))


def candidates(keys, q_rows, k=30, chunk=40_000, truth=None, ranker=None, keep_evidence=False):
    """Top-k candidates per S1 in q_rows. Returns (pairs, stats) — stats only when truth is given."""
    xk = keys.filter(pl.col("source") != 1).select("r", "cc", "kh", "kt", "w")
    qk_all = keys.filter(pl.col("r").is_in(q_rows)).select("r", "cc", "kh")
    q_rows = np.sort(q_rows)
    out, stats = [], []
    for a in range(0, len(q_rows), chunk):
        qk = qk_all.filter(pl.col("r").is_in(q_rows[a:a + chunk]))
        m = (qk.join(xk, on=["cc", "kh"], how="inner")
             .select(pl.col("r").alias("i"), pl.col("r_right").alias("j"), "kt", "w"))
        g = _pair_evidence(m)
        score = ranker.predict(g.select(BLK_FEATS).to_numpy()) if ranker is not None else g["w_sum"].to_numpy()
        g = g.with_columns(pl.Series("blk_score", score.astype(np.float32)))
        g = g.with_columns(pl.col("blk_score").rank("ordinal", descending=True).over("i").alias("blk_rank"))
        if truth is not None:
            stats.append(g.join(truth, on=["i", "j"], how="inner").select("i", "j", "blk_rank", *[f"n_{t}" for t in KT]))
        g = g.filter(pl.col("blk_rank") <= k)
        out.append(g if keep_evidence else g.select("i", "j", "blk_score", "blk_rank", "n_keys", "w_max"))
    pairs = pl.concat(out)
    return pairs, (pl.concat(stats) if truth is not None else None)


def train_ranker(keys, truth, q_rows, max_pairs=6_000_000):
    """Supervised meta-blocking: LightGBM on per-key-type evidence of ALL key-sharing pairs."""
    t = time.time()
    xk = keys.filter(pl.col("source") != 1).select("r", "cc", "kh", "kt", "w")
    qk = keys.filter(pl.col("r").is_in(q_rows)).select("r", "cc", "kh")
    g = _pair_evidence(qk.join(xk, on=["cc", "kh"]).select(pl.col("r").alias("i"), pl.col("r_right").alias("j"), "kt", "w"))
    g = g.join(truth.with_columns(pl.lit(1).alias("y")), on=["i", "j"], how="left").with_columns(pl.col("y").fill_null(0))
    if g.height > max_pairs:
        g = g.sample(max_pairs, seed=C.SEED)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.9, verbose=-1, num_threads=C.N_JOBS, seed=C.SEED)
    model = lgb.train(params, lgb.Dataset(g.select(BLK_FEATS).to_numpy(), g["y"].to_numpy()), 300)
    model.save_model(RANKER)
    print(f"ranker trained on {g.height} pairs ({g['y'].sum()} positive) ({time.time() - t:.0f}s)", flush=True)
    return model


def load_ranker():
    return lgb.Booster(model_file=RANKER) if os.path.exists(RANKER) else None


def add_competition(pairs):
    """Candidate-side features over ALL S1s that retrieved the same candidate."""
    v = pl.col("blk_score")
    return pairs.with_columns(
        v.rank("min", descending=True).over("j").cast(pl.Float32).alias("blk_rank_c"),
        (v.max().over("j") - v).cast(pl.Float32).alias("blk_gap_c"),
        pl.len().over("j").cast(pl.Float32).alias("n_s1_per_cand"),
        (v.max().over("i") - v).cast(pl.Float32).alias("blk_gap_s1"),
    )


def add_dense(pairs, dense_path, k_add=C.DENSE_K, k_look=C.DENSE_LOOKUP):
    """Union key candidates (already carrying add_competition features) with the dense retriever's top-k_add.

    Dense S1-side features (cos, dense_rank, cos gap to the S1's best) come from the dense top-k_look list for
    every pair, key or dense. Candidate-side competition features stay KEY-ONLY: on train the dense run covered
    only the sampled S1s, on test all S1s, so counting dense pairs per candidate would mean different things."""
    d = (pl.read_parquet(dense_path).filter(pl.col("dense_rank") <= k_look)
         .with_columns((pl.col("cos").max().over("i") - pl.col("cos")).alias("cos_gap_s1"))
         .select("i", "j", pl.col("cos").cast(pl.Float32), pl.col("dense_rank").cast(pl.Float32),
                 pl.col("cos_gap_s1").cast(pl.Float32)))
    d = d.filter(pl.col("i").is_in(pairs["i"].unique().implode()))
    per_j = pairs.group_by("j").agg(pl.col("n_s1_per_cand").first())
    new = (d.filter(pl.col("dense_rank") <= k_add).select("i", "j")
           .join(pairs.select("i", "j"), on=["i", "j"], how="anti")
           .join(per_j, on="j", how="left")
           .with_columns(pl.col("n_s1_per_cand").fill_null(0)))
    out = pl.concat([pairs.with_columns(pl.lit(True).alias("from_key")),
                     new.with_columns(pl.lit(False).alias("from_key"))], how="diagonal_relaxed")
    # key pairs outside the dense top-k_look keep null cos / dense_rank (LightGBM treats them as missing)
    out = out.join(d, on=["i", "j"], how="left").with_columns(pl.col("from_key").cast(pl.Float32))
    print(f"dense: +{new.height} pairs from dense top-{k_add} (key pairs {pairs.height})", flush=True)
    return out


def truth_pairs(recs, split="train", s1_rows=None):
    """Ground truth as (i, j) record-index pairs."""
    gt = pl.read_csv(os.path.join(C.DATA_DIR, split, f"{split}_ground_truth.tsv"), separator="\t",
                     quote_char=None, infer_schema=False).fill_null("")
    gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .filter(pl.col("matched_entity_ids") != ""))
    ids = recs.select("entity_id", "r")
    t = (gt.join(ids, left_on="source1_entity_id", right_on="entity_id")
         .join(ids, left_on="matched_entity_ids", right_on="entity_id", suffix="_j")
         .select(pl.col("r").alias("i"), pl.col("r_j").alias("j")))
    return t if s1_rows is None else t.filter(pl.col("i").is_in(s1_rows))


def report(stats, n_true, n_s1, pairs):
    print(f"\n{n_s1} S1 queries, {n_true} true pairs")
    print(f"any key (no top-K cut): recall {stats.height / n_true:.4f}")
    for t in KT:
        hit = stats.filter(pl.col(f"n_{t}") > 0).height
        print(f"  key {t} ({KEY_TYPES[t]:<36}): recall alone {hit / n_true:.4f}")
    for k in (5, 10, 15, 20, 30, 50):
        print(f"  top-{k:<3}: recall {stats.filter(pl.col('blk_rank') <= k).height / n_true:.4f}")
    per = pairs.group_by("i").len()["len"]
    print(f"kept pairs {pairs.height} | cands/S1 mean {per.mean():.1f} | S1 with no candidate "
          f"{1 - per.len() / n_s1:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--train-ranker", action="store_true")
    a = ap.parse_args()
    t = time.time()
    recs = load_records(a.split)
    keys = build_keys(recs)
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    rng = np.random.default_rng(C.SEED)
    perm = rng.permutation(s1)
    q = np.sort(perm[: int(len(s1) * a.sample)])
    ranker = None
    if a.train_ranker:  # ranker trained on a DIFFERENT set of S1 than the one evaluated
        rq = np.sort(perm[-60_000:])
        ranker = train_ranker(keys, truth_pairs(recs, "train", rq), rq)
    elif a.split == "train":
        ranker = load_ranker()
    truth = truth_pairs(recs, a.split, q) if a.split == "train" else None
    for name, rk in ([("summed IDF", None)] if ranker is None else [("summed IDF", None), ("learned ranker", ranker)]):
        pairs, stats = candidates(keys, q, k=a.k, truth=truth, ranker=rk)
        print(f"\n######## ranking: {name} ({time.time() - t:.0f}s)")
        if truth is not None:
            report(stats, truth.height, len(q), pairs)
            cc = recs.select(pl.col("r").alias("i"), "country_key")
            tt = truth.join(cc, on="i").group_by("country_key").len()
            hh = stats.filter(pl.col("blk_rank") <= a.k).join(cc, on="i").group_by("country_key").len()
            print(tt.join(hh, on="country_key", suffix="_hit").with_columns((pl.col("len_hit") / pl.col("len")).alias("recall")))


if __name__ == "__main__":
    main()

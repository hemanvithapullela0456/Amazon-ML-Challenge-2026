"""Country-invariant word-role features, learned from each country's own UNLABELLED records.

The in-domain matcher knows, per country, which extra name words are copy noise (India: Sri/Dr/Center) and which
mark a sibling business (Exports/Developers, Private/Public). A matcher for an unseen country lacks these lists.
Here every country gets its own word statistics from its unlabelled records, computed the same way everywhere:
  ins(t) : t is the ONLY extra word between two records at the same address (house number + street key)
           -> copies of one business: noise words score high
  sub(t) : t is swapped for another word between two records on the same street (same address or not)
           -> sibling businesses: sibling-marker words score high
Pair features then describe the words that differ between S1 and candidate through these rates, so the model
learns a country-independent rule ("extra word with high insertion rate = harmless, swapped word with high
substitution rate = sibling") and, for France, applies French word roles without any French label.

    python src/word_roles.py --split train      -> work/word_roles_train.parquet
    python src/word_roles.py --split test       -> work/word_roles_test.parquet
"""
import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # one BLAS thread per process: workers x threads exhausted memory

import argparse
import os
import time

import numpy as np
import polars as pl

import config as C
from prep import load_norm

ROLE_FEATS = ["wr_n_extra", "wr_pure_insert", "wr_ins_max", "wr_ins_min", "wr_ins_mean", "wr_sub_max", "wr_sub_mean",
              "wr_role_min", "wr_role_mean", "wr_same_if_noise_dropped", "wr_extra_df_min", "wr_extra_df_max"]


def _pairs_in_groups(recs, prefix, max_group=25, max_pairs=4_000_000, seed=0):
    """record pairs that share a key of the given type (same address 'h:' / same street 's:')"""
    k = (recs.select("r", "cc", "keys").explode("keys").filter(pl.col("keys").str.starts_with(prefix))
         .unique(["r", "keys"]))
    size = k.group_by("cc", "keys").len()
    k = k.join(size.filter((pl.col("len") >= 2) & (pl.col("len") <= max_group)), on=["cc", "keys"]).select("r", "cc", "keys")
    p = k.join(k, on=["cc", "keys"]).filter(pl.col("r") < pl.col("r_right")).select("r", "r_right", "cc").unique(["r", "r_right"])
    if p.height > max_pairs:
        p = p.sample(max_pairs, seed=seed)
    return p


def word_stats(split):
    t = time.time()
    recs = (load_norm(split, columns=["country_key", "name_core", "keys", "addr_nums"]).with_row_index("r")
            .with_columns(pl.col("r").cast(pl.UInt32), pl.col("country_key").alias("cc")))
    names = recs["name_core"].fill_null("").to_list()
    first_num = [(x or "").split()[0] if (x or "").split() else "" for x in recs["addr_nums"].to_list()]
    out = []
    for cc in recs["cc"].unique().to_list():
        rc = recs.filter(pl.col("cc") == cc)
        df = (rc.select(pl.col("name_core").fill_null("").str.split(" ").list.unique().alias("w")).explode("w")
              .filter(pl.col("w").str.len_chars() >= 2).group_by("w").len().rename({"len": "df"}))
        ins, sub = {}, {}
        for prefix, same_addr in (("h:", True), ("s:", False)):
            pr = _pairs_in_groups(rc, prefix)
            for a, b in zip(pr["r"].to_list(), pr["r_right"].to_list()):
                A, B = set(names[a].split()), set(names[b].split())
                if not A or not B:
                    continue
                da, db = A - B, B - A
                if same_addr and ((len(da) == 1 and not db) or (len(db) == 1 and not da)):
                    w = next(iter(da or db))
                    ins[w] = ins.get(w, 0) + 1
                elif len(da) == 1 and len(db) == 1 and (same_addr or first_num[a] != first_num[b]):
                    for w in (next(iter(da)), next(iter(db))):
                        sub[w] = sub.get(w, 0) + 1
        s = df.with_columns(pl.col("w").replace_strict(ins, default=0, return_dtype=pl.Int64).alias("ins"),
                            pl.col("w").replace_strict(sub, default=0, return_dtype=pl.Int64).alias("sub"),
                            pl.lit(cc).alias("cc"))
        out.append(s)
        top_i = s.sort("ins", descending=True).head(12)["w"].to_list()
        top_s = s.sort("sub", descending=True).head(12)["w"].to_list()
        print(f"[{split}/{cc}] {rc.height} records | noise-like (insertions): {top_i}\n"
              f"      sibling-like (substitutions): {top_s} ({time.time() - t:.0f}s)", flush=True)
    st = pl.concat(out).with_columns(
        ((pl.col("ins") + 0.5) / (pl.col("df") + 20)).alias("ins_rate"),
        ((pl.col("sub") + 0.5) / (pl.col("df") + 20)).alias("sub_rate"),
        ((pl.col("ins") + 1) / (pl.col("ins") + pl.col("sub") + 2)).alias("role"))   # 1 = noise-like, 0 = sibling-like
    st.write_parquet(os.path.join(C.WORK_DIR, f"word_roles_{split}.parquet"))
    return st


_STATS = {}


def _init(stats):
    global _STATS
    _STATS = stats


def _one(args):
    n1, n2, cc = args
    stats = _STATS
    out = np.full((len(n1), len(ROLE_FEATS)), np.nan, np.float32)
    for k, (a, b, c) in enumerate(zip(n1, n2, cc)):
        A, B = set((a or "").split()), set((b or "").split())
        if not A or not B:
            continue
        da, db = A - B, B - A
        extra = list(da | db)
        st = stats.get(c, {})
        if not extra:
            out[k] = (0, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, np.nan, np.nan)
            continue
        v = [st.get(w, (0.0, 0.0, 0.5, 0)) for w in extra]
        ins = [x[0] for x in v]; sub = [x[1] for x in v]; role = [x[2] for x in v]; dfv = [x[3] for x in v]
        noise = {w for w, x in zip(extra, v) if x[2] >= 0.6}
        same = float((A - noise) == (B - noise) and len(A - noise) > 0)
        out[k] = (len(extra), float(not da or not db), max(ins), min(ins), float(np.mean(ins)), max(sub),
                  float(np.mean(sub)), min(role), float(np.mean(role)), same, float(np.log1p(min(dfv))),
                  float(np.log1p(max(dfv))))
    return out


def pair_features(pairs, norm, stats_df, n_jobs=4, chunk=200_000):
    """pairs: (i, j) record indices of one split. norm: r, name_core, country_key. -> pairs + ROLE_FEATS
    Word statistics go to each worker once (initializer); chunks are generated lazily to bound memory."""
    from multiprocessing import Pool
    stats = {}
    for cc, w, i_, s_, r_, d_ in stats_df.filter(pl.col("df") >= 2).select("cc", "w", "ins_rate", "sub_rate", "role", "df").iter_rows():
        stats.setdefault(cc, {})[w] = (i_, s_, r_, d_)
    d = (pairs.select("i", "j")
         .join(norm.select(pl.col("r").alias("i"), pl.col("name_core").alias("n1"), pl.col("country_key").alias("cc")), on="i", how="left")
         .join(norm.select(pl.col("r").alias("j"), pl.col("name_core").alias("n2")), on="j", how="left"))

    def jobs():
        for a in range(0, d.height, chunk):
            yield (d["n1"].slice(a, chunk).to_list(), d["n2"].slice(a, chunk).to_list(), d["cc"].slice(a, chunk).to_list())
    with Pool(n_jobs, initializer=_init, initargs=(stats,)) as pool:
        parts = list(pool.imap(_one, jobs(), chunksize=1))
    f = pl.DataFrame(np.vstack(parts) if parts else np.zeros((0, len(ROLE_FEATS)), np.float32), schema=ROLE_FEATS)
    return pl.concat([d.select("i", "j"), f], how="horizontal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    word_stats(ap.parse_args().split)

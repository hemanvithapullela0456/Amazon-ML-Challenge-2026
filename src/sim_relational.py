"""Collective (relational) second stage on the US/India OOF: sibling evidence for missed copies.

55% of the validation loss is S1s where some copies were found and others missed (median prob of a missed
copy 0.25). Copies of one business resemble EACH OTHER at least as much as they resemble the S1, so for a
band pair (i, k) we add: how similar k is to i's confidently matched copies (anchors, prob >= 0.9),
plus per-S1 context (rank, number of anchors, expected count) and whether another sampled S1 claims k.
A LightGBM on the band, grouped CV by S1 with the OOF folds, then threshold + one-to-one as before.

    python src/sim_relational.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config as C
import decide
from prep import load_norm

W = C.WORK_DIR
LO, HI, ANCHOR = 0.01, 0.99, 0.9
N_ROUNDS = int(os.environ.get("ROUNDS", 3))


def sibling(pairs, anchors, norm):
    """max / mean similarity of k to the S1's anchors j (j != k)."""
    x = pairs.select("i", pl.col("j").alias("k")).join(anchors.select("i", "j"), on="i").filter(pl.col("j") != pl.col("k"))
    t = norm.select("r", "name_core", "addr_core", "postcode", "addr_nums", "source")
    x = (x.join(t, left_on="k", right_on="r").join(t, left_on="j", right_on="r", suffix="_a"))
    feats = {
        "s_name": cpdist(x["name_core"].to_list(), x["name_core_a"].to_list(), scorer=fuzz.token_set_ratio, workers=4),
        "s_addr": cpdist(x["addr_core"].to_list(), x["addr_core_a"].to_list(), scorer=fuzz.token_set_ratio, workers=4),
    }
    x = x.with_columns(*[pl.Series(k, v.astype(np.float32)) for k, v in feats.items()],
                       ((pl.col("addr_nums") == pl.col("addr_nums_a")) & (pl.col("addr_nums") != "")).cast(pl.Float32).alias("s_num"),
                       (pl.col("source") == pl.col("source_a")).cast(pl.Float32).alias("s_same_src"))
    x = x.with_columns(((pl.col("s_name") + pl.col("s_addr")) / 2).alias("s_both"))
    sib = x.group_by("i", "k").agg(
        pl.col("s_name").max().alias("sib_name_max"), pl.col("s_addr").max().alias("sib_addr_max"),
        pl.col("s_both").max().alias("sib_both_max"), pl.col("s_both").mean().alias("sib_both_mean"),
        pl.col("s_both").min().alias("sib_both_min"),
        pl.col("s_num").max().alias("sib_num"),
        pl.col("s_same_src").max().alias("sib_same_src")).rename({"k": "j"})
    return sib.join(consensus(pairs, anchors, norm), on=["i", "j"], how="left")


def _tokens(df, col, who):
    return df.select("i", pl.col(col).str.split(" ").alias("w")).explode("w").filter(pl.col("w") != "").with_columns(pl.lit(who).alias("who"))


def consensus(pairs, anchors, norm):
    """entity consensus = S1 + its anchors. For candidate k: share of k's name / address words and numbers that
    the consensus contains, and how many consensus members support each word (majority evidence)."""
    t = norm.select("r", "name_core", "addr_core", "addr_nums")
    members = pl.concat([pairs.select("i").unique().with_columns(pl.col("i").alias("m")),
                         anchors.select("i", pl.col("j").alias("m"))]).unique()
    n_mem = members.group_by("i").len().rename({"len": "n_mem"})
    self_m = members.join(pairs.select("i", pl.col("j").alias("m")), on=["i", "m"], how="semi").select(
        "i", pl.col("m").alias("j"), pl.lit(1).alias("self"))
    out = pairs.select("i", "j")
    for col in ("name_core", "addr_core", "addr_nums"):
        mw = (members.join(t.select("r", col), left_on="m", right_on="r")
              .select("i", "m", pl.col(col).str.split(" ").alias("w")).explode("w").filter(pl.col("w") != "")
              .unique(["i", "m", "w"]).group_by("i", "w").len().rename({"len": "support"}))
        kw = (pairs.select("i", "j").join(t.select("r", col), left_on="j", right_on="r")
              .select("i", "j", pl.col(col).str.split(" ").alias("w")).explode("w").filter(pl.col("w") != "").unique(["i", "j", "w"]))
        # the candidate itself never counts as a consensus member (it may be an anchor)
        f = (kw.join(mw, on=["i", "w"], how="left").join(n_mem, on="i").join(self_m, on=["i", "j"], how="left")
             .with_columns(pl.col("self").fill_null(0))
             .with_columns((pl.col("support").fill_null(0) - pl.col("self")).alias("support"),
                           (pl.col("n_mem") - pl.col("self")).alias("n_mem"))
             .with_columns((pl.col("support") / pl.col("n_mem")).alias("frac"))
             .group_by("i", "j").agg((pl.col("support").fill_null(0) > 0).mean().alias(f"cov_{col}"),
                                     pl.col("frac").mean().alias(f"sup_{col}"),
                                     pl.col("frac").min().alias(f"supmin_{col}")))
        # consensus words the candidate lacks (weighted by support): missing majority evidence
        miss = (mw.join(n_mem, on="i").join(pairs.select("i", "j"), on="i")
                .join(self_m, on=["i", "j"], how="left").with_columns(pl.col("self").fill_null(0))
                .join(kw.with_columns(pl.lit(1).alias("has")), on=["i", "j", "w"], how="left")
                .with_columns((pl.col("support") - pl.col("self") * pl.col("has").fill_null(0)).alias("support"))
                .filter(pl.col("support") / (pl.col("n_mem") - pl.col("self")) >= 0.5)
                .group_by("i", "j").agg(pl.col("has").is_null().mean().alias(f"miss_{col}")))
        out = out.join(f, on=["i", "j"], how="left").join(miss, on=["i", "j"], how="left")
    return (out.join(n_mem, on="i", how="left").join(self_m, on=["i", "j"], how="left")
            .with_columns((pl.col("n_mem") - pl.col("self").fill_null(0)).cast(pl.Float32).alias("sib_n_mem")).drop("n_mem", "self"))


def context(o):
    return o.with_columns(
        pl.col("prob").rank("ordinal", descending=True).over("i").cast(pl.Float32).alias("rank_i"),
        (pl.col("prob") >= ANCHOR).sum().over("i").cast(pl.Float32).alias("n_anchor"),
        pl.col("prob").sum().over("i").alias("sum_p"),
        (pl.col("prob").max().over("i") - pl.col("prob")).alias("gap_top"),
        pl.len().over("j").cast(pl.Float32).alias("n_i_j"),
        (pl.col("prob").max().over("j") - pl.col("prob")).alias("gap_j"),
        pl.col("prob").rank("ordinal", descending=True).over("j").cast(pl.Float32).alias("rank_j"))


def main():
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G.index)
    ids = G.index.values
    o = pl.read_parquet(os.path.join(W, "train_oof.parquet"))
    band_ij = o.filter((pl.col("prob") > LO) & (pl.col("prob") < HI)).select("i", "j")
    norm = load_norm("train", columns=["name_core", "addr_core", "postcode", "addr_nums", "source"]).with_row_index("r")
    norm = norm.with_columns(pl.col("r").cast(o["j"].dtype), *[pl.col(c).fill_null("") for c in ("name_core", "addr_core", "postcode", "addr_nums")])
    s1f = [c for c in pd.read_json(os.path.join(W, "decision.json"), typ="series")["features"]]
    f1 = pl.read_parquet(os.path.join(W, "train_feats.parquet"), columns=["i", "j", *s1f]).join(band_ij, on=["i", "j"], how="semi").to_pandas()
    ctx = ["prob", "rank_i", "n_anchor", "sum_p", "gap_top", "n_i_j", "gap_j", "rank_j"]
    base = o.to_pandas()[["i", "j", "label", "prob"]]

    def evaluate(name, d):
        best = max((decide.score_pairs(d.assign(pred=decide.predict_mask(d, dict(method="threshold", t=t, one_to_one=True))),
                                       "pred", G, ids), t) for t in np.arange(0.4, 0.96, 0.05))
        cfg = dict(method="threshold", t=best[1], one_to_one=True)
        per = {c: decide.score_pairs(d[d.i.isin(gc.index[gc == c])].assign(pred=decide.predict_mask(d[d.i.isin(gc.index[gc == c])], cfg)),
                                     "pred", G, gc.index[gc == c]) for c in ("us", "india")}
        print(f"{name}: F0.5 {best[0]:.5f} @t={best[1]:.2f} | us {per['us']:.4f} india {per['india']:.4f}", flush=True)

    evaluate("stage-1 OOF baseline", base)
    cur = o  # current probabilities: stage-1 outside the band, round-r predictions inside it
    for rnd in range(1, N_ROUNDS + 1):
        c = context(cur)
        band = c.join(band_ij, on=["i", "j"], how="semi")
        sib = sibling(band, c.filter(pl.col("prob") >= ANCHOR), norm)
        b = band.join(sib, on=["i", "j"], how="left").to_pandas().merge(f1, on=["i", "j"], how="left")
        sibf = [x for x in sib.columns if x.split("_")[0] in ("sib", "cov", "sup", "supmin", "miss")]
        # round r's model sees the ORIGINAL stage-1 prob plus the round-(r-1) prob
        b = b.merge(base[["i", "j", "prob"]].rename(columns={"prob": "prob_s1"}), on=["i", "j"])
        cols = s1f + ctx + sibf + (["prob_s1"] if rnd > 1 else [])
        p = np.zeros(len(b))
        for f in sorted(b.fold.unique()):
            tr, va = (b.fold != f).values, (b.fold == f).values
            dtr = lgb.Dataset(b.loc[tr, cols], b.loc[tr, "label"].astype(int))
            dva = lgb.Dataset(b.loc[va, cols], b.loc[va, "label"].astype(int), reference=dtr)
            m = lgb.train(C.LGB_PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            p[va] = m.predict(b.loc[va, cols], num_iteration=m.best_iteration)
        upd = pl.DataFrame({"i": b.i.values, "j": b.j.values, "p_new": p}).with_columns(
            pl.col("i").cast(o["i"].dtype), pl.col("j").cast(o["j"].dtype))
        cur = (o.drop("prob").join(cur.select("i", "j", "prob"), on=["i", "j"]).join(upd, on=["i", "j"], how="left")
               .with_columns(pl.coalesce("p_new", "prob").alias("prob")).drop("p_new"))
        if rnd == 1:
            for lab in (True, False):
                x = b[b.label == lab]
                print(f"  label {lab}: cov_name {x.cov_name_core.mean():.3f} miss_addr {x.miss_addr_core.mean():.3f} "
                      f"cov_nums {x.cov_addr_nums.mean():.3f}")
        evaluate(f"round {rnd} (anchors {c.filter(pl.col('prob') >= ANCHOR).height})", cur.to_pandas()[["i", "j", "label", "prob"]])
    cur.select("i", "j", "prob").write_parquet(os.path.join(W, "sim_relational_oof.parquet"))


if __name__ == "__main__":
    main()

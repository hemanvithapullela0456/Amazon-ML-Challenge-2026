"""Experiment: do entity-level sibling features (per-source copy counts, script mix of the member's copies,
similarity to the member's copy names) improve same-name owner choice for name-only copies?"""
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
import owner as O
from prep import load_norm

W = C.WORK_DIR
NONLATIN = "[" + "".join(f"{a}-{b}" for a, b in O.SCRIPTS) + "]"


def sibling_table(n, own_pairs):
    """own_pairs: (o, j) address-bearing copies believed to belong to S1 o -> per-o sibling stats"""
    s = own_pairs.join(n.select(pl.col("r").alias("j"), "business_name", "source", "name_legal"), on="j")
    s = s.with_columns(nl=pl.col("business_name").str.contains(NONLATIN))
    return s.group_by("o").agg(
        pl.len().alias("sib_n"), (pl.col("source") == 2).sum().alias("sib_n2"), (pl.col("source") == 3).sum().alias("sib_n3"),
        pl.col("nl").mean().alias("sib_nl"), pl.col("business_name").str.to_lowercase().alias("sib_names"),
        pl.col("name_legal").alias("sib_legs"))


def add_sib(x, sib):
    x = x.join(sib.rename({"o": "c"}), on="c", how="left").with_columns(
        pl.col("sib_n", "sib_n2", "sib_n3").fill_null(0), pl.col("sib_nl").fill_null(-1))
    x = x.with_columns(
        sib_same=pl.when(pl.col("src") == 2).then("sib_n2").otherwise("sib_n3"),
        sib_other=pl.when(pl.col("src") == 2).then("sib_n3").otherwise("sib_n2"),
        rnl=pl.col("rbn").str.contains(NONLATIN).cast(pl.Int32),
        sib_leg_any=pl.col("sib_legs").list.contains(pl.col("rleg")).fill_null(False).cast(pl.Int32),
        sib_name_eq=pl.col("sib_names").list.contains(pl.col("rbnl")).fill_null(False).cast(pl.Int32))
    key = ["j", "country_key", "g"]
    x = x.with_columns(
        (pl.col("sib_same") - pl.col("sib_same").mean().over(key)).alias("sib_same_rel"),
        (pl.col("sib_n") - pl.col("sib_n").mean().over(key)).alias("sib_n_rel"),
        (pl.col("sib_nl") - pl.col("sib_nl").mean().over(key)).alias("sib_nl_rel"),
        (pl.col("sib_n") == 0).cast(pl.Int32).sum().over(key).alias("sib_zero_g"),
        pl.col("sib_leg_any").sum().over(key).alias("sib_leg_any_g"))
    return x


SIB = ["sib_n", "sib_same", "sib_other", "sib_nl", "rnl", "sib_leg_any", "sib_name_eq", "sib_same_rel", "sib_n_rel",
       "sib_nl_rel", "sib_zero_g", "sib_leg_any_g"]


def main():
    n, s1, m, rec = O.tables("train")
    n2 = load_norm("train", columns=["entity_id", "business_address", "name_legal"]).with_row_index("r")
    gt = (pl.read_parquet(os.path.join(W, "_gt_long.parquet"))
          .join(n2.select(pl.col("r").alias("o"), pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id")
          .join(n2.select(pl.col("r").alias("j"), pl.col("entity_id").alias("matched_entity_ids")), on="matched_entity_ids")
          .select("o", "j"))
    addr = n.filter(pl.col("source") != 1).select(pl.col("r").alias("j"), "noaddr")
    gt = gt.join(addr, on="j")
    sib = sibling_table(n.with_columns(n2["name_legal"]), gt.filter(~pl.col("noaddr")).select("o", "j"))
    own = rec.select("j", "rcore").join(gt.filter("noaddr").select("o", "j"), on="j", how="left")
    gname = n.filter(pl.col("source") == 1).select(pl.col("r").alias("own"), "country_key", pl.col("name_core").alias("g"))
    g1 = own.rename({"o": "own"}).join(gname, on="own").select("j", "country_key", "g")
    g2 = own.join(n.select(pl.col("r").alias("j"), "country_key"), on="j").select("j", "country_key", pl.col("rcore").alias("g"))
    groups = pl.concat([g1, g2]).unique()
    x = O.expand(groups, s1, m, rec).join(own.select("j", pl.col("o").alias("own")), on="j", how="left")
    x = x.with_columns(y=(pl.col("c") == pl.col("own")).fill_null(False), fold=O.fold_of())
    x = x.filter(pl.col("m") >= 2)
    x = add_sib(x, sib)
    print(f"rows {x.height} | records {x['j'].n_unique()} | positives {x['y'].sum()}", flush=True)
    y, fo = x["y"].to_numpy(), x["fold"].to_numpy()
    for name, feats in (("base", O.FEATS), ("base+sib", O.FEATS + SIB)):
        X = x.select([pl.col(f).cast(pl.Float32) for f in feats]).to_numpy()
        oof = np.zeros(len(y))
        for f in range(C.N_FOLDS):
            tr = fo != f
            mdl = lgb.train(O.PARAMS, lgb.Dataset(X[tr], y[tr]), 300)
            oof[~tr] = mdl.predict(X[~tr])
        xo = x.select("j", "m", "y").with_columns(p=pl.Series(oof))
        top = xo.sort("p", descending=True).group_by("j", maintain_order=True).first()
        msg = [f"{name}: top-1 {top['y'].mean():.4f} (1/m {(1 / top['m']).mean():.4f})"]
        for th in (0.7, 0.8, 0.9):
            s = top.filter(pl.col("p") >= th)
            msg.append(f"p>={th}: n {s.height} acc {s['y'].mean():.3f} tp {int(s['y'].sum())}")
        print(" | ".join(msg), flush=True)
        if name == "base+sib":
            imp = sorted(zip(mdl.feature_importance("gain"), feats), reverse=True)[:14]
            print("importance:", ", ".join(f"{k}:{v:.0f}" for v, k in imp))
            xo.with_columns(x["c"]).write_parquet(os.path.join(W, "_own_sib_oof.parquet"))


if __name__ == "__main__":
    main()

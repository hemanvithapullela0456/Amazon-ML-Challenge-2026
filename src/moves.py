"""Generator-move features (component 1) + alias clusters (component 2).

The data is synthetic: records are distorted copies of latent entities. Measured on train truth: 12.9% of true
copies share NO name word with their S1 (name replaced by an alias / translated / other script); when an entity has
>= 2 alias-named copies they share ONE alias 56.7% of the time; an alias is used by a median of 1 record (p90 3).
Decoys: same name with a shifted house number, or same address with another name.
Every feature here describes WHICH MOVE turns the S1 into the candidate, not which words it uses, so it means the
same in every country (France included).

  move features  : name relation (equal / noised / disjoint-alias / other script), address dropped, house-number
                   equal / shifted by delta, number-set overlap, city-tail overlap
  alias features : global frequency of the candidate's exact name among S2/S3; whether the candidate's (rare) name
                   equals the name of one of the S1's confident copies; micro-cluster (same exact name + same house
                   number) best probability among the S1's other candidates

    python src/moves.py --eval            (US/India OOF band + France-like simulation)
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import polars as pl

import config as C

W = C.WORK_DIR
NONLATIN = r"[^\x00-\x7FÀ-ɏ]"


def first_num(col):
    return pl.col(col).str.extract(r"(\d+)", 1).cast(pl.Int64, strict=False)


def move_features(pairs, norm):
    """pairs: (i, j, prob) ; norm: r, name_core, addr_core, addr_nums, business_name, business_address, source"""
    t = norm.select("r", "name_core", "addr_core", "addr_nums", "business_name", "business_address")
    x = (pairs.join(t.rename({c: c + "_i" for c in t.columns if c != "r"}), left_on="i", right_on="r")
         .join(t.rename({c: c + "_j" for c in t.columns if c != "r"}), left_on="j", right_on="r"))
    wi, wj = pl.col("name_core_i").fill_null("").str.split(" "), pl.col("name_core_j").fill_null("").str.split(" ")
    ai, aj = pl.col("addr_core_i").fill_null(""), pl.col("addr_core_j").fill_null("")
    hi, hj = first_num("addr_nums_i"), first_num("addr_nums_j")
    ni, nj = pl.col("addr_nums_i").fill_null("").str.split(" "), pl.col("addr_nums_j").fill_null("").str.split(" ")
    tail = lambda c: pl.col(c).fill_null("").str.split(" ").list.tail(2)
    x = x.with_columns(
        (pl.col("name_core_i") == pl.col("name_core_j")).cast(pl.Float32).alias("mv_name_equal"),
        (wi.list.set_intersection(wj).list.len() / wi.list.set_union(wj).list.len().clip(1)).cast(pl.Float32).alias("mv_name_jacc"),
        (wi.list.set_intersection(wj).list.len() == 0).cast(pl.Float32).alias("mv_name_disjoint"),
        ((wj.list.len() == 1) & (wi.list.set_intersection(wj).list.len() == 0)
         & pl.col("name_core_j").fill_null("").str.contains(r"^[a-z]{5,}$")).cast(pl.Float32).alias("mv_alias_word"),
        (pl.col("business_name_j").fill_null("").str.contains(NONLATIN) != pl.col("business_name_i").fill_null("").str.contains(NONLATIN)).cast(pl.Float32).alias("mv_script_diff"),
        (aj == "").cast(pl.Float32).alias("mv_addr_dropped"),
        (ai == "").cast(pl.Float32).alias("mv_s1_addr_empty"),
        (hi == hj).cast(pl.Float32).alias("mv_hn_equal"),
        (hj - hi).abs().cast(pl.Float32).alias("mv_hn_delta"),
        (((hj - hi).abs() > 0) & ((hj - hi).abs() <= 20)).cast(pl.Float32).alias("mv_hn_small_shift"),
        (ni.list.set_intersection(nj).list.len() / ni.list.set_union(nj).list.len().clip(1)).cast(pl.Float32).alias("mv_nums_jacc"),
        (tail("addr_core_i").list.set_intersection(tail("addr_core_j")).list.len()).cast(pl.Float32).alias("mv_city_tail"),
    )
    # joint move codes the trees can split on directly: "same name + shifted number" = decoy signature
    x = x.with_columns(
        ((pl.col("mv_name_jacc") >= 0.8) & (pl.col("mv_hn_small_shift") == 1)).cast(pl.Float32).alias("mv_decoy_shift"),
        ((pl.col("mv_name_disjoint") == 1) & (pl.col("mv_hn_equal") == 1)).cast(pl.Float32).alias("mv_alias_same_hn"),
    )
    return x


def alias_features(x, norm, anchor=0.9):
    """global alias frequency + alias shared with a confident copy + micro-cluster propagation (within S1)."""
    freq = norm.filter(pl.col("source") != 1).group_by("name_core").len().rename({"len": "al_name_freq"})
    x = x.join(freq, left_on="name_core_j", right_on="name_core", how="left").with_columns(
        pl.col("al_name_freq").fill_null(1).cast(pl.Float32))
    anc = x.filter(pl.col("prob") >= anchor).select("i", pl.col("j").alias("k"), pl.col("name_core_j").alias("nm"),
                                                   pl.col("addr_nums_j").alias("nums_k"))
    # candidate j shares its exact (rare) name with ANOTHER confident copy k of the same S1
    sh = (x.select("i", "j", "name_core_j").join(anc, left_on=["i", "name_core_j"], right_on=["i", "nm"])
          .filter(pl.col("j") != pl.col("k")).group_by("i", "j").len().rename({"len": "al_shared_with_anchor"}))
    x = x.join(sh, on=["i", "j"], how="left").with_columns(pl.col("al_shared_with_anchor").fill_null(0).cast(pl.Float32))
    # micro-cluster = same exact name + same first house number among this S1's candidates -> best other prob
    x = x.with_columns(first_num("addr_nums_j").fill_null(-1).alias("_hn"))
    x = x.with_columns(
        pl.len().over("i", "name_core_j", "_hn").cast(pl.Float32).alias("al_cluster_size"),
        pl.col("prob").max().over("i", "name_core_j", "_hn").alias("_cmax"),
        pl.col("prob").sum().over("i", "name_core_j", "_hn").alias("_csum"),
    ).with_columns(
        pl.when(pl.col("al_cluster_size") > 1).then(pl.max_horizontal(
            pl.when(pl.col("_cmax") > pl.col("prob")).then(pl.col("_cmax")).otherwise(
                (pl.col("_csum") - pl.col("prob")) / (pl.col("al_cluster_size") - 1)))).otherwise(None)
        .cast(pl.Float32).alias("al_cluster_other_max"))
    return x.drop("_hn", "_cmax", "_csum")


FEATS = None


def feature_names(x):
    return [c for c in x.columns if c.startswith("mv_") or c.startswith("al_")]


def build(split, pairs):
    from prep import load_norm
    norm = (load_norm(split, columns=["name_core", "addr_core", "addr_nums", "business_name", "business_address", "source"])
            .with_row_index("r").with_columns(pl.col("r").cast(pairs["j"].dtype),
                                              *[pl.col(c).fill_null("") for c in ("name_core", "addr_core", "addr_nums")]))
    x = alias_features(move_features(pairs, norm), norm)
    return x.select("i", "j", *feature_names(x))


def evaluate():
    import lightgbm as lgb
    import decide
    from scipy.stats import rankdata
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    ids = G.index.values
    o = pl.read_parquet(os.path.join(W, "train_oof.parquet"))
    mv = build("train", o.select("i", "j", "prob"))
    mv.write_parquet(os.path.join(W, "moves_train.parquet"))
    fn = [c for c in mv.columns if c not in ("i", "j")]
    band = o.filter((pl.col("prob") > 0.01) & (pl.col("prob") < 0.99))
    s1f = list(pd.read_json(os.path.join(W, "decision.json"), typ="series")["features"])
    f1 = pl.read_parquet(os.path.join(W, "train_feats.parquet"), columns=["i", "j", *s1f]).join(band.select("i", "j"), on=["i", "j"], how="semi")
    b = band.join(f1, on=["i", "j"], how="left").join(mv, on=["i", "j"], how="left").to_pandas()
    for lab in (True, False):
        s = b[b.label == lab]
        print(f"  label {lab}: " + " ".join(f"{c} {s[c].mean():.3f}" for c in
              ("mv_name_disjoint", "mv_alias_word", "mv_addr_dropped", "mv_hn_small_shift", "mv_decoy_shift", "al_shared_with_anchor")))
    base = o.to_pandas()[["i", "j", "label", "prob"]]

    def run(cols, name):
        p = np.zeros(len(b))
        for f in sorted(b.fold.unique()):
            tr, va = (b.fold != f).values, (b.fold == f).values
            dtr = lgb.Dataset(b.loc[tr, cols], b.loc[tr, "label"].astype(int))
            dva = lgb.Dataset(b.loc[va, cols], b.loc[va, "label"].astype(int), reference=dtr)
            m = lgb.train(C.LGB_PARAMS, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
            p[va] = m.predict(b.loc[va, cols], num_iteration=m.best_iteration)
        d = base.merge(b[["i", "j"]].assign(p2=p), on=["i", "j"], how="left")
        d["prob"] = d.p2.fillna(d.prob)
        best = max((decide.score_pairs(d.assign(pred=decide.predict_mask(d, dict(method="threshold", t=t, one_to_one=True))),
                                       "pred", G, ids), round(t, 2)) for t in np.arange(0.5, 0.96, 0.05))
        print(f"{name}: F0.5 {best[0]:.5f} @t={best[1]}", flush=True)
    run(s1f + ["prob"], "US/India band: stage-1 features + prob")
    run(s1f + ["prob"] + fn, "US/India band: + move/alias features")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate()

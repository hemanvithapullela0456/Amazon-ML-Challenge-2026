"""US/India corrector: re-score uncertain ADDRESS pairs of the final stack with co-location / name-frequency features.

Trained on the stack's out-of-fold scores (stack_oof_*_owner), 5 folds by S1; applied to the US/India pairs of
test_scored_stage2_hyb2 (name-only pairs and France untouched). Validation (43.6k stack S1s, t 0.725):
0.98565 -> ~0.9864.

    python src/coloc_corr.py            -> work/test_scored_stage2_hyb3.parquet + decision_stage2_hyb3.json
"""
import argparse
import json
import os
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import coloc
import config as C
from decide import one_to_one

W = C.WORK_DIR
LO, HI = 0.005, 0.9995
FEATS = ["lg"] + [f for f in coloc.PAIR_FEATS if f not in ("r2_name_ps1", "r2_addr_rel", "na2")] + \
        ["co_n_other", "co_best_other", "co_gap"]
P = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, verbose=-1, num_threads=8, seed=0)


def build(pairs, split):
    rec = pl.read_parquet(os.path.join(W, f"coloc_rec_{split}.parquet"))
    top, na_cnt, n_addr = coloc.extra_tables(rec)
    x = coloc.extra_features(coloc.pair_features(pairs, rec), rec, top, na_cnt, n_addr)
    p = pl.col("prob").clip(1e-6, 1 - 1e-6)
    return x.with_columns(lg=(p / (1 - p)).log())


def f05(d, col, t, s1, G):
    d = one_to_one(d.assign(prob=d[col]))
    pr = d[d.prob >= t]
    g = pr.groupby("i").agg(tp=("label", "sum"), k=("label", "size")).reindex(s1, fill_value=0)
    Gv = G.reindex(s1, fill_value=0).values
    tp, k = g.tp.values, g.k.values
    return np.where(Gv == 0, (k == 0) * 1., np.where(k == 0, 0., 1.25 * tp / np.maximum(k + 0.25 * Gv, 1e-9))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--src", default="test_scored_stage2_hyb2.parquet")
    ap.add_argument("--out_tag", default="_hyb3")
    ap.add_argument("--oof", default="stack_oof_set_qwen_f0_mdeb_f0_q17_f0_graph_moves_owner.parquet")
    a = ap.parse_args()

    s = pl.read_parquet(os.path.join(W, a.oof))
    x = build(s, "train").to_pandas()
    band = (x.prob > LO) & (x.prob < HI) & (x.na2 == 0)
    s1 = np.unique(x.i.values)
    fold = pd.Series(np.random.RandomState(0).randint(0, 5, len(s1)), index=s1).reindex(x.i.values).values
    x["p2"] = x.prob
    for k in range(5):
        tr, te = band & (fold != k), band & (fold == k)
        m = lgb.train(P, lgb.Dataset(x.loc[tr, FEATS], x.loc[tr, "label"]), a.rounds)
        x.loc[te, "p2"] = m.predict(x.loc[te, FEATS])
    G = pl.read_parquet(os.path.join(W, "train_G.parquet")).to_pandas().set_index("i")["len"]
    for t in (0.7, 0.725, 0.75):
        print(f"val t={t}: stack {f05(x, 'prob', t, s1, G):.5f} -> corrected {f05(x, 'p2', t, s1, G):.5f}", flush=True)
    chg = x[band & ((x.prob >= 0.725) != (x.p2 >= 0.725))]
    print(f"val: {band.sum()} band pairs, {len(chg)} flip at 0.725 ({chg.i.nunique() / len(s1):.4f} of S1s)", flush=True)

    model = lgb.train(P, lgb.Dataset(x.loc[band, FEATS], x.loc[band, "label"]), a.rounds)
    del x
    t = pl.read_parquet(os.path.join(W, a.src))
    ck = pl.read_parquet(os.path.join(W, "coloc_rec_test.parquet"), columns=["r", "country_key"])
    ui = ck.filter(pl.col("country_key").is_in(["us", "india"])).select(pl.col("r").alias("i"))
    cand = t.join(ui, on="i").filter((pl.col("prob") > LO) & (pl.col("prob") < HI))
    y = build(cand, "test").filter(pl.col("na2") == 0)
    newp = model.predict(y.select(FEATS).to_pandas())
    y = y.select("i", "j", "prob").with_columns(p2=pl.Series(newp.astype(np.float32)))
    n_s1 = ui.height
    fl = y.filter((pl.col("prob") >= 0.725) != (pl.col("p2") >= 0.725))
    print(f"test: {y.height} band address pairs, {fl.height} flip at 0.725 ({fl['i'].n_unique() / n_s1:.4f} of "
          f"US/India S1s); up {(fl['p2'] >= 0.725).sum()} down {(fl['p2'] < 0.725).sum()}", flush=True)
    out = (t.join(y.select("i", "j", "p2"), on=["i", "j"], how="left")
           .with_columns(prob=pl.coalesce("p2", "prob").cast(t["prob"].dtype)).drop("p2"))
    out.write_parquet(os.path.join(W, f"test_scored_stage2{a.out_tag}.parquet"))
    shutil.copy(os.path.join(W, "decision_stage2_hyb2.json"), os.path.join(W, f"decision_stage2{a.out_tag}.json"))
    print("wrote", f"test_scored_stage2{a.out_tag}.parquet", flush=True)


if __name__ == "__main__":
    main()

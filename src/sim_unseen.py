"""Unseen-country simulation (a labelled stand-in for France).

France is absent from training, so its decision rule cannot be tuned on labels. Here India plays France:
the dense retriever was trained on US only (dense.py train --country us -> work/dense_train_us.parquet) and the
matcher is trained on US S1s only, then scored on India S1s. Reports India F0.5 for
  - the matcher with / without the dense-retriever features (cos, dense_rank, cos_gap_s1)
  - a sweep of decision thresholds (the --unseen_t rule of run_pipeline.py)
Caveat: the key-blocking ranker was trained on both countries (a small leak that favours key blocking).

    python src/sim_unseen.py --dense work/dense_train_us.parquet
Artifacts go to work/sim_us2in/ (the real model in work/ is not touched).
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide
from blocking import add_competition, add_dense, build_keys, candidates, load_ranker, load_records, truth_pairs
from features import feature_cols, make_features
from train import norm_table

DENSE_FEATS = ["cos", "dense_rank", "cos_gap_s1"]


def build(dense_path, out_dir, n_s1=200_000, k=30):
    fpath = os.path.join(out_dir, "feats.parquet")
    if os.path.exists(fpath):
        print("loading cached simulation features")
        return pl.read_parquet(fpath)
    t = time.time()
    recs = load_records("train")
    keys = build_keys(recs)
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    sample = np.sort(np.random.default_rng(C.SEED).choice(s1, min(n_s1, len(s1)), replace=False))  # = train.py
    truth = truth_pairs(recs, "train")
    pairs, _ = candidates(keys, s1, k=k, ranker=load_ranker())
    del keys
    pairs = add_competition(pairs).filter(pl.col("i").is_in(sample))
    pairs = add_dense(pairs, dense_path)
    print(f"{pairs.height} pairs ({time.time() - t:.0f}s)", flush=True)
    feats = make_features(pairs, norm_table("train"))
    feats = (feats.join(truth.with_columns(pl.lit(True).alias("label")), on=["i", "j"], how="left")
             .with_columns(pl.col("label").fill_null(False))
             .join(recs.select(pl.col("r").alias("i"), "country_key"), on="i", how="left"))
    feats.write_parquet(fpath)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default=os.path.join(C.WORK_DIR, "dense_train_us.parquet"))
    ap.add_argument("--train_country", default="us")
    ap.add_argument("--eval_country", default="india")
    ap.add_argument("--rounds", type=int, default=2000)
    a = ap.parse_args()
    out_dir = os.path.join(C.WORK_DIR, "sim_us2in")
    os.makedirs(out_dir, exist_ok=True)

    df = build(a.dense, out_dir).to_pandas()
    G_pl = pl.read_parquet(os.path.join(C.WORK_DIR, "train_G.parquet"))  # true-match counts incl. blocking misses
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G_pl["i"].to_numpy())
    ev_ids = gc.index[gc.values == a.eval_country]
    tr, ev = df[df["country_key"] == a.train_country], df[df["country_key"] == a.eval_country].copy()
    print(f"train {a.train_country}: {len(tr)} pairs | eval {a.eval_country}: {len(ev)} pairs, "
          f"candidate recall {ev['label'].sum() / G.loc[ev_ids].sum():.4f}", flush=True)

    all_feats = feature_cols(df)
    for name, feats in [("with dense feats", all_feats),
                        ("WITHOUT dense feats", [f for f in all_feats if f not in DENSE_FEATS])]:
        t = time.time()
        m = lgb.train(C.LGB_PARAMS, lgb.Dataset(tr[feats], tr["label"].astype(int)), a.rounds)
        ev["prob"] = m.predict(ev[feats])
        print(f"\n== matcher {name} ({len(feats)} features, {time.time() - t:.0f}s)")
        for t_ in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99):
            d = ev.assign(pred=decide.predict_mask(ev, dict(method="threshold", t=t_, one_to_one=True)))
            print(f"  t={t_:<5} {a.eval_country} F0.5 {decide.score_pairs(d, 'pred', G, ev_ids):.4f}", flush=True)
        ev[["i", "j", "label", "prob"]].to_parquet(os.path.join(out_dir, f"eval_{name.split()[0].lower()}.parquet"))


if __name__ == "__main__":
    main()

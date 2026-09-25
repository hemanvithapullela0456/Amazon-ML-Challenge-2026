"""Stage-1 training: blocking for all train S1 -> features for a sample of S1 -> grouped CV LightGBM
-> decision tuning -> final model.

Usage:  python src/train.py [--n_s1 200000] [--k 30] [--no-cache] [--country-check]
Needs work/train_norm.parquet (python src/prep.py --split train).
Artifacts in work/: train_feats.parquet, train_oof.parquet, model.txt, decision.json
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

import config as C
import decide
from blocking import add_competition, add_dense, build_keys, candidates, load_ranker, load_records, report, truth_pairs
from features import FIELDS, feature_cols, make_features
from prep import load_norm


def norm_table(split):
    return (load_norm(split, columns=["source", "country_key", *FIELDS]).with_row_index("r")
            .with_columns(pl.col("r").cast(pl.UInt32)))


def build_train(n_s1, k, use_cache=True, quick=False, dense=""):
    fpath = os.path.join(C.WORK_DIR, "train_feats.parquet")
    gpath = os.path.join(C.WORK_DIR, "train_G.parquet")
    if use_cache and os.path.exists(fpath):
        print("loading cached train features")
        return pl.read_parquet(fpath), pl.read_parquet(gpath)
    t = time.time()
    recs = load_records("train")
    keys = build_keys(recs)
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    rng = np.random.default_rng(C.SEED)
    sample = np.sort(rng.choice(s1, min(n_s1, len(s1)), replace=False))
    if quick:  # smoke test: block only the sample (competition features then under-count)
        s1 = sample
    truth = truth_pairs(recs, "train")
    ranker = load_ranker()
    assert ranker is not None, "train the blocking ranker first: python src/blocking.py --train-ranker"
    pairs, stats = candidates(keys, s1, k=k, truth=truth, ranker=ranker)
    report(stats, truth.height, len(s1), pairs)
    del keys, stats
    pairs = add_competition(pairs)  # over ALL S1s, as on test
    pairs = pairs.filter(pl.col("i").is_in(sample))
    if dense:
        pairs = add_dense(pairs, dense)
    print(f"sampled {len(sample)} S1 -> {pairs.height} pairs ({time.time() - t:.0f}s)", flush=True)

    feats = make_features(pairs, norm_table("train"))
    feats = (feats.join(truth.with_columns(pl.lit(True).alias("label")), on=["i", "j"], how="left")
             .with_columns(pl.col("label").fill_null(False)))
    cc = recs.select(pl.col("r").alias("i"), "country_key")
    feats = feats.join(cc, on="i", how="left")
    # number of true matches per sampled S1 (incl. those blocking missed) - needed for exact F0.5
    G = (pl.DataFrame({"i": sample}).with_columns(pl.col("i").cast(pl.UInt32))
         .join(truth.group_by("i").len(), on="i", how="left").with_columns(pl.col("len").fill_null(0))
         .join(cc, on="i", how="left"))
    feats.write_parquet(fpath)
    G.write_parquet(gpath)
    return feats, G


def fit(X, y, Xva=None, yva=None, rounds=3000):
    dtr = lgb.Dataset(X, y.astype(int), free_raw_data=True)
    if Xva is None:
        return lgb.train(C.LGB_PARAMS, dtr, rounds)
    dva = lgb.Dataset(Xva, yva.astype(int), reference=dtr)
    return lgb.train(C.LGB_PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])


def cross_validate(df, feats, G, s1_ids):
    rng = np.random.default_rng(C.SEED)
    fold_of = pd.Series(rng.permutation(len(s1_ids)) % C.N_FOLDS, index=s1_ids)
    df["fold"] = fold_of.reindex(df["i"].values).values
    df["prob"] = np.nan
    iters = []
    for f in range(C.N_FOLDS):
        tr, va = (df["fold"] != f).values, (df["fold"] == f).values
        m = fit(df.loc[tr, feats], df.loc[tr, "label"], df.loc[va, feats], df.loc[va, "label"])
        df.loc[va, "prob"] = m.predict(df.loc[va, feats], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        ids_f = fold_of.index[fold_of.values == f]
        d = df[va].assign(pred=df.loc[va, "prob"] >= 0.5)
        print(f"fold {f}: AUC {roc_auc_score(df.loc[va, 'label'], df.loc[va, 'prob']):.4f} | "
              f"F0.5@0.5 {decide.score_pairs(d, 'pred', G, ids_f):.4f} | iters {m.best_iteration}", flush=True)
    print(f"OOF AUC {roc_auc_score(df['label'], df['prob']):.4f}")
    return df, int(np.mean(iters) * 1.1) + 1


def country_check(df, feats, G, gcountry, cfg):
    """Train on one country, evaluate on another: proxy for the unseen test country (France)."""
    for c in df["country_key"].unique():
        tr, va = (df["country_key"] != c).values, (df["country_key"] == c).values
        if va.sum() < 1000 or tr.sum() < 1000:
            continue
        m = fit(df.loc[tr, feats], df.loc[tr, "label"], rounds=cfg["n_rounds"])
        d = df.loc[va].copy()
        d["prob"] = m.predict(d[feats])
        ids_c = gcountry.index[gcountry.values == c]
        d["pred"] = decide.predict_mask(d, cfg)
        ind = df.loc[va].assign(pred=decide.predict_mask(df.loc[va], cfg))
        print(f"  held-out {c!r}: F0.5 {decide.score_pairs(d, 'pred', G, ids_c):.4f} "
              f"(in-domain OOF {decide.score_pairs(ind, 'pred', G, ids_c):.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_s1", type=int, default=200_000)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--country-check", action="store_true")
    ap.add_argument("--no-expected", action="store_true")
    ap.add_argument("--dense", default="", help="dense pairs parquet to union in (e.g. work/dense_train.parquet)")
    ap.add_argument("--quick", action="store_true", help="smoke test: block only the sampled S1s")
    a = ap.parse_args()

    feats_pl, G_pl = build_train(a.n_s1, a.k, use_cache=not a.no_cache, quick=a.quick, dense=a.dense)
    df = feats_pl.to_pandas()
    del feats_pl
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gcountry = pd.Series(G_pl["country_key"].to_numpy(), index=G_pl["i"].to_numpy())
    s1_ids = G.index.values
    feats = feature_cols(df)
    print(f"{len(df)} pairs, {df['label'].sum()} positives, {len(feats)} features, "
          f"blocking ceiling: {df['label'].sum() / G.sum():.4f} of true pairs")

    df, n_rounds = cross_validate(df, feats, G, s1_ids)
    print("decision tuning (top configs):")
    cfg = decide.tune(df, G, s1_ids, try_expected=not a.no_expected)
    cfg["n_rounds"] = n_rounds
    print(f"best: {cfg}")
    for c in gcountry.unique():
        ids_c = gcountry.index[gcountry.values == c]
        d = df[df["i"].isin(ids_c)]
        print(f"  OOF F0.5 {c!r}: {decide.score_pairs(d.assign(pred=decide.predict_mask(d, cfg)), 'pred', G, ids_c):.4f}")
    if a.country_check:
        country_check(df, feats, G, gcountry, cfg)

    model = fit(df[feats], df["label"], rounds=n_rounds)
    model.save_model(os.path.join(C.WORK_DIR, "model.txt"))
    cfg["features"] = feats
    with open(os.path.join(C.WORK_DIR, "decision.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    pl.from_pandas(df[["i", "j", "label", "prob", "fold"]]).write_parquet(os.path.join(C.WORK_DIR, "train_oof.parquet"))
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("top features:\n" + imp.head(25).round(0).to_string())


if __name__ == "__main__":
    main()

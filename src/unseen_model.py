"""Matcher for test countries that never appear in training (France in this test set).

sim_unseen.py showed that on an unseen country the matcher does better WITHOUT the dense-retriever features
(cos, dense_rank, cos_gap_s1): the retriever's similarity is calibrated on the training countries only
(US-only retriever -> India: dense feats F0.5 0.823, no dense feats 0.849). The dense CANDIDATES are kept.

    python src/unseen_model.py --dense work/dense_test.parquet
1. trains LightGBM on work/train_feats.parquet (all training countries) without the dense features
2. rebuilds test candidates for the unseen-country S1s only (same blocking as run_pipeline.py; candidates and
   candidate-side competition never cross countries, so this subset is exact), scores them
-> work/test_scored_unseen.parquet (i, j, prob), used by run_pipeline.py --unseen_scores
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
from blocking import add_competition, add_dense, build_keys, candidates, load_ranker, load_records
from features import idf_tables, make_features
from train import norm_table

DENSE_FEATS = ["cos", "dense_rank", "cos_gap_s1"]
SYNTH_DROP = DENSE_FEATS + ["from_key"]   # from_key is constant in synthetic data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default=os.path.join(C.WORK_DIR, "dense_test.parquet"))
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=15_000)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--synth", default="", help="work/<dir> from synth.py: add its labelled synthetic rows to training")
    ap.add_argument("--synth_weight", type=float, default=1.0)
    ap.add_argument("--roles", action="store_true", help="add word-role features (word_roles.py) to the matcher")
    ap.add_argument("--tag", default="", help="suffix for model / scores files, e.g. _synth")
    a = ap.parse_args()
    t = time.time()
    W = C.WORK_DIR
    with open(os.path.join(W, "decision.json")) as f:
        cfg = json.load(f)
    drop = SYNTH_DROP if a.synth else DENSE_FEATS
    feats = [x for x in cfg["features"] if x not in drop]
    base_feats = list(feats)
    if a.roles:
        from word_roles import ROLE_FEATS, pair_features
        feats = feats + ROLE_FEATS

    def with_roles(df, split):
        if not a.roles:
            return df
        path = os.path.join(W, f"roles_{split}_pairs.parquet")
        if not os.path.exists(path):
            nt = norm_table(split).select("r", "name_core", "country_key")
            pair_features(df.select("i", "j"), nt, pl.read_parquet(os.path.join(W, f"word_roles_{split}.parquet"))).write_parquet(path)
        return df.join(pl.read_parquet(path), on=["i", "j"], how="left")

    mpath = os.path.join(W, f"model_nodense{a.tag}.txt")
    if os.path.exists(mpath):
        model = lgb.Booster(model_file=mpath)
    else:
        tr = with_roles(pl.read_parquet(os.path.join(W, "train_feats.parquet"), columns=["i", "j"] + base_feats + ["label"]), "train")
        X, y, w = tr.select(feats).to_numpy(), tr["label"].cast(pl.Int32).to_numpy(), np.ones(tr.height)
        if a.synth:
            sy = pl.read_parquet(os.path.join(W, a.synth, "feats.parquet"), columns=feats + ["label"])
            X = np.vstack([X, sy.select(feats).to_numpy()])
            y = np.concatenate([y, sy["label"].cast(pl.Int32).to_numpy()])
            w = np.concatenate([w, np.full(sy.height, a.synth_weight)])
            print(f"+ {sy.height} synthetic rows from {a.synth} (weight {a.synth_weight})", flush=True)
        model = lgb.train(C.LGB_PARAMS, lgb.Dataset(X, y, weight=w, feature_name=feats), cfg["n_rounds"])
        model.save_model(mpath)
        del tr, X
        print(f"unseen-country matcher trained ({len(feats)} features, {cfg['n_rounds']} rounds, {time.time() - t:.0f}s)", flush=True)

    fcache = os.path.join(W, "test_feats_unseen.parquet")
    if not os.path.exists(fcache):   # France test features do not depend on the model: compute once, then cache
        seen = set(pl.read_parquet(os.path.join(W, "train_G.parquet"))["country_key"].unique().to_list())
        recs = load_records("test")
        s1 = recs.filter((pl.col("source") == 1) & ~pl.col("country_key").is_in(list(seen)))["r"].to_numpy()
        print(f"unseen countries {sorted(set(recs['country_key'].unique().to_list()) - seen)}: {len(s1)} S1", flush=True)
        keys = build_keys(recs)
        pairs, _ = candidates(keys, s1, k=a.k, ranker=load_ranker())
        del keys
        pairs = add_dense(add_competition(pairs), a.dense)
        norm = norm_table("test")
        idfs = idf_tables(norm)
        fparts = []
        all_feats = list(dict.fromkeys([x for x in cfg["features"] if x not in DENSE_FEATS] + ["from_key"]))
        for a0 in range(0, len(s1), a.chunk):
            p = pairs.filter(pl.col("i").is_in(s1[a0:a0 + a.chunk]))
            f = make_features(p, norm, idfs=idfs, n_jobs=a.n_jobs)
            fparts.append(f.select("i", "j", *all_feats))
            print(f"  featurised unseen S1 {min(a0 + a.chunk, len(s1))}/{len(s1)} ({time.time() - t:.0f}s)", flush=True)
        pl.concat(fparts).write_parquet(fcache)
        del pairs, norm, fparts
    f = with_roles(pl.read_parquet(fcache), "test")
    out = f.select("i", "j").with_columns(pl.Series("prob", model.predict(f.select(feats).to_numpy()).astype(np.float32)))
    out.write_parquet(os.path.join(W, f"test_scored_unseen{a.tag}.parquet"))
    print(f"{out.height} pairs scored -> work/test_scored_unseen{a.tag}.parquet ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()

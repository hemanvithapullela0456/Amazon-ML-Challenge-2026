"""Hidden-owner simulation: does the test's missing-S1 condition hurt the in-domain matcher?

Test has ~5.7 S2/S3 records per S1 vs 4.67 in train, and only ~60% of S2/S3 records are "owned" (vs 74%):
roughly 19% of the S1s that own copies were withheld, so their copies sit in the pool as orphans. The only
features that see other S1s are the candidate-side competition features (blk_rank_c, blk_gap_c,
n_s1_per_cand): hiding owners changes nothing else (key IDF counts S2/S3 only; each S1's top-k is local).

  --step block : block ALL train S1s once, save (i, j, blk_score) -> work/sim_hidden_blk.parquet
  --step cv    : recompute competition features with --frac of the non-sampled S1s hidden, then grouped CV:
                 full/full (baseline), full-trained -> hidden-scored (what test looks like now),
                 hidden/hidden (retrained in the test condition)
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide

W = C.WORK_DIR
BLK = os.path.join(W, "sim_hidden_blk.parquet")
COMP = ["blk_rank_c", "blk_gap_c", "n_s1_per_cand"]


def block():
    from blocking import build_keys, candidates, load_ranker, load_records
    recs = load_records("train")
    keys = build_keys(recs)
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    pairs, _ = candidates(keys, s1, k=30, ranker=load_ranker())
    pairs.select("i", "j", "blk_score").write_parquet(BLK)
    print(f"blocked {len(s1)} S1 -> {pairs.height} pairs -> {BLK}")


def comp_features(blk, sample, hidden):
    p = blk.filter(~pl.col("i").is_in(hidden)) if hidden is not None else blk
    v = pl.col("blk_score")
    p = p.with_columns(v.rank("min", descending=True).over("j").cast(pl.Float32).alias("blk_rank_c"),
                       (v.max().over("j") - v).cast(pl.Float32).alias("blk_gap_c"),
                       pl.len().over("j").cast(pl.Float32).alias("n_s1_per_cand"))
    per_j = p.group_by("j").agg(pl.col("n_s1_per_cand").first().alias("npj"))
    return p.filter(pl.col("i").is_in(sample)).select("i", "j", *COMP), per_j


def apply_comp(feats, comp, per_j):
    """replace competition columns; dense-only pairs keep null rank/gap and get the per-candidate count"""
    out = (feats.drop(COMP).join(comp, on=["i", "j"], how="left").join(per_j, on="j", how="left")
           .with_columns(pl.coalesce("n_s1_per_cand", "npj").fill_null(0).alias("n_s1_per_cand")).drop("npj"))
    return out.select(feats.columns)


def fit(X, y, Xva, yva):
    dtr = lgb.Dataset(X, y.astype(int))
    dva = lgb.Dataset(Xva, yva.astype(int), reference=dtr)
    return lgb.train(C.LGB_PARAMS, dtr, 3000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])


def report(name, df, G, gc, s1_ids):
    cfg = dict(method="threshold", t=0.75, one_to_one=True)
    parts = []
    for c in ("us", "india"):
        ids = gc.index[gc.values == c]
        d = df[df.i.isin(ids)]
        parts.append(f"{c} {decide.score_pairs(d.assign(pred=decide.predict_mask(d, cfg)), 'pred', G, ids):.4f}")
    full = decide.score_pairs(df.assign(pred=decide.predict_mask(df, cfg)), "pred", G, s1_ids)
    best = max((decide.score_pairs(df.assign(pred=decide.predict_mask(df, dict(cfg, t=t))), "pred", G, s1_ids), t)
               for t in np.arange(0.5, 0.96, 0.05))
    print(f"{name}: F0.5@0.75 {full:.4f} ({', '.join(parts)}) | best {best[0]:.4f} @t={best[1]:.2f}", flush=True)


def cv(a):
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G_pl["i"].to_numpy())
    sample = G_pl["i"].to_numpy()
    blk = pl.read_parquet(BLK)
    s1_all = blk["i"].unique().to_numpy()
    others = np.setdiff1d(s1_all, sample)
    n_hide = int(round(a.frac * len(s1_all)))
    hidden = np.random.default_rng(a.seed).choice(others, min(n_hide, len(others)), replace=False)
    print(f"S1s blocked {len(s1_all)} | sampled {len(sample)} | hidden {len(hidden)} ({len(hidden) / len(s1_all):.1%})")

    feats = pl.read_parquet(os.path.join(W, "train_feats.parquet"))
    # top-30 cut-offs fall inside blocks of tied scores and polars breaks ties arbitrarily, so a fresh blocking
    # run keeps a slightly different candidate set: add the cached key pairs it lacks, then compute BOTH
    # conditions from this same union (only the hidden S1s differ between them)
    extra = (feats.filter(pl.col("from_key") == 1).select("i", "j", "blk_score")
             .join(blk.select("i", "j"), on=["i", "j"], how="anti"))
    print(f"cached key pairs missing from the fresh blocking: {extra.height}")
    blk = pl.concat([blk, extra.with_columns(pl.col("i").cast(blk["i"].dtype), pl.col("j").cast(blk["j"].dtype),
                                             pl.col("blk_score").cast(blk["blk_score"].dtype))])
    c_full, pj_full = comp_features(blk, sample, None)
    c_hid, pj_hid = comp_features(blk, sample, hidden)
    del blk
    full = apply_comp(feats, c_full, pj_full)
    hid = apply_comp(feats, c_hid, pj_hid)
    chk = feats.select(*COMP).to_numpy() - full.select(*COMP).to_numpy()
    print("recomputed full vs cached, share equal: " + ", ".join(
        f"{c} {np.mean(np.nan_to_num(chk[:, k], nan=0) == 0):.3f}" for k, c in enumerate(COMP)))
    moved = (hid["n_s1_per_cand"] != full["n_s1_per_cand"]).mean()
    print(f"pairs whose candidate lost at least one competing S1: {moved:.3f}", flush=True)
    feat_cols = [c for c in pd.read_json(os.path.join(W, "decision.json"), typ="series")["features"]]
    df_f = full.select("i", "label", *feat_cols).to_pandas()
    df_h = hid.select(*COMP).to_pandas()
    ij = feats.select("i", "j").to_pandas()
    del feats, hid, full
    Xh = df_f[feat_cols].copy()
    Xh[COMP] = df_h[COMP].values
    del df_h

    rng = np.random.default_rng(C.SEED)
    fold_of = pd.Series(rng.permutation(len(sample)) % a.folds, index=sample)
    fold = fold_of.reindex(df_f["i"].values).values
    y = df_f["label"].values
    p_ff, p_fh, p_hh = (np.full(len(y), np.nan) for _ in range(3))
    for f in range(a.folds):
        tr, va = fold != f, fold == f
        m = fit(df_f.loc[tr, feat_cols], y[tr], df_f.loc[va, feat_cols], y[va])
        p_ff[va] = m.predict(df_f.loc[va, feat_cols], num_iteration=m.best_iteration)
        p_fh[va] = m.predict(Xh.loc[va], num_iteration=m.best_iteration)
        m = fit(Xh.loc[tr], y[tr], Xh.loc[va], y[va])
        p_hh[va] = m.predict(Xh.loc[va], num_iteration=m.best_iteration)
        print(f"fold {f} done", flush=True)
    base = ij.assign(label=y)
    for name, p in (("full-trained, full-scored  ", p_ff), ("full-trained, hidden-scored", p_fh),
                    ("hidden-trained, hidden-scored", p_hh)):
        report(name, base.assign(prob=p), G, gc, sample)
    base.assign(p_ff=p_ff, p_fh=p_fh, p_hh=p_hh).to_parquet(os.path.join(W, f"sim_hidden_oof_{a.frac}.parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["block", "cv"], required=True)
    ap.add_argument("--frac", type=float, default=0.19)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    block() if a.step == "block" else cv(a)

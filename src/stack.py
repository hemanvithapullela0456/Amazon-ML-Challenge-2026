"""Stage 2: combine stage-1 probability with cross-encoder scores, re-tune the F0.5 decision, write output.

Needs work/train_oof.parquet, work/test_scored.parquet, work/ce_train.parquet, work/ce_test.parquet.
Pairs the cross-encoder did not score keep ce = NaN (LightGBM handles it).
Usage:  python src/stack.py --tag _xlmr        (full: ce_train_xlmr + ce_test_xlmr -> test_scored_stage2_xlmr)
        python src/stack.py --suffix _eval_x   (evaluation only, on the S1s the eval run scored)
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide

# Only within-S1 features: a candidate-side "who else wants this record" feature would be computed over the
# 200k training sample but over all 1.7M S1s at test time, a distribution mismatch. (Stage 1 already carries
# candidate-side competition, and the decision rule applies one-to-one over everything.)
BASE_FEATS = ["prob", "prob_rank", "prob_gap", "prob_n_above"]
CE_FEATS = ["ce", "ce_rank", "ce_gap", "ce_max_other", "ce_available"]


def feats_for(names):
    """one block of reranker features per reranker (names: column suffixes, e.g. ['', '_mdeb'])"""
    return BASE_FEATS + [f + n for n in names for f in CE_FEATS]


FEATS = feats_for([""])


def stack_features(df, ce, names=("",)):
    """ce: one table (i, j, ce) or a list of them, one per reranker; names: suffix for each reranker's columns"""
    ces = ce if isinstance(ce, (list, tuple)) else [ce]
    d = df
    for t, n in zip(ces, names):
        d = d.join(t.rename({"ce": "ce" + n}), on=["i", "j"], how="left")
    p = pl.col("prob")
    d = d.with_columns(
        p.rank("min", descending=True).over("i").cast(pl.Float32).alias("prob_rank"),
        (p.max().over("i") - p).alias("prob_gap"),
        (p > 0.5).sum().over("i").cast(pl.Float32).alias("prob_n_above"))
    for n in names:
        c = pl.col("ce" + n)
        d = d.with_columns(
            c.rank("min", descending=True).over("i").cast(pl.Float32).alias("ce_rank" + n),
            (c.max().over("i") - c).alias("ce_gap" + n),
            c.is_not_null().cast(pl.Float32).alias("ce_available" + n),
        ).with_columns(  # best score of the S1's OTHER candidates
            pl.when(pl.col("ce_rank" + n) == 1).then(c.sort(descending=True, nulls_last=True).slice(1, 1).first().over("i"))
            .otherwise(c.max().over("i")).alias("ce_max_other" + n))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="evaluation-only run on ce_train<suffix> (no test output)")
    ap.add_argument("--tag", default="", help="full run: ce_train<tag>/ce_test<tag> -> test_scored_stage2<tag>")
    ap.add_argument("--eval_only", action="store_true", help="with --tag: stop after the validation score")
    ap.add_argument("--restrict", action="store_true",
                    help="full run: fit the stack only on S1s the FIRST reranker scored (one that has OOF on one fold only)")
    ap.add_argument("--graph", action="store_true", help="add graph/consensus features (work/graph_{train,test}.parquet)")
    ap.add_argument("--moves", action="store_true", help="add generator-move / alias features (work/moves_{train,test}.parquet)")
    ap.add_argument("--owner", action="store_true",
                    help="add same-name owner-choice features for name-only copies (work/owner_{train,test}.parquet)")
    ap.add_argument("--out_prefix", default="", help="prefix for the test score file (reproducibility checks)")
    ap.add_argument("--owner_file", default="owner_train.parquet")
    ap.add_argument("--owner_val", default="",
                    help="evaluation only: owner features used when PREDICTING the held-out folds (e.g. computed with "
                         "hidden S1s), while the model is trained on --owner_file")
    ap.add_argument("--no-expected", action="store_true", help="skip the (slow) expected-F0.5 selector while tuning")
    a = ap.parse_args()
    W = C.WORK_DIR
    oof = pl.read_parquet(os.path.join(W, "train_oof.parquet"))
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    tags = (a.suffix or a.tag).split(",")          # several rerankers: --tag _qwen,_mdeb
    names = [""] if len(tags) == 1 else tags
    feats = feats_for(names)
    ce_trs = [pl.read_parquet(os.path.join(W, f"ce_train{t}.parquet")).drop_nulls("ce").filter(pl.col("ce").is_not_nan())
              for t in tags]
    ce_tr = ce_trs[0]
    tr = stack_features(oof, ce_trs, names)
    gfe = []
    if a.graph:
        g_tr = pl.read_parquet(os.path.join(W, "graph_train.parquet"))
        gfe = [c for c in g_tr.columns if c not in ("i", "j")]
        tr = tr.join(g_tr, on=["i", "j"], how="left")
        feats = feats + gfe
        print(f"graph features: {len(gfe)} on {g_tr.height} band pairs")
    if a.moves:
        m_tr = pl.read_parquet(os.path.join(W, "moves_train.parquet"))
        mfe = [c for c in m_tr.columns if c not in ("i", "j")]
        tr = tr.join(m_tr, on=["i", "j"], how="left")
        feats = feats + mfe
        print(f"move features: {len(mfe)}")
    if a.owner:
        o_tr = pl.read_parquet(os.path.join(W, a.owner_file))
        ofe = [c for c in o_tr.columns if c not in ("i", "j")]
        tr = tr.join(o_tr, on=["i", "j"], how="left")
        if a.owner_val:
            o_va = pl.read_parquet(os.path.join(W, a.owner_val)).select(["i", "j"] + ofe)
            tr = tr.join(o_va.rename({c: c + "__v" for c in ofe}), on=["i", "j"], how="left")
        feats = feats + ofe
        print(f"owner features: {len(ofe)} on {o_tr.height} name-only pairs")
    if a.suffix or a.restrict:  # evaluate / fit only on S1s the first CE saw
        tr = tr.filter(pl.col("i").is_in(ce_tr["i"].unique()))
    df = tr.to_pandas()
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    s1_ids = df["i"].unique()

    # baseline: stage-1 decision on the same S1s
    with open(os.path.join(W, "decision.json")) as f:
        cfg1 = json.load(f)
    base = decide.score_pairs(df.assign(pred=decide.predict_mask(df, cfg1)), "pred", G, s1_ids)

    rng = np.random.default_rng(C.SEED + 1)
    fold_of = pd.Series(rng.permutation(len(s1_ids)) % C.N_FOLDS, index=s1_ids)
    df["sfold"] = fold_of.reindex(df["i"].values).values
    df["prob1"] = df["prob"]
    params = dict(objective="binary", learning_rate=float(os.environ.get("STACK_LR", 0.05)),
                  num_leaves=int(os.environ.get("STACK_LEAVES", 31)), min_data_in_leaf=int(os.environ.get("STACK_MINLEAF", 200)),
                  feature_fraction=float(os.environ.get("STACK_FF", 0.9)), verbose=-1, num_threads=C.N_JOBS, seed=C.SEED)
    oof2, iters = np.zeros(len(df)), []
    for f in range(C.N_FOLDS):
        trm, vam = (df["sfold"] != f).values, (df["sfold"] == f).values
        dtr = lgb.Dataset(df.loc[trm, feats], df.loc[trm, "label"].astype(int))
        dva = lgb.Dataset(df.loc[vam, feats], df.loc[vam, "label"].astype(int), reference=dtr)
        m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        if a.owner and a.owner_val:
            xv = df.loc[vam, feats].copy()
            for c in ofe:
                xv[c] = df.loc[vam, c + "__v"].values
            oof2[vam] = m.predict(xv, num_iteration=m.best_iteration)
        else:
            oof2[vam] = m.predict(df.loc[vam, feats], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
    df["prob"] = oof2
    if a.suffix or a.eval_only or a.out_prefix:   # out-of-fold stage-2 probabilities for decision-layer experiments
        df[["i", "j", "label", "prob"]].to_parquet(os.path.join(W, f"{a.out_prefix}stack_oof{(a.suffix or a.tag).replace(',', '')}"
                                                       f"{'_graph' if a.graph else ''}{'_moves' if a.moves else ''}{'_owner' if a.owner else ''}.parquet"))
    print("stage-2 decision tuning:")
    cfg = decide.tune(df, G, s1_ids, try_expected=not a.no_expected)
    print(f"stage-1 F0.5 on these S1: {base:.5f}  ->  stage-2 F0.5: {cfg['cv_f05']:.5f}")
    if a.suffix or a.eval_only:
        return

    n_rounds = int(np.mean(iters) * 1.1) + 1
    # BUGFIX 27 Sep: "prob" was overwritten by the stage-2 OOF probability above, so the final model used to be
    # trained on stage-2 probs but applied to stage-1 probs at test time (compressed, stage-1-driven test scores).
    df["prob"] = df["prob1"]
    model = lgb.train(params, lgb.Dataset(df[feats], df["label"].astype(int)), n_rounds)
    # test set: 49.6M pairs, so score in S1 chunks (all features are within-S1, so chunking is exact)
    scored = pl.read_parquet(os.path.join(W, "test_scored.parquet"))
    ce_tes = [pl.read_parquet(os.path.join(W, f"ce_test{t}.parquet")) for t in tags]
    g_te = pl.read_parquet(os.path.join(W, "graph_test.parquet")) if a.graph else None
    m_te = pl.read_parquet(os.path.join(W, "moves_test.parquet")) if a.moves else None
    o_te = pl.read_parquet(os.path.join(W, "owner_test.parquet")) if a.owner else None
    s1 = scored["i"].unique().sort()
    parts, step = [], 300_000
    for a0 in range(0, len(s1), step):
        ch = scored.filter(pl.col("i").is_in(s1[a0:a0 + step].implode()))
        f = stack_features(ch, ce_tes, names)
        if g_te is not None:
            f = f.join(g_te, on=["i", "j"], how="left")
        if m_te is not None:
            f = f.join(m_te, on=["i", "j"], how="left")
        if o_te is not None:
            f = f.join(o_te, on=["i", "j"], how="left")
        p2 = model.predict(f.select(feats).to_numpy())
        parts.append(f.select("i", "j").with_columns(pl.Series("prob", p2.astype(np.float32))))
        print(f"  stage-2 scored S1 {min(a0 + step, len(s1))}/{len(s1)}", flush=True)
    pl.concat(parts).write_parquet(os.path.join(W, f"{a.out_prefix}test_scored_stage2{a.tag.replace(',', '')}{'_graph' if a.graph else ''}{'_moves' if a.moves else ''}{'_owner' if a.owner else ''}.parquet"))
    cfg.update(features=feats, n_rounds=n_rounds, stage=2)
    with open(os.path.join(W, f"decision_stage2{a.tag.replace(',', '')}{'_graph' if a.graph else ''}{'_moves' if a.moves else ''}{'_owner' if a.owner else ''}.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    print(f"saved work/test_scored_stage2{a.tag.replace(',', '')}.parquet -> python src/run_pipeline.py --reuse --stage2 --stage2_tag {a.tag.replace(',', '')}")


if __name__ == "__main__":
    main()

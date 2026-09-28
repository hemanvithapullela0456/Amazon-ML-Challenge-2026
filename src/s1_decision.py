"""Per-S1 decision layer on the US/India stage-2 stack (out-of-fold), cross-validated over S1s.

The pair model + global threshold makes all-or-nothing mistakes per S1: S1s WITH copies for which nothing is
predicted, and S1s WITHOUT copies for which something is predicted (each costs a full 1.0 of F0.5). A per-S1 model
reads the whole scored candidate list (probability profile, generator moves of the top candidates) and predicts
P(S1 has >= 1 copy). Policy: rescue the top candidate of an empty S1 when P(has copy) is high; abstain on a
predicted S1 when P(no copy) is high. Thresholds are chosen on the training folds, reported on held-out folds.

    python src/s1_decision.py work/stack_oof_....parquet
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide

W = C.WORK_DIR


def s1_table(d, t):
    d = d.sort_values(["i", "prob"], ascending=[True, False]).copy()
    d["rk"] = d.groupby("i").cumcount()
    piv = d[d.rk < 3].pivot(index="i", columns="rk", values="prob").reindex(columns=[0, 1, 2]).fillna(0)
    piv.columns = ["p1", "p2", "p3"]
    agg = d.groupby("i").agg(n_t=("prob", lambda x: (x >= t).sum()), n_50=("prob", lambda x: (x >= 0.5).sum()),
                             n_20=("prob", lambda x: (x >= 0.2).sum()), n_c=("prob", "size"), p_sum=("prob", "sum"))
    f = piv.join(agg)
    f["gap12"] = f.p1 - f.p2
    top = d[d.rk == 0][["i", "j"]]
    mv = pl.read_parquet(os.path.join(W, "moves_train.parquet")).to_pandas()
    keep = ["mv_name_equal", "mv_name_jacc", "mv_name_disjoint", "mv_addr_dropped", "mv_s1_addr_empty", "mv_hn_equal",
            "mv_hn_small_shift", "mv_nums_jacc", "mv_city_tail", "al_name_freq", "al_cluster_size"]
    f = f.join(top.merge(mv[["i", "j"] + keep], on=["i", "j"], how="left").set_index("i")[keep])
    return f.reset_index()


def apply(d, base_pred, p_has, rescue_tau, abstain_tau):
    pred = base_pred.copy()
    n_pred = pd.Series(pred, index=d.index).groupby(d.i).transform("sum").values
    ph = d.i.map(p_has).values
    top = (d.groupby("i")["prob"].transform("max") == d.prob).values
    pred = np.where((n_pred == 0) & (ph >= rescue_tau) & top, True, pred)
    pred = np.where((n_pred > 0) & (ph <= 1 - abstain_tau), False, pred)
    return pred


def main(path):
    d = pd.read_parquet(path)
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    ids = d.i.unique()
    t = max(((decide.score_pairs(d.assign(pred=decide.predict_mask(d, dict(method="threshold", t=tt, one_to_one=True))),
                                 "pred", G, ids), tt) for tt in np.arange(0.5, 0.9, 0.025)))[1]
    base_pred = decide.predict_mask(d, dict(method="threshold", t=t, one_to_one=True))
    base = decide.score_pairs(d.assign(pred=base_pred), "pred", G, ids)
    print(f"{len(ids)} S1s | stack OOF F0.5 {base:.5f} @t={t:.3f}")
    f = s1_table(d, t)
    f["has"] = (G.reindex(f.i).values > 0).astype(int)
    feats = [c for c in f.columns if c not in ("i", "has")]
    fold = np.random.default_rng(3).permutation(len(f)) % 5
    f["p_has"] = 0.0
    for k in range(5):
        m = lgb.train(dict(C.LGB_PARAMS, num_leaves=31, min_data_in_leaf=40), lgb.Dataset(f.loc[fold != k, feats], f.has[fold != k]), 400)
        f.loc[fold == k, "p_has"] = m.predict(f.loc[fold == k, feats])
    from sklearn.metrics import roc_auc_score
    print(f"P(has copy) AUC {roc_auc_score(f.has, f.p_has):.4f} | S1s without copies {int((f.has == 0).sum())}")
    p_has = dict(zip(f.i, f.p_has))
    # choose thresholds on 4 folds, report on the 5th (nested, so the reported gain is honest)
    s1_fold = dict(zip(f.i, fold))
    dfold = d.i.map(s1_fold).values
    total_new = []
    grid = [(r, a) for r in (0.8, 0.9, 0.95, 0.98, 1.01) for a in (0.8, 0.9, 0.95, 0.98, 1.01)]
    for k in range(5):
        tr_ids, va_ids = f.i[fold != k].values, f.i[fold == k].values
        trm, vam = dfold != k, dfold == k
        best = max(grid, key=lambda ra: decide.score_pairs(d[trm].assign(pred=apply(d[trm], base_pred[trm], p_has, *ra)), "pred", G, tr_ids))
        new = decide.score_pairs(d[vam].assign(pred=apply(d[vam], base_pred[vam], p_has, *best)), "pred", G, va_ids)
        old = decide.score_pairs(d[vam].assign(pred=base_pred[vam]), "pred", G, va_ids)
        total_new.append((new, old, len(va_ids)))
        print(f"  fold {k}: rescue>={best[0]} abstain>={best[1]} | held-out F0.5 {old:.5f} -> {new:.5f}", flush=True)
    w = np.array([x[2] for x in total_new])
    print(f"held-out F0.5: {np.average([x[1] for x in total_new], weights=w):.5f} -> {np.average([x[0] for x in total_new], weights=w):.5f}")
    for r, a in [(0.9, 1.01), (1.01, 0.9), (0.9, 0.9), (0.95, 0.95)]:
        print(f"  fixed policy rescue>={r} abstain>={a}: {decide.score_pairs(d.assign(pred=apply(d, base_pred, p_has, r, a)), 'pred', G, ids):.5f}")


if __name__ == "__main__":
    main(sys.argv[1])

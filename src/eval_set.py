"""Evaluate the one-to-set model on its validation fold (work/set_val.parquet: i, j, ce).

Prints F0.5 on the fold-0 S1s for: stage 1 | set model ALONE (its sigmoid replaces the band probabilities;
also rank-mapped onto stage-1's band distribution) | writes ce_train_set / ce_train_{qwen,mdeb}_f0 so that
stack.py --suffix can compare stacks on exactly these S1s.
"""
import os

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

import config as C
import decide

W = C.WORK_DIR
LO, HI = 0.01, 0.99


def main():
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G.index)
    sv = pl.read_parquet(os.path.join(W, "set_val.parquet"))
    ids = sv["i"].unique().to_numpy()
    o = pl.read_parquet(os.path.join(W, "train_oof.parquet"), columns=["i", "j", "label", "prob"]).filter(pl.col("i").is_in(ids))
    d = o.join(sv, on=["i", "j"], how="left").to_pandas()
    band = d.prob.gt(LO) & d.prob.lt(HI) & d.ce.notna()
    from sklearn.metrics import roc_auc_score
    print(f"{len(ids)} validation S1s | band pairs {band.sum()} | AUC set {roc_auc_score(d.label[band], d.ce[band]):.4f} "
          f"vs stage-1 {roc_auc_score(d.label[band], d.prob[band]):.4f}")

    def f05(x, name):
        res = []
        for t in np.arange(0.3, 0.96, 0.05):
            m = decide.predict_mask(x, dict(method="threshold", t=t, one_to_one=True))
            res.append((decide.score_pairs(x.assign(pred=m), "pred", G, ids), round(t, 2)))
        best = max(res)
        cfg = dict(method="threshold", t=best[1], one_to_one=True)
        per = []
        for c in ("us", "india"):
            ic = np.intersect1d(ids, gc.index[gc == c])
            xc = x[x.i.isin(ic)]
            per.append(f"{c} {decide.score_pairs(xc.assign(pred=decide.predict_mask(xc, cfg)), 'pred', G, ic):.4f}")
        print(f"{name:34s} F0.5 {best[0]:.5f} @t={best[1]} | {' '.join(per)}", flush=True)

    f05(d, "stage 1")
    x = d.copy(); x.loc[band, "prob"] = 1 / (1 + np.exp(-x.loc[band, "ce"])); f05(x, "set model alone (sigmoid on band)")
    x = d.copy(); qs = np.sort(d.loc[band, "prob"].values)
    x.loc[band, "prob"] = qs[(rankdata(d.loc[band, "ce"], method="ordinal") - 1).astype(int)]
    f05(x, "set model alone (rank-mapped)")
    x = d.copy(); x.loc[band, "prob"] = qs[(rankdata(0.5 * rankdata(d.loc[band, "ce"]) + 0.5 * rankdata(d.loc[band, "prob"]),
                                                     method="ordinal") - 1).astype(int)]
    f05(x, "set + stage-1 rank average")
    sv.write_parquet(os.path.join(W, "ce_train_set.parquet"))
    for t in ("_qwen", "_mdeb"):
        p = os.path.join(W, f"ce_train{t}.parquet")
        if os.path.exists(p):
            pl.read_parquet(p).filter(pl.col("i").is_in(ids)).write_parquet(os.path.join(W, f"ce_train{t}_f0.parquet"))
    print("wrote ce_train_set / ce_train_{qwen,mdeb}_f0 -> stack.py --suffix _set | _set,_qwen_f0,_mdeb_f0 | _qwen_f0,_mdeb_f0")


if __name__ == "__main__":
    main()

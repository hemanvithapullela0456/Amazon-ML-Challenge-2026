"""Apply the simulation-trained per-S1 "no match" model to France and write an adjusted France score file.

Train: all France-like simulation S1s (Latin-script India, run13-equivalent recipe), label = S1 has no true copy.
Apply: France S1s scored with run13's recipe (work/test_scored_unseen_mdeb_set4.parquet + set-model ranks).
Predictions of S1s with P(no copy) >= tau are zeroed. Prints abstention rates on both sides as a transfer check.

    python src/abstain_france.py --tau 0.5
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

import config as C
import sim_abstain as A

W = C.WORK_DIR


def france_frame(scores_file):
    sc = pl.read_parquet(os.path.join(W, scores_file)).to_pandas()
    st = pd.read_parquet(os.path.join(W, "ce_test_set.parquet")).rename(columns={"ce": "set"})
    band = (sc.prob > 0.01) & (sc.prob < 0.99)
    b = sc[band].merge(st, on=["i", "j"], how="left")
    m = b.set.notna()
    b.loc[m, "set_r"] = rankdata(b.loc[m, "set"]) / m.sum()
    return sc.merge(b[["i", "j", "set_r"]], on=["i", "j"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--scores", default="test_scored_unseen_mdeb_set4.parquet")
    ap.add_argument("--out", default="test_scored_unseen_abstain.parquet")
    a = ap.parse_args()
    ev, G, lat = A.sim_scores()
    fs = A.s1_features(ev)
    fs["g0"] = (G.reindex(fs.i).values == 0).astype(int)
    feats = [c for c in fs.columns if c not in ("i", "j", "g0")]
    m = lgb.train(dict(C.LGB_PARAMS, num_leaves=31, min_data_in_leaf=50), lgb.Dataset(fs[feats], fs.g0), 400)
    fr = france_frame(a.scores)
    import moves
    d = fr.sort_values(["i", "prob"], ascending=[True, False])
    d["rk"] = d.groupby("i").cumcount()
    top = d[d.rk == 0][["i", "j", "prob", "set_r"]].rename(columns={"prob": "p1", "set_r": "set1"})
    sec = d[d.rk == 1][["i", "prob"]].rename(columns={"prob": "p2"})
    agg = d.groupby("i").agg(n_t=("prob", lambda x: (x >= A.T).sum()), n_50=("prob", lambda x: (x >= 0.5).sum()),
                             n_c=("prob", "size"), p_sum=("prob", "sum"))
    ff = top.merge(sec, on="i", how="left").merge(agg, on="i")
    ff["gap12"] = ff.p1 - ff.p2.fillna(0)
    mv = moves.build("test", pl.from_pandas(top[["i", "j", "p1"]].rename(columns={"p1": "prob"})).with_columns(
        pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))).to_pandas()
    ff = ff.merge(mv[["i", "j"] + [c for c in feats if c.startswith(("mv_", "al_"))]], on=["i", "j"], how="left")
    ff["p_none"] = m.predict(ff[feats])
    fs["p_none"] = m.predict(fs[feats])
    rate = lambda f: float(((f.p_none >= a.tau) & (f.n_t > 0)).sum() / max((f.n_t > 0).sum(), 1))
    print(f"S1s with a prediction that get abstained: simulation {rate(fs):.3%} (in-sample) | France {rate(ff):.3%}")
    print(f"feature means, simulation vs France: " + " ".join(
        f"{c} {fs[c].mean():.3f}/{ff[c].mean():.3f}" for c in ("p1", "gap12", "n_t", "mv_name_equal", "mv_hn_small_shift")))
    drop = set(ff.i[(ff.p_none >= a.tau) & (ff.n_t > 0)])
    out = pl.read_parquet(os.path.join(W, a.scores)).with_columns(
        pl.when(pl.col("i").is_in(list(drop))).then(0.0).otherwise(pl.col("prob")).cast(pl.Float32).alias("prob"))
    out.write_parquet(os.path.join(W, a.out))
    print(f"France S1s abstained: {len(drop)} of {ff.i.nunique()} -> work/{a.out}")


if __name__ == "__main__":
    main()

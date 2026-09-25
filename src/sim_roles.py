"""Gate for the word-role features: US-trained matcher scored on France-like (Latin-only) India, true labels.
Word statistics for each country come from that country's unlabelled records (word_roles.py).

    python src/sim_roles.py
"""
import os as _os
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # one BLAS thread per process: workers x threads exhausted memory

import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide
from blocking import load_records, truth_pairs
from features import feature_cols
from prep import load_norm
from train import norm_table
from word_roles import ROLE_FEATS, pair_features

DROP = {"cos", "dense_rank", "cos_gap_s1"}


def best(ev, G, ids, ts=(0.6, 0.7, 0.8, 0.85, 0.9, 0.95)):
    res = {t: decide.score_pairs(ev.assign(pred=decide.predict_mask(ev, dict(method="threshold", t=t, one_to_one=True))),
                                 "pred", G, ids) for t in ts}
    t = max(res, key=res.get)
    return res[t], t, res


def main():
    W = C.WORK_DIR
    t0 = time.time()
    df = pl.read_parquet(os.path.join(W, "sim_us2in", "feats.parquet"))
    rp = os.path.join(W, "sim_us2in", "roles.parquet")
    if not os.path.exists(rp):
        stats = pl.read_parquet(os.path.join(W, "word_roles_train.parquet"))
        norm = norm_table("train").select("r", "name_core", "country_key")
        pair_features(df.select("i", "j"), norm, stats).write_parquet(rp)
        print(f"role features computed ({time.time() - t0:.0f}s)", flush=True)
    df = df.join(pl.read_parquet(rp), on=["i", "j"], how="left")
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    ids = G_pl.filter(pl.col("country_key") == "india")["i"].to_numpy()
    us_ids = G_pl.filter(pl.col("country_key") == "us")["i"].to_numpy()
    nl = (load_norm("train", columns=["business_name", "business_address"]).with_row_index("r")
          .select(pl.col("r").cast(pl.UInt32), (pl.col("business_name").str.contains(r"[ऀ-෿]")
                  | pl.col("business_address").str.contains(r"[ऀ-෿]")).alias("nl")))
    tp = truth_pairs(load_records("train"), "train", ids).join(nl.rename({"r": "j"}), on="j")
    bad = set(tp.filter(pl.col("nl"))["i"].to_list()) | set(nl.filter(pl.col("nl") & pl.col("r").is_in(ids))["r"].to_list())
    lat = np.array([i for i in ids if i not in bad])
    base = [f for f in feature_cols(df.head(1).to_pandas()) if f not in DROP and f not in ROLE_FEATS]
    us = df.filter(pl.col("country_key") == "us").to_pandas()
    ind = df.filter(pl.col("country_key") == "india").to_pandas()
    lat_ev = ind[ind["i"].isin(lat)].copy()
    params = dict(C.LGB_PARAMS, learning_rate=0.1)
    for name, feats in [("baseline (no dense feats)", base), ("+ word-role features", base + ROLE_FEATS)]:
        t = time.time()
        m = lgb.train(params, lgb.Dataset(us[feats], us["label"].astype(int)), 700)
        lat_ev["prob"] = m.predict(lat_ev[feats])
        f, thr, res = best(lat_ev, G, lat)
        m2 = lgb.train(params, lgb.Dataset(ind[feats], ind["label"].astype(int)), 700)
        us["prob"] = m2.predict(us[feats])
        f2, t2, _ = best(us, G, us_ids, ts=(0.6, 0.7, 0.8, 0.9))
        print(f"{name:<28} US->India(Latin) {f:.4f} @t={thr} | India->US {f2:.4f} @t={t2} | "
              + " ".join(f"{k}:{v:.3f}" for k, v in res.items()) + f" ({time.time() - t:.0f}s)", flush=True)
        if feats is not base:
            imp = pd.Series(m.feature_importance("gain"), index=feats)
            print("  role-feature gain share (US model): " + f"{imp[ROLE_FEATS].sum() / imp.sum():.3f} | "
                  + ", ".join(f"{k} {v:.0f}" for k, v in imp[ROLE_FEATS].sort_values(ascending=False).head(5).items()))


if __name__ == "__main__":
    main()

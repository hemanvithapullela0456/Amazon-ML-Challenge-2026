"""Which matcher settings transfer to an unseen Latin-script country? Two France-like tests, both with true labels:
  US -> India(Latin) : trained on US, scored on India S1s whose record and all true matches are Latin script
  India -> US        : trained on India, scored on US
Candidates come from the US-only retriever run (sim_unseen.py); dense-retriever features are never used.
Variants: dropping feature groups, and monotone constraints (a match can only get likelier when a similarity rises
and less likely when a conflict appears) - a standard way to stop a model learning country-specific quirks.

    python src/sim_transfer.py
"""
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

DROP = {"cos", "dense_rank", "cos_gap_s1", "from_key"}
GROUPS = {
    "lengths": lambda f: "_len_" in f,
    "numbers": lambda f: f in {"pc_eq", "num_jac", "num_conflict", "hn_eq"},
    "region/landmark": lambda f: f.startswith("lm_") or f == "region_eq",
    "legal-form": lambda f: f.startswith("legal_"),
    "relative-to-S1": lambda f: f.endswith("_rank_s1") or f.endswith("_gap_s1") or f == "n_cands_s1",
    "cand_source": lambda f: f == "cand_source",
}
INCREASING = {"n_ratio", "n_partial", "n_tsort", "n_tset", "n_norm_ratio", "n_jw", "n_alias_best", "n_first_eq",
              "n_exact", "n_contain", "n_acronym", "n_idf_jac", "n_idf_ovl", "legal_jac", "a_ratio", "a_tset",
              "a_tsort", "a_partial", "a_idf_jac", "a_idf_ovl", "pc_eq", "num_jac", "hn_eq", "lm_tset", "lm_in_b",
              "lm_in_a", "region_eq", "combo_tset", "blk_score", "w_max", "n_keys"}
DECREASING = {"n_unmatched_idf", "a_unmatched_idf", "legal_conflict", "num_conflict", "n_len_diff", "blk_rank",
              "blk_gap_s1", "blk_gap_c", "blk_rank_c"}


def best(ev, G, ids):
    res = {t: decide.score_pairs(ev.assign(pred=decide.predict_mask(ev, dict(method="threshold", t=t, one_to_one=True))),
                                 "pred", G, ids) for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)}
    t = max(res, key=res.get)
    return res[t], t


def main():
    W = C.WORK_DIR
    df = pl.read_parquet(os.path.join(W, "sim_us2in", "feats.parquet"))
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G_pl["i"].to_numpy())
    in_ids, us_ids = gc.index[gc.values == "india"], gc.index[gc.values == "us"]
    nl = (load_norm("train", columns=["business_name", "business_address"]).with_row_index("r")
          .select(pl.col("r").cast(pl.UInt32), (pl.col("business_name").str.contains(r"[ऀ-෿]")
                  | pl.col("business_address").str.contains(r"[ऀ-෿]")).alias("nl")))
    tp = truth_pairs(load_records("train"), "train", in_ids).join(nl.rename({"r": "j"}), on="j")
    bad = set(tp.filter(pl.col("nl"))["i"].to_list()) | set(nl.filter(pl.col("nl") & pl.col("r").is_in(in_ids))["r"].to_list())
    lat_ids = np.array([i for i in in_ids if i not in bad])
    base = [f for f in feature_cols(df.head(1).to_pandas()) if f not in DROP]
    us = df.filter(pl.col("country_key") == "us").select(["i", "j", "label"] + base).to_pandas()
    ind = df.filter(pl.col("country_key") == "india").select(["i", "j", "label"] + base).to_pandas()
    ind_lat = ind[ind["i"].isin(lat_ids)].copy()
    del df
    params = dict(C.LGB_PARAMS, learning_rate=0.15)
    variants = [("baseline", base, {})]
    variants += [(f"drop {g}", [f for f in base if not fn(f)], {}) for g, fn in GROUPS.items()]
    variants += [("monotone constraints", base, {"_mono": True}),
                 ("monotone + drop lengths", [f for f in base if "_len_" not in f], {"_mono": True})]
    print(f"France-like tests: India(Latin) {len(lat_ids)} S1 | US {len(us_ids)} S1", flush=True)
    for name, feats, extra in variants:
        t = time.time()
        extra = dict(extra)
        if extra.pop("_mono", False):
            extra["monotone_constraints"] = [1 if f in INCREASING else -1 if f in DECREASING else 0 for f in feats]
            extra["monotone_constraints_method"] = "advanced"
        p = dict(params, **extra)
        m1 = lgb.train(p, lgb.Dataset(us[feats], us["label"].astype(int)), 450)
        ind_lat["prob"] = m1.predict(ind_lat[feats])
        f1, t1 = best(ind_lat, G, lat_ids)
        m2 = lgb.train(p, lgb.Dataset(ind[feats], ind["label"].astype(int)), 450)
        us["prob"] = m2.predict(us[feats])
        f2, t2 = best(us, G, us_ids)
        print(f"{name:<26} {len(feats):>2} feats | US->India(Latin) {f1:.4f} @t={t1} | India->US {f2:.4f} @t={t2} | "
              f"mean {(f1 + f2) / 2:.4f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

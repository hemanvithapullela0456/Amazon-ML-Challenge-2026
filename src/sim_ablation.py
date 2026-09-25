"""Which feature groups / model settings transfer to an unseen country? (India plays the unseen country.)

Matcher trained on US only (candidates from the US-only retriever, no dense features = the unseen-country setup),
each variant scored on India with the TRUE labels. A variant that raises India F0.5 without hurting in-domain
validation is a candidate for the France matcher.

    python src/sim_ablation.py        (needs work/sim_us2in/feats.parquet from sim_unseen.py)
"""
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide
from features import feature_cols

DENSE = {"cos", "dense_rank", "cos_gap_s1"}
GROUPS = {
    "blocking-ranker": lambda f: f.startswith("blk_") or f in {"n_keys", "w_max", "n_s1_per_cand"},
    "relative-to-S1": lambda f: f.endswith("_rank_s1") or f.endswith("_gap_s1") or f == "n_cands_s1",
    "lengths": lambda f: "_len_" in f,
    "legal-form": lambda f: f.startswith("legal_"),
    "landmark/region": lambda f: f.startswith("lm_") or f == "region_eq",
    "numbers": lambda f: f in {"pc_eq", "num_jac", "num_conflict", "hn_eq"},
    "source/from_key": lambda f: f in {"cand_source", "from_key"},
}


def best_f05(ev, G, ids):
    res = {}
    for t in (0.4, 0.5, 0.6, 0.7, 0.8):
        d = ev.assign(pred=decide.predict_mask(ev, dict(method="threshold", t=t, one_to_one=True)))
        res[t] = decide.score_pairs(d, "pred", G, ids)
    t = max(res, key=res.get)
    return res[t], t


def main():
    W = C.WORK_DIR
    df = pl.read_parquet(os.path.join(W, "sim_us2in", "feats.parquet")).to_pandas()
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G_pl["i"].to_numpy())
    ids = gc.index[gc.values == "india"]
    base = [f for f in feature_cols(df) if f not in DENSE]
    tr = df[df["country_key"] == "us"]
    ev = df[df["country_key"] == "india"][["i", "j", "label"] + base].copy()
    del df
    fast = dict(C.LGB_PARAMS, learning_rate=0.1)
    variants = [("baseline (no dense feats)", base, {})]
    variants += [(f"drop {g}", [f for f in base if not fn(f)], {}) for g, fn in GROUPS.items()]
    variants += [("simpler trees (15 leaves, min_data 500)", base, dict(num_leaves=15, min_data_in_leaf=500)),
                 ("strong regularisation (lambda_l2 20, feature_fraction 0.5)", base, dict(lambda_l2=20.0, feature_fraction=0.5))]
    # per-country quantile normalisation: every feature becomes its percentile among that country's candidate
    # pairs, so a similarity that is "ordinary" in a country with generic names/addresses no longer looks strong
    qtr = tr[base].rank(pct=True).astype("float32")
    qev = ev[base].rank(pct=True).astype("float32")
    variants += [("per-country quantile-normalised features", base, {"_quantile": True})]
    for name, feats, extra in variants:
        t = time.time()
        q = extra.pop("_quantile", False)
        Xtr, Xev = (qtr[feats], qev[feats]) if q else (tr[feats], ev[feats])
        m = lgb.train(dict(fast, **extra), lgb.Dataset(Xtr, tr["label"].astype(int)), 700)
        ev["prob"] = m.predict(Xev)
        f, thr = best_f05(ev, G, ids)
        print(f"{name:<58} {len(feats):>2} feats | India F0.5 {f:.4f} @ t={thr} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

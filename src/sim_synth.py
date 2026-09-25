"""Approach A, gate: does synthetic target-country data close the unseen-country gap? (India plays France.)

Train variants, all scored on REAL India pairs with the TRUE labels (candidates from the US-only retriever):
  US only                      - today's unseen-country matcher (baseline)
  US + synthetic India         - approach A
  synthetic India only         - fidelity check: a model that never saw a real India label
    python src/sim_synth.py --synth synth_india
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

import config as C
import decide
from features import feature_cols

DROP = {"cos", "dense_rank", "cos_gap_s1", "from_key"}   # dense features + a flag that is constant in synthetic data


def best_f05(ev, G, ids):
    res = {}
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        d = ev.assign(pred=decide.predict_mask(ev, dict(method="threshold", t=t, one_to_one=True)))
        res[t] = decide.score_pairs(d, "pred", G, ids)
    t = max(res, key=res.get)
    return res[t], t, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="synth_india")
    ap.add_argument("--rounds", type=int, default=700)
    a = ap.parse_args()
    W = C.WORK_DIR
    real = pl.read_parquet(os.path.join(W, "sim_us2in", "feats.parquet"))
    syn = pl.read_parquet(os.path.join(W, a.synth, "feats.parquet"))
    feats = [f for f in feature_cols(real.to_pandas().head(1)) if f not in DROP and f in syn.columns]
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G_pl["i"].to_numpy())
    ids = gc.index[gc.values == "india"]
    # France-like subset: India S1s whose record and all true matches are in Latin script (France has no
    # native-script records; 77% of India's unseen-country misses are native-script pairs)
    from blocking import load_records, truth_pairs
    from prep import load_norm
    nl = (load_norm("train", columns=["business_name", "business_address"]).with_row_index("r")
          .select(pl.col("r").cast(pl.UInt32), (pl.col("business_name").str.contains(r"[ऀ-෿]")
                  | pl.col("business_address").str.contains(r"[ऀ-෿]")).alias("nl")))
    tp = truth_pairs(load_records("train"), "train", ids).join(nl.rename({"r": "j"}), on="j")
    bad = set(tp.filter(pl.col("nl"))["i"].to_list()) | set(nl.filter(pl.col("nl") & pl.col("r").is_in(ids))["r"].to_list())
    lat_ids = np.array([i for i in ids if i not in bad])
    print(f"France-like (Latin-only) India S1: {len(lat_ids)} of {len(ids)}", flush=True)
    us = real.filter(pl.col("country_key") == "us").select(feats + ["label"]).to_pandas()
    ev = real.filter(pl.col("country_key") == "india").select(["i", "j", "label"] + feats).to_pandas()
    sy = syn.select(feats + ["label"]).to_pandas()
    print(f"features {len(feats)} | US {len(us)} pairs | synthetic {len(sy)} pairs ({sy['label'].mean():.3f} pos) | "
          f"real India eval {len(ev)} pairs", flush=True)
    params = dict(C.LGB_PARAMS, learning_rate=0.1)
    variants = [("US only (baseline)", [us], [1.0]),
                ("US + synthetic India", [us, sy], [1.0, 1.0]),
                ("US + synthetic India (x2 weight)", [us, sy], [1.0, 2.0]),
                ("synthetic India only (fidelity)", [sy], [1.0])]
    for name, parts, ws in variants:
        t = time.time()
        X = pd.concat([p[feats] for p in parts])
        y = np.concatenate([p["label"].astype(int).values for p in parts])
        w = np.concatenate([np.full(len(p), wt) for p, wt in zip(parts, ws)])
        m = lgb.train(params, lgb.Dataset(X, y, weight=w), a.rounds)
        ev["prob"] = m.predict(ev[feats])
        f, thr, res = best_f05(ev, G, ids)
        el = ev[ev["i"].isin(lat_ids)]
        fl, thl, resl = best_f05(el, G, lat_ids)
        print(f"{name:<36} India F0.5 {f:.4f} @ t={thr} | LATIN-ONLY India F0.5 {fl:.4f} @ t={thl} | "
              + " ".join(f"{k}:{v:.3f}" for k, v in resl.items()) + f" ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

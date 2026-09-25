"""Does self-training on an unseen country help? Tested on the labelled stand-in (India unseen, see sim_unseen.py).

Round 0: matcher trained on US only (no dense-retriever features: the unseen-country setup), scores India.
Round r: India pairs the previous model is confident about become pseudo-labels (prob >= hi -> match,
prob <= lo -> non-match; one-to-one resolved first), the matcher is retrained on US + pseudo-labelled India and
India is re-scored. India's TRUE labels are used only to report F0.5, never for training.

    python src/sim_selftrain.py            (needs work/sim_us2in/feats.parquet from sim_unseen.py)
"""
import argparse
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide
from features import feature_cols

DENSE_FEATS = ["cos", "dense_rank", "cos_gap_s1"]


def f05_sweep(ev, G, ids, ts=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)):
    out = {}
    for t in ts:
        d = ev.assign(pred=decide.predict_mask(ev, dict(method="threshold", t=t, one_to_one=True)))
        out[t] = decide.score_pairs(d, "pred", G, ids)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", type=float, default=0.97)
    ap.add_argument("--lo", type=float, default=0.03)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--trees", type=int, default=1500)
    ap.add_argument("--weight", type=float, default=1.0, help="sample weight of pseudo-labelled pairs")
    a = ap.parse_args()
    W = C.WORK_DIR
    df = pl.read_parquet(os.path.join(W, "sim_us2in", "feats.parquet")).to_pandas()
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    gc = pd.Series(G_pl["country_key"].to_numpy(), index=G_pl["i"].to_numpy())
    ids = gc.index[gc.values == "india"]
    from blocking import load_records, truth_pairs
    from prep import load_norm
    nl = (load_norm("train", columns=["business_name", "business_address"]).with_row_index("r")
          .select(pl.col("r").cast(pl.UInt32), (pl.col("business_name").str.contains(r"[ऀ-෿]")
                  | pl.col("business_address").str.contains(r"[ऀ-෿]")).alias("nl")))
    tp = truth_pairs(load_records("train"), "train", ids).join(nl.rename({"r": "j"}), on="j")
    bad = set(tp.filter(pl.col("nl"))["i"].to_list()) | set(nl.filter(pl.col("nl") & pl.col("r").is_in(ids))["r"].to_list())
    lat_ids = np.array([i for i in ids if i not in bad])
    feats = [f for f in feature_cols(df) if f not in DENSE_FEATS]
    tr = df[df["country_key"] == "us"]
    ev = df[df["country_key"] == "india"].copy()
    del df

    X, y, w = tr[feats], tr["label"].astype(int).values, np.ones(len(tr))
    for r in range(a.rounds + 1):
        t = time.time()
        m = lgb.train(C.LGB_PARAMS, lgb.Dataset(X, y, weight=w), a.trees)
        ev["prob"] = m.predict(ev[feats])
        res = f05_sweep(ev, G, ids)
        best = max(res, key=res.get)
        rl = f05_sweep(ev[ev["i"].isin(lat_ids)], G, lat_ids, ts=(0.6, 0.75, 0.85, 0.9, 0.95))
        bl = max(rl, key=rl.get)
        print(f"round {r}: India F0.5 best {res[best]:.4f} @ t={best} | LATIN-ONLY best {rl[bl]:.4f} @ t={bl} | " +
              " ".join(f"{k}:{v:.4f}" for k, v in rl.items()) + f" ({time.time() - t:.0f}s)", flush=True)
        if r == a.rounds:
            break
        # pseudo-labels from this round (one-to-one first, like the decision rule)
        keep = decide.one_to_one(ev)
        pos = keep[keep["prob"] >= a.hi]
        neg = ev[ev["prob"] <= a.lo]
        pseudo = pd.concat([pos.assign(pl_label=1), neg.assign(pl_label=0)])
        acc = (pseudo["pl_label"].astype(bool) == pseudo["label"].astype(bool)).mean()
        print(f"  pseudo-labels: {len(pos)} matches, {len(neg)} non-matches | agree with truth {acc:.4f}", flush=True)
        X = pd.concat([tr[feats], pseudo[feats]])
        y = np.concatenate([tr["label"].astype(int).values, pseudo["pl_label"].values])
        w = np.concatenate([np.ones(len(tr)), np.full(len(pseudo), a.weight)])


if __name__ == "__main__":
    main()

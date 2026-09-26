"""Graph (collective) features for the stage-2 stack: each S1 + its confident copies form an entity consensus;
every uncertain candidate is compared with that consensus and with each confident sibling copy.

Validated in src/sim_relational.py (US/India OOF, stage-1 only): 0.98294 -> 0.98402.
All features are within-S1 (anchors = the S1's own pairs with stage-1 prob >= 0.9), so they mean the same on
the 200k training sample and on all test S1s.

    python src/graph_feats.py --split train   -> work/graph_train.parquet   (from train_oof)
    python src/graph_feats.py --split test    -> work/graph_test.parquet    (from test_scored)
"""
import argparse
import os

import numpy as np
import polars as pl

import config as C
from prep import load_norm
from sim_relational import ANCHOR, HI, LO, sibling

W = C.WORK_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--chunk", type=int, default=250_000, help="S1s per chunk")
    a = ap.parse_args()
    src = "train_oof.parquet" if a.split == "train" else "test_scored.parquet"
    o = pl.read_parquet(os.path.join(W, src), columns=["i", "j", "prob"])
    o = o.filter(pl.col("prob") > LO)   # only band pairs and anchors are needed
    norm = load_norm(a.split, columns=["name_core", "addr_core", "postcode", "addr_nums", "source"]).with_row_index("r")
    norm = norm.with_columns(pl.col("r").cast(o["j"].dtype),
                             *[pl.col(c).fill_null("") for c in ("name_core", "addr_core", "postcode", "addr_nums")])
    s1 = o.filter(pl.col("prob") < HI)["i"].unique().sort().to_numpy()
    parts = []
    for k in range(0, len(s1), a.chunk):
        ch = o.filter(pl.col("i").is_in(s1[k:k + a.chunk]))
        band = ch.filter(pl.col("prob") < HI).select("i", "j")
        rows = pl.concat([ch.select("i"), ch.select(pl.col("j").alias("i"))]).unique()["i"]
        nm = norm.filter(pl.col("r").is_in(rows.implode()))
        parts.append(sibling(band, ch.filter(pl.col("prob") >= ANCHOR), nm))
        print(f"  S1 {min(k + a.chunk, len(s1))}/{len(s1)}", flush=True)
    out = pl.concat(parts)
    out = out.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    path = os.path.join(W, f"graph_{a.split}.parquet")
    out.write_parquet(path)
    print(f"{out.height} band pairs, {out.width - 2} graph features -> {path}")


if __name__ == "__main__":
    main()

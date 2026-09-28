"""Co-location features as stack inputs (instead of a corrector on top): pairs with stage-1 prob in (0.005, 0.9995).
    python src/coloc_stack_feats.py --split train|test  -> work/coloc_pairs_{split}.parquet
"""
import argparse
import os

import polars as pl

import coloc
import config as C

W = C.WORK_DIR
FEATS = [f for f in coloc.PAIR_FEATS if f not in ("r2_name_ps1", "r2_addr_rel", "na2")] + ["co_n_other", "co_best_other", "co_gap"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    src = "train_oof.parquet" if a.split == "train" else "test_scored.parquet"
    p = pl.read_parquet(os.path.join(W, src), columns=["i", "j", "prob"]).filter((pl.col("prob") > 0.005) & (pl.col("prob") < 0.9995))
    if a.split == "train":
        s = pl.read_parquet(os.path.join(W, "ce_train_set.parquet"), columns=["i"]).unique()
        p = p.join(s, on="i")
    rec = pl.read_parquet(os.path.join(W, f"coloc_rec_{a.split}.parquet"))
    top, na_cnt, n_addr = coloc.extra_tables(rec)
    x = coloc.extra_features(coloc.pair_features(p.select("i", "j"), rec), rec, top, na_cnt, n_addr)
    x.select(["i", "j"] + [pl.col(f).alias("cl_" + f) for f in FEATS]).write_parquet(os.path.join(W, f"coloc_pairs_{a.split}.parquet"))
    print(a.split, x.height)

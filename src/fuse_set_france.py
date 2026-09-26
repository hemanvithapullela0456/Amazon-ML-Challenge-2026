"""Upload-2 variant: add the one-to-set model's scores to run7's France scores on the uncertain band.

run7's France probabilities (work/test_scored_unseen_fused.parquet) are rank-fused with the set model on the
band (LO < prob < HI) and mapped back onto run7's own band probabilities, so only the ORDER of band pairs changes
and the France threshold (0.90) keeps its meaning. Unvalidated on France (the set model was trained on India,
so the France-like simulation cannot score it); the weight is a moderate prior.

    python src/fuse_set_france.py --w_set 0.4
"""
import argparse
import os

import numpy as np
import polars as pl
from scipy.stats import rankdata

import config as C

W = C.WORK_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="test_scored_unseen_fused.parquet")
    ap.add_argument("--set", default="ce_test_set.parquet")
    ap.add_argument("--w_set", type=float, default=0.4)
    ap.add_argument("--out", default="test_scored_unseen_set.parquet")
    a = ap.parse_args()
    base = pl.read_parquet(os.path.join(W, a.base))
    st = pl.read_parquet(os.path.join(W, a.set)).select("i", "j", pl.col("ce").alias("set"))
    b = (base.filter((pl.col("prob") > 0.01) & (pl.col("prob") < 0.99))
         .join(st, on=["i", "j"], how="inner").to_pandas())
    R = lambda x: rankdata(x) / len(x)
    score = (1 - a.w_set) * R(b.prob.values) + a.w_set * R(b.set.values)
    qs = np.sort(b.prob.values)
    b["p_new"] = qs[(rankdata(score, method="ordinal") - 1).astype(int)].astype(np.float32)
    upd = pl.from_pandas(b[["i", "j", "p_new"]]).with_columns(pl.col("i").cast(base["i"].dtype), pl.col("j").cast(base["j"].dtype))
    out = (base.join(upd, on=["i", "j"], how="left").with_columns(pl.coalesce("p_new", "prob").alias("prob")).drop("p_new"))
    out.write_parquet(os.path.join(W, a.out))
    moved = float(np.mean(np.abs(b.p_new - b.prob) > 0.2))
    cross = float(np.mean((b.p_new >= 0.9) != (b.prob >= 0.9)))
    print(f"France band pairs fused with the set model: {len(b)} (of {base.height}) | moved > 0.2: {moved:.3f} | "
          f"decision at 0.90 flipped: {cross:.3f} -> work/{a.out}")


if __name__ == "__main__":
    main()

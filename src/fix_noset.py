"""Pairs with no set-model score and stage-1 prob >= 0.01 never occur in stack training (set coverage ~1.0 there),
so the (fixed) stack extrapolates badly on them; keep their stage-1 prob.
    python src/fix_noset.py <in stack parquet> <out parquet>
"""
import sys

import polars as pl

import config as C

W = C.WORK_DIR + "/"
src, out = sys.argv[1], sys.argv[2]
s = pl.read_parquet(W + src)
p1 = pl.read_parquet(W + "test_scored.parquet", columns=["i", "j", "prob"]).rename({"prob": "p1"})
cs = pl.read_parquet(W + "ce_test_set.parquet", columns=["i", "j"]).with_columns(hs=pl.lit(True))
x = s.join(p1, on=["i", "j"], how="left").join(cs, on=["i", "j"], how="left")
m = pl.col("hs").is_null() & (pl.col("p1") >= 0.01)
print("pairs reverted to stage-1:", x.filter(m).height, " of which p1>=0.725:", x.filter(m & (pl.col("p1") >= 0.725)).height)
x.with_columns(prob=pl.when(m).then(pl.col("p1")).otherwise(pl.col("prob")).cast(s["prob"].dtype)).select("i", "j", "prob").write_parquet(W + out)

"""France co-location rule (label-free transfer of a US/India train finding).

US/India train (stage-1 OOF): an alias (no name word shared with the S1) at the S1's exact address is 88-97% true at
p 0.3-0.9 when that S1 is the only S1 at the address, but only 42-50% true at p 0.9-0.95 when 2+ S1s share it.
France: promote the first class (best S1 for the record) to 0.901, demote the second to 0.85.

    python src/france_coloc.py --src ../work/test_scored_unseen_c_alias80.parquet --out ../work/test_scored_unseen_c_coloc.parquet
"""
import argparse

import polars as pl

import coloc
import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.3)
    a = ap.parse_args()
    rec = pl.read_parquet(f"{C.WORK_DIR}/coloc_rec_test.parquet")
    f = pl.read_parquet(a.src)
    x = coloc.pair_features(f.filter(pl.col("prob") > 0.02), rec)
    x = x.with_columns(best=pl.col("prob") == pl.col("prob").max().over("j"))
    alias = (pl.col("aeq") == 1) & (pl.col("ov") == 0)
    up = x.filter(alias & (pl.col("k1_addr") == 1) & pl.col("best") & (pl.col("prob") >= a.lo) & (pl.col("prob") < 0.9))
    dn = x.filter(alias & (pl.col("k1_addr") >= 2) & (pl.col("prob") >= 0.9) & (pl.col("prob") < 0.95))
    print(f"promote {up.height}, demote {dn.height}")
    ch = pl.concat([up.select("i", "j", pl.lit(0.901).alias("p_new")), dn.select("i", "j", pl.lit(0.85).alias("p_new"))])
    out = (f.join(ch, on=["i", "j"], how="left").with_columns(prob=pl.coalesce("p_new", "prob").cast(f["prob"].dtype))
           .drop("p_new"))
    out.write_parquet(a.out)


if __name__ == "__main__":
    main()

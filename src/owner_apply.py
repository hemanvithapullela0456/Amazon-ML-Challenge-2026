"""Apply the same-name owner choice (src/owner.py) to a score file without a labelled stack (France).

For a name-only copy j and an S1 i whose name is shared by m >= 2 S1s of the country:
  exact name (j's name_core == i's):  prob = own_pn                      (P(owner in the group) ~ 0.98)
  noisy name:                          prob = min(1, prob * m) * own_pn   (prob * m ~ P(owner in the group))
On the labelled US/India stack fold this rule scores 0.98481 vs 0.98313 (halving ambiguous exact copies: 0.98342).

    python src/owner_apply.py --scores test_scored_unseen_decoy_all_u.parquet --owner owner_test_unseen.parquet
"""
import argparse
import os

import polars as pl

import config as C

W = C.WORK_DIR


def apply_owner(sc, ow):
    d = sc.join(ow.select("i", "j", "own_pn", "own_m", "own_exact"), on=["i", "j"], how="left")
    amb = pl.col("own_m") >= 2
    ex = pl.col("own_exact") == 1
    new = (pl.when(amb & ex).then(pl.col("own_pn").clip(0, 0.9999))
           .when(amb).then((pl.col("prob") * pl.col("own_m")).clip(0, 0.9999) * pl.col("own_pn"))
           .otherwise(pl.col("prob")))
    d = d.with_columns(new.cast(sc.schema["prob"]).alias("prob_new"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--owner", default="owner_test_unseen.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t", type=float, default=0.9, help="decision threshold, for the change report only")
    a = ap.parse_args()
    sc = pl.read_parquet(os.path.join(W, a.scores)).with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))
    d = apply_owner(sc, pl.read_parquet(os.path.join(W, a.owner)))
    best_old = pl.col("prob") == pl.col("prob").max().over("j")
    best_new = pl.col("prob_new") == pl.col("prob_new").max().over("j")
    d = d.with_columns(old=(pl.col("prob") >= a.t) & best_old, new=(pl.col("prob_new") >= a.t) & best_new)
    print(f"pairs changed: {int((d['prob'] != d['prob_new']).sum())} | predictions: {int(d['old'].sum())} -> "
          f"{int(d['new'].sum())} (removed {int((d['old'] & ~d['new']).sum())}, added {int((~d['old'] & d['new']).sum())})")
    d.select("i", "j", pl.col("prob_new").alias("prob")).write_parquet(os.path.join(W, a.out))
    print(f"-> work/{a.out}")


if __name__ == "__main__":
    main()

"""Print true pairs that blocking misses (no shared key) or ranks low, with their normalised fields/keys.

Usage:  python src/diag_blocking.py [--n 25]
"""
import argparse
import sys

import numpy as np
import polars as pl

import config as C
from blocking import build_index, candidates, load_records, truth_pairs
from prep import load_norm

sys.stdout.reconfigure(encoding="utf-8")


def show(norm, rows, title, n):
    print(f"\n===== {title}: {rows.height} pairs (showing {min(n, rows.height)})")
    for i, j in rows.sample(min(n, rows.height), seed=1).select("i", "j").iter_rows():
        a, b = norm.row(i, named=True), norm.row(j, named=True)
        print(f"[{a['country']}] S1: {a['business_name']} | {a['business_address']}")
        print(f"      {b['entity_id'][:2]}: {b['business_name']} | {b['business_address']}")
        print(f"   norm S1: {a['name_core']} || {a['addr_core']}")
        print(f"   norm {b['entity_id'][:2]}: {b['name_core']} || {b['addr_core']}")
        common = set(a["keys"]) & set(b["keys"])
        print(f"   shared keys: {sorted(common)[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sample", type=float, default=0.02)
    a = ap.parse_args()
    recs = load_records("train")
    xk = build_index(recs)
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    q = np.sort(np.random.default_rng(1).choice(s1, int(len(s1) * a.sample), replace=False))
    truth = truth_pairs(recs, "train", q)
    pairs, stats = candidates(recs, xk, q, k=50, truth=truth)
    del xk
    norm = load_norm("train", columns=["entity_id", "business_name", "business_address", "country",
                                        "name_core", "addr_core", "keys"])
    missed = truth.join(stats.select("i", "j"), on=["i", "j"], how="anti")
    low = stats.filter(pl.col("blk_rank") > 50)
    cc = recs.select(pl.col("r").alias("i"), "country_key")
    for c in ["india", "us"]:
        show(norm, missed.join(cc, on="i").filter(pl.col("country_key") == c), f"NO SHARED KEY ({c})", a.n)
    show(norm, low, "FOUND BUT RANK > 50", a.n // 2)
    # what do the higher-ranked wrong candidates look like for those?
    ex = low.head(3)
    for i, j in ex.select("i", "j").iter_rows():
        top = pairs.filter(pl.col("i") == i).sort("blk_rank").head(5)
        print(f"\n--- S1 {norm.row(i, named=True)['business_name']} | {norm.row(i, named=True)['business_address']}")
        for jj, sc in top.select("j", "blk_score").iter_rows():
            r = norm.row(jj, named=True)
            print(f"   top cand score {sc:.1f}: {r['business_name']} | {r['business_address']}")


if __name__ == "__main__":
    main()

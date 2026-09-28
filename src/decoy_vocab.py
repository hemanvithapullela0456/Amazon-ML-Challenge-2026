"""Demote "vocabulary-swap" look-alikes in unseen-country scores (label-free, per country).

The generator's decoys include a record at the S1's address whose name keeps the S1's stem but carries a different
ordinary business word ("Bordeaux Ecole SARL" -> "Bordeaux Ehpad SARL", "Tourcoing Maternelle" -> "Tourcoing
Amicale"). In US/India such same-address swaps to a real vocabulary word are mostly negatives (23-57% true), while
swaps to the generator's inserted words (Services, Center, Group ...) are 99.9% true. In France the name templates
(city/initials + generic word) make these decoys common, and the unseen-country matcher scores them ~0.99.

Word roles are learned from the unseen country's own unlabelled records: a generator-inserted word appears in S2/S3
names far more often than in S1 names (ratio >= 1.3); an ordinary vocabulary word appears at the normal rate
(ratio < 1.3) and in >= 50 S1 names. The country's own name is treated as an inserted word ("... France SA").

Label-free check (France): true copies sit on S1s that have other copies; decoys land on S1s regardless of their
copy count. Predictions with a new vocabulary word are an S1's only match 4.5% of the time vs 1.8% for all others
(decoys: ~7%), and within every probability bin they are worse; below prob 0.999 the implied precision is ~0.1-0.35.

    python src/decoy_vocab.py --scores test_scored_unseen_mdeb_set4.parquet --max_prob 0.999
"""
import argparse
import os

import polars as pl

import config as C
from prep import load_norm

W = C.WORK_DIR


def word_roles(norm, ck, min_s1=50, max_ratio=1.3):
    """ordinary vocabulary words of country ck: in >= min_s1 S1 names and not over-represented in S2/S3 names"""
    n = norm.filter(pl.col("country_key") == ck)
    w = (n.select((pl.col("source") == 1).alias("s1"), pl.col("name_core").str.split(" ").list.unique().alias("w"))
         .explode("w").filter(pl.col("w").str.len_chars() > 0))
    t = w.group_by("w").agg(n1=pl.col("s1").sum(), n23=(~pl.col("s1")).sum())
    ns1 = int(n["source"].eq(1).sum())
    ns23 = n.height - ns1
    t = t.with_columns(r=(pl.col("n23") / ns23) / ((pl.col("n1") + 1) / ns1))
    return t.filter((pl.col("n1") >= min_s1) & (pl.col("r") < max_ratio) & (pl.col("w") != ck))["w"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="test_scored_unseen_mdeb_set4.parquet")
    ap.add_argument("--out", default="test_scored_unseen_decoy.parquet")
    ap.add_argument("--max_prob", type=float, default=0.999, help="demote only pairs below this probability")
    ap.add_argument("--factor", type=float, default=0.5, help="demoted prob = prob * factor")
    ap.add_argument("--skip_vocab", action="store_true",
                    help="only the name-only rule (e.g. on the US/India stage-2 scores, where the stack already learned "
                         "vocabulary swaps from labels)")
    ap.add_argument("--nameonly_m", type=int, default=0,
                    help="also demote name-only copies (empty address) whose exact name_core is shared by >= this many "
                         "S1s of the country (train: P(true) ~ 1/m, below the F0.5 break-even); 0 = off")
    ap.add_argument("--promote_unique", type=float, default=0.0,
                    help="raise name-only copies whose exact name_core belongs to exactly one S1 of the country (and "
                         "equals this S1's) to at least this prob (train: P(true) ~ 0.97); 0 = off")
    a = ap.parse_args()
    sc = pl.read_parquet(os.path.join(W, a.scores))
    norm = load_norm("test", columns=["name_core", "business_address", "source", "country_key"]).with_row_index("r")
    if a.skip_vocab:  # lean path for large score files: only pairs whose record is an ambiguous name-only copy
        m = norm.filter(pl.col("source") == 1).group_by("country_key", "name_core").len().rename({"len": "m"})
        rec = (norm.filter(pl.col("business_address").str.strip_chars() == "")
               .join(m, on=["country_key", "name_core"]).filter(pl.col("m") >= a.nameonly_m)
               .select(pl.col("r").alias("_j"), pl.col("name_core").alias("b")))
        keys = (sc.select(pl.col("i").cast(pl.UInt32).alias("_i"), pl.col("j").cast(pl.UInt32).alias("_j"), "i", "j")
                .join(rec, on="_j").join(norm.select(pl.col("r").alias("_i"), pl.col("name_core").alias("a")), on="_i")
                .filter(pl.col("a") == pl.col("b")).select("i", "j", pl.lit(True).alias("amb")))
        out = sc.join(keys, on=["i", "j"], how="left").with_columns(
            pl.when(pl.col("amb")).then(pl.col("prob") * a.factor).otherwise(pl.col("prob")).alias("prob"))
        print(f"ambiguous name-only pairs (shared by >= {a.nameonly_m} S1s) demoted: {keys.height}")
        out.select(sc.columns).write_parquet(os.path.join(W, a.out))
        print(f"-> work/{a.out}")
        return
    s1c = norm.select(pl.col("r").alias("i"), "country_key")
    d = sc.with_columns(pl.col("i").cast(pl.UInt32).alias("_i"), pl.col("j").cast(pl.UInt32).alias("_j"))
    d = (d.join(s1c.rename({"i": "_i"}), on="_i", how="left")
         .join(norm.select(pl.col("r").alias("_i"), pl.col("name_core").alias("a")), on="_i", how="left")
         .join(norm.select(pl.col("r").alias("_j"), pl.col("name_core").alias("b")), on="_j", how="left"))
    d = d.with_columns(new=pl.col("b").str.split(" ").list.unique()
                       .list.set_difference(pl.col("a").str.split(" ").list.unique()))
    flag = pl.lit(False)
    for ck in ([] if a.skip_vocab else d["country_key"].unique().to_list()):
        voc = word_roles(norm, ck)
        print(f"{ck}: {voc.len()} vocabulary words (e.g. {', '.join(voc.head(8).to_list())})")
        flag = flag | ((pl.col("country_key") == ck) & pl.col("new").list.eval(pl.element().is_in(voc.to_list())).list.any())
    d = d.with_columns(flag.alias("voc"))
    hit = pl.col("voc") & (pl.col("prob") < a.max_prob)
    if a.nameonly_m:
        m = norm.filter(pl.col("source") == 1).group_by("country_key", "name_core").len().rename({"len": "m"})
        rec = (norm.join(m, on=["country_key", "name_core"], how="left")
               .select(pl.col("r").alias("_j"), (pl.col("business_address").str.strip_chars() == "").alias("noaddr"),
                       pl.col("m").fill_null(0)))
        d = d.join(rec, on="_j", how="left")
        amb = pl.col("noaddr") & (pl.col("a") == pl.col("b")) & (pl.col("m") >= a.nameonly_m)
        print(f"ambiguous name-only pairs (shared by >= {a.nameonly_m} S1s): {int(d.select(amb.sum()).item())}")
        hit = hit | amb
        if a.promote_unique:
            uni = pl.col("noaddr") & (pl.col("a") == pl.col("b")) & (pl.col("m") == 1)
            print(f"unique-name name-only pairs promoted to >= {a.promote_unique}: "
                  f"{int(d.select((uni & (pl.col('prob') < a.promote_unique)).sum()).item())}")
            d = d.with_columns(pl.when(uni).then(pl.max_horizontal("prob", pl.lit(a.promote_unique, pl.Float32)))
                               .otherwise(pl.col("prob")).alias("prob"))
    best = pl.col("prob") == pl.col("prob").max().over("j")
    print(f"pairs with a new vocabulary word: {int(d['voc'].sum())} | demoted (prob < {a.max_prob}): "
          f"{int(d.select(hit.sum()).item())} | of them currently predicted (>= 0.9, best for record): "
          f"{int(d.select((hit & best & (pl.col('prob') >= 0.9)).sum()).item())}")
    out = d.with_columns(pl.when(hit).then(pl.col("prob") * a.factor).otherwise(pl.col("prob")).alias("prob"))
    out.select(sc.columns).write_parquet(os.path.join(W, a.out))
    print(f"-> work/{a.out}")


if __name__ == "__main__":
    main()

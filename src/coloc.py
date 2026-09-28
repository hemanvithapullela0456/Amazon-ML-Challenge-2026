"""Co-location / name-frequency features for a pair corrector (full-data counts, identical on train and test).

The pair models never see how many S1s share an address or a name. An alias (made-up name) at the S1's exact
address is 99.6% true when that S1 is the only one at the address, but ~51-76% when 2-3 S1s share it.

    python src/coloc.py --split train   -> work/coloc_rec_train.parquet  (per record)
    python src/coloc.py --split test    -> work/coloc_rec_test.parquet
"""
import argparse
import os

import polars as pl

import config as C
from prep import load_norm

W = C.WORK_DIR


def rec_table(split):
    n = load_norm(split, columns=["addr_core", "addr_nums", "name_core", "source", "country_key",
                                  "business_address"]).with_row_index("r")
    n = n.with_columns(
        akey=pl.col("addr_core").fill_null("").str.split(" ").list.unique().list.sort().list.join(" "),
        num1=pl.col("addr_nums").fill_null("").str.split(" ").list.first(),
        ncore=pl.col("name_core").fill_null(""),
        na=pl.col("business_address").fill_null("").str.strip_chars() == "")
    s1 = n.filter(pl.col("source") == 1)
    ns1 = s1.group_by("country_key").len().rename({"len": "ns1"})
    k_addr = s1.group_by("country_key", "akey").len().rename({"len": "k_addr"})
    k_name = s1.group_by("country_key", "ncore").len().rename({"len": "k_name"})
    rec = n.filter(pl.col("source") != 1)
    r_name = rec.group_by("country_key", "ncore").len().rename({"len": "r_name"})
    r_addr = rec.filter(~pl.col("na")).group_by("country_key", "akey").len().rename({"len": "r_addr"})
    # S1 name-token document frequency (per country, as a share of S1s)
    tok = (s1.select("country_key", pl.col("ncore").str.split(" ").list.unique().alias("t")).explode("t")
           .group_by("country_key", "t").len().join(ns1, on="country_key")
           .with_columns(tf=(pl.col("len") / pl.col("ns1")).log10()).select("country_key", "t", "tf"))
    tt = (n.select("r", "country_key", pl.col("ncore").str.split(" ").alias("t")).explode("t")
          .join(tok, on=["country_key", "t"], how="left").with_columns(pl.col("tf").fill_null(-7.0))
          .group_by("r").agg(tf_min=pl.col("tf").min(), tf_mean=pl.col("tf").mean()))
    out = (n.join(k_addr, on=["country_key", "akey"], how="left").join(k_name, on=["country_key", "ncore"], how="left")
           .join(r_name, on=["country_key", "ncore"], how="left").join(r_addr, on=["country_key", "akey"], how="left")
           .join(ns1, on="country_key").join(tt, on="r", how="left")
           .with_columns(pl.col("k_addr", "k_name", "r_name", "r_addr").fill_null(0))
           .with_columns(r_name_per_s1=pl.col("r_name") / pl.col("ns1") * 1e6,
                         r_addr_rel=pl.col("r_addr") / (pl.col("k_addr") + 1))
           .select("r", "source", "country_key", "akey", "num1", "ncore", "na", "k_addr", "k_name", "r_name", "r_addr",
                   "r_name_per_s1", "r_addr_rel", "tf_min", "tf_mean"))
    return out


PAIR_FEATS = ["aeq", "num_eq", "ov", "ov_frac", "neq", "k1_addr", "k1_name", "k2_addr", "k2_name", "r2_name_ps1",
              "r2_addr_rel", "tf2_min", "tf2_mean", "tf1_min", "na2"]


def pair_features(pairs, rec):
    a = rec.select(pl.col("r").alias("i"), pl.col("akey").alias("ak1"), pl.col("num1").alias("nu1"),
                   pl.col("ncore").alias("c1"), pl.col("k_addr").alias("k1_addr"), pl.col("k_name").alias("k1_name"),
                   pl.col("tf_min").alias("tf1_min"))
    b = rec.select(pl.col("r").alias("j"), pl.col("akey").alias("ak2"), pl.col("num1").alias("nu2"),
                   pl.col("ncore").alias("c2"), pl.col("k_addr").alias("k2_addr"), pl.col("k_name").alias("k2_name"),
                   pl.col("r_name_per_s1").alias("r2_name_ps1"), pl.col("r_addr_rel").alias("r2_addr_rel"),
                   pl.col("tf_min").alias("tf2_min"), pl.col("tf_mean").alias("tf2_mean"), pl.col("na").alias("na2"))
    x = pairs.join(a, on="i", how="left").join(b, on="j", how="left")
    t1, t2 = pl.col("c1").str.split(" "), pl.col("c2").str.split(" ")
    x = x.with_columns(
        aeq=(pl.col("ak1") == pl.col("ak2")) & ~pl.col("na2"),
        num_eq=(pl.col("nu1") == pl.col("nu2")) & (pl.col("nu1") != ""),
        ov=t1.list.set_intersection(t2).list.len(),
        neq=pl.col("c1") == pl.col("c2"))
    x = x.with_columns(ov_frac=pl.col("ov") / t2.list.len().clip(1, None))
    return x.with_columns([pl.col(f).cast(pl.Float32) for f in PAIR_FEATS])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    a = ap.parse_args()
    rec_table(a.split).write_parquet(os.path.join(W, f"coloc_rec_{a.split}.parquet"))


EXTRA_FEATS = ["co_n_other", "co_best_other", "co_gap", "mates_at_i", "mates_addr", "mates_at_i_frac"]


def extra_tables(rec, kmax=30):
    """co-located competitors (other S1s at the record's address) and alias-mates (records with the record's name)."""
    rec = rec.with_columns(tok=pl.col("ncore").str.split(" "))
    s1 = rec.filter((pl.col("source") == 1) & (pl.col("k_addr") <= kmax)).select(
        pl.col("r").alias("c"), "country_key", "akey", pl.col("tok").alias("tc"))
    ra = rec.filter((pl.col("source") != 1) & ~pl.col("na")).select("r", "country_key", "akey", "tok")
    co = (ra.join(s1, on=["country_key", "akey"]).with_columns(ov=pl.col("tok").list.set_intersection(pl.col("tc")).list.len())
          .select("r", "c", "ov"))
    co = co.sort(["r", "ov"], descending=[False, True])
    top = co.group_by("r", maintain_order=True).agg(cbest=pl.col("c").first(), o1=pl.col("ov").first(),
                                                    o2=pl.col("ov").slice(1, 1).first(), nco=pl.len())
    # alias-mates: records sharing the record's name core, counted per (name, address key)
    rr = rec.filter((pl.col("source") != 1) & (pl.col("ncore") != ""))
    na_cnt = rr.filter(~pl.col("na")).group_by("country_key", "ncore", "akey").len().rename({"len": "cna"})
    n_addr = rr.filter(~pl.col("na")).group_by("country_key", "ncore").len().rename({"len": "cn_addr"})
    return top, na_cnt, n_addr


def load_norm_country(rec):
    return rec["country_key"] if "country_key" in rec.columns else None


def extra_features(x, rec, top, na_cnt, n_addr):
    """x: pair table from pair_features (needs i, j, ak1, ak2, c2, na2)."""
    ck = rec.select(pl.col("r").alias("j"), "country_key")
    x = x.join(ck, on="j", how="left").join(top.rename({"r": "j"}), on="j", how="left")
    x = x.with_columns(
        co_n_other=(pl.col("nco").fill_null(0) - pl.when(pl.col("ak1") == pl.col("ak2")).then(1).otherwise(0)).clip(0, None),
        co_best_other=pl.when(pl.col("cbest") == pl.col("i")).then(pl.col("o2")).otherwise(pl.col("o1")).fill_null(0))
    x = x.with_columns(co_gap=pl.col("ov") - pl.col("co_best_other"))
    x = (x.join(na_cnt.rename({"ncore": "c2", "akey": "ak1"}), on=["country_key", "c2", "ak1"], how="left")
         .join(n_addr.rename({"ncore": "c2"}), on=["country_key", "c2"], how="left"))
    self_at_i = ((pl.col("ak2") == pl.col("ak1")) & (pl.col("na2") == 0)).cast(pl.Int32)
    x = x.with_columns(mates_at_i=(pl.col("cna").fill_null(0) - self_at_i).clip(0, None),
                       mates_addr=(pl.col("cn_addr").fill_null(0) - (pl.col("na2") == 0).cast(pl.Int32)).clip(0, None))
    x = x.with_columns(mates_at_i_frac=pl.col("mates_at_i") / pl.col("mates_addr").clip(1, None))
    return x.with_columns([pl.col(f).cast(pl.Float32) for f in EXTRA_FEATS]).drop("country_key", "cbest", "o1", "o2", "nco",
                                                                                   "cna", "cn_addr")

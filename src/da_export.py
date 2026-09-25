"""Bundles for da_encoder.py (domain-adapted cross-encoder).

--mode proxy : source = US uncertain pairs (labelled); target = India uncertain pairs of the US-trained matcher
               (sim_unseen.py), labels kept ONLY for scoring, plus a 'latin' flag for the France-like subset
--mode france: source = US + India uncertain pairs (labelled); target = France uncertain pairs of the unseen-country
               matcher (unseen_model.py) - no labels exist
Text = "name | address" (no country string). Record keys are "tr:<row>" (train) / "te:<row>" (test).

    python src/da_export.py --mode proxy     -> work/da_proxy/
    python src/da_export.py --mode france    -> work/da_france/
"""
import argparse
import os

import numpy as np
import polars as pl

import config as C
from blocking import load_records, truth_pairs
from prep import load_norm


def texts(split, rows, tag):
    raw = load_norm(split, columns=["business_name", "business_address"]).with_row_index("r").with_columns(pl.col("r").cast(pl.UInt32))
    raw = raw.filter(pl.col("r").is_in(rows.implode()))
    return raw.select((pl.lit(tag + ":") + pl.col("r").cast(pl.Utf8)).alias("k"),
                      (pl.col("business_name").fill_null("").str.strip_chars() + " | "
                       + pl.col("business_address").fill_null("").str.strip_chars()).alias("text"))


def keyed(df, tag):
    return df.with_columns((pl.lit(tag + ":") + pl.col("i").cast(pl.Utf8)).alias("ka"),
                           (pl.lit(tag + ":") + pl.col("j").cast(pl.Utf8)).alias("kb"))


def pseudo(df, tag, n=150_000, hi=0.97, lo=0.03):
    """Teacher pseudo-labels on the target country: confident one-to-one matches -> 1, confident non-matches -> 0
    (half of them 'hard': prob between 0.001 and lo). Never touches a label column."""
    best = df.filter(pl.col("prob") >= pl.col("prob").max().over("j"))
    pos = best.filter(pl.col("prob") >= hi)
    neg = df.filter(pl.col("prob") <= lo)
    hard = neg.filter(pl.col("prob") >= 0.001)
    pos = pos.sample(min(n, pos.height), seed=1)
    neg = pl.concat([hard.sample(min(n // 2, hard.height), seed=1), neg.sample(min(n // 2, neg.height), seed=2)]).unique(["i", "j"])
    out = pl.concat([pos.with_columns(pl.lit(1.0).alias("pl_label")), neg.with_columns(pl.lit(0.0).alias("pl_label"))])
    return keyed(out.select("i", "j", "pl_label"), tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["proxy", "france"], required=True)
    ap.add_argument("--lo", type=float, default=0.01)
    ap.add_argument("--hi", type=float, default=0.99)
    a = ap.parse_args()
    W = C.WORK_DIR
    out = os.path.join(W, f"da_{a.mode}")
    os.makedirs(out, exist_ok=True)
    bundle = pl.read_parquet(os.path.join(W, "ce_bundle", "train_pairs.parquet"))
    if a.mode == "proxy":
        src = keyed(bundle.filter(pl.col("country_key") == "us"), "tr").select("ka", "kb", "label")
        ev = pl.read_parquet(os.path.join(W, "sim_us2in", "eval_without.parquet"))
        tgt = ev.filter((pl.col("prob") >= a.lo) & (pl.col("prob") <= a.hi))
        G = pl.read_parquet(os.path.join(W, "train_G.parquet"))
        ids = G.filter(pl.col("country_key") == "india")["i"].to_numpy()
        nl = (load_norm("train", columns=["business_name", "business_address"]).with_row_index("r")
              .select(pl.col("r").cast(pl.UInt32), (pl.col("business_name").str.contains(r"[ऀ-෿]")
                      | pl.col("business_address").str.contains(r"[ऀ-෿]")).alias("nl")))
        tp = truth_pairs(load_records("train"), "train", ids).join(nl.rename({"r": "j"}), on="j")
        bad = set(tp.filter(pl.col("nl"))["i"].to_list()) | set(nl.filter(pl.col("nl") & pl.col("r").is_in(ids))["r"].to_list())
        tgt = keyed(tgt, "tr").with_columns(pl.col("i").is_in(list(bad)).not_().alias("latin"),
                                            pl.col("prob").alias("p_lgbm")).select("ka", "kb", "i", "j", "label", "p_lgbm", "latin")
        ps = pseudo(ev.select("i", "j", "prob"), "tr")
        acc = ps.join(ev.select("i", "j", "label"), on=["i", "j"]).select(
            ((pl.col("pl_label") > 0.5) == pl.col("label")).mean()).item()
        print(f"[proxy] teacher pseudo-labels {ps.height} (agree with the hidden truth {acc:.4f} - reported only)")
        rows = pl.concat([bundle.filter(pl.col("country_key") == "us")["i"], bundle.filter(pl.col("country_key") == "us")["j"],
                          tgt["i"], tgt["j"], ps["i"], ps["j"]]).unique()
        tx = texts("train", rows, "tr")
    else:
        src = keyed(bundle, "tr").select("ka", "kb", "label")
        sc = pl.read_parquet(os.path.join(W, "test_scored_unseen.parquet"))
        tgt = keyed(sc.filter((pl.col("prob") >= a.lo) & (pl.col("prob") <= a.hi)), "te").with_columns(
            pl.col("prob").alias("p_lgbm")).select("ka", "kb", "i", "j", "p_lgbm")
        ps = pseudo(sc, "te")
        tx = pl.concat([texts("train", pl.concat([bundle["i"], bundle["j"]]).unique(), "tr"),
                        texts("test", pl.concat([tgt["i"], tgt["j"], ps["i"], ps["j"]]).unique(), "te")])
    src.write_parquet(os.path.join(out, "src_pairs.parquet"))
    tgt.write_parquet(os.path.join(out, "tgt_pairs.parquet"))
    tx.write_parquet(os.path.join(out, "texts.parquet"))
    ps.select("ka", "kb", "pl_label").write_parquet(os.path.join(out, "tgt_pseudo.parquet"))
    extra = f", latin {int(tgt['latin'].sum())}, match rate {tgt['label'].mean():.3f}" if a.mode == "proxy" else ""
    print(f"[{a.mode}] source {src.height} labelled pairs | target {tgt.height} pairs{extra} | texts {tx.height} -> {out}")


if __name__ == "__main__":
    main()

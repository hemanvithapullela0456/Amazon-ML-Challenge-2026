"""Approach C, step 1 (laptop): export labelled hard pairs to test a NON-fine-tuned LLM judge on Kaggle.

A zero/few-shot LLM never trained on US/India has no home country, so its accuracy on labelled US/India pairs is
an honest estimate of its accuracy on France. Exported for comparison on the same pairs:
  p_stage1 : in-domain stage-1 probability (out-of-fold)
  p_unseen : the unseen-country matcher's probability (India only; US-trained model from sim_unseen.py)

    python src/llm_export.py --n 2000
-> work/llm_eval/{eval_pairs.parquet, fewshot.json}  (upload the folder to Kaggle as a dataset)
"""
import argparse
import json
import os

import polars as pl

import config as C
from prep import load_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="pairs per country")
    ap.add_argument("--lo", type=float, default=0.05)
    ap.add_argument("--hi", type=float, default=0.97)
    a = ap.parse_args()
    W = C.WORK_DIR
    out = os.path.join(W, "llm_eval")
    os.makedirs(out, exist_ok=True)
    raw = (load_norm("train", columns=["business_name", "business_address", "country"]).with_row_index("r")
           .with_columns(pl.col("r").cast(pl.UInt32),
                         (pl.col("business_name").fill_null("").str.strip_chars() + " | "
                          + pl.col("business_address").fill_null("").str.strip_chars() + " | "
                          + pl.col("country").fill_null("")).alias("text"))
           .select("r", "text"))
    o = pl.read_parquet(os.path.join(W, "train_oof.parquet"), columns=["i", "j", "label", "prob"])
    cc = pl.read_parquet(os.path.join(W, "train_G.parquet")).select("i", "country_key")
    band = o.filter((pl.col("prob") >= a.lo) & (pl.col("prob") <= a.hi)).join(cc, on="i")
    unseen = pl.read_parquet(os.path.join(W, "sim_us2in", "eval_without.parquet")).select(
        "i", "j", pl.col("prob").alias("p_unseen"))
    parts = [band.filter(pl.col("country_key") == c).sample(a.n, seed=C.SEED) for c in ("us", "india")]
    ev = pl.concat(parts).rename({"prob": "p_stage1"}).join(unseen, on=["i", "j"], how="left")
    shots = band.join(ev.select("i", "j"), on=["i", "j"], how="anti")
    shots = pl.concat([shots.filter(pl.col("label")).sample(4, seed=7), shots.filter(~pl.col("label")).sample(4, seed=7)]).sample(fraction=1.0, shuffle=True, seed=3)

    def with_text(d):
        return (d.join(raw.rename({"r": "i", "text": "text_a"}), on="i")
                 .join(raw.rename({"r": "j", "text": "text_b"}), on="j"))
    ev = with_text(ev).select("i", "j", "country_key", "label", "p_stage1", "p_unseen", "text_a", "text_b")
    ev.write_parquet(os.path.join(out, "eval_pairs.parquet"))
    shots = with_text(shots)
    json.dump([{"a": r["text_a"], "b": r["text_b"], "same": bool(r["label"])} for r in shots.iter_rows(named=True)],
              open(os.path.join(out, "fewshot.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{ev.height} eval pairs (match rate {ev['label'].mean():.3f}), 8 few-shot examples -> {out}")


if __name__ == "__main__":
    main()

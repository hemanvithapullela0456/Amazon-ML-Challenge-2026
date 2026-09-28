"""Assemble the US/India stage-2 hybrid: address pairs from the moves stack, name-only pairs from the owner stack.
(test_scored_stage2_hyb2 = buggy-stack version; _hybfix = stacks refit after the stack.py prob-column fix.)

    python src/make_hyb.py --moves fix_test_scored_stage2_set_qwen_mdeb_q17_graph_moves.parquet \
        --owner fix_test_scored_stage2_set_qwen_mdeb_q17_graph_moves_owner.parquet --tag _hybfix
"""
import argparse
import os
import shutil

import polars as pl

import config as C

W = C.WORK_DIR

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    na = pl.read_parquet(os.path.join(W, "coloc_rec_test.parquet"), columns=["r", "na"]).rename({"r": "j"})
    m = pl.read_parquet(os.path.join(W, a.moves))
    o = pl.read_parquet(os.path.join(W, a.owner)).rename({"prob": "po"})
    h = (m.join(na, on="j", how="left").join(o, on=["i", "j"], how="left")
         .with_columns(prob=pl.when(pl.col("na")).then(pl.col("po")).otherwise(pl.col("prob")).cast(m["prob"].dtype))
         .select("i", "j", "prob"))
    assert h["prob"].null_count() == 0
    h.write_parquet(os.path.join(W, f"test_scored_stage2{a.tag}.parquet"))
    shutil.copy(os.path.join(W, "decision_stage2_hyb2.json"), os.path.join(W, f"decision_stage2{a.tag}.json"))
    print("wrote", h.height)

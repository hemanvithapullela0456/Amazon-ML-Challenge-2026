"""Lightweight ID-existence check: does every ID in matching_results.tsv actually exist in the test set?

A cheap polars-based alternative to validate_submission.py --check-ids (which uses plain Python sets
and needs "a few GB"). Only reads the entity_id column, so it stays light even on the full ~1.7M-entity
test set.
"""
import csv
import sys

import polars as pl

import config as C


def main():
    ids = (pl.read_csv(f"{C.DATA_DIR}/test/test_source2.tsv", separator="\t", columns=["entity_id"], quote_char=None)
           .to_series())
    ids3 = (pl.read_csv(f"{C.DATA_DIR}/test/test_source3.tsv", separator="\t", columns=["entity_id"], quote_char=None)
           .to_series())
    valid = set(ids.to_list()) | set(ids3.to_list())
    print(f"valid S2/S3 test IDs: {len(valid)}")

    m = pl.read_csv(f"{C.OUT_DIR}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False)
    all_matched = (m.filter(pl.col("matched_entity_ids") != "")["matched_entity_ids"]
                  .str.split(",").explode())
    n_total = all_matched.len()
    bad = all_matched.filter(~all_matched.is_in(list(valid)))
    print(f"total matched IDs in output: {n_total}")
    print(f"IDs that DON'T exist in test set: {bad.len()} ({bad.len() / max(n_total, 1):.4%})")
    if bad.len():
        print("examples:", bad.head(10).to_list())
    else:
        print("ALL matched IDs exist in the test set. Clean.")

    # sanity: every S2/S3 id belongs to at most one S1 (self-consistency, not a scorer rule)
    dup = (all_matched.value_counts().filter(pl.col("count") > 1))
    print(f"IDs matched to more than one S1 (should be 0 if blocking assumption held): {dup.height}")


if __name__ == "__main__":
    main()

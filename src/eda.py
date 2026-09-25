"""Quick data checks that decide pipeline settings (run first).

Usage:  python src/eda.py
"""
from collections import Counter

import numpy as np
import pandas as pd

from io_utils import load_gt, load_sources


def main():
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    for split in ["train", "test"]:
        recs = load_sources(split)
        print(f"\n=== {split}: {len(recs)} records")
        print(recs.groupby(["source", "country"]).size().unstack(fill_value=0))
        for c in ["business_name", "business_address", "country"]:
            print(f"  empty {c}: {(recs[c].str.strip() == '').mean():.4f}")
        dup = recs["entity_id"].duplicated().sum()
        print(f"  duplicate entity_ids: {dup}")

    recs = load_sources("train")
    gt = load_gt("train")
    rid = recs.set_index("entity_id")
    n_m = np.array([len(v) for v in gt.values()])
    print(f"\n=== ground truth: {len(gt)} S1 entities")
    print(f"  singletons {np.mean(n_m == 0):.4f} | mean matches {n_m.mean():.2f} | max {n_m.max()}")
    print("  matches-per-S1 distribution:", dict(sorted(Counter(np.minimum(n_m, 10)).items())))
    s1_in_file = set(rid.index[rid.source == 1])
    print(f"  S1 in GT but not in file: {len(set(gt) - s1_in_file)} | in file not in GT: {len(s1_in_file - set(gt))}")

    all_m = [m for v in gt.values() for m in v]
    cnt = Counter(all_m)
    print(f"  matched ids: {len(all_m)} | S2 share {np.mean([m.startswith('S2') for m in all_m]):.3f}")
    print(f"  ONE-TO-ONE CHECK: ids appearing in >1 S1 list: {sum(c > 1 for c in cnt.values())}")
    unknown = [m for m in cnt if m not in rid.index]
    print(f"  matched ids missing from source files: {len(unknown)}")
    for s in (2, 3):
        ids = set(rid.index[rid.source == s])
        print(f"  S{s}: {len(ids)} records, {len(ids & set(cnt)) / max(len(ids), 1):.3f} of them matched to some S1")

    same = [rid.at[s, "country"] == rid.at[m, "country"] for s, v in gt.items() for m in v if m in rid.index]
    print(f"  SAME-COUNTRY CHECK: matched pairs with equal country label: {np.mean(same):.4f}")

    print("\n=== sample matches")
    shown = 0
    for s, v in gt.items():
        if v and shown < 8:
            print(f"S1 [{rid.at[s, 'country']}] {rid.at[s, 'business_name']} | {rid.at[s, 'business_address']}")
            for m in sorted(v):
                if m in rid.index:
                    print(f"   {m[:2]} {rid.at[m, 'business_name']} | {rid.at[m, 'business_address']}")
            shown += 1


if __name__ == "__main__":
    main()

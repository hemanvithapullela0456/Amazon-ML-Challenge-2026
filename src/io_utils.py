"""Reading the source TSVs / ground truth and writing the two submission files."""
import csv
import os

import pandas as pd

from config import DATA_DIR

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    # QUOTE_NONE: names/addresses may contain stray quotes; the files use tabs, not quoting.
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_sources(split, data_dir=DATA_DIR):
    """Returns one DataFrame with all records of the split plus a 'source' column (1/2/3)."""
    parts = []
    for s in (1, 2, 3):
        df = read_tsv(os.path.join(data_dir, split, f"{split}_source{s}.tsv"))
        for c in COLS:
            if c not in df:
                df[c] = ""
        df = df[COLS].copy()
        df["source"] = s
        parts.append(df)
    recs = pd.concat(parts, ignore_index=True)
    recs["entity_id"] = recs["entity_id"].str.strip()
    return recs


def load_gt(split="train", data_dir=DATA_DIR):
    """dict: S1 id -> set of matched S2/S3 ids (empty set for singletons)."""
    df = read_tsv(os.path.join(data_dir, split, f"{split}_ground_truth.tsv"))
    return {r.source1_entity_id.strip(): parse_id_list(r.matched_entity_ids) for r in df.itertuples()}


def parse_id_list(s):
    return {x.strip() for x in str(s).split(",") if x.strip()}


def _sort_key(eid):
    return (eid[:3], eid)


def write_id_lists(path, s1_ids, lists, col):
    """One row per S1 id (in the given order), ids sorted & deduplicated, empty when none."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for s1 in s1_ids:
            ids = sorted(set(lists.get(s1, ())), key=_sort_key)
            f.write(f"{s1}\t{','.join(ids)}\n")
    print(f"wrote {path} ({len(s1_ids)} rows)")


def archive_run(matching_path, note, runs_dir=None):
    """Copy a finished matching_results.tsv to <out>/runs/runN_matching_results.tsv (N = next free number)
    and append a line to runs/RUNS.md, so every submission candidate is kept under a stable name."""
    import re
    import shutil
    import time
    from config import OUT_DIR
    runs_dir = runs_dir or os.path.join(OUT_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    nums = [int(m.group(1)) for f in os.listdir(runs_dir) if (m := re.match(r"run(\d+)_", f))]
    n = max(nums, default=0) + 1
    dst = os.path.join(runs_dir, f"run{n}_matching_results.tsv")
    shutil.copyfile(matching_path, dst)
    log = os.path.join(runs_dir, "RUNS.md")
    if not os.path.exists(log):
        with open(log, "w", encoding="utf-8") as f:
            f.write("| run | saved | what | leaderboard |\n|---|---|---|---|\n")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"| run{n} | {time.strftime('%Y-%m-%d %H:%M')} | {note.replace('|', ';')} | |\n")
    print(f"archived -> {dst}")
    return dst


def write_submission(out_dir, s1_ids, matches, candidates):
    # final matches must be a subset of candidates
    for s1, m in matches.items():
        assert set(m) <= set(candidates.get(s1, ())), f"{s1}: match outside candidate set"
    write_id_lists(os.path.join(out_dir, "matching_results.tsv"), s1_ids, matches, "matched_entity_ids")
    write_id_lists(os.path.join(out_dir, "candidate_pairs.tsv"), s1_ids, candidates, "candidate_entity_ids")

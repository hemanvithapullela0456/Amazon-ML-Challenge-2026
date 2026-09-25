"""Test inference: blocking -> features (chunked by S1) -> LightGBM -> decision -> output/*.tsv.

Usage:  python src/run_pipeline.py [--k 30] [--chunk 250000]
Needs work/test_norm (python src/prep.py --split test) and a trained model (python src/train.py).
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
import decide
from blocking import add_competition, add_dense, build_keys, candidates, load_ranker, load_records
from features import idf_tables, make_features
from io_utils import archive_run, write_id_lists
from train import norm_table


def run_validator():
    if not os.path.exists(C.VALIDATOR):
        print(f"validator not found at {C.VALIDATOR}; run it manually")
        return
    cmd = [sys.executable, C.VALIDATOR,
           "--matching", os.path.join(C.OUT_DIR, "matching_results.tsv"),
           "--candidate", os.path.join(C.OUT_DIR, "candidate_pairs.tsv"),
           "--test-dir", os.path.join(C.DATA_DIR, "test")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:])


def score_test(k, chunk, n_jobs, resume=True, dense=""):
    """Scores test pairs in S1 batches, checkpointing each batch to work/test_scored_parts/ so a crash
    (e.g. an out-of-memory worker) only loses the batch in progress, not the whole run."""
    t = time.time()
    with open(os.path.join(C.WORK_DIR, "decision.json")) as f:
        cfg = json.load(f)
    model = lgb.Booster(model_file=os.path.join(C.WORK_DIR, "model.txt"))
    recs = load_records("test")
    keys = build_keys(recs)
    s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    pairs, _ = candidates(keys, s1, k=k, ranker=load_ranker())
    del keys
    pairs = add_competition(pairs)
    if dense:
        pairs = add_dense(pairs, dense)
    print(f"test blocking: {pairs.height} pairs for {len(s1)} S1 ({time.time() - t:.0f}s)", flush=True)
    norm = norm_table("test")
    idfs = idf_tables(norm)

    part_dir = os.path.join(C.WORK_DIR, "test_scored_parts" + ("_dense" if dense else ""))  # never mix checkpoints
    os.makedirs(part_dir, exist_ok=True)
    for a in range(0, len(s1), chunk):
        part_path = os.path.join(part_dir, f"part-{a:09d}.parquet")
        if resume and os.path.exists(part_path):
            print(f"  S1 {min(a + chunk, len(s1))}/{len(s1)} already scored, skipping", flush=True)
            continue
        p = pairs.filter(pl.col("i").is_in(s1[a:a + chunk]))
        f = make_features(p, norm, idfs=idfs, n_jobs=n_jobs)
        prob = model.predict(f.select(cfg["features"]).to_numpy())
        f.select("i", "j").with_columns(pl.Series("prob", prob.astype(np.float32))).write_parquet(part_path)
        del p, f, prob
        gc.collect()
        print(f"  scored S1 {min(a + chunk, len(s1))}/{len(s1)} ({time.time() - t:.0f}s)", flush=True)
    del pairs, norm
    gc.collect()
    scored = pl.read_parquet(os.path.join(part_dir, "*.parquet"))
    scored.write_parquet(os.path.join(C.WORK_DIR, "test_scored.parquet"))
    return recs, s1, scored, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=15_000, help="S1 businesses per batch (lower = less RAM)")
    ap.add_argument("--n_jobs", type=int, default=4, help="worker processes for feature scoring")
    ap.add_argument("--no-resume", action="store_true", help="ignore work/test_scored_parts checkpoints")
    ap.add_argument("--dense", default="", help="dense pairs parquet to union in (e.g. work/dense_test.parquet)")
    ap.add_argument("--unseen_t", type=float, default=None,
                    help="stricter match threshold for test countries that never appear in training")
    ap.add_argument("--note", default="", help="description for output/runs/RUNS.md")
    ap.add_argument("--reuse", action="store_true", help="reuse work/test_scored.parquet (only re-decide)")
    ap.add_argument("--stage2", action="store_true", help="with --reuse: use the stacked (cross-encoder) scores")
    a = ap.parse_args()

    if a.reuse:
        tag = "_stage2" if a.stage2 else ""
        with open(os.path.join(C.WORK_DIR, f"decision{tag}.json")) as f:
            cfg = json.load(f)
        recs = load_records("test")
        s1 = recs.filter(pl.col("source") == 1)["r"].to_numpy()
        scored = pl.read_parquet(os.path.join(C.WORK_DIR, f"test_scored{tag}.parquet"))
    else:
        recs, s1, scored, cfg = score_test(a.k, a.chunk, a.n_jobs, resume=not a.no_resume, dense=a.dense)
    print(f"decision: { {k: v for k, v in cfg.items() if k != 'features'} }")

    df = scored.to_pandas()
    df["pred"] = decide.predict_mask(df, cfg)
    if a.unseen_t is not None:
        # countries absent from training are a distribution shift (their look-alike businesses fool a model
        # tuned on the training countries), so demand a higher probability there. Decided per country label
        # seen/unseen in train - no country is named in the code.
        seen = set(pl.read_parquet(os.path.join(C.WORK_DIR, "train_G.parquet"))["country_key"].unique().to_list())
        s1r = recs.filter(pl.col("source") == 1)
        cmap = dict(zip(s1r["r"].to_list(), s1r["country_key"].to_list()))
        unseen = ~df["i"].map(cmap).isin(seen).values
        before = int(df["pred"].sum())
        df["pred"] = df["pred"].values & (~unseen | (df["prob"].values >= a.unseen_t))
        print(f"unseen-country threshold {a.unseen_t}: countries {sorted(set(cmap.values()) - seen)} | "
              f"dropped {before - int(df['pred'].sum())} predicted pairs")
    ids = recs["entity_id"].to_numpy()
    s1_ids = ids[s1]
    cand = df.groupby("i")["j"].apply(lambda s: ids[s.values].tolist()).to_dict()
    match = df[df["pred"]].groupby("i")["j"].apply(lambda s: ids[s.values].tolist()).to_dict()
    cand = {ids[i]: v for i, v in cand.items()}
    match = {ids[i]: v for i, v in match.items()}

    n = np.array([len(match.get(s, ())) for s in s1_ids])
    ctry = recs["country_key"].to_numpy()[s1]
    print(f"test: {len(s1)} S1 | predicted singletons {np.mean(n == 0):.3f} | mean matches {n.mean():.2f}")
    for c in np.unique(ctry):
        m = ctry == c
        print(f"  {c}: {m.sum()} S1 | singletons {np.mean(n[m] == 0):.3f} | mean matches {n[m].mean():.2f}")
    write_id_lists(os.path.join(C.OUT_DIR, "matching_results.tsv"), s1_ids, match, "matched_entity_ids")
    write_id_lists(os.path.join(C.OUT_DIR, "candidate_pairs.tsv"), s1_ids, cand, "candidate_entity_ids")
    run_validator()
    stats = " ".join(f"{c} {np.mean(n[ctry == c] == 0):.3f}/{n[ctry == c].mean():.2f}" for c in np.unique(ctry))
    archive_run(os.path.join(C.OUT_DIR, "matching_results.tsv"),
                a.note or f"{'stage2 ' if a.stage2 else 'stage1 '}"
                          f"cv {cfg.get('cv_f05', float('nan')):.4f}; singletons/matches {stats}")


if __name__ == "__main__":
    main()

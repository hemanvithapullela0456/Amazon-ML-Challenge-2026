"""Normalise every record of a split (in parallel) and build its blocking keys.

Usage:  python src/prep.py --split train      (-> work/train_norm.parquet)
        python src/prep.py --split test
Row order = S1 file, then S2, then S3; the row number is the record index used everywhere else.
"""
import argparse
import os
import time

import polars as pl
from joblib import Parallel, delayed

import config as C
from normalize import NORM_FIELDS, STREET_WORDS, NAME_STOP, norm_record

# ---------------------------------------------------------------- blocking keys
# Several key families; blocking.py measures which ones actually retrieve true matches.
KEY_TYPES = {
    "h": "house number + next street word",
    "p": "two consecutive number tokens",
    "s": "street-name bigram after a number",
    "n": "single name token",
    "b": "first two name tokens (sorted)",
    "c": "whole compact name",
    "m": "name token + house number",
}


def make_keys(name_core, name_alias, addr_core):
    keys = set()
    at = addr_core.split()
    nums = [t for t in at if any(ch.isdigit() for ch in t)]
    for k, t in enumerate(at):
        if not any(ch.isdigit() for ch in t):
            continue
        nxt = [w for w in at[k + 1:k + 4] if w.isalpha() and len(w) >= 3 and w not in STREET_WORDS]
        if nxt:
            keys.add(f"h:{t}:{nxt[0]}")
            if len(nxt) >= 2:
                keys.add(f"s:{nxt[0]}:{nxt[1]}")
        if k + 1 < len(at) and any(ch.isdigit() for ch in at[k + 1]):
            keys.add(f"p:{t}:{at[k + 1]}")
    for name in (name_core, name_alias):
        nt = [t for t in name.split() if len(t) >= 3 and t not in NAME_STOP]
        if not nt:
            continue
        for t in nt:
            keys.add(f"n:{t}")
            for h in nums[:2]:
                keys.add(f"m:{t}:{h}")
        if len(nt) >= 2:
            keys.add("b:" + ":".join(sorted(nt[:2])))
        keys.add("c:" + "".join(name.split()))
    return list(keys)


def _norm_batch(names, addrs):
    rows = [norm_record(n, a) for n, a in zip(names, addrs)]
    cols = list(zip(*rows)) if rows else [[] for _ in NORM_FIELDS]
    df = pl.DataFrame({f: list(c) for f, c in zip(NORM_FIELDS, cols)}, schema={f: pl.Utf8 for f in NORM_FIELDS})
    keys = [make_keys(nc, na, ac) for nc, na, ac in zip(df["name_core"], df["name_alias"], df["addr_core"])]
    return df.with_columns(pl.Series("keys", keys, dtype=pl.List(pl.Utf8)))


def read_source(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8-lossy")


def prep(split, n_jobs=C.N_JOBS, batch=50_000, part_rows=1_000_000):
    """Writes work/<split>_norm/part-XXX.parquet (row order preserved) to keep memory bounded."""
    t = time.time()
    parts = []
    for s in (1, 2, 3):
        df = read_source(os.path.join(C.DATA_DIR, split, f"{split}_source{s}.tsv"))
        df = df.select([pl.col(c).fill_null("") for c in ["entity_id", "business_name", "business_address", "country"]])
        parts.append(df.with_columns(pl.lit(s, dtype=pl.Int8).alias("source")))
    raw = pl.concat(parts)
    del parts
    print(f"[{split}] read {raw.height} records ({time.time() - t:.0f}s)", flush=True)

    out_dir = os.path.join(C.WORK_DIR, f"{split}_norm")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    with Parallel(n_jobs=n_jobs, batch_size=1, return_as="generator") as par:
        for p, a in enumerate(range(0, raw.height, part_rows)):
            r = raw.slice(a, part_rows)
            nm, ad = r["business_name"].to_list(), r["business_address"].to_list()
            norm = pl.concat(list(par(delayed(_norm_batch)(nm[b:b + batch], ad[b:b + batch])
                                      for b in range(0, len(nm), batch))))
            df = pl.concat([r, norm], how="horizontal").with_columns(
                pl.col("country").str.strip_chars().str.to_lowercase().alias("country_key"))
            df.write_parquet(os.path.join(out_dir, f"part-{p:03d}.parquet"))
            print(f"[{split}] {a + r.height}/{raw.height} done ({time.time() - t:.0f}s)", flush=True)
    print(f"[{split}] normalised + keys -> {out_dir} ({time.time() - t:.0f}s)", flush=True)


def load_norm(split, columns=None):
    return pl.read_parquet(os.path.join(C.WORK_DIR, f"{split}_norm", "part-*.parquet"), columns=columns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_jobs", type=int, default=C.N_JOBS)
    a = ap.parse_args()
    prep(a.split, a.n_jobs)

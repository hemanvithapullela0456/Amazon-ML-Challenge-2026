"""Pairwise features for (S1 record i, candidate S2/S3 record j), computed in parallel chunks.

No country one-hot: every feature is a similarity / agreement signal or a rank relative to the
other candidates, so the model can transfer to a country unseen in training (France).
"""
import math
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import config as C

FIELDS = ["name_norm", "name_core", "name_alias", "name_acr", "name_legal",
          "addr_core", "postcode", "addr_nums", "landmark", "region"]

STR_FEATS = ["n_ratio", "n_partial", "n_tsort", "n_tset", "n_norm_ratio", "n_jw", "n_alias_best",
             "n_first_eq", "n_exact", "n_contain", "n_acronym", "n_idf_jac", "n_idf_ovl", "n_unmatched_idf",
             "n_len_a", "n_len_b", "n_len_diff", "legal_jac", "legal_conflict",
             "a_ratio", "a_tset", "a_tsort", "a_partial", "a_idf_jac", "a_idf_ovl", "a_unmatched_idf",
             "pc_eq", "num_jac", "num_conflict", "hn_eq", "a_len_a", "a_len_b",
             "lm_tset", "lm_in_b", "lm_in_a", "region_eq", "combo_tset"]

_IDF_N, _IDF_A = {}, {}


def _init(idf_n, idf_a):
    global _IDF_N, _IDF_A
    _IDF_N, _IDF_A = idf_n, idf_a


def idf_table(series, min_df=3):
    """token -> idf over all records (tokens rarer than min_df get the default = max idf)."""
    n = series.len()
    s = series.str.split(" ").list.unique().explode().drop_nulls()
    vc = s.filter(s != "").value_counts()
    vc = vc.filter(vc["count"] >= min_df)
    toks, cnt = vc[vc.columns[0]].to_list(), vc["count"].to_numpy()
    idf = np.log((n + 1) / (cnt + 1)) + 1
    return dict(zip(toks, idf.tolist())), float(math.log(n + 1) + 1)


def _idf_overlap(a, b, idf, default):
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return np.nan, np.nan, np.nan
    wa = sum(idf.get(t, default) for t in A)
    wb = sum(idf.get(t, default) for t in B)
    ws = sum(idf.get(t, default) for t in A & B)
    unmatched = [idf.get(t, default) for t in A ^ B]
    return ws / (wa + wb - ws), ws / min(wa, wb), max(unmatched) if unmatched else 0.0


def _set_feats(a, b):
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return np.nan, np.nan
    inter = len(A & B)
    return inter / len(A | B), float(inter == 0)


def _nan_if(cond, f, *args):
    return f(*args) if cond else np.nan


def _chunk(d):
    idf_n, dn = _IDF_N
    idf_a, da = _IDF_A
    out = np.empty((len(d["a_name_core"]), len(STR_FEATS)), np.float32)
    for r, (an, bn, ac, bc, aal, bal, aacr, bacr, al, bl, aa, ba, ap, bp, anum, bnum, alm, blm, arg, brg) in enumerate(zip(
            d["a_name_norm"], d["b_name_norm"], d["a_name_core"], d["b_name_core"], d["a_name_alias"],
            d["b_name_alias"], d["a_name_acr"], d["b_name_acr"], d["a_name_legal"], d["b_name_legal"],
            d["a_addr_core"], d["b_addr_core"], d["a_postcode"], d["b_postcode"], d["a_addr_nums"],
            d["b_addr_nums"], d["a_landmark"], d["b_landmark"], d["a_region"], d["b_region"])):
        acc, bcc = ac.replace(" ", ""), bc.replace(" ", "")
        va = [x for x in (ac, aal) if x]
        vb = [x for x in (bc, bal) if x]
        alias_best = max((fuzz.token_set_ratio(x, y) for x in va for y in vb), default=np.nan)
        n_jac, n_ovl, n_unm = _idf_overlap(ac, bc, idf_n, dn)
        a_jac, a_ovl, a_unm = _idf_overlap(aa, ba, idf_a, da)
        num_jac, num_conf = _set_feats(anum, bnum)
        leg_jac, leg_conf = _set_feats(al, bl)
        at, bt = ac.split(), bc.split()
        an_, bn_ = anum.split(), bnum.split()
        has_a = bool(aa and ba)
        out[r] = (
            fuzz.ratio(ac, bc), fuzz.partial_ratio(ac, bc), fuzz.token_sort_ratio(ac, bc),
            fuzz.token_set_ratio(ac, bc), fuzz.ratio(an, bn), JaroWinkler.normalized_similarity(acc, bcc),
            alias_best,
            float(bool(at) and bool(bt) and at[0] == bt[0]),
            float(acc == bcc and acc != ""),
            float(len(acc) >= 4 and len(bcc) >= 4 and (acc in bcc or bcc in acc)),
            float((aacr != "" and aacr == bcc) or (bacr != "" and bacr == acc)),
            n_jac, n_ovl, n_unm, len(at), len(bt), abs(len(acc) - len(bcc)) / max(len(acc), len(bcc), 1),
            leg_jac, leg_conf,
            _nan_if(has_a, fuzz.ratio, aa, ba), _nan_if(has_a, fuzz.token_set_ratio, aa, ba),
            _nan_if(has_a, fuzz.token_sort_ratio, aa, ba), _nan_if(has_a, fuzz.partial_ratio, aa, ba),
            a_jac, a_ovl, a_unm,
            (float(ap == bp) if ap and bp else np.nan),
            num_jac, num_conf,
            (float(an_[0] == bn_[0]) if an_ and bn_ else np.nan),
            len(aa.split()), len(ba.split()),
            _nan_if(bool(alm and blm), fuzz.token_set_ratio, alm, blm),
            _nan_if(bool(alm and ba), fuzz.partial_ratio, alm, ba),
            _nan_if(bool(blm and aa), fuzz.partial_ratio, blm, aa),
            (float(arg == brg) if arg and brg else np.nan),
            fuzz.token_set_ratio(ac + " " + aa, bc + " " + ba),
        )
    return out


def attach_text(pairs, norm):
    """Adds a_<field> (S1 side) and b_<field> (candidate side) text columns."""
    a = norm.select(pl.col("r").alias("i"), *[pl.col(f).alias("a_" + f) for f in FIELDS])
    b = norm.select(pl.col("r").alias("j"), *[pl.col(f).alias("b_" + f) for f in FIELDS])
    return pairs.join(a, on="i", how="left").join(b, on="j", how="left")


REL_BASE = ["blk_score", "n_tset", "a_tset", "combo_tset", "n_idf_jac", "a_idf_jac"]


def add_relative(df):
    """Rank / gap within each S1's candidate list. (Candidate-side competition features are computed
    in blocking.add_competition over ALL S1s, so they mean the same on a training sample and on test.)"""
    exprs = []
    for c in REL_BASE:
        v = pl.col(c).fill_nan(-1).fill_null(-1)
        exprs += [
            v.rank("min", descending=True).over("i").cast(pl.Float32).alias(f"{c}_rank_s1"),
            (v.max().over("i") - v).cast(pl.Float32).alias(f"{c}_gap_s1"),
        ]
    exprs += [pl.len().over("i").cast(pl.Float32).alias("n_cands_s1")]
    return df.with_columns(exprs)


def idf_tables(norm):
    return idf_table(norm["name_core"]), idf_table(norm["addr_core"])


def make_features(pairs, norm, n_jobs=C.N_JOBS, chunk=100_000, idfs=None):
    """pairs: polars (i, j, blk_*). norm: polars with r + FIELDS + source (all records of the split).
    idfs: precomputed idf_tables(norm), to avoid recomputing them for every chunk."""
    t = time.time()
    idf_n, idf_a = idfs or idf_tables(norm)
    txt = attach_text(pairs.select("i", "j"), norm)
    cols = ["a_" + f for f in FIELDS] + ["b_" + f for f in FIELDS]
    # generator, not a list: only ~n_jobs job dicts (each up to `chunk` Python strings x 20 columns)
    # exist at once. Building them all upfront before Pool.map runs blew past 16 GB on large batches.
    def jobs():
        for a in range(0, txt.height, chunk):
            yield {c: txt[c].slice(a, chunk).fill_null("").to_list() for c in cols}
    with Pool(n_jobs, initializer=_init, initargs=(idf_n, idf_a)) as pool:
        parts = list(pool.imap(_chunk, jobs(), chunksize=1))
    del txt
    sf = pl.DataFrame(np.vstack(parts) if parts else np.zeros((0, len(STR_FEATS)), np.float32), schema=STR_FEATS)
    src = norm.select(pl.col("r").alias("j"), pl.col("source").cast(pl.Float32).alias("cand_source"))
    df = pl.concat([pairs, sf], how="horizontal").join(src, on="j", how="left")
    df = add_relative(df)
    print(f"features: {df.height} pairs x {df.width} cols ({time.time() - t:.0f}s)", flush=True)
    return df


META_COLS = {"i", "j", "s1_id", "cand_id", "label", "prob", "fold", "country_key"}


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]

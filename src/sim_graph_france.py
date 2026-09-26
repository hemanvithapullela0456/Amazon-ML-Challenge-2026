"""Does the graph (consensus) re-scorer transfer to an unseen country? France-like simulation.

Train: US band pairs of the in-domain OOF (US labels only) -> LightGBM on within-S1 prob context + graph features.
Apply: Latin-script India S1s scored by the US-only matcher (work/sim_us2in/eval_without.parquet), graph features
built from THOSE probabilities. Compare F0.5 with the matcher alone, and optionally on top of a fused prob file.

    python src/sim_graph_france.py [--base work/sim_fused.parquet]
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

import config as C
import decide
from prep import load_norm
from sim_relational import ANCHOR, HI, LO, sibling

W = C.WORK_DIR


def ctx(o):
    p = pl.col("prob")
    return o.with_columns(p.rank("min", descending=True).over("i").cast(pl.Float32).alias("prob_rank"),
                          (p.max().over("i") - p).alias("prob_gap"),
                          (p > 0.5).sum().over("i").cast(pl.Float32).alias("prob_n_above"))


def latin_india(G_pl):
    from blocking import load_records, truth_pairs
    ids = G_pl.filter(pl.col("country_key") == "india")["i"].to_numpy()
    nl = (load_norm("train", columns=["business_name", "business_address"]).with_row_index("r")
          .select(pl.col("r").cast(pl.UInt32), (pl.col("business_name").str.contains(r"[ऀ-෿]")
                  | pl.col("business_address").str.contains(r"[ऀ-෿]")).alias("nl")))
    tp = truth_pairs(load_records("train"), "train", ids).join(nl.rename({"r": "j"}), on="j")
    bad = set(tp.filter(pl.col("nl"))["i"].to_list()) | set(nl.filter(pl.col("nl") & pl.col("r").is_in(ids))["r"].to_list())
    return np.array([i for i in ids if i not in bad])


def graph(o, norm):
    band = o.filter((pl.col("prob") > LO) & (pl.col("prob") < HI))
    g = sibling(band.select("i", "j"), o.filter(pl.col("prob") >= ANCHOR).select("i", "j"), norm)
    return ctx(o).join(band.select("i", "j"), on=["i", "j"], how="semi").join(g, on=["i", "j"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="optional (i, j, prob) file replacing the matcher's sim probabilities")
    a = ap.parse_args()
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    us = G_pl.filter(pl.col("country_key") == "us")["i"]
    lat = latin_india(G_pl)
    norm = load_norm("train", columns=["name_core", "addr_core", "postcode", "addr_nums", "source"]).with_row_index("r")
    norm = norm.with_columns(pl.col("r").cast(pl.UInt32), *[pl.col(c).fill_null("") for c in ("name_core", "addr_core", "postcode", "addr_nums")])

    tr_o = pl.read_parquet(os.path.join(W, "train_oof.parquet"), columns=["i", "j", "label", "prob"]).filter(pl.col("i").is_in(us.implode()))
    tr = graph(tr_o, norm).to_pandas()
    ev_o = pl.read_parquet(os.path.join(W, "sim_us2in", "eval_without.parquet"), columns=["i", "j", "label", "prob"])
    ev_o = ev_o.with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32)).filter(pl.col("i").is_in(lat))
    if a.base:
        b = pl.read_parquet(a.base).select("i", "j", pl.col("prob").alias("pb")).with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))
        ev_o = ev_o.join(b, on=["i", "j"], how="left").with_columns(pl.coalesce("pb", "prob").alias("prob")).drop("pb")
    ev = graph(ev_o, norm).to_pandas()
    feats = ["prob", "prob_rank", "prob_gap", "prob_n_above"] + [c for c in tr.columns if c.split("_")[0] in ("sib", "cov", "sup", "supmin", "miss")]
    print(f"train (US band) {len(tr)} | eval (Latin India band) {len(ev)} | features {len(feats)}", flush=True)
    m = lgb.train(C.LGB_PARAMS, lgb.Dataset(tr[feats], tr["label"].astype(int)), 600)
    ev["p_new"] = m.predict(ev[feats])
    base = ev_o.to_pandas()

    def score(d):
        return max((decide.score_pairs(d.assign(pred=decide.predict_mask(d, dict(method="threshold", t=t, one_to_one=True))),
                                       "pred", G, lat), round(t, 2)) for t in (0.7, 0.75, 0.8, 0.85, 0.9, 0.95))

    print(f"matcher alone{' (base file)' if a.base else ''}: {score(base)}")
    d = base.merge(ev[["i", "j", "p_new"]], on=["i", "j"], how="left")
    for w in (1.0, 0.7, 0.5):
        from scipy.stats import rankdata
        x = d.copy()
        mk = x.p_new.notna()
        if w < 1:   # rank blend with the original order, mapped back onto the band's probabilities
            r = w * rankdata(x.loc[mk, "p_new"]) / mk.sum() + (1 - w) * rankdata(x.loc[mk, "prob"]) / mk.sum()
            qs = np.sort(x.loc[mk, "prob"].values)
            x.loc[mk, "prob"] = qs[(rankdata(r, method="ordinal") - 1).astype(int)]
        else:
            x.loc[mk, "prob"] = x.loc[mk, "p_new"]
        print(f"graph re-scored (weight {w}): {score(x)}", flush=True)


if __name__ == "__main__":
    main()

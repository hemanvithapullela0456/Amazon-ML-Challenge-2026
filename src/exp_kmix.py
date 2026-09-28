"""Label-free decoy share per pair class from the S1 copy-count distribution (validated on train, applied to France).

A true copy sits on an S1 with k predicted copies with probability ~ k * P(k) (size-biased); a decoy lands on an S1
regardless of its copy count, so its S1 shows k = k0 + 1 with k0 ~ P(k). For each class of predicted pairs the observed
histogram of k is a mixture; the decoy share f is fitted by maximum likelihood."""
import sys

import numpy as np
import polars as pl

KMAX = 9


def refs(k_s1):
    """k_s1: predicted copies per S1 (all S1s, incl. 0) -> (size-biased true ref, decoy ref) over k = 1..KMAX"""
    h = np.bincount(np.clip(k_s1, 0, KMAX), minlength=KMAX + 1).astype(float)
    p = h / h.sum()
    ks = np.arange(KMAX + 1)
    true = ks * p
    true /= true.sum()
    dec = np.zeros(KMAX + 1)
    dec[1:] = p[:-1]  # decoy on an S1 with k0 other copies -> k0 + 1
    dec[KMAX] += p[KMAX]
    dec /= dec.sum()
    return true[1:], dec[1:]


def fit_f(obs, t, d):
    """obs: counts over k=1..KMAX -> MLE decoy share"""
    fs = np.linspace(0, 1, 1001)
    ll = [(obs * np.log((1 - f) * t + f * d + 1e-12)).sum() for f in fs]
    return fs[int(np.argmax(ll))]


def classes(d):
    return d.with_columns(
        name=pl.when(pl.col("n_exact") > 0).then(pl.lit("exact")).when(pl.col("n_tset") >= 80).then(pl.lit("close"))
        .when(pl.col("n_tset") >= 40).then(pl.lit("partial")).otherwise(pl.lit("diff")),
        addr=pl.when(pl.col("a_len_b") == 0).then(pl.lit("none"))
        .when((pl.col("a_tset") >= 90) & (pl.col("num_conflict") == 0)).then(pl.lit("same"))
        .when(pl.col("num_conflict") > 0).then(pl.lit("numconf")).when(pl.col("a_tset") >= 60).then(pl.lit("near"))
        .otherwise(pl.lit("diff")),
        pb=pl.col("prob").cut([0.95, 0.99, 0.999]).cast(pl.Utf8))


COLS = ["i", "j", "n_exact", "n_tset", "a_tset", "num_conflict", "a_len_b"]


def report(pred, all_s1, label=None):
    k = pred.group_by("i").len().rename({"len": "k"})
    ks = all_s1.join(k, on="i", how="left")["k"].fill_null(0).to_numpy()
    t, d = refs(ks)
    x = pred.join(k, on="i")
    rows = []
    for key, g in x.group_by(["name", "addr", "pb"]):
        if g.height < 300:
            continue
        obs = np.bincount(np.clip(g["k"].to_numpy(), 1, KMAX) - 1, minlength=KMAX).astype(float)
        f = fit_f(obs, t, d)
        r = dict(name=key[0], addr=key[1], pb=key[2], n=g.height, f_est=round(f, 3),
                 only=round(float((g["k"] == 1).mean()), 4))
        if label is not None:
            r["fp_true"] = round(1 - float(g["label"].mean()), 4)
        rows.append(r)
    return pl.DataFrame(rows).sort("n", descending=True)


def main():
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(200)
    W = "work/"
    if "train" in sys.argv:
        f = pl.read_parquet(W + "train_feats.parquet", columns=COLS + ["label"])
        o = pl.read_parquet(W + "train_oof.parquet", columns=["i", "j", "prob"])
        f = f.join(o, on=["i", "j"]).with_columns(best=pl.col("prob") == pl.col("prob").max().over("j"))
        pred = classes(f.filter((pl.col("prob") >= 0.725) & pl.col("best")))
        print(report(pred, o.select("i").unique(), label=True))
    else:
        p = pl.read_parquet(W + "_fr_run25.parquet").filter(pl.col("pred") == 1)
        f = pl.read_parquet(W + "test_feats_unseen.parquet", columns=COLS)
        pred = classes(p.join(f, on=["i", "j"]))
        print(report(pred, pl.read_parquet(W + "_fr_run25.parquet").select("i").unique()))


if __name__ == "__main__":
    main()

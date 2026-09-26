"""France-simulation decision layer: (1) per-S1 "no match" abstention, (2) count-aware selection.

On the France-like simulation (Latin-script India, US-trained matcher) 22.5% of the F0.5 loss is S1s with NO true
copy for which we still predict one, and 17.1% is retrieved copies left just under the threshold. Both are
decisions, not ranking. Per-S1 features are language-free summaries of the S1's scored candidate list (probability
profile, set-model agreement, generator moves of the top candidate), so the model can move to France.
Validated by 5-fold CV over simulation S1s (the gain is measured on held-out S1s).

    python src/sim_abstain.py
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

import config as C
import decide
from fuse_v3 import fuse, load_students
from sim_graph_france import latin_india

W = C.WORK_DIR
T = 0.84


def sim_scores():
    """simulation probabilities under run13's France recipe (students + set model w 0.4)"""
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    lat = latin_india(G_pl)
    ev = pd.read_parquet(os.path.join(W, "sim_us2in", "eval_without.parquet"))
    ev = ev[ev.i.isin(lat)][["i", "j", "label", "prob"]].copy()
    band = pl.read_parquet(os.path.join(W, "da_proxy", "tgt_pairs.parquet")).select("ka", "kb", "i", "j")
    bs = load_students("sim", "band")
    for s in ("mdeb", "mdeb7"):
        band = band.join(bs[s], on=["ka", "kb"], how="inner")
    b = band.to_pandas().merge(ev[["i", "j", "prob"]], on=["i", "j"])
    b = b.merge(pd.read_parquet(os.path.join(W, "set_test_simus.parquet")).rename(columns={"ce": "set"}), on=["i", "j"], how="left")
    m = b.set.notna().values
    R = lambda x: rankdata(x) / len(x)
    p = np.asarray(fuse(b, ["mdeb", "mdeb7"], 0.1, {"mdeb": 0.45, "mdeb7": 0.45}))
    s = 0.6 * R(p[m]) + 0.4 * R(b.set.values[m])
    p[m] = np.sort(p[m])[(rankdata(s, method="ordinal") - 1).astype(int)]
    mp = dict(zip(zip(b.i, b.j), p))
    ev["prob"] = [mp.get(k, q) for k, q in zip(zip(ev.i, ev.j), ev.prob)]
    setr = dict(zip(zip(b.i, b.j), np.where(m, R(np.nan_to_num(b.set.values)), np.nan)))
    ev["set_r"] = [setr.get(k, np.nan) for k in zip(ev.i, ev.j)]
    return ev, G, lat


def s1_features(ev):
    import moves
    d = ev.sort_values(["i", "prob"], ascending=[True, False])
    d["rk"] = d.groupby("i").cumcount()
    top = d[d.rk == 0][["i", "j", "prob", "set_r"]].rename(columns={"prob": "p1", "set_r": "set1"})
    sec = d[d.rk == 1][["i", "prob"]].rename(columns={"prob": "p2"})
    agg = d.groupby("i").agg(n_t=("prob", lambda x: (x >= T).sum()), n_50=("prob", lambda x: (x >= 0.5).sum()),
                             n_c=("prob", "size"), p_sum=("prob", "sum"))
    f = top.merge(sec, on="i", how="left").merge(agg, on="i")
    f["gap12"] = f.p1 - f.p2.fillna(0)
    mv = moves.build("train", pl.from_pandas(top[["i", "j", "p1"]].rename(columns={"p1": "prob"})).with_columns(
        pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))).to_pandas()
    keep = ["mv_name_equal", "mv_name_jacc", "mv_name_disjoint", "mv_addr_dropped", "mv_s1_addr_empty", "mv_hn_equal",
            "mv_hn_small_shift", "mv_nums_jacc", "mv_city_tail", "mv_decoy_shift", "al_name_freq"]
    f = f.merge(mv[["i", "j"] + keep], on=["i", "j"], how="left")
    return f


def main():
    ev, G, lat = sim_scores()
    cfg = dict(method="threshold", t=T, one_to_one=True)
    base_pred = decide.predict_mask(ev, cfg)
    base = decide.score_pairs(ev.assign(pred=base_pred), "pred", G, lat)
    print(f"simulation F0.5, run13 recipe at t={T}: {base:.4f}")
    # (2) count-aware selection: expected-F0.5 selector with a calibration shift
    for sh in (-1.0, -0.5, 0.0, 0.5):
        pm = decide.predict_mask(ev, dict(method="expected", shift=sh, one_to_one=True))
        print(f"  expected-F0.5 selection, shift {sh}: {decide.score_pairs(ev.assign(pred=pm), 'pred', G, lat):.4f}")
    # (1) per-S1 abstention, 5-fold CV over S1s
    f = s1_features(ev)
    f["g0"] = (G.reindex(f.i).values == 0).astype(int)
    feats = [c for c in f.columns if c not in ("i", "j", "g0")]
    fold = pd.Series(np.random.default_rng(0).permutation(len(f)) % 5, index=f.index)
    f["p_none"] = 0.0
    for k in range(5):
        tr, va = fold != k, fold == k
        m = lgb.train(dict(C.LGB_PARAMS, num_leaves=31, min_data_in_leaf=50), lgb.Dataset(f.loc[tr, feats], f.loc[tr, "g0"]), 400)
        f.loc[va, "p_none"] = m.predict(f.loc[va, feats])
    from sklearn.metrics import roc_auc_score
    has_pred = f.n_t > 0
    print(f"S1s with a prediction: {has_pred.sum()} | of them truly without copies: {f.g0[has_pred].sum()} | "
          f"'no match' AUC among them {roc_auc_score(f.g0[has_pred], f.p_none[has_pred]):.4f}")
    pn = dict(zip(f.i, f.p_none))
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        drop = ev.i.map(pn).fillna(0).values >= tau
        pm = base_pred & ~drop
        print(f"  abstain when P(no copy) >= {tau}: F0.5 {decide.score_pairs(ev.assign(pred=pm), 'pred', G, lat):.4f} "
              f"(S1s abstained {int(sum(v >= tau for v in pn.values()))})", flush=True)
    f[["i", "p_none"]].to_parquet(os.path.join(W, "sim_pnone.parquet"))


if __name__ == "__main__":
    main()

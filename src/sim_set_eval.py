"""Score set-model variants on the France-like simulation (Latin-script India, true labels).

For each score file (i, j, ce) on the simulation lists: band AUC, and F0.5 when rank-fused into run10's France
recipe (LightGBM 0.1 + mDeBERTa students) at several weights and thresholds. Files can be averaged with '+'.

    python src/sim_set_eval.py set_test_simus.parquet set_test_simscr.parquet set_test_simus.parquet+set_test_simseed1.parquet
"""
import os
import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import config as C
import decide
from fuse_v3 import fuse, load_students
from sim_graph_france import latin_india

W = C.WORK_DIR
R = lambda x: rankdata(x) / len(x)


def main(files):
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    lat = latin_india(G_pl)
    ev = pd.read_parquet(os.path.join(W, "sim_us2in", "eval_without.parquet"))
    ev = ev[ev.i.isin(lat)]
    band = pl.read_parquet(os.path.join(W, "da_proxy", "tgt_pairs.parquet")).select("ka", "kb", "i", "j")
    bs = load_students("sim", "band")
    for s in ("mdeb", "mdeb7"):
        band = band.join(bs[s], on=["ka", "kb"], how="inner")
    b = band.to_pandas().merge(ev[["i", "j", "prob", "label"]], on=["i", "j"])
    p10 = np.asarray(fuse(b, ["mdeb", "mdeb7"], 0.1, {"mdeb": 0.45, "mdeb7": 0.45}))

    def f05(newp):
        e = ev.copy()
        mp = dict(zip(zip(b.i, b.j), newp))
        e["prob"] = [mp.get(k, p) for k, p in zip(zip(e.i, e.j), e.prob)]
        return max((round(decide.score_pairs(e.assign(pred=decide.predict_mask(e, dict(method="threshold", t=t, one_to_one=True))),
                                             "pred", G, lat), 4), t) for t in (0.8, 0.82, 0.84, 0.86, 0.88, 0.9))

    for spec in files:
        sc = None
        for f in spec.split("+"):
            d = pd.read_parquet(os.path.join(W, f))
            d["ce"] = rankdata(d["ce"]) / len(d)
            sc = d if sc is None else sc.merge(d, on=["i", "j"], suffixes=("", "_2")).assign(ce=lambda x: (x.ce + x.ce_2) / 2).drop(columns="ce_2")
        x = b.merge(sc.rename(columns={"ce": "set"}), on=["i", "j"], how="left")
        m = x.set.notna().values
        res = []
        for w in (0.3, 0.4, 0.5, 0.6):
            p = p10.copy()
            s = (1 - w) * R(p10[m]) + w * R(x.set.values[m])
            p[m] = np.sort(p10[m])[(rankdata(s, method="ordinal") - 1).astype(int)]
            res.append((f05(p), w))
        (best, t), w = max(res)
        print(f"{spec}: band AUC {roc_auc_score(x.label[m], x.set[m]):.4f} | fused F0.5 by weight "
              + " ".join(f"{ww}:{r[0]:.4f}" for r, ww in res) + f" | best {best:.4f} (w {w}, t {t})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])

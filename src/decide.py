"""Turn pair probabilities into per-S1 match lists, optimising macro F0.5.

Works on a pandas DataFrame with columns i (S1 record index), j (candidate index), prob [, label].
Strategies (tuned on out-of-fold predictions, the best is kept):
  threshold : keep candidates with prob >= t
  expected  : per S1, pick the top-k subset (k may be 0) maximising expected F0.5 under
              independent Bernoulli(prob) labels; `shift` adds a logit offset to calibrate.
Optionally one-to-one first: a candidate is kept only for the S1 that scores it highest.
"""
import numpy as np
import pandas as pd

import config as C


def one_to_one(df):
    best = df.groupby("j")["prob"].transform("max")
    return df[df["prob"] >= best]


def score_pairs(df, pred_col, G, s1_ids, beta2=C.BETA2):
    """Vectorised macro F0.5. G: pd.Series i -> number of true matches (incl. ones blocking missed)."""
    g = (df.assign(tp=df["label"].astype(bool) & df[pred_col].astype(bool))
         .groupby("i").agg(tp=("tp", "sum"), k=(pred_col, "sum")).reindex(s1_ids, fill_value=0))
    Gv = G.reindex(s1_ids, fill_value=0).values
    tp, k = g["tp"].values, g["k"].values
    f = np.where(Gv == 0, (k == 0).astype(float),
                 np.where(k == 0, 0.0, (1 + beta2) * tp / np.maximum(k + beta2 * Gv, 1e-9)))
    return float(f.mean())


# ------------------------------------------------------------ expected-F selection
def _pmf(ps):
    pmf = np.array([1.0])
    for p in ps:
        pmf = np.append(pmf * (1 - p), 0) + np.append(0, pmf * p)
    return pmf


def _best_k(p, beta2=C.BETA2, kmax=12):
    """p sorted descending. Returns k maximising E[F_beta] when predicting the top-k."""
    best_k, best_e = 0, float(np.prod(1 - p))  # empty prediction scores 1 only if there is no true match
    for k in range(1, min(len(p), kmax) + 1):
        tp, rest = _pmf(p[:k]), _pmf(p[k:])
        a = np.arange(len(tp))[:, None]
        b = np.arange(len(rest))[None, :]
        e = float((tp[:, None] * rest[None, :] * (1 + beta2) * a / (k + beta2 * (a + b))).sum())
        if e > best_e:
            best_k, best_e = k, e
    return best_k


def expected_select(df, shift=0.0, pmin=0.01):
    p = df["prob"].values
    if shift:
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        p = 1 / (1 + np.exp(-(lg + shift)))
    order = np.lexsort((-p, df["i"].values))
    ii, pp = df["i"].values[order], p[order]
    keep = np.zeros(len(df), bool)
    starts = np.flatnonzero(np.r_[True, ii[1:] != ii[:-1]])
    ends = np.r_[starts[1:], len(ii)]
    for s, e in zip(starts, ends):
        grp = pp[s:e]
        grp = grp[grp >= pmin]
        k = _best_k(grp) if len(grp) else 0
        if k:
            keep[order[s:s + k]] = True
    return keep


# ------------------------------------------------------------ tuning / applying
def predict_mask(df, cfg):
    d = one_to_one(df) if cfg.get("one_to_one") else df
    m = d["prob"].values >= cfg["t"] if cfg["method"] == "threshold" else expected_select(d, cfg["shift"])
    mask = pd.Series(False, index=df.index)
    mask.loc[d.index[m]] = True
    return mask.values


def tune(df, G, s1_ids, try_expected=True):
    """df: OOF pairs with i, j, label, prob. Returns the best config dict."""
    results = []
    for o2o in ([True, False] if C.ONE_TO_ONE else [False]):
        d = one_to_one(df) if o2o else df
        for t in np.arange(0.05, 0.96, 0.025):
            d2 = d.assign(pred=d["prob"].values >= t)
            results.append((score_pairs(d2, "pred", G, s1_ids), dict(method="threshold", t=round(float(t), 3), one_to_one=o2o)))
        if try_expected:
            for shift in (-0.5, 0.0, 0.5):
                d2 = d.assign(pred=expected_select(d, shift))
                results.append((score_pairs(d2, "pred", G, s1_ids), dict(method="expected", shift=shift, one_to_one=o2o)))
    results.sort(key=lambda r: -r[0])
    for s, cfg in results[:8]:
        print(f"  {s:.5f}  {cfg}")
    best_score, best = results[0]
    best["cv_f05"] = best_score
    return best

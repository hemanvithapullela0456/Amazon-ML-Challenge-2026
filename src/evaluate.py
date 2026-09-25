"""Macro F0.5 exactly as defined in the problem statement, plus blocking diagnostics."""
import numpy as np

from config import BETA2


def entity_f(pred, true, beta2=BETA2):
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    # F_beta = (1+b2) TP / ((1+b2) TP + b2 FN + FP) = (1+b2) TP / (|pred| + b2 |true|)
    return (1 + beta2) * tp / (len(pred) + beta2 * len(true))


def macro_f(pred, gt, s1_ids=None):
    s1_ids = list(gt) if s1_ids is None else s1_ids
    return float(np.mean([entity_f(pred.get(s, ()), gt.get(s, ())) for s in s1_ids]))


def blocking_report(cands, gt, s1_ids=None, name="blocking"):
    s1_ids = list(gt) if s1_ids is None else s1_ids
    n_true = sum(len(gt[s]) for s in s1_ids)
    hit = sum(len(gt[s] & set(cands.get(s, ()))) for s in s1_ids)
    full = np.mean([gt[s] <= set(cands.get(s, ())) for s in s1_ids])
    # best achievable macro F0.5 if the matcher were perfect on these candidates
    ceiling = macro_f({s: gt[s] & set(cands.get(s, ())) for s in s1_ids}, gt, s1_ids)
    n_c = np.array([len(cands.get(s, ())) for s in s1_ids])
    print(f"[{name}] pair recall {hit / max(n_true, 1):.4f} | S1 with all matches covered {full:.4f} | "
          f"F0.5 ceiling {ceiling:.4f} | cands/S1 mean {n_c.mean():.1f} median {np.median(n_c):.0f} max {n_c.max()} | "
          f"total pairs {n_c.sum()}")
    return dict(pair_recall=hit / max(n_true, 1), full_cover=full, ceiling=ceiling, mean_cands=n_c.mean())


def _selftest():
    assert abs(entity_f(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"]) - 0.714) < 1e-3
    assert entity_f([], []) == 1.0
    assert entity_f(["S2-1"], []) == 0.0
    assert entity_f([], ["S2-1"]) == 0.0
    assert entity_f(["S2-1"], ["S2-1"]) == 1.0
    print("evaluate selftest OK")


if __name__ == "__main__":
    _selftest()

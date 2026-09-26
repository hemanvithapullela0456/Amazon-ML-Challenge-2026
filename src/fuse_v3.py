"""Unseen-country fusion v3: student ensemble on the uncertain band + student veto of confident teacher matches.

--mode sim    : tune on the France-like simulation (Latin-only India, TRUE labels) and print the best recipe
--mode france : apply a recipe to France and write work/test_scored_unseen_v3.parquet

Students: any of qwen / xlmr / mdeb whose files exist (work/sim3_<s>_band.parquet, work/sim3_<s>_extra.parquet,
and fr3_<s>_... for France). Band fusion = weighted rank average of LightGBM and students, mapped back onto
LightGBM's own probabilities (only the order changes). Veto = among the teacher's confident one-to-one matches
(prob > 0.99), the fraction q with the lowest student-ensemble score is dropped.

    python src/fuse_v3.py --mode sim
    python src/fuse_v3.py --mode france --w_lgbm 0.2 --veto 0.02
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata

import config as C
import decide

W = C.WORK_DIR
STUDENTS = ["qwen", "xlmr", "mdeb"]
R = lambda x: rankdata(x) / len(x)


def load_students(prefix, kind):
    """prefix 'sim' or 'fr': finds work/<prefix>3_<name>_<kind>.parquet and <prefix>4_... (name = student)"""
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(W, f"{prefix}[34]_*_{kind}.parquet"))):
        name = os.path.basename(p)[len(prefix) + 2:-len(f"_{kind}.parquet")]
        out[name] = pl.read_parquet(p).rename({"ce": name})
    return out


def consistency(prefix, bundle_dir):
    """transitive consistency: for each checked (i, j), the max / mean student score of j against the S1's
    other confident copies k (implied pairs). Returns (i, j, cons_max, cons_mean) or None."""
    import glob
    fs = sorted(glob.glob(os.path.join(W, f"{prefix}[34]_*_implied.parquet")))
    ip = os.path.join(W, bundle_dir, "implied_pairs.parquet")
    if not fs or not os.path.exists(ip):
        return None
    imp = pl.read_parquet(ip).select("i", "j", "ka", "kb")
    sc = None
    for f in fs:   # average several students' implied scores (rank-normalised)
        d = pl.read_parquet(f)
        d = d.with_columns(pl.Series("ce", rankdata(d["ce"].to_numpy()) / d.height))
        sc = d if sc is None else sc.join(d, on=["ka", "kb"], suffix="_x").with_columns(
            ((pl.col("ce") + pl.col("ce_x")) / 2).alias("ce")).drop("ce_x")
    x = imp.join(sc, on=["ka", "kb"], how="inner")
    return x.group_by("i", "j").agg(pl.col("ce").max().alias("cons_max"), pl.col("ce").mean().alias("cons_mean"))


def fuse(pairs, studs, w_lgbm, w_st, w_cons=0.0):
    """pairs: DataFrame with 'prob' + student columns (+ cons_max). Returns new prob (same distribution, new order)."""
    score = w_lgbm * R(pairs["prob"].values)
    for s, w in w_st.items():
        score = score + w * R(pairs[s].values)
    if w_cons > 0:
        c = pairs["cons_max"].fillna(pairs["cons_max"].median()).values
        score = score + w_cons * R(c)
    qs = np.sort(pairs["prob"].values)
    return qs[np.clip((rankdata(score) - 1).astype(int), 0, len(qs) - 1)]


def veto_mask(ex, studs, q, w_cons=0.0):
    if q <= 0 or ex.empty:
        return np.zeros(len(ex), bool)
    s = sum(R(ex[c].values) for c in studs) / len(studs)
    if w_cons > 0 and "cons_max" in ex:
        s = (1 - w_cons) * s + w_cons * R(ex["cons_max"].fillna(ex["cons_max"].median()).values)
    return s <= np.quantile(s, q)


def sim():
    from blocking import load_records, truth_pairs
    from prep import load_norm
    G_pl = pl.read_parquet(os.path.join(W, "train_G.parquet"))
    G = pd.Series(G_pl["len"].to_numpy(), index=G_pl["i"].to_numpy())
    ids = G_pl.filter(pl.col("country_key") == "india")["i"].to_numpy()
    nl = (load_norm("train", columns=["business_name", "business_address"]).with_row_index("r")
          .select(pl.col("r").cast(pl.UInt32), (pl.col("business_name").str.contains(r"[ऀ-෿]")
                  | pl.col("business_address").str.contains(r"[ऀ-෿]")).alias("nl")))
    tp = truth_pairs(load_records("train"), "train", ids).join(nl.rename({"r": "j"}), on="j")
    bad = set(tp.filter(pl.col("nl"))["i"].to_list()) | set(nl.filter(pl.col("nl") & pl.col("r").is_in(ids))["r"].to_list())
    lat = np.array([i for i in ids if i not in bad])
    ev = pd.read_parquet(os.path.join(W, "sim_us2in", "eval_without.parquet"))
    ev = ev[ev.i.isin(lat)].copy()
    band = pl.read_parquet(os.path.join(W, "da_proxy", "tgt_pairs.parquet")).select("ka", "kb", "i", "j")
    extra = pl.read_parquet(os.path.join(W, "da_proxy", "extra_pairs.parquet")).select("ka", "kb", "i", "j")
    bs, es = load_students("sim", "band"), load_students("sim", "extra")
    names = sorted(bs)
    print(f"students with band scores: {names} | with veto scores: {sorted(es)}")
    for s in names:
        band = band.join(bs[s], on=["ka", "kb"], how="inner")
    cons = consistency("sim", "da_proxy")
    b = band.to_pandas().merge(ev[["i", "j", "prob"]], on=["i", "j"])
    vs = sorted(es)
    ex = extra
    for s in vs:
        ex = ex.join(es[s], on=["ka", "kb"], how="inner")
    ex = ex.to_pandas().merge(ev[["i", "j"]], on=["i", "j"])
    if cons is not None:
        cp = cons.to_pandas()
        b, ex = b.merge(cp, on=["i", "j"], how="left"), ex.merge(cp, on=["i", "j"], how="left")
        print(f"consistency signal: band coverage {b.cons_max.notna().mean():.3f} | veto coverage {ex.cons_max.notna().mean():.3f}")

    def score(w_lgbm, w_st, q, w_cons=0.0, v_cons=0.0, vstud=None):
        e = ev.copy()
        m = dict(zip(zip(b.i, b.j), fuse(b, names, w_lgbm, w_st, w_cons)))
        e["prob"] = [m.get(k, p) for k, p in zip(zip(e.i, e.j), e.prob)]
        if vs and q > 0:
            vm = veto_mask(ex, vstud or vs, q, v_cons)
            drop = set(zip(ex.i[vm], ex.j[vm]))
            e.loc[[k in drop for k in zip(e.i, e.j)], "prob"] = 0.0
        res = {t: decide.score_pairs(e.assign(pred=decide.predict_mask(e, dict(method="threshold", t=t, one_to_one=True))),
                                     "pred", G, lat) for t in (0.8, 0.85, 0.9, 0.95)}
        t = max(res, key=res.get)
        return res[t], t

    print(f"LightGBM alone: {score(1.0, {}, 0)[0]:.4f}")
    combos = [c for k in (1, 2) for c in itertools.combinations(names, k)]
    if len(names) > 2:
        combos.append(tuple(n for n in names if n.startswith("mdeb")))
    results = []
    for w_lgbm in (0.1, 0.2):
        for combo in dict.fromkeys(combos):
            if not combo:
                continue
            f, t = score(w_lgbm, {s: (1 - w_lgbm) / len(combo) for s in combo}, 0)
            results.append((f, t, w_lgbm, combo))
            print(f"  lgbm {w_lgbm} + {'+'.join(combo)}: F0.5 {f:.4f} @t={t}", flush=True)
    f, t, w_lgbm, combo = max(results)
    print(f"best band fusion: F0.5 {f:.4f} (lgbm {w_lgbm}, students {combo})")
    best_c = 0.0
    if cons is not None:
        for wc in (0.1, 0.2, 0.3):
            rest = 1 - w_lgbm - wc
            fc, tc = score(w_lgbm, {s: rest / len(combo) for s in combo}, 0, w_cons=wc)
            print(f"  + consistency weight {wc}: F0.5 {fc:.4f} @t={tc}", flush=True)
            if fc > f:
                f, best_c = fc, wc
    w_st = {s: (1 - w_lgbm - best_c) / len(combo) for s in combo}
    vstud = [s for s in combo if s in es] or vs
    if vs:
        for vc in ((0.0, 0.3) if cons is not None else (0.0,)):
            for q in (0.01, 0.02, 0.03):
                fq, tq = score(w_lgbm, w_st, q, w_cons=best_c, v_cons=vc, vstud=vstud)
                print(f"  + veto {q:.0%} (students {vstud}, consistency in veto {vc}): F0.5 {fq:.4f} @t={tq}", flush=True)
    print(f"RECIPE: --students {','.join(combo)} --w_lgbm {w_lgbm} --w_cons {best_c} (pick veto / veto_cons from the lines above)")


def france(a):
    lg = pl.read_parquet(os.path.join(W, a.lgbm))
    band = pl.read_parquet(os.path.join(W, "da_france", "tgt_pairs.parquet")).select("ka", "kb", "i", "j")
    extra = pl.read_parquet(os.path.join(W, "da_france", "extra_pairs.parquet")).select("ka", "kb", "i", "j")
    names = [s for s in a.students.split(",") if s]
    bs, es = load_students("fr", "band"), load_students("fr", "extra")
    for s in names:
        band = band.join(bs[s], on=["ka", "kb"], how="inner")
    b = band.join(lg, on=["i", "j"], how="inner").to_pandas()
    cons = consistency("fr", "da_france") if (a.w_cons > 0 or a.veto_cons > 0) else None
    if cons is not None:
        b = b.merge(cons.to_pandas(), on=["i", "j"], how="left")
    w_st = {s: (1 - a.w_lgbm - a.w_cons) / len(names) for s in names}
    b["p_new"] = fuse(b, names, a.w_lgbm, w_st, a.w_cons)
    out = lg.join(pl.from_pandas(b[["i", "j", "p_new"]]).with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32)),
                  on=["i", "j"], how="left").with_columns(pl.coalesce("p_new", "prob").alias("prob")).drop("p_new")
    n_drop = 0
    vnames = [s for s in names if s in es]
    if a.veto > 0 and vnames:
        ex = extra
        for s in vnames:
            ex = ex.join(es[s], on=["ka", "kb"], how="inner")
        ex = ex.to_pandas()
        if cons is not None:
            ex = ex.merge(cons.to_pandas(), on=["i", "j"], how="left")
        m = veto_mask(ex, vnames, a.veto, a.veto_cons)
        drop = pl.DataFrame({"i": ex.i[m].astype("uint32").values, "j": ex.j[m].astype("uint32").values, "vetoed": True})
        out = out.join(drop, on=["i", "j"], how="left").with_columns(
            pl.when(pl.col("vetoed")).then(0.0).otherwise(pl.col("prob")).cast(pl.Float32).alias("prob")).drop("vetoed")
        n_drop = int(m.sum())
    out.write_parquet(os.path.join(W, a.out))
    print(f"France: band fused {len(b)} pairs ({names}, lgbm {a.w_lgbm}, consistency {a.w_cons}) | vetoed {n_drop} "
          f"confident matches -> work/{a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sim", "france"], required=True)
    ap.add_argument("--lgbm", default="test_scored_unseen_roles.parquet")
    ap.add_argument("--students", default="qwen,xlmr,mdeb")
    ap.add_argument("--w_lgbm", type=float, default=0.2)
    ap.add_argument("--veto", type=float, default=0.0)
    ap.add_argument("--w_cons", type=float, default=0.0)
    ap.add_argument("--veto_cons", type=float, default=0.0)
    ap.add_argument("--out", default="test_scored_unseen_v3.parquet")
    a = ap.parse_args()
    sim() if a.mode == "sim" else france(a)

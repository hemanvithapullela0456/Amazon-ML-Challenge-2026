"""Owner choice for name-only copies among same-name S1s (label-trained, applies to every country).

A name-only copy (empty address) of an S1 whose name is shared by m S1s of the country has P(true) ~ 1/m under the
pair model, because every same-name S1 gets the same pair features. But the generator keeps some of the owner's
surface form: the legal suffix (Corp vs LLC vs Pvt Ltd), casing and punctuation of the raw name. Among same-name S1s
the copy's owner is far more often the one with the same legal form (train: top-1 0.50 vs 1/m 0.32; where the
model is >= 0.8 sure, 94% correct).

For every (S1 i, name-only record j) pair, the group is the S1s of i's country with i's exact name_core. A LightGBM
scores each group member, trained on train truth over groups that contain the owner; normalising over the group gives
own_pn = P(i is the owner | the owner is in i's name group). These features go into the US/India stack
(stack.py --owner) and adjust France's unseen-country scores.

    python src/owner.py --split train      # fit the owner model (5 folds by record), OOF features for train_oof pairs
    python src/owner.py --split test       # features for test pairs (stage-1 and unseen-country score files)
"""
import argparse
import os

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C
from prep import load_norm

W = C.WORK_DIR
MAX_M = 1000
PMIN = 0.005
BOOL = ["bn_eq", "bn_ieq", "bn_seq", "leg_eq", "norm_eq", "case_eq", "script_eq"]
FEATS = BOOL + [f + "_g" for f in BOOL] + ["m", "exact", "rleg_e", "cleg_e", "rcase", "ccase", "rscript", "caddr_e",
                                           "caddr_len", "clen", "rlen", "src"]
SCRIPTS = [("ऀ", "ॿ"), ("ঀ", "৿"), ("਀", "੿"), ("઀", "૿"),
           ("଀", "୿"), ("஀", "௿"), ("ఀ", "౿"), ("ಀ", "೿"), ("ഀ", "ൿ")]


def _case(c):
    s = pl.col(c)
    return (pl.when(s == s.str.to_uppercase()).then(0).when(s == s.str.to_lowercase()).then(1)
            .when(s == s.str.to_titlecase()).then(2).otherwise(3))


def _script(c):
    e = pl.lit(0)
    for k, (a, b) in enumerate(SCRIPTS):
        e = pl.when(pl.col(c).str.contains(f"[{a}-{b}]")).then(k + 1).otherwise(e)
    return e


def tables(split):
    n = load_norm(split, columns=["business_name", "business_address", "name_core", "name_legal", "name_norm",
                                  "source", "country_key"]).with_row_index("r")
    n = n.with_columns(noaddr=pl.col("business_address").str.strip_chars() == "",
                       bnl=pl.col("business_name").str.to_lowercase(),
                       bns=pl.col("business_name").str.to_lowercase().str.replace_all(r"[^\w]", ""),
                       case=_case("business_name"), script=_script("business_name"),
                       alen=pl.col("business_address").str.len_chars(), nlen=pl.col("business_name").str.len_chars())
    s1 = n.filter(pl.col("source") == 1).select(
        pl.col("r").alias("c"), "country_key", pl.col("name_core").alias("g"), pl.col("business_name").alias("cbn"),
        pl.col("bnl").alias("cbnl"), pl.col("bns").alias("cbns"), pl.col("name_legal").alias("cleg"),
        pl.col("name_norm").alias("cnorm"), pl.col("case").alias("ccase"), pl.col("script").alias("cscript"),
        pl.col("alen").alias("caddr_len"), pl.col("nlen").alias("clen"))
    m = s1.group_by("country_key", "g").len().rename({"len": "m"})
    rec = n.filter((pl.col("source") != 1) & pl.col("noaddr")).select(
        pl.col("r").alias("j"), pl.col("business_name").alias("rbn"), pl.col("bnl").alias("rbnl"),
        pl.col("bns").alias("rbns"), pl.col("name_legal").alias("rleg"), pl.col("name_norm").alias("rnorm"),
        pl.col("name_core").alias("rcore"), pl.col("case").alias("rcase"), pl.col("script").alias("rscript"),
        pl.col("nlen").alias("rlen"), pl.col("source").cast(pl.Int32).alias("src"))
    return n, s1, m, rec


NONLATIN = "[" + "".join(f"{a}-{b}" for a, b in SCRIPTS) + "]"
SIB = ["sib_n", "sib_same", "sib_other", "sib_nl", "rnl", "sib_leg_any", "sib_name_eq", "sib_same_rel", "sib_n_rel",
       "sib_nl_rel", "sib_zero_g", "sib_leg_any_g"]
SIBT = None  # per-S1 sibling table (sibling_table); None = no sibling features


def sibling_table(n, own_pairs):
    """own_pairs: (o, j) address-bearing copies of S1 o (train: truth; test: predicted) -> per-S1 sibling stats.
    Entity-level cues for the owner of a name-only copy: the member's copy count in the record's source (copies per
    source are budgeted), the script its copies use (e.g. Devanagari for Hindi-belt entities), legal forms / raw names
    its copies carry."""
    s = own_pairs.join(n.select(pl.col("r").alias("j"), "business_name", "source", "name_legal"), on="j")
    s = s.with_columns(nl=pl.col("business_name").str.contains(NONLATIN))
    return s.group_by("o").agg(
        pl.len().alias("sib_n"), (pl.col("source") == 2).sum().alias("sib_n2"), (pl.col("source") == 3).sum().alias("sib_n3"),
        pl.col("nl").mean().alias("sib_nl"), pl.col("business_name").str.to_lowercase().alias("sib_names"),
        pl.col("name_legal").alias("sib_legs")).rename({"o": "c"})


def add_sib(x):
    x = x.join(SIBT, on="c", how="left").with_columns(
        pl.col("sib_n", "sib_n2", "sib_n3").fill_null(0), pl.col("sib_nl").fill_null(-1))
    x = x.with_columns(
        sib_same=pl.when(pl.col("src") == 2).then("sib_n2").otherwise("sib_n3"),
        sib_other=pl.when(pl.col("src") == 2).then("sib_n3").otherwise("sib_n2"),
        rnl=pl.col("rbn").str.contains(NONLATIN).cast(pl.Int32),
        sib_leg_any=pl.col("sib_legs").list.contains(pl.col("rleg")).fill_null(False).cast(pl.Int32),
        sib_name_eq=pl.col("sib_names").list.contains(pl.col("rbnl")).fill_null(False).cast(pl.Int32))
    key = ["j", "country_key", "g"]
    return x.with_columns(
        (pl.col("sib_same") - pl.col("sib_same").mean().over(key)).alias("sib_same_rel"),
        (pl.col("sib_n") - pl.col("sib_n").mean().over(key)).alias("sib_n_rel"),
        (pl.col("sib_nl") - pl.col("sib_nl").mean().over(key)).alias("sib_nl_rel"),
        (pl.col("sib_n") == 0).cast(pl.Int32).sum().over(key).alias("sib_zero_g"),
        pl.col("sib_leg_any").sum().over(key).alias("sib_leg_any_g")).drop("sib_names", "sib_legs", "sib_n2", "sib_n3")


def feat_names():
    return FEATS + (SIB if SIBT is not None else [])


def expand(groups, s1, m, rec):
    """groups: (j, country_key, g) -> one row per group member c, with features"""
    x = (groups.join(m, on=["country_key", "g"]).filter(pl.col("m") <= MAX_M)
         .join(s1, on=["country_key", "g"]).join(rec, on="j"))
    x = x.with_columns(bn_eq=pl.col("cbn") == pl.col("rbn"), bn_ieq=pl.col("cbnl") == pl.col("rbnl"),
                       bn_seq=pl.col("cbns") == pl.col("rbns"), leg_eq=pl.col("cleg") == pl.col("rleg"),
                       norm_eq=pl.col("cnorm") == pl.col("rnorm"), case_eq=pl.col("ccase") == pl.col("rcase"),
                       script_eq=pl.col("cscript") == pl.col("rscript"), exact=pl.col("g") == pl.col("rcore"),
                       rleg_e=pl.col("rleg") == "", cleg_e=pl.col("cleg") == "", caddr_e=pl.col("caddr_len") == 0)
    key = ["j", "country_key", "g"]
    x = x.with_columns([pl.col(f).cast(pl.Int32).sum().over(key).alias(f + "_g") for f in BOOL])
    return add_sib(x) if SIBT is not None else x


def fmat(x):
    return x.select([pl.col(f).cast(pl.Float32) for f in feat_names()]).to_numpy()


def normalise(x, p):
    key = ["j", "country_key", "g"]
    x = x.with_columns(p=pl.Series(p, dtype=pl.Float32))
    # own_pn: P(member is the owner), unconditional (trained with ~20% of S1s hidden, as in test, so "the owner is
    # not in S1 at all" is a possible outcome); own_share: normalised within the group
    x = x.with_columns(own_pn=pl.col("p"), own_share=pl.col("p") / pl.col("p").sum().over(key))
    mx = pl.col("own_pn").max().over(key)
    second = pl.col("own_pn").filter(pl.col("own_pn") < pl.col("own_pn").max()).max().over(key).fill_null(0)
    top_n = (pl.col("own_pn") == pl.col("own_pn").max()).sum().over(key)
    x = x.with_columns(own_other=pl.when((pl.col("own_pn") < mx) | (top_n > 1)).then(mx).otherwise(second),
                       own_rank=pl.col("own_pn").rank("ordinal", descending=True).over(key))
    return x


def pair_features(pairs, s1, m, rec, predict):
    """pairs: (i, j) with j name-only -> (i, j, own_*) ; predict(x) -> raw member scores"""
    ig = s1.select(pl.col("c").alias("i"), "country_key", "g")
    p = pairs.join(ig, on="i")
    groups = p.select("j", "country_key", "g").unique()
    x = expand(groups, s1, m, rec)
    x = normalise(x, predict(x))
    out = p.join(x.select(pl.col("c").alias("i"), "j", "own_pn", "own_share", "own_other", "own_rank", "m", "exact", "leg_eq",
                          "leg_eq_g", "bn_ieq", "bn_ieq_g"), on=["i", "j"], how="left")
    return out.select("i", "j", pl.col("own_pn").cast(pl.Float32), pl.col("own_share").cast(pl.Float32), pl.col("own_other").cast(pl.Float32),
                      pl.col("own_rank").cast(pl.Float32), pl.col("m").cast(pl.Float32).alias("own_m"),
                      pl.col("exact").cast(pl.Float32).alias("own_exact"),
                      pl.col("leg_eq").cast(pl.Float32).alias("own_leg_eq"),
                      pl.col("leg_eq_g").cast(pl.Float32).alias("own_leg_eq_g"),
                      pl.col("bn_ieq").cast(pl.Float32).alias("own_bn_ieq"),
                      pl.col("bn_ieq_g").cast(pl.Float32).alias("own_bn_ieq_g"))


def superset_feats(pairs, n, max_df=3000):
    """record-side competition for name-only copies with noisy / truncated names: how many S1s of the country contain
    the record's two rarest known name tokens (sup_cnt), and whether S1 i is one of them (sup_in)"""
    s1t = (n.filter(pl.col("source") == 1).select(pl.col("r").alias("c"), "country_key",
                                                    pl.col("name_core").str.split(" ").list.unique().alias("t"))
           .explode("t").filter(pl.col("t").str.len_chars() > 0))
    df = s1t.group_by("country_key", "t").len().rename({"len": "df"})
    js = pairs.select("j").unique()
    rt = (js.join(n.select(pl.col("r").alias("j"), "country_key", pl.col("name_core").str.split(" ").list.unique().alias("t")),
                  on="j").explode("t").join(df, on=["country_key", "t"]).sort("df")
          .group_by("j", maintain_order=True).agg("country_key", pl.col("t").head(2), pl.col("df").first().alias("df1"))
          .with_columns(pl.col("country_key").list.first(), t1=pl.col("t").list.get(0),
                        t2=pl.col("t").list.get(1, null_on_oob=True)).drop("t"))
    post = rt.filter(pl.col("df1") <= max_df).join(s1t.rename({"t": "t1"}), on=["country_key", "t1"])
    both = post.join(s1t.rename({"t": "t2"}), on=["country_key", "c", "t2"], how="semi")
    sup = pl.concat([both.filter(pl.col("t2").is_not_null()),
                     post.filter(pl.col("t2").is_null())]).select("j", "c")
    cnt = sup.group_by("j").len().rename({"len": "sup_cnt"})
    out = (pairs.select("i", "j").join(cnt, on="j", how="left")
           .join(sup.select(pl.col("c").alias("i"), "j", pl.lit(1.0, pl.Float32).alias("sup_in")), on=["i", "j"], how="left")
           .join(rt.select("j", "df1"), on="j", how="left"))
    return out.select("i", "j", pl.col("sup_cnt").cast(pl.Float32), pl.col("sup_in").fill_null(0.0),
                      pl.col("df1").cast(pl.Float32).alias("sup_df1"))


PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=100, verbose=-1,
              num_threads=C.N_JOBS, seed=C.SEED)


def fold_of(col="j"):
    return (pl.col(col).hash(7) % C.N_FOLDS).cast(pl.Int32)


def train(a):
    if a.sup_only:  # add superset features to an existing owner_train.parquet
        n = load_norm("train", columns=["name_core", "source", "country_key"]).with_row_index("r")
        f = pl.read_parquet(os.path.join(W, "owner_train.parquet"))
        f = f.select([c for c in f.columns if not c.startswith("sup_")])
        f = f.join(superset_feats(f.select("i", "j"), n), on=["i", "j"], how="left")
        f.write_parquet(os.path.join(W, "owner_train.parquet"))
        print(f.select("sup_cnt", "sup_in").describe())
        return
    n, s1, m, rec = tables("train")
    ent = load_norm("train", columns=["entity_id"]).with_row_index("r")
    gt = (pl.read_csv(os.path.join(C.DATA_DIR, "train", "train_ground_truth.tsv"), separator="	")
          .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
          .filter(pl.col("matched_entity_ids") != "")
          .join(ent.rename({"r": "own", "entity_id": "source1_entity_id"}), on="source1_entity_id")
          .join(ent.rename({"r": "j", "entity_id": "matched_entity_ids"}), on="matched_entity_ids")
          .select("own", "j"))
    if a.sib:  # siblings = the S1's address-bearing true copies (test: its predicted address copies)
        global SIBT
        addr = n.filter((pl.col("source") != 1) & ~pl.col("noaddr")).select(pl.col("r").alias("j"))
        SIBT = sibling_table(n, gt.join(addr, on="j").select(pl.col("own").alias("o"), "j"))
    keep = pl.read_parquet(os.path.join(W, "train_oof.parquet"), columns=["i"])["i"].unique()
    # test has ~20% fewer S1s per S2/S3 record than train (0.172-0.181 vs 0.214): owners withheld from S1.
    # Hide a random 20% of the train S1s that are never scored (the train_oof sample stays visible).
    hid = s1.filter(~pl.col("c").is_in(keep.implode())).select("c").sample(fraction=a.hide, seed=C.SEED)
    s1 = s1.join(hid, on="c", how="anti")
    m = s1.group_by("country_key", "g").len().rename({"len": "m"})
    print(f"hidden S1s: {hid.height} | visible: {s1.height}")
    own =rec.select("j", "rcore").join(gt, on="j", how="left")
    gname = n.filter(pl.col("source") == 1).select(pl.col("r").alias("own"), "country_key", pl.col("name_core").alias("g"))
    # groups: the owner's name group (owner visible or not) and the record's own exact-name group
    g1 = own.join(gname, on="own").select("j", "country_key", "g")
    g2 = own.join(n.select(pl.col("r").alias("j"), "country_key"), on="j").select("j", "country_key", pl.col("rcore").alias("g"))
    groups = pl.concat([g1, g2]).unique()
    x = expand(groups, s1, m, rec).join(own.select("j", "own"), on="j", how="left")
    x = x.with_columns(y=(pl.col("c") == pl.col("own")).fill_null(False), fold=fold_of())
    print(f"owner-model rows: {x.height} | groups: {groups.height} | positives: {x['y'].sum()}")
    X, y, fo = fmat(x), x["y"].to_numpy(), x["fold"].to_numpy()
    models, oof = [], np.zeros(len(y))
    for f in range(C.N_FOLDS):
        tr = fo != f
        mdl = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), 300)
        oof[~tr] = mdl.predict(X[~tr])
        mdl.save_model(os.path.join(W, f"owner_model_{f}.txt"))
        models.append(mdl)
    xo = normalise(x, oof)
    top = xo.filter(pl.col("m") >= 2).sort("own_pn", descending=True).group_by("j", maintain_order=True).first()
    print(f"groups m>=2: top-1 acc {top['y'].mean():.4f} vs 1/m {(1 / top['m']).mean():.4f}")
    for th in (0.6, 0.7, 0.8, 0.9):
        s = top.filter(pl.col("own_pn") >= th)
        print(f"  pn >= {th}: {s.height} records, acc {s['y'].mean():.4f}")
    imp = sorted(zip(models[0].feature_importance("gain"), feat_names()), reverse=True)[:10]
    print("importance:", ", ".join(f"{k}:{v:.0f}" for v, k in imp))

    def pred_oof(xx):
        fo2 = xx.select(fold_of())["j"].to_numpy()
        Xx, out = fmat(xx), np.zeros(xx.height)
        for f in range(C.N_FOLDS):
            mk = fo2 == f
            if mk.any():
                out[mk] = models[f].predict(Xx[mk])
        return out

    pairs = (pl.read_parquet(os.path.join(W, "train_oof.parquet"), columns=["i", "j", "prob"])
             .filter(pl.col("prob") >= PMIN).join(rec.select("j"), on="j").select("i", "j"))
    feats = pair_features(pairs, s1, m, rec, pred_oof)
    feats.write_parquet(os.path.join(W, "owner_train.parquet"))
    print(f"-> work/owner_train.parquet ({feats.height} pairs, {feats['own_pn'].is_not_null().sum()} with own_pn)")


def test(a):
    n, s1, m, rec = tables("test")
    if a.sib:  # siblings = the S1's predicted address-bearing copies (a submission's pairs, (i, j) record indices)
        global SIBT
        pred = pl.read_parquet(os.path.join(W, a.sib_pairs)).select(pl.col("i").alias("o"), "j")
        addr = n.filter((pl.col("source") != 1) & ~pl.col("noaddr")).select(pl.col("r").alias("j"))
        SIBT = sibling_table(n, pred.join(addr, on="j"))
    models = [lgb.Booster(model_file=os.path.join(W, f"owner_model_{f}.txt")) for f in range(C.N_FOLDS)]

    def pred(xx):
        Xx = fmat(xx)
        return np.mean([mdl.predict(Xx) for mdl in models], axis=0)

    for src, dst in (("test_scored.parquet", "owner_test.parquet"), (a.unseen, "owner_test_unseen.parquet")):
        pairs = (pl.read_parquet(os.path.join(W, src), columns=["i", "j", "prob"]).filter(pl.col("prob") >= PMIN)
                 .with_columns(pl.col("i").cast(pl.UInt32), pl.col("j").cast(pl.UInt32))
                 .join(rec.select("j"), on="j").select("i", "j").unique())
        feats = pair_features(pairs, s1, m, rec, pred)
        feats.write_parquet(os.path.join(W, dst))
        print(f"-> work/{dst} ({feats.height} pairs, {feats['own_pn'].is_not_null().sum()} with own_pn)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--unseen", default="test_scored_unseen_mdeb_set4.parquet",
                    help="unseen-country (France) score file whose pairs also get owner features")
    ap.add_argument("--sup_only", action="store_true")
    ap.add_argument("--hide", type=float, default=0.2, help="share of unscored train S1s hidden while training")
    ap.add_argument("--sib", action="store_true", help="add entity-level sibling features (see sibling_table)")
    ap.add_argument("--sib_pairs", default="_run25_pairs.parquet", help="test: predicted pairs giving the siblings")
    a = ap.parse_args()
    train(a) if a.split == "train" else test(a)

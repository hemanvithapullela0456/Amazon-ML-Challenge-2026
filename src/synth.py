"""Approach A: labelled training data for a country that has no labels, by emulating the data generator.

The generator is the same in every country (identical matches-per-S1 distribution in US and India; France's
predictions follow it too), only the vocabulary differs. So the target country's own S1 records are used as
clean seed entities, and the noise the generator applies (read off the training pairs) produces:
  * copies  (label 1 for their seed): S2 style (UPPER, ##, NULL fields, [LEGAL], accents, typos) or S3 style
    (Title, "3142.", suffix words, l/1 swaps), name-only copies, renames, DBA/fka, website names, reordered /
    abbreviated / dropped address parts, house numbers with a dropped digit or letter suffix;
  * siblings (label 0): hidden businesses with no S1 record, derived from a seed - same street other number,
    other street in the same city, other name at the same address - each with its own noisy copies.
The synthetic world goes through the real pipeline (normalise -> keys -> blocking -> features), so its training
rows look exactly like the target country's real candidate pairs, with correct labels.

    python src/synth.py --split train --country india --out synth_india --feat_s1 120000
    python src/synth.py --split test  --country france --out synth_france --feat_s1 150000
-> work/<out>/{records.parquet, truth.parquet, norm.parquet, feats.parquet}
Uses only the TEXT of the seed S1 records (and of the target country's S2/S3 records, for region spellings);
never any label.
"""
import argparse
import os
import random
import re
import time

import numpy as np
import polars as pl
from joblib import Parallel, delayed

import config as C
from blocking import add_competition, build_keys, candidates, load_ranker
from features import make_features
from prep import _norm_batch, load_norm

# matches per S1 in training (US and India agree to 3 decimals)
N_COPIES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
P_COPIES = np.array([0.056, 0.055, 0.169, 0.241, 0.219, 0.147, 0.075, 0.028, 0.008, 0.002])
P_COPIES = P_COPIES / P_COPIES.sum()
ACCENT = {"a": "àâá", "e": "éèê", "i": "íî", "o": "ôó", "u": "úû", "c": "ç", "n": "ñ",
          "A": "ÀÂÁ", "E": "ÉÈÊ", "I": "ÍÎ", "O": "ÔÓ", "U": "ÚÛ", "C": "Ç"}
CONFUSE = {"l": "1", "I": "l", "i": "l", "o": "0", "O": "0", "s": "5", "e": "3"}
KEYB = "qwertyuiopasdfghjklzxcvbnm"
STREET_ABBR = {  # display abbreviations the sources use
    "street": ["St", "St.", "Str"], "road": ["Rd", "Rd."], "avenue": ["Ave", "Av", "Av.", "AVE"], "drive": ["Dr", "Dr."],
    "lane": ["Ln", "Ln."], "court": ["Ct", "Ct."], "boulevard": ["Blvd", "Bd", "Bd."], "place": ["Pl", "Pl."],
    "circle": ["Cir"], "highway": ["Hwy"], "parkway": ["Pkwy"], "terrace": ["Ter"], "trail": ["Trl"],
    "rue": ["R", "R.", "RUE"], "allée": ["All.", "All"], "allee": ["All.", "All"], "chemin": ["Ch.", "Chem."],
    "impasse": ["Imp.", "Imp"], "route": ["Rte", "Rte."], "quai": ["Qu."], "faubourg": ["Fbg", "Fg"],
    "nagar": ["Ngr"], "sector": ["Sec", "Sect"], "marg": ["Rd"], "near": ["Nr", "Nr."], "opposite": ["Opp", "Opp."],
    "floor": ["Fl", "Flr"], "building": ["Bldg"], "suite": ["Ste"], "apartment": ["Apt"],
}
SUFFIX = ["Services", "Group", "Center", "International", "Holdings", "Solutions", "Co", "Enterprises",
          "Groupe", "Distribution", "Participations", "Développement", "France", "India", "Global", "Systems"]
LEGAL_VARIANTS = {"llc": ["LLC", "L.L.C.", "Llc"], "inc": ["Inc", "Inc.", "INC", "Incorporated"],
                  "ltd": ["Ltd", "Ltd.", "Limited", "LTD."], "limited": ["Ltd", "Ltd.", "Limited", "LTD."],
                  "pvt": ["Pvt", "Private", "Pvt."], "private": ["Pvt", "Private", "Pvt."],
                  "corp": ["Corp", "Corp.", "Corporation"], "corporation": ["Corp", "Corp.", "Corporation"],
                  "sarl": ["SARL", "S.A.R.L.", "Sàrl", "Sarl"], "sas": ["SAS", "S.A.S.", "S.A.S", "Sas"],
                  "sa": ["SA", "S.A."], "eurl": ["EURL", "E.U.R.L."], "sci": ["SCI", "S.C.I."], "snc": ["SNC", "Snc"],
                  "sasu": ["SASU", "S.A.S.U."], "co": ["Co", "Co.", "Company"], "company": ["Co", "Co.", "Company"]}
OTHER_LEGAL = {"sarl": ["SAS", "SA", "SCI", "SNC", "EURL"], "sas": ["SARL", "SA", "SASU", "SCI"], "sa": ["SAS", "SARL"],
               "sci": ["SARL", "SAS", "SNC"], "snc": ["SARL", "SCI"], "eurl": ["SARL", "SASU"], "sasu": ["SAS", "EURL"],
               "llc": ["Inc", "Corp", "LP", "Ltd"], "inc": ["LLC", "Corp", "Co"], "corp": ["Inc", "LLC"],
               "ltd": ["Private Limited", "LLP"], "limited": ["Private Limited", "LLP"], "pvt": ["Limited", "LLP"],
               "private": ["Public", "LLP"]}
SYL = ["ver", "sol", "pyra", "nex", "brix", "faye", "nyla", "dova", "umbra", "vionex", "delta", "lum", "quo", "lyre",
       "lant", "ara", "zen", "kor", "vel", "tri", "mira", "vox", "ter", "nova", "quin", "rax", "ilo", "sen"]


# ------------------------------------------------------------------ string noise
def typo(w, r):
    if len(w) < 3:
        return w
    k = r.randrange(1, len(w) - 1)
    op = r.random()
    if op < 0.3:
        return w[:k] + r.choice(KEYB) + w[k + 1:]
    if op < 0.55:
        return w[:k] + w[k + 1:]
    if op < 0.8:
        return w[:k] + r.choice(KEYB) + w[k:]
    return w[:k - 1] + w[k] + w[k - 1] + w[k + 1:]


def noisy_word(w, r, p_typo=0.5, p_acc=0.35, p_conf=0.15):
    if r.random() < p_typo:
        w = typo(w, r)
    if r.random() < p_acc:
        idx = [k for k, ch in enumerate(w) if ch in ACCENT]
        if idx:
            k = r.choice(idx)
            w = w[:k] + r.choice(ACCENT[w[k]]) + w[k + 1:]
    if r.random() < p_conf:
        idx = [k for k, ch in enumerate(w) if ch in CONFUSE]
        if idx:
            k = r.choice(idx)
            w = w[:k] + CONFUSE[w[k]] + w[k + 1:]
    return w


def invented(r):
    return "".join(r.choice(SYL) for _ in range(r.choice([2, 3]))).capitalize()


def name_variant(name, r, style, vocab):
    toks = name.split()
    if not toks:
        return name
    low = [t.lower().strip(".,") for t in toks]
    legal_idx = [k for k, t in enumerate(low) if t in LEGAL_VARIANTS]
    core_idx = [k for k in range(len(toks)) if k not in legal_idx]
    # word-level edits
    if r.random() < 0.35 and core_idx:                        # typo / accent / confusion in one or two words
        for k in r.sample(core_idx, min(len(core_idx), r.choice([1, 1, 2]))):
            toks[k] = noisy_word(toks[k], r)
    if r.random() < 0.12 and len(core_idx) >= 3:              # drop a non-first word
        del toks[r.choice(core_idx[1:])]
        low = [t.lower().strip(".,") for t in toks]
        legal_idx = [k for k, t in enumerate(low) if t in LEGAL_VARIANTS]
    if r.random() < 0.08 and len(toks) >= 2:                  # swap two adjacent words
        k = r.randrange(len(toks) - 1)
        toks[k], toks[k + 1] = toks[k + 1], toks[k]
    if legal_idx and r.random() < 0.45:                       # legal form: rewrite / bracket / drop / move to front
        k = legal_idx[0]
        form = r.choice(LEGAL_VARIANTS[low[k]])
        op = r.random()
        if op < 0.35:
            toks[k] = form
        elif op < 0.6:
            toks[k] = f"[{form}]"
        elif op < 0.8:
            del toks[k]
        else:
            del toks[k]
            toks.insert(0, form)
    if r.random() < 0.12:                                     # suffix word
        toks.append(r.choice(SUFFIX if r.random() < 0.6 else vocab))
    out = " ".join(toks)
    if r.random() < 0.05:
        out = out.replace(" ", "-", 1)
    if r.random() < 0.08:
        out = out.replace(" ", "  ", 1)
    if r.random() < 0.01:
        out += f" (ID: {r.randrange(10000, 99999)})"
    # whole-name alternatives
    u = r.random()
    if u < 0.03:
        out = f"{invented(r)} {r.choice(['DBA', 'DBA:', 'fka', 'd/b/a', 'F/K/A'])} {out}"
    elif u < 0.055:
        out = re.sub(r"[^a-z0-9]", "", out.lower().encode("ascii", "ignore").decode()) + r.choice([".com", ".Com", ".com"])
    elif u < 0.095:
        out = invented(r)                                     # renamed copy, keeps the address
    # case by source style
    c = r.random()
    if style == 2:
        out = out.upper() if c < 0.45 else (out.lower() if c < 0.52 else out)
    else:
        out = out.lower() if c < 0.08 else (out.upper() if c < 0.12 else out)
    return out


def split_addr(addr):
    return [p.strip() for p in addr.split(",") if p.strip()]


def number_variant(part, r):
    m = re.search(r"\d+", part)
    if not m:
        return part
    n = m.group(0)
    op = r.random()
    if op < 0.35 and len(n) >= 2:
        k = r.randrange(len(n))
        new = n[:k] + n[k + 1:]                               # a digit dropped (909 -> 90, 677 -> 77)
    elif op < 0.55:
        new = n + r.choice(["B", "b", "A", " bis", "C"])      # letter suffix
    elif op < 0.75:
        new = str(int(n) + r.choice([-1, 1, 100, -100])) if int(n) > 100 else n  # a near/transposed number
    else:
        new = str(r.randrange(1, 9999))                       # an unrelated number
    return part[:m.start()] + new + part[m.end():]


def abbreviate(part, r):
    def rep(mt):
        w = mt.group(0)
        opts = STREET_ABBR.get(w.lower())
        return r.choice(opts) if opts and r.random() < 0.7 else w
    return re.sub(r"[A-Za-zÀ-ÿ]+", rep, part)


def addr_variant(addr, r, style, region_alts):
    parts = split_addr(addr)
    if not parts:
        return ""
    num_k = next((k for k, p in enumerate(parts) if re.search(r"\d", p)), None)
    if r.random() < 0.55:
        parts = [abbreviate(p, r) for p in parts]
    if num_k is not None:
        p = parts[num_k]
        if r.random() < 0.05:
            p = number_variant(p, r)
        if r.random() < 0.3:
            pre = r.choice(["##", "#", "# ", "No. ", "N°", "Nº ", "No "] if style == 2 else ["#", "N°", "No. ", "(", "0", "00"])
            p = re.sub(r"(\d+)", (lambda m: f"({m.group(1)})") if pre == "(" else (lambda m: pre + m.group(1)), p, count=1)
        elif style == 3 and r.random() < 0.1:
            p = re.sub(r"(\d+)", lambda m: m.group(1) + ".", p, count=1)
        if r.random() < 0.12:                                 # a street-word typo
            ws = p.split()
            cand = [k for k, w in enumerate(ws) if w.isalpha() and len(w) >= 4]
            if cand:
                k = r.choice(cand)
                ws[k] = typo(ws[k], r)
                p = " ".join(ws)
        parts[num_k] = p
    last = len(parts) - 1
    if len(parts) >= 3 and r.random() < 0.3:                  # region: dropped or spelled another way
        if region_alts and r.random() < 0.5:
            parts[last] = r.choice(region_alts)
        else:
            del parts[last]
    if len(parts) >= 3 and r.random() < 0.08:                 # a middle part dropped (landmark, locality)
        del parts[r.randrange(1, len(parts) - 1)]
    if r.random() < 0.45:
        r.shuffle(parts)
    if style == 2 and r.random() < 0.08:
        parts.insert(r.randrange(len(parts) + 1), r.choice(["NULL", "<NULL>", "N/A"]))
    out = ", ".join(parts)
    if style == 2:
        out = out.upper() if r.random() < 0.8 else out
    return out


# ------------------------------------------------------------------ siblings (hidden look-alike businesses)
def sibling(name, addr, r, vocab, streets):
    toks = name.split()
    parts = split_addr(addr)
    num_k = next((k for k, p in enumerate(parts) if re.search(r"\d", p)), None)
    kind = r.random()
    if kind < 0.45 and num_k is not None:                     # same street, another house number
        def bump(m):
            n = int(m.group(0))
            return str(max(1, n + r.choice([-3, -2, -1, 1, 2, 3, 5, 10, 11]))) if r.random() < 0.7 else str(r.randrange(1, 400))
        parts[num_k] = re.sub(r"\d+", bump, parts[num_k], count=1)
    elif kind < 0.75 and num_k is not None and streets:       # another street in the same city
        parts[num_k] = r.choice(streets)
    else:                                                     # same address, another business
        return (invented(r) if r.random() < 0.5 else " ".join([r.choice(vocab).capitalize()] + toks[1:] or toks)), addr
    # the look-alike's name: one word swapped for a common word of this country, or the legal form changed
    low = [t.lower().strip(".,") for t in toks]
    u = r.random()
    core = [k for k, t in enumerate(low) if t not in LEGAL_VARIANTS]
    if u < 0.55 and len(core) >= 2:
        toks[r.choice(core[1:])] = r.choice(vocab).capitalize()
    elif u < 0.8:
        leg = [k for k, t in enumerate(low) if t in OTHER_LEGAL]
        if leg:
            toks[leg[0]] = r.choice(OTHER_LEGAL[low[leg[0]]])
        elif core:
            toks.append(r.choice(SUFFIX))
    return " ".join(toks), ", ".join(parts)


# ------------------------------------------------------------------ world builder
def _generate(chunk, seed, vocab, streets_by_city, region_alts_by_city, p_sib):
    r = random.Random(seed)
    npr = np.random.default_rng(seed)
    recs, truth = [], []
    for eid, name, addr, country in chunk:
        parts = split_addr(addr)
        city = parts[-2].lower() if len(parts) >= 2 else ""
        ralts = region_alts_by_city.get(city, [])
        k = int(npr.choice(N_COPIES, p=P_COPIES))
        for c in range(k):
            style = 2 if r.random() < 0.6 else 3
            nm = name_variant(name, r, style, vocab)
            ad = "" if r.random() < 0.045 else addr_variant(addr, r, style, ralts)
            cid = f"S{style}-x{eid}-{c}"
            recs.append((cid, nm, ad, country, style))
            truth.append((eid, cid))
        if r.random() < p_sib:
            for s in range(r.choice([1, 1, 1, 2, 2, 3])):
                sn, sa = sibling(name, addr, r, vocab, streets_by_city.get(city, []))
                for c in range(r.choice([1, 1, 2, 2, 3])):
                    style = 2 if r.random() < 0.6 else 3
                    nm = name_variant(sn, r, style, vocab)
                    ad = "" if r.random() < 0.045 else addr_variant(sa, r, style, ralts)
                    recs.append((f"S{style}-y{eid}-{s}-{c}", nm, ad, country, style))
    return recs, truth


def build(a):
    t = time.time()
    W = C.WORK_DIR
    out = os.path.join(W, a.out)
    os.makedirs(out, exist_ok=True)
    raw = load_norm(a.split, columns=["entity_id", "business_name", "business_address", "country", "country_key", "source"])
    tgt = raw.filter(pl.col("country_key") == a.country)
    seeds = tgt.filter(pl.col("source") == 1)
    if a.max_seeds and seeds.height > a.max_seeds:
        seeds = seeds.sample(a.max_seeds, seed=C.SEED)
    # country vocabulary, streets per city and alternative region spellings, all read off the TEXT
    words = (seeds.select(pl.col("business_name").str.to_lowercase().str.extract_all(r"[a-zà-ÿ]{4,}").alias("w"))
             .explode("w").drop_nulls().group_by("w").len().sort("len", descending=True))
    vocab = [w for w in words["w"].head(400).to_list() if w not in LEGAL_VARIANTS]
    sp = seeds.select(pl.col("business_address").str.split(",").alias("p")).with_columns(
        pl.col("p").list.eval(pl.element().str.strip_chars()))
    streets_by_city, city_set = {}, set()
    for parts in sp["p"].to_list():
        parts = [p for p in parts if p]
        if len(parts) < 2:
            continue
        city = parts[-2].lower()
        city_set.add(city)
        st = next((p for p in parts if re.search(r"\d", p)), None)
        if st:
            lst = streets_by_city.setdefault(city, [])
            if len(lst) < 300:
                lst.append(st)
    region_alts = {}
    other = tgt.filter(pl.col("source") != 1)
    other = other.sample(min(400_000, other.height), seed=C.SEED)
    for addr in other["business_address"].to_list():
        parts = [p.strip() for p in (addr or "").split(",") if p.strip()]
        low = [p.lower() for p in parts]
        cities = [c for c in low if c in city_set]
        if not cities:
            continue
        for p, lp in zip(parts, low):
            if lp not in city_set and not re.search(r"\d", p) and len(p) < 30:
                lst = region_alts.setdefault(cities[0], [])
                if len(lst) < 50:
                    lst.append(p)
    print(f"[{a.out}] {seeds.height} seeds | vocab {len(vocab)} | cities {len(city_set)} | "
          f"cities with alt regions {len(region_alts)} ({time.time() - t:.0f}s)", flush=True)

    rows = list(zip(seeds["entity_id"].to_list(), seeds["business_name"].to_list(),
                    seeds["business_address"].to_list(), seeds["country"].to_list()))
    step = 20_000
    res = Parallel(n_jobs=C.N_JOBS)(delayed(_generate)(rows[s:s + step], C.SEED + s, vocab, streets_by_city,
                                                       region_alts, a.p_sib) for s in range(0, len(rows), step))
    syn = [x for r_, _ in res for x in r_]
    truth = pl.DataFrame([x for _, t_ in res for x in t_], schema=["s1_id", "cid"], orient="row")
    s1 = seeds.select("entity_id", "business_name", "business_address", "country", pl.lit(1, dtype=pl.Int8).alias("source"))
    s23 = pl.DataFrame(syn, schema=["entity_id", "business_name", "business_address", "country", "source"], orient="row") \
        .with_columns(pl.col("source").cast(pl.Int8))
    records = pl.concat([s1, s23])
    print(f"[{a.out}] {s1.height} S1 + {s23.height} synthetic S2/S3 (copies {truth.height}, "
          f"look-alikes {s23.height - truth.height}) ({time.time() - t:.0f}s)", flush=True)
    records.write_parquet(os.path.join(out, "records.parquet"))
    truth.write_parquet(os.path.join(out, "truth.parquet"))

    # the real normalisation + keys, in parallel
    nm, ad = records["business_name"].to_list(), records["business_address"].to_list()
    b = 50_000
    norm = pl.concat(Parallel(n_jobs=C.N_JOBS)(delayed(_norm_batch)(nm[s:s + b], ad[s:s + b]) for s in range(0, len(nm), b)))
    norm = pl.concat([records, norm], how="horizontal").with_columns(
        pl.col("country").str.strip_chars().str.to_lowercase().alias("country_key")).with_row_index("r").with_columns(
        pl.col("r").cast(pl.UInt32))
    norm.write_parquet(os.path.join(out, "norm.parquet"))
    print(f"[{a.out}] normalised ({time.time() - t:.0f}s)", flush=True)

    recs = norm.select("r", "entity_id", "source", "country_key", "keys", "addr_core").with_columns(pl.lit(0, dtype=pl.UInt8).alias("cc"))
    keys = build_keys(recs)
    s1_rows = recs.filter(pl.col("source") == 1)["r"].to_numpy()
    pairs, _ = candidates(keys, s1_rows, k=a.k, ranker=load_ranker())
    del keys
    pairs = add_competition(pairs)
    rng = np.random.default_rng(C.SEED)
    fs = np.sort(rng.choice(s1_rows, min(a.feat_s1, len(s1_rows)), replace=False))
    pairs = pairs.filter(pl.col("i").is_in(fs))
    print(f"[{a.out}] blocking: {pairs.height} pairs for {len(fs)} S1 ({time.time() - t:.0f}s)", flush=True)

    ids = norm.select("r", "entity_id")
    tr = (truth.join(ids.rename({"r": "i", "entity_id": "s1_id"}), on="s1_id")
               .join(ids.rename({"r": "j", "entity_id": "cid"}), on="cid").select("i", "j"))
    ntab = norm.select("r", "source", "country_key", "name_norm", "name_core", "name_alias", "name_acr", "name_legal",
                       "addr_core", "postcode", "addr_nums", "landmark", "region")
    feats = make_features(pairs, ntab)
    feats = (feats.join(tr.with_columns(pl.lit(True).alias("label")), on=["i", "j"], how="left")
                  .with_columns(pl.col("label").fill_null(False), pl.lit(a.country).alias("country_key")))
    n_true = tr.filter(pl.col("i").is_in(fs)).height
    print(f"[{a.out}] features {feats.height} pairs | positives {int(feats['label'].sum())} | "
          f"synthetic candidate recall {feats['label'].sum() / max(n_true, 1):.4f} ({time.time() - t:.0f}s)", flush=True)
    feats.write_parquet(os.path.join(out, "feats.parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--country", default="india")
    ap.add_argument("--out", default="synth_india")
    ap.add_argument("--max_seeds", type=int, default=0)
    ap.add_argument("--feat_s1", type=int, default=120_000)
    ap.add_argument("--p_sib", type=float, default=0.55)
    ap.add_argument("--k", type=int, default=30)
    build(ap.parse_args())

"""Transductive knowledge-graph matcher (Phase 1).

Knowledge graph over ALL records of a bundle (source country with labels + unseen target country without):
    record --has_name_word / has_addr_word / has_number--> token ;  token --subword--> char-3gram
Token meaning is learned from the graph itself, per domain:
  * rarity: each token carries its document frequency WITHIN ITS OWN DOMAIN (target stats come from target records)
  * context: every refresh, each token gets the mean embedding of the records that contain it (record -> token ->
    record message passing over the whole graph, target records included), so a target-country abbreviation that
    lives in the same record neighbourhoods as its long form ends up close to it without any labels.
Record encoder: 2-layer transformer over [word + char-3gram + graph context + field + rarity] token vectors.
Pair scorer = mixture of experts with a learned gate:
  E1 embedding expert  MLP[h_a, h_b, |h_a-h_b|, h_a*h_b]
  E2 alignment expert  soft token alignment (per-field coverage both ways, number agreement) -> MLP
Losses: BCE on labelled source pairs + contrastive self-supervision on ALL records (SelfKG / Sudowoodo style:
two corrupted views of a record must find each other among in-batch negatives) [+ optional teacher pseudo-labels].
Refs: EmbDI (SIGMOD'20), Contextual Graph Embeddings (2025), SelfKG (WWW'22), Sudowoodo (ICDE'23), HGT, MoE.

GPU:  python src/kg_matcher.py --bundle b_proxy --out out_kg --tag sim5_kg      (France-like simulation)
      python src/kg_matcher.py --bundle b_france --out out_kg --tag fr5_kg      (France)
Writes <out>/<tag>_band.parquet (ka, kb, ce) and <tag>_extra.parquet, the same format as the students.
"""
import argparse
import os
import re
import time
import zlib

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

L, G, NB = 32, 6, 1 << 20          # tokens per record, char-grams per token, gram hash buckets
WORD = re.compile(r"[^\W_]+", re.UNICODE)


# ------------------------------------------------------------------ graph construction
def tokenize(text):
    name, _, addr = (text or "").partition(" | ")
    out = []
    for fld, s in ((0, name), (1, addr)):
        for w in WORD.findall(s.lower()):
            f = 2 if any(c.isdigit() for c in w) else fld
            out.append((f, w))
    return out[:L]


def grams(w):
    s = f"<{w}>"
    return [zlib.crc32(s[k:k + 3].encode()) % (NB - 1) + 1 for k in range(max(1, len(s) - 2))][:G]   # 0 = padding


class Graph:
    """padded token arrays for every record + domain-specific rarity + record-token incidence."""

    def __init__(self, keys, texts, domain):
        t0 = time.time()
        self.key_ix = {k: n for n, k in enumerate(keys)}
        N = len(keys)
        vocab = {}
        self.word = np.zeros((N, L), np.int64)
        self.field = np.full((N, L), 3, np.int64)          # 3 = padding
        self.gram = np.zeros((N, L, G), np.int64)
        for n, tx in enumerate(texts):
            for p, (f, w) in enumerate(tokenize(tx)):
                wid = vocab.setdefault((min(f, 2), w), len(vocab) + 1)
                self.word[n, p], self.field[n, p] = wid, f
                g = grams(w)
                self.gram[n, p, :len(g)] = g
        self.V = len(vocab) + 1
        self.mask = self.field < 3
        # document frequency of each token within each domain -> log relative frequency per (record, position)
        self.domain = np.asarray(domain, np.int64)
        self.logdf = np.zeros((N, L), np.float32)
        for d in np.unique(self.domain):
            rows = np.where(self.domain == d)[0]
            m = self.mask[rows]
            key = np.unique(np.repeat(rows[:, None], L, 1)[m].astype(np.int64) * self.V + self.word[rows][m])
            df = np.bincount(key % self.V, minlength=self.V)
            self.logdf[rows] = np.log((df[self.word[rows]] + 1) / (len(rows) + 1)) * m
        r, p = np.nonzero(self.mask)
        self.inc_r, self.inc_t = r, self.word[r, p]
        self.mask[:, 0] = True   # empty records keep one (padding) position so attention never sees an all-masked row
        print(f"graph: {N} records, {self.V} tokens, {len(r)} record-token edges, domains "
              f"{dict(zip(*np.unique(self.domain, return_counts=True)))} ({time.time() - t0:.0f}s)", flush=True)

    def batch(self, idx, dev, drop=0.0, rng=None):
        w, f, g, ld = (torch.from_numpy(a[idx]) for a in (self.word, self.field, self.gram, self.logdf))
        m = torch.from_numpy(self.mask[idx])
        if drop > 0:   # corrupted view: drop tokens, truncate some words to a prefix (abbreviation-like)
            keep = torch.from_numpy(rng.random(m.shape) > drop) & m
            keep[:, 0] |= m[:, 0]
            m = keep
            trunc = torch.from_numpy(rng.random(m.shape) < 0.1) & m
            g = g.clone(); g[..., 2:] = torch.where(trunc.unsqueeze(-1), torch.zeros_like(g[..., 2:]), g[..., 2:])
        return w.to(dev), f.to(dev), g.to(dev), ld.to(dev), m.to(dev)


# ------------------------------------------------------------------ model
class KGMatcher(nn.Module):
    def __init__(self, V, d=192):
        super().__init__()
        self.wemb = nn.Embedding(V, d, padding_idx=0)
        self.gemb = nn.EmbeddingBag(NB, d, mode="sum", padding_idx=0)
        self.femb = nn.Embedding(4, d)
        self.df_proj = nn.Linear(1, d)
        self.ctx_proj = nn.Linear(d, d)
        self.register_buffer("ctx", torch.zeros(V, d))    # graph context per token (refreshed, no grad)
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 2)
        self.pool = nn.Linear(d, 1)
        self.e1 = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.e2 = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(8, 32), nn.GELU(), nn.Linear(32, 2))
        self.imp = nn.Linear(d, 1)   # token importance for alignment
        self.d = d

    def base(self, w, g):
        B, T, Gn = g.shape
        return self.wemb(w) + self.gemb(g.reshape(-1, Gn)).reshape(B, T, -1)

    def tokens(self, w, f, g, ld, m):
        x = self.base(w, g) + self.ctx_proj(self.ctx[w]) + self.femb(f) + self.df_proj(ld.unsqueeze(-1))
        x = self.enc(x, src_key_padding_mask=~m)
        a = self.pool(x).squeeze(-1).masked_fill(~m, -1e4).softmax(-1)
        h = F.normalize((a.unsqueeze(-1) * x).sum(1), dim=-1)
        return x, h

    @torch.no_grad()
    def refresh_context(self, graph, dev, bs=8192):
        """record -> token message passing over the WHOLE graph: ctx[t] = mean pooled base vector of records with t"""
        N = graph.word.shape[0]
        rec = torch.zeros(N, self.d, device=dev)
        for s in range(0, N, bs):
            idx = np.arange(s, min(N, s + bs))
            w, f, g, ld, m = graph.batch(idx, dev)
            wt = (-ld).clamp(min=0) * m                          # rarer tokens weigh more (domain-specific)
            v = self.base(w, g)
            rec[idx] = F.normalize((wt.unsqueeze(-1) * v).sum(1) / wt.sum(1, keepdim=True).clamp(min=1e-6), dim=-1)
        ctx = torch.zeros_like(self.ctx)
        cnt = torch.zeros(self.ctx.shape[0], device=dev)
        for s in range(0, len(graph.inc_r), 1_000_000):   # edge chunks keep GPU memory small
            r = torch.from_numpy(graph.inc_r[s:s + 1_000_000]).to(dev); t = torch.from_numpy(graph.inc_t[s:s + 1_000_000]).to(dev)
            ctx.index_add_(0, t, rec[r]); cnt.index_add_(0, t, torch.ones_like(t, dtype=torch.float))
        self.ctx.copy_(ctx / cnt.clamp(min=1).unsqueeze(-1))

    def pair(self, A, B):
        (xa, ha, fa, ma, lda), (xb, hb, fb, mb, ldb) = A, B
        e1 = self.e1(torch.cat([ha, hb, (ha - hb).abs(), ha * hb], -1)).squeeze(-1)
        na, nb = F.normalize(xa, dim=-1), F.normalize(xb, dim=-1)
        S = torch.einsum("bid,bjd->bij", na, nb)
        feats = []
        for fld in (0, 1, 2):
            sa = S.masked_fill(~(mb & (fb == fld)).unsqueeze(1), -1).max(-1).values      # best match of each a-token
            sb = S.masked_fill(~(ma & (fa == fld)).unsqueeze(2), -1).max(1).values
            wa = (self.imp(xa).squeeze(-1).clamp(max=8).exp() * (ma & (fa == fld)))
            wb = (self.imp(xb).squeeze(-1).clamp(max=8).exp() * (mb & (fb == fld)))
            feats += [(wa * sa).sum(1) / wa.sum(1).clamp(min=1e-6), (wb * sb).sum(1) / wb.sum(1).clamp(min=1e-6),
                      ((ma & (fa == fld)).any(1) & (mb & (fb == fld)).any(1)).float()]
        z = torch.stack(feats, -1)
        e2 = self.e2(z).squeeze(-1)
        gin = torch.stack([(ma & (fa == k)).sum(1).float() / 8 for k in range(3)] + [(mb & (fb == k)).sum(1).float() / 8 for k in range(3)]
                          + [lda.sum(1) / ma.sum(1).clamp(min=1), ldb.sum(1) / mb.sum(1).clamp(min=1)], -1)
        gw = self.gate(gin).softmax(-1)
        return gw[:, 0] * e1 + gw[:, 1] * e2

    def encode(self, graph, idx, dev, drop=0.0, rng=None):
        w, f, g, ld, m = graph.batch(idx, dev, drop, rng)
        x, h = self.tokens(w, f, g, ld, m)
        return x, h, f, m, ld


# ------------------------------------------------------------------ training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="out_kg")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--cl_w", type=float, default=0.5, help="contrastive self-supervision weight")
    ap.add_argument("--pseudo_w", type=float, default=0.0, help="teacher pseudo-label weight (0 = pure KG)")
    ap.add_argument("--refresh", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    os.makedirs(a.out, exist_ok=True)
    B = a.bundle
    src = pl.read_parquet(f"{B}/src_pairs.parquet")
    tgt = pl.read_parquet(f"{B}/tgt_pairs.parquet")
    ps = pl.read_parquet(f"{B}/tgt_pseudo.parquet")
    ex = pl.read_parquet(f"{B}/extra_pairs.parquet") if os.path.exists(f"{B}/extra_pairs.parquet") else None
    tx = pl.read_parquet(f"{B}/texts.parquet")
    src_keys = set(src["ka"].to_list()) | set(src["kb"].to_list())
    keys = tx["k"].to_list()
    graph = Graph(keys, tx["text"].to_list(), [0 if k in src_keys else 1 for k in keys])
    ix = graph.key_ix
    to_ix = lambda df: (np.array([ix[k] for k in df["ka"].to_list()]), np.array([ix[k] for k in df["kb"].to_list()]))
    sa, sb = to_ix(src); sy = src["label"].to_numpy().astype(np.float32)
    pa, pb = to_ix(ps); py = ps["pl_label"].to_numpy().astype(np.float32)
    tgt_rows = np.where(graph.domain == 1)[0]
    model = KGMatcher(graph.V).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    t0, run = time.time(), []
    for step in range(a.steps):
        if step % a.refresh == 0:
            model.eval(); model.refresh_context(graph, dev); model.train()
        k = rng.integers(0, len(sy), a.bs)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model.pair(model.encode(graph, sa[k], dev), model.encode(graph, sb[k], dev))
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(sy[k]).to(dev))
            if a.pseudo_w > 0:
                q = rng.integers(0, len(py), a.bs)
                lp = model.pair(model.encode(graph, pa[q], dev), model.encode(graph, pb[q], dev))
                loss = loss + a.pseudo_w * F.binary_cross_entropy_with_logits(lp.float(), torch.from_numpy(py[q]).to(dev))
            if a.cl_w > 0:   # half the contrastive batch from the unseen target domain
                r = np.concatenate([rng.integers(0, graph.word.shape[0], a.bs), rng.choice(tgt_rows, a.bs)])
                _, h1, *_ = model.encode(graph, r, dev, 0.2, rng)
                _, h2, *_ = model.encode(graph, r, dev, 0.2, rng)
                sim = h1.float() @ h2.float().T / 0.07
                lab = torch.arange(len(r), device=dev)
                loss = loss + a.cl_w * 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
        run.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}/{a.steps} loss {np.mean(run[-500:]):.4f} ({time.time() - t0:.0f}s)", flush=True)

    model.eval(); model.refresh_context(graph, dev)

    def score(df):
        xa, xb = to_ix(df)
        out = np.zeros(len(xa), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(xa), 2048):
                out[s:s + 2048] = model.pair(model.encode(graph, xa[s:s + 2048], dev),
                                             model.encode(graph, xb[s:s + 2048], dev)).float().cpu().numpy()
        return df.select("ka", "kb").with_columns(pl.Series("ce", out))

    ts = score(tgt)
    ts.write_parquet(f"{a.out}/{a.tag}_band.parquet")
    if "label" in tgt.columns and tgt["label"].n_unique() > 1:
        from sklearn.metrics import roc_auc_score
        y = tgt["label"].to_numpy()
        lat = tgt["latin"].to_numpy() if "latin" in tgt.columns else np.ones(len(y), bool)
        print(f"target band AUC: KG {roc_auc_score(y[lat], ts['ce'].to_numpy()[lat]):.4f} | "
              f"LightGBM {roc_auc_score(y[lat], tgt['p_lgbm'].to_numpy()[lat]):.4f}", flush=True)
    if ex is not None:
        score(ex).write_parquet(f"{a.out}/{a.tag}_extra.parquet")
    print(f"saved {a.out}/{a.tag}_band.parquet ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
